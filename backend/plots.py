from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
import plotly.io as pio  # noqa: E402
import plotly.express as px  # noqa: E402

import shap  # noqa: E402
import lightgbm as lgb  # noqa: E402
from . import model as mdl  # noqa: E402
from .database import get_connection  # noqa: E402
from statsmodels.tsa.stattools import ccf, acf, pacf  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402


STATIC_DIR = Path(__file__).resolve().parent / "static"
GEN_DIR = STATIC_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)


# Plotly defaults for consistent styling and sizing across all charts
PLOTLY_TEMPLATE = "plotly_white"
DEFAULT_WIDTH = 1100
DEFAULT_HEIGHT = 600
DEFAULT_MARGIN = dict(l=60, r=30, t=60, b=60, pad=2)
DEFAULT_SCALE = 2

# Apply a global default template
pio.templates.default = PLOTLY_TEMPLATE


def _to_json(fig: go.Figure) -> Dict[str, Any]:
    """Convert Plotly figure to JSON-compatible dict."""
    return json.loads(fig.to_json())


def generate_eda_images(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate EDA plots as Plotly JSON objects."""
    out: Dict[str, Any] = {}

    # 1) Master time series plot
    y = df["daily_sales"].astype(float)
    y_roll = y.rolling(30).mean()
    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=df.index,
            y=y,
            mode="lines",
            name="Daily Sales",
            line=dict(width=2),
            opacity=0.85,
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=df.index,
            y=y_roll,
            mode="lines",
            name="30D Rolling Mean",
            line=dict(width=2, dash="dash"),
        )
    )
    fig_ts.update_layout(
        title="Daily E-commerce Sales",
        xaxis_title="Date",
        yaxis_title="USD",
        template=PLOTLY_TEMPLATE,
    )
    out["historical"] = _to_json(fig_ts)

    # 1a) Distribution of daily sales (histogram)
    fig_hist = px.histogram(df, x="daily_sales", nbins=50, opacity=0.9)
    sales_data = df["daily_sales"].dropna().astype(float)
    kde = gaussian_kde(sales_data)
    x_kde = np.linspace(sales_data.min(), sales_data.max(), 200)
    y_kde = kde(x_kde)
    n = len(sales_data)
    bin_width = (sales_data.max() - sales_data.min()) / 50
    y_kde_scaled = y_kde * n * bin_width
    fig_hist.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde_scaled,
            mode="lines",
            name="Trend",
            line=dict(color="red", width=2),
        )
    )
    fig_hist.update_layout(
        title="Distribution of Daily Sales",
        xaxis_title="Daily Sales (USD)",
        yaxis_title="Frequency",
    )
    out["sales_distribution"] = _to_json(fig_hist)

    # 2) Day-of-week boxplot
    if "day_of_week" in df.columns:
        order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        tmp = df.reset_index()
        fig_box_dow = px.box(
            tmp,
            x="day_of_week",
            y="daily_sales",
            category_orders={"day_of_week": order},
        )
        fig_box_dow.update_layout(
            title="Sales by Day of Week",
            xaxis_title="Day of Week",
            yaxis_title="Daily Sales",
        )
        out["dow_boxplot"] = _to_json(fig_box_dow)

    # 3) Holiday vs non-holiday boxplot
    if "is_holiday" in df.columns:
        tmp = df.reset_index()
        fig_box_h = px.box(tmp, x="is_holiday", y="daily_sales")
        fig_box_h.update_layout(
            title="Sales: Holiday (1) vs Regular Day (0)",
            xaxis_title="is_holiday",
            yaxis_title="Daily Sales",
        )
        out["holiday_boxplot"] = _to_json(fig_box_h)

    # 4) STL decomposition (weekly)
    stl = STL(df["daily_sales"].astype(float), period=7)
    result = stl.fit()
    fig_stl = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    )
    fig_stl.add_trace(
        go.Scatter(x=df.index, y=result.observed, mode="lines", name="Observed"),
        row=1,
        col=1,
    )
    fig_stl.add_trace(
        go.Scatter(x=df.index, y=result.trend, mode="lines", name="Trend"), row=2, col=1
    )
    fig_stl.add_trace(
        go.Scatter(x=df.index, y=result.seasonal, mode="lines", name="Seasonal"),
        row=3,
        col=1,
    )
    fig_stl.add_trace(
        go.Scatter(x=df.index, y=result.resid, mode="lines", name="Residual"),
        row=4,
        col=1,
    )
    fig_stl.update_layout(title="STL Decomposition", xaxis4_title="Date")
    out["stl"] = _to_json(fig_stl)

    # 5) ACF/PACF on original (non-stationary) sales
    series = df["daily_sales"].astype(float).dropna()
    lags = 40
    acf_vals = acf(series, nlags=lags, fft=False)
    pacf_vals = pacf(series, nlags=lags, method="ywm")
    conf = 1.96 / (len(series) ** 0.5)
    xlags = list(range(0, lags + 1))
    fig_acfpacf_raw = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("ACF of Original Sales Data", "PACF of Original Sales Data"),
    )
    fig_acfpacf_raw.add_trace(go.Bar(x=xlags, y=acf_vals, name="ACF"), row=1, col=1)
    fig_acfpacf_raw.add_hline(y=conf, line_dash="dash", line_color="red", row=1, col=1)
    fig_acfpacf_raw.add_hline(y=-conf, line_dash="dash", line_color="red", row=1, col=1)
    fig_acfpacf_raw.add_trace(go.Bar(x=xlags, y=pacf_vals, name="PACF"), row=2, col=1)
    fig_acfpacf_raw.add_hline(y=conf, line_dash="dash", line_color="red", row=2, col=1)
    fig_acfpacf_raw.add_hline(y=-conf, line_dash="dash", line_color="red", row=2, col=1)
    fig_acfpacf_raw.update_xaxes(title_text="Lag", row=2, col=1)
    fig_acfpacf_raw.update_yaxes(title_text="Correlation", row=1, col=1)
    fig_acfpacf_raw.update_yaxes(title_text="Correlation", row=2, col=1)
    out["acf_pacf_raw"] = _to_json(fig_acfpacf_raw)

    # 6) CCF after differencing (ΔSales vs ΔMarketing)
    sales_diff = df["daily_sales"].diff(1).dropna()
    marketing_diff = df["marketing_spend"].diff(1).dropna()
    n = int(min(len(sales_diff), len(marketing_diff)))
    sales_diff = sales_diff.iloc[-n:]
    marketing_diff = marketing_diff.iloc[-n:]
    vals = ccf(sales_diff.values, marketing_diff.values, adjusted=False)
    vals = vals[: (lags + 1)]
    xlags = list(range(0, len(vals)))
    fig_ccf = go.Figure()
    fig_ccf.add_trace(go.Bar(x=xlags, y=vals, name="CCF"))
    fig_ccf.add_hline(y=0, line_color="black", line_width=1)
    conf_ccf = 1.96 / (n**0.5)
    fig_ccf.add_hline(y=conf_ccf, line_dash="dash", line_color="red")
    fig_ccf.add_hline(y=-conf_ccf, line_dash="dash", line_color="red")
    fig_ccf.update_layout(
        title="Cross-Correlation: ΔSales(t) vs. ΔMarketing(t-k)",
        xaxis_title="Lag (k) in Days",
        yaxis_title="Cross-Correlation",
    )
    out["ccf_diff"] = _to_json(fig_ccf)

    # 7) ACF/PACF on stationary series (d=1, D=1, m=7)
    sales_d1 = df["daily_sales"].diff(1)
    sales_d1_D1 = sales_d1.diff(7)
    stationary = sales_d1_D1.dropna()
    acf_vals_s = acf(stationary, nlags=lags, fft=False)
    pacf_vals_s = pacf(stationary, nlags=lags, method="ywm")
    conf_s = 1.96 / (len(stationary) ** 0.5)
    xlags = list(range(0, lags + 1))
    fig_acfpacf_s = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "ACF of Stationary Sales (d=1, D=1, m=7)",
            "PACF of Stationary Sales (d=1, D=1, m=7)",
        ),
    )
    fig_acfpacf_s.add_trace(go.Bar(x=xlags, y=acf_vals_s, name="ACF"), row=1, col=1)
    fig_acfpacf_s.add_hline(y=conf_s, line_dash="dash", line_color="red", row=1, col=1)
    fig_acfpacf_s.add_hline(y=-conf_s, line_dash="dash", line_color="red", row=1, col=1)
    fig_acfpacf_s.add_trace(go.Bar(x=xlags, y=pacf_vals_s, name="PACF"), row=2, col=1)
    fig_acfpacf_s.add_hline(y=conf_s, line_dash="dash", line_color="red", row=2, col=1)
    fig_acfpacf_s.add_hline(y=-conf_s, line_dash="dash", line_color="red", row=2, col=1)
    fig_acfpacf_s.update_xaxes(title_text="Lag", row=2, col=1)
    fig_acfpacf_s.update_yaxes(title_text="Correlation", row=1, col=1)
    fig_acfpacf_s.update_yaxes(title_text="Correlation", row=2, col=1)
    out["acf_pacf_stationary"] = _to_json(fig_acfpacf_s)

    return out


def generate_lgbm_explainability_images(
    model: lgb.LGBMRegressor, X_train: pd.DataFrame
) -> Dict[str, Any]:
    """Create feature importance and SHAP-based explainability charts for LGBM."""
    out: Dict[str, Any] = {}

    # 1) Feature importance (gain)
    booster = model.booster_
    gain_vals = booster.feature_importance(importance_type="gain")
    feature_names = list(X_train.columns)
    df_gain = pd.DataFrame({"feature": feature_names, "importance": gain_vals})
    df_gain = df_gain.sort_values("importance", ascending=False).head(20)
    fig_gain = go.Figure(
        go.Bar(x=df_gain["importance"], y=df_gain["feature"], orientation="h")
    )
    fig_gain.update_layout(
        title="LightGBM | Feature Importance (Gain)",
        xaxis_title="Gain",
        yaxis_title="Feature",
    )
    out["lgbm_gain"] = _to_json(fig_gain)

    # 2) Feature importance (split)
    split_vals = booster.feature_importance(importance_type="split")
    df_split = pd.DataFrame({"feature": feature_names, "importance": split_vals})
    df_split = df_split.sort_values("importance", ascending=False).head(20)
    fig_split = go.Figure(
        go.Bar(x=df_split["importance"], y=df_split["feature"], orientation="h")
    )
    fig_split.update_layout(
        title="LightGBM | Feature Importance (Split)",
        xaxis_title="#Splits",
        yaxis_title="Feature",
    )
    out["lgbm_split"] = _to_json(fig_split)

    # SHAP calculations
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_idx_sorted = np.argsort(mean_abs_shap)[::-1]
    top_features = [X_train.columns[i] for i in top_idx_sorted]

    # 3) SHAP feature dynamics
    abs_shap_df = pd.DataFrame(
        np.abs(shap_values.values), index=X_train.index, columns=X_train.columns
    )
    fig_dyn = go.Figure()
    for feat in top_features[:7]:
        series_smoothed = abs_shap_df[feat].rolling(7, min_periods=1).mean()
        fig_dyn.add_trace(
            go.Scatter(
                x=series_smoothed.index,
                y=series_smoothed.values,
                mode="lines",
                name=feat,
            )
        )
    fig_dyn.update_layout(
        title="SHAP | Feature Importance Dynamics (7-Day Rolling Mean)",
        xaxis_title="Date",
        yaxis_title="Mean Absolute SHAP Value",
    )
    out["shap_dynamics"] = _to_json(fig_dyn)

    # 4) SHAP interaction heatmap (Top features)
    n_samples = X_train.shape[0]
    sample_size = min(200, n_samples)
    X_sample = X_train.iloc[-sample_size:]
    interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_sample)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]
    mean_abs_interactions = np.mean(np.abs(interaction_values), axis=0)
    top_k = min(10, mean_abs_interactions.shape[0])
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
    top_names = [X_train.columns[i] for i in top_idx]
    sub_mat = mean_abs_interactions[np.ix_(top_idx, top_idx)]
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=sub_mat, x=top_names, y=top_names, colorscale="RdBu", reversescale=True
        )
    )
    fig_heat.update_layout(title="SHAP | Feature Interaction Heatmap (Top Features)")
    out["shap_interaction"] = _to_json(fig_heat)

    # 5) SHAP scatter plots for top 2 features (shap value vs feature value)
    for i, feat in enumerate(top_features[:2]):
        idx = list(X_train.columns).index(feat)
        shap_col = shap_values.values[:, idx]
        feat_vals = X_train.iloc[:, idx].values
        fig_scatter = go.Figure(
            data=go.Scatter(
                x=feat_vals,
                y=shap_col,
                mode="markers",
                marker=dict(color=shap_col, colorscale="RdBu", showscale=True),
            )
        )
        fig_scatter.update_layout(
            title=f"SHAP | Feature Contribution ({feat})",
            xaxis_title=f"{feat}",
            yaxis_title="SHAP value",
        )
        out[f"shap_scatter_{i + 1}"] = _to_json(fig_scatter)

    # 6) SHAP beeswarm-like (violin with points for top 20)
    fig_bee = go.Figure()
    for feat in top_features[:20][::-1]:  # reverse so most important on top
        idx = list(X_train.columns).index(feat)
        vals = shap_values.values[:, idx]
        fig_bee.add_trace(
            go.Violin(
                y=[feat] * len(vals),
                x=vals,
                orientation="h",
                name=feat,
                points="all",
                pointpos=0.0,
                jitter=0.3,
                spanmode="hard",
            )
        )
    fig_bee.update_layout(
        title="SHAP | Beeswarm", xaxis_title="SHAP value", yaxis_title="Feature"
    )
    out["lgbm_shap_beeswarm"] = _to_json(fig_bee)

    # 7) SHAP bar summary
    df_bar = pd.DataFrame(
        {"feature": list(X_train.columns), "mean_abs_shap": mean_abs_shap}
    )
    df_bar = df_bar.sort_values("mean_abs_shap", ascending=False).head(20)
    fig_bar = go.Figure(
        go.Bar(x=df_bar["mean_abs_shap"], y=df_bar["feature"], orientation="h")
    )
    fig_bar.update_layout(
        title="SHAP | Summary (Bar)", xaxis_title="Mean |SHAP|", yaxis_title="Feature"
    )
    out["lgbm_shap_bar"] = _to_json(fig_bar)

    # 8) SHAP dependence for top feature
    top_feature_name = top_features[0]
    idx0 = list(X_train.columns).index(top_feature_name)
    shap_col = shap_values.values[:, idx0]
    feat_vals = X_train.iloc[:, idx0].values
    fig_dep = go.Figure(
        data=go.Scatter(
            x=feat_vals,
            y=shap_col,
            mode="markers",
            marker=dict(color=feat_vals, colorscale="Viridis", showscale=True),
        )
    )
    fig_dep.update_layout(
        title=f"SHAP | Dependence ({top_feature_name})",
        xaxis_title=f"{top_feature_name}",
        yaxis_title="SHAP value",
    )
    out["lgbm_shap_dependence"] = _to_json(fig_dep)

    return out


def generate_holdout_overlay(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a holdout comparison plot overlaying actuals vs model predictions (Plotly)."""
    holdout_days = 60
    split_date = df.index.max() - pd.DateOffset(days=holdout_days - 1)
    df_train = df[df.index < split_date]
    df_test = df[df.index >= split_date]

    # Actuals
    actual = df_test["daily_sales"].astype(float).values

    # Prophet stored as JSON; read path from registry
    yhat_prophet = None
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT path FROM models_registry WHERE model_name = ?", ("prophet",))
    row = cur.fetchone()
    conn.close()
    if row:
        from prophet.serialize import model_from_json  # type: ignore

        with open(row["path"], "r") as f:
            m = model_from_json(f.read())
        future = pd.concat(
            [
                df_train.reset_index()[["date", "marketing_spend"]].rename(
                    columns={"date": "ds"}
                ),
                df_test.reset_index()[["date", "marketing_spend"]].rename(
                    columns={"date": "ds"}
                ),
            ]
        )
        fc = m.predict(future)
        yhat_prophet = np.expm1(fc.loc[fc["ds"] >= split_date, "yhat"].values)

    # LGBM
    yhat_lgb = None
    bundle = mdl.load_model("lgbm")
    model_lgb = bundle["model"]
    feat_cols = bundle["features"]
    preds_lgb = mdl.recursive_forecast_lgb(model_lgb, df_train, df_test, feat_cols)
    yhat_lgb = np.array(preds_lgb, dtype=float)

    # SARIMAX
    yhat_sa = None
    res = mdl.load_model("sarimax")
    exog_te = df_test[["is_holiday"]].astype(int)
    fc = res.get_forecast(steps=len(df_test), exog=exog_te)
    yhat_sa = fc.predicted_mean.values.astype(float)

    # Plot (Plotly)
    idx = df_test.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=actual,
            name="Actual Sales",
            mode="lines",
            line=dict(color="black", width=3),
        )
    )
    if yhat_prophet is not None:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=yhat_prophet,
                name="Prophet (Baseline)",
                mode="lines",
                line=dict(color="#2563eb", width=2, dash="dash"),
            )
        )
    if yhat_lgb is not None:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=yhat_lgb,
                name="LightGBM (w/ Feature Engineering)",
                mode="lines",
                line=dict(color="#dc2626", width=2, dash="dot"),
            )
        )
    if yhat_sa is not None:
        fig.add_trace(
            go.Scatter(
                x=idx,
                y=yhat_sa,
                name="SARIMAX (w/ is_holiday)",
                mode="lines",
                line=dict(color="#16a34a", width=2, dash="longdash"),
            )
        )
    fig.update_layout(
        title="Sales | Actuals vs. Predictions",
        xaxis_title="Date",
        yaxis_title="Daily Sales (USD)",
    )
    return _to_json(fig)


def generate_forecast_overlay(df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """Generate a combined forecast plot with actuals and future predictions (Plotly)."""
    horizon = int(max(1, horizon))

    # Context window for actuals
    context_days = 180
    start_idx = max(0, len(df) - context_days)
    df_ctx = df.iloc[start_idx:]

    dates_future = None
    preds: Dict[str, np.ndarray] = {}

    # LGBM
    df_lgb = mdl.forecast("lgbm", horizon)
    preds["LightGBM (w/ Feature Engineering)"] = df_lgb["yhat"].values.astype(float)
    dates_future = df_lgb["date"].values

    # SARIMAX
    df_sa = mdl.forecast("sarimax", horizon)
    preds["SARIMAX (w/ is_holiday)"] = df_sa["yhat"].values.astype(float)
    if dates_future is None:
        dates_future = df_sa["date"].values

    # Prophet
    df_p = mdl.forecast("prophet", horizon)
    preds["Prophet (Baseline)"] = df_p["yhat"].values.astype(float)
    if dates_future is None:
        dates_future = df_p["date"].values

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_ctx.index,
            y=df_ctx["daily_sales"].values.astype(float),
            name="Actual Sales",
            mode="lines",
            line=dict(color="black", width=2),
        )
    )
    if dates_future is not None:
        for label, arr in preds.items():
            fig.add_trace(go.Scatter(x=dates_future, y=arr, name=label, mode="lines"))
    fig.update_layout(
        title="Forecasts | Actuals + Predictions",
        xaxis_title="Date",
        yaxis_title="Daily Sales (USD)",
    )
    return _to_json(fig)
