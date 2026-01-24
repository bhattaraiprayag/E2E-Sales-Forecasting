"""
Sales forecasting script using Prophet (baseline), LightGBM  with recursive forecasting and SARIMAX models.

This script:
- Loads `ecommerce_sales_data.xlsx`
- Builds a Prophet baseline on log-transformed sales
- Engineers features for a LightGBM regressor on a differenced log target
- Trains with early stopping and reconstructs forecasts to the original scale using recursive forecasting
- Prints metrics and plots actuals vs predictions on a holdout period
"""

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import lightgbm as lgb
from prophet import Prophet
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------
# Plotting Style Configuration
# -----------------------------
sns.set_style("ticks")
plt.rc("figure", figsize=(12, 6))
plt.rc("axes", titlesize=15, labelsize=12, titleweight="bold")
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
plt.rc("legend", fontsize=10)


# -----------------------------
# Metrics and Feature Utilities
# -----------------------------
def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return RMSE, MAE, and MAPE on the original sales scale."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = 1e-15
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


def encode_cyclical(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    """Encode a cyclical feature (e.g., month, weekday) using sine/cosine."""
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def create_gbdt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for LightGBM on a differenced log-sales target."""
    df_feat = df.copy()

    df_feat["sales_log"] = np.log1p(df_feat["daily_sales"])
    df_feat["target_sales_diff_log"] = df_feat["sales_log"].diff(1)

    df_feat["month_of_year"] = df_feat.index.month
    df_feat["day_of_month"] = df_feat.index.day
    df_feat["day_of_week_num"] = df_feat.index.dayofweek
    df_feat["week_of_year"] = df_feat.index.isocalendar().week.astype(int)
    df_feat["quarter"] = df_feat.index.quarter
    df_feat["is_month_start"] = df_feat.index.is_month_start.astype(int)
    df_feat["is_month_end"] = df_feat.index.is_month_end.astype(int)
    df_feat["is_quarter_start"] = df_feat.index.is_quarter_start.astype(int)
    df_feat["is_quarter_end"] = df_feat.index.is_quarter_end.astype(int)

    df_feat = encode_cyclical(df_feat, "day_of_week_num", 7)
    df_feat = encode_cyclical(df_feat, "month_of_year", 12)
    df_feat = encode_cyclical(df_feat, "day_of_month", 31)
    df_feat = encode_cyclical(df_feat, "week_of_year", 52)

    for lag in [1, 2, 3, 7, 14, 28]:
        df_feat[f"lag_target_{lag}"] = df_feat["target_sales_diff_log"].shift(lag)

    for window in [7, 14, 30]:
        df_feat[f"rolling_mean_target_{window}"] = (
            df_feat["target_sales_diff_log"].shift(1).rolling(window).mean()
        )
        df_feat[f"rolling_std_target_{window}"] = (
            df_feat["target_sales_diff_log"].shift(1).rolling(window).std()
        )

    for lag in [1, 3, 7]:
        df_feat[f"marketing_lag_{lag}"] = df_feat["marketing_spend"].shift(lag)

    df_feat["marketing_rolling_mean_7"] = (
        df_feat["marketing_spend"].shift(1).rolling(7).mean()
    )
    df_feat["marketing_x_holiday"] = df_feat["marketing_spend"].shift(1) * df_feat[
        "is_holiday"
    ].shift(1)

    cols_to_drop = [
        "day_of_week",
        "month_of_year",
        "day_of_month",
        "day_of_week_num",
        "week_of_year",
        "quarter",
    ]
    df_feat = df_feat.drop(columns=cols_to_drop)
    return df_feat


# -----------------------------
# Data Loading and Preparation
# -----------------------------
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load Excel data, set daily frequency, and clean basic issues."""
    try:
        df = pd.read_excel(file_path, parse_dates=["date"])
        df = df.set_index("date")
    except FileNotFoundError:
        print("Error: 'ecommerce_sales_data.xlsx' not found.")
        print("Please make sure the file is in the same directory as the script.")
        raise

    df = df.sort_index()
    df = df.asfreq("D")
    df["daily_sales"] = df["daily_sales"].interpolate(method="linear")
    df["marketing_spend"] = df["marketing_spend"].interpolate(method="linear")
    df["day_of_week"] = df["day_of_week"].ffill()
    df["is_holiday"] = df["is_holiday"].ffill()

    if "product_category" in df.columns and df["product_category"].nunique() == 1:
        df = df.drop(columns=["product_category"])
    return df


# -----------------------------
# Prophet Baseline
# -----------------------------
def train_prophet_baseline(
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    holdout_days: int,
    split_date: pd.Timestamp,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Train Prophet with log target and marketing_spend regressor; return preds and metrics."""
    df_prophet_train = df_train.reset_index().rename(
        columns={"date": "ds", "daily_sales": "y"}
    )
    df_prophet_train["y"] = np.log1p(df_prophet_train["y"])
    df_prophet_test = df_test.reset_index().rename(
        columns={"date": "ds", "daily_sales": "y"}
    )

    holidays_df = df_full[df_full["is_holiday"] == 1].reset_index()[["date"]]
    holidays_df = holidays_df.rename(columns={"date": "ds"})
    holidays_df["holiday"] = "special_day"

    model_prophet = Prophet(holidays=holidays_df)
    model_prophet.add_regressor("marketing_spend")
    model_prophet.fit(df_prophet_train)

    future = model_prophet.make_future_dataframe(periods=holdout_days)
    future_regressors = pd.concat(
        [
            df_prophet_train[["ds", "marketing_spend"]],
            df_prophet_test[["ds", "marketing_spend"]],
        ]
    )
    future = pd.merge(future, future_regressors, on="ds", how="left")

    forecast = model_prophet.predict(future)
    y_pred_prophet_log = forecast[forecast["ds"] >= split_date]["yhat"]
    y_pred_prophet = np.expm1(y_pred_prophet_log)
    metrics = calculate_metrics(df_test["daily_sales"], y_pred_prophet)
    return y_pred_prophet.values, metrics


# -----------------------------
# LightGBM Training and Explainability
# -----------------------------
def prepare_lgbm_training(
    df_train: pd.DataFrame, validation_window: int = 60
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], str]:
    """Create features, split validation tail, and return matrices and metadata."""
    df_train_feat = create_gbdt_features(df_train.copy())
    df_train_feat_clean = df_train_feat.dropna()

    target_column = "target_sales_diff_log"
    feature_columns = [
        col
        for col in df_train_feat_clean.columns
        if col
        not in [
            "daily_sales",
            "marketing_spend",
            "is_holiday",
            target_column,
            "sales_log",
        ]
    ]

    X = df_train_feat_clean[feature_columns]
    y = df_train_feat_clean[target_column]

    X_train_main = X.iloc[:-validation_window]
    y_train_main = y.iloc[:-validation_window]
    X_val = X.iloc[-validation_window:]
    y_val = y.iloc[-validation_window:]
    return X_train_main, y_train_main, X_val, y_val, feature_columns, target_column


def get_lgbm_model() -> lgb.LGBMRegressor:
    """Get the LightGBM model with default parameters."""
    return lgb.LGBMRegressor(
        random_state=42,
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=15,
        objective="regression_l1",
        metric="mae",
        n_jobs=-1,
        verbose=-1,
        colsample_bytree=0.7,
        subsample=0.7,
    )


def train_lgbm(
    X_train_main: pd.DataFrame,
    y_train_main: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 100,
) -> lgb.LGBMRegressor:
    """Train the LightGBM model with early stopping."""
    model = get_lgbm_model()
    model.fit(
        X_train_main,
        y_train_main,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def analyze_lgbm_explainability(
    model_lgb: lgb.LGBMRegressor, X_train_main: pd.DataFrame
) -> None:
    """Analyze the LightGBM model's explainability using SHAP."""
    try:
        lgb.plot_importance(
            model_lgb,
            importance_type="gain",
            max_num_features=20,
            figsize=(10, 8),
            grid=False,
            title="LightGBM | Feature Importance (Gain)",
        )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Feature Importance (Gain): {e}")

    try:
        lgb.plot_importance(
            model_lgb,
            importance_type="split",
            max_num_features=20,
            figsize=(10, 8),
            grid=False,
            title="LightGBM | Feature Importance (Split)",
        )
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Feature Importance (Split): {e}")

    """SHAP explainability visuals"""
    try:
        explainer = shap.Explainer(model_lgb, X_train_main)
        shap_values = explainer(X_train_main)

        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title("LightGBM | Feature Importance (Beeswarm)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Feature Importance (Beeswarm): {e}")

    try:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_feature_index = int(np.argmax(mean_abs_shap))
        top_feature_name = X_train_main.columns[top_feature_index]
        shap.dependence_plot(
            top_feature_name,
            shap_values.values,
            X_train_main,
            interaction_index="auto",
            show=False,
        )
        plt.title(f"LightGBM | Feature Dependence ({top_feature_name})")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Feature Dependence: {e}")

    try:
        shap.plots.violin(shap_values, max_display=20, show=False)
        plt.title("LightGBM | Feature Importance Violin")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Feature Importance Violin: {e}")

    try:
        top_idx_sorted = np.argsort(mean_abs_shap)[::-1]
        top_features = [X_train_main.columns[i] for i in top_idx_sorted[:5]]
        for feat in top_features[:2]:
            shap.plots.scatter(shap_values[:, feat], color=shap_values, show=False)
            plt.title(f"LightGBM | Feature Contribution ({feat})")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Feature Contribution ({feat}): {e}")

    try:
        n_samples = shap_values.values.shape[0]
        sample_size = int(min(100, max(10, n_samples * 0.2)))
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

        base_val = None
        try:
            base_val = getattr(explainer, "expected_value", None)
        except Exception:
            base_val = None
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.mean(base_val))
        if base_val is None:
            base_val = float(np.mean(shap_values.base_values))

        shap.decision_plot(
            base_val,
            shap_values.values[sample_idx],
            features=X_train_main.iloc[sample_idx],
            feature_names=list(X_train_main.columns),
            show=False,
        )
        plt.title("LightGBM | Prediction Explanation (Decision Plot)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Prediction Explanation (Decision Plot): {e}")

    try:
        shap.plots.waterfall(shap_values[-1], max_display=20, show=False)
        plt.title("LightGBM | Prediction Explanation (Waterfall)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Prediction Explanation (Waterfall): {e}")

    try:
        tree_explainer = shap.TreeExplainer(model_lgb)
        n_samples = X_train_main.shape[0]
        sample_size = min(200, n_samples)
        X_sample = X_train_main.iloc[-sample_size:]
        interaction_values = tree_explainer.shap_interaction_values(X_sample)
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[0]
        mean_abs_interactions = np.mean(np.abs(interaction_values), axis=0)
        top_k = min(10, mean_abs_interactions.shape[0])
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
        top_names = [X_train_main.columns[i] for i in top_idx]
        sub_mat = mean_abs_interactions[np.ix_(top_idx, top_idx)]
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            sub_mat, xticklabels=top_names, yticklabels=top_names, cmap="coolwarm"
        )
        plt.title("LightGBM | Feature Interaction Heatmap")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Feature Interaction Heatmap: {e}")

    try:
        abs_shap_df = pd.DataFrame(
            np.abs(shap_values.values),
            index=X_train_main.index,
            columns=X_train_main.columns,
        )
        plt.figure(figsize=(12, 6))
        for feat in top_features:
            series_smoothed = abs_shap_df[feat].rolling(7, min_periods=1).mean()
            plt.plot(series_smoothed.index, series_smoothed.values, label=feat)
        plt.title("LightGBM | Feature Importance Dynamics (7-Day Rolling Mean)")
        plt.xlabel("Date")
        plt.ylabel("Mean Absolute SHAP Value")
        plt.legend()
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(
            f"Skipping LightGBM | Feature Importance Dynamics (7-Day Rolling Mean): {e}"
        )

    try:
        feature_names = list(X_train_main.columns)
        group_to_cols = {
            "Lag/Rolling target": [
                c
                for c in feature_names
                if c.startswith("lag_target_")
                or c.startswith("rolling_mean_target_")
                or c.startswith("rolling_std_target_")
            ],
            "Marketing": [c for c in feature_names if c.startswith("marketing_")],
            "Calendar": [
                c
                for c in feature_names
                if c.endswith("_sin")
                or c.endswith("_cos")
                or c.startswith("is_month_")
                or c.startswith("is_quarter_")
            ],
        }
        grouped_cols = set(sum(group_to_cols.values(), []))
        group_to_cols["Other"] = [c for c in feature_names if c not in grouped_cols]

        mean_abs_per_feature = np.mean(np.abs(shap_values.values), axis=0)
        importances_by_group = {}
        for g, cols in group_to_cols.items():
            if not cols:
                continue
            idxs = [feature_names.index(c) for c in cols]
            importances_by_group[g] = float(mean_abs_per_feature[idxs].sum())

        if importances_by_group:
            groups, vals = zip(
                *sorted(
                    importances_by_group.items(), key=lambda kv: kv[1], reverse=True
                )
            )
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(vals), y=list(groups), orient="h", palette="Blues_r")
            plt.title("LightGBM | Grouped Feature Importance")
            plt.xlabel("Sum of Mean Absolute SHAP Values")
            plt.ylabel("Feature Group")
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Skipping LightGBM | Grouped Feature Importance: {e}")


# -----------------------------
# SARIMAX Training and Forecasting
# -----------------------------
def train_sarimax_forecast(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    order: Tuple[int, int, int] = (1, 1, 1),  # p, d, q: see EDA_and_Tests.ipynb
    seasonal_order: Tuple[int, int, int, int] = (
        1,
        1,
        1,
        7,
    ),  # P, D, Q, m: see EDA_and_Tests.ipynb
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Train SARIMAX with is_holiday exog and forecast the holdout."""
    endog_train = df_train["daily_sales"]
    exog_train = df_train[["is_holiday"]].astype(int)
    exog_test = df_test[["is_holiday"]].astype(int)

    model = sm.tsa.SARIMAX(
        endog=endog_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=len(df_test), exog=exog_test)
    y_pred = forecast.predicted_mean.values
    metrics = calculate_metrics(df_test["daily_sales"].values, y_pred)
    return y_pred, metrics


# -----------------------------
# Recursive Forecasting for LightGBM
# -----------------------------
def recursive_forecast_lgb(
    model_lgb: lgb.LGBMRegressor,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str],
) -> List[float]:
    """Recursive forecast using LightGBM."""
    df_history = df_train.copy()
    predictions: List[float] = []
    last_known_log_sale = np.log1p(df_history["daily_sales"].iloc[-1])

    for date in df_test.index:
        new_row_template = df_test.loc[[date]]
        df_for_features = pd.concat([df_history, new_row_template])
        df_features_step = create_gbdt_features(df_for_features)
        X_step = df_features_step[feature_columns].iloc[[-1]]
        pred_diff_log = model_lgb.predict(X_step)[0]
        pred_log = last_known_log_sale + pred_diff_log
        pred_sales = np.expm1(pred_log)
        predictions.append(float(pred_sales))
        new_row_template["daily_sales"] = pred_sales
        df_history = pd.concat([df_history, new_row_template])
        last_known_log_sale = pred_log
    return predictions


# -----------------------------
# Results and Plotting
# -----------------------------
def build_results_df(
    df_test: pd.DataFrame,
    prophet_pred: np.ndarray,
    lgb_pred: List[float],
    sarimax_pred: np.ndarray,
) -> pd.DataFrame:
    """Build a results dataframe with actual, Prophet, and LightGBM predictions."""
    results_df = df_test[["daily_sales"]].copy()
    results_df.rename(columns={"daily_sales": "Actuals"}, inplace=True)
    results_df["Prophet_Pred"] = prophet_pred
    results_df["LGBM_Pred"] = lgb_pred
    results_df["SARIMAX_Pred"] = sarimax_pred
    return results_df


def plot_forecasts(results_df: pd.DataFrame) -> None:
    """Plot the forecasts."""
    plt.figure(figsize=(18, 8))
    plt.plot(
        results_df.index,
        results_df["Actuals"],
        label="Actual Sales",
        color="black",
        linewidth=2.5,
        alpha=0.8,
    )
    plt.plot(
        results_df.index,
        results_df["Prophet_Pred"],
        label="Prophet (Baseline)",
        color="blue",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        results_df.index,
        results_df["LGBM_Pred"],
        label="LightGBM (w/ Feature Engineering)",
        color="red",
        linestyle="-.",
        linewidth=2,
    )
    plt.plot(
        results_df.index,
        results_df["SARIMAX_Pred"],
        label="SARIMAX (w/ is_holiday)",
        color="green",
        linestyle="-.",
        linewidth=2,
    )
    plt.title("Forecasts | Actuals vs. Predictions (Holdout Period)")
    plt.ylabel("Daily Sales (USD)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Orchestration
# -----------------------------
def main() -> None:
    """Main function to orchestrate the forecasting pipeline."""
    file_path = "../data/ecommerce_sales_data.xlsx"
    df = load_and_prepare_data(file_path)

    HOLD_OUT_DAYS = 60
    split_date = df.index.max() - pd.DateOffset(days=HOLD_OUT_DAYS - 1)
    df_train = df[df.index < split_date]
    df_test = df[df.index >= split_date]

    prophet_pred, metrics_prophet = train_prophet_baseline(
        df_full=df,
        df_train=df_train,
        df_test=df_test,
        holdout_days=HOLD_OUT_DAYS,
        split_date=split_date,
    )

    (
        X_train_main,
        y_train_main,
        X_val,
        y_val,
        feature_columns,
        _,
    ) = prepare_lgbm_training(
        df_train,
        validation_window=60,
    )
    model_lgb = train_lgbm(
        X_train_main, y_train_main, X_val, y_val, early_stopping_rounds=100
    )
    analyze_lgbm_explainability(model_lgb, X_train_main)

    lgb_recursive_pred = recursive_forecast_lgb(
        model_lgb, df_train, df_test, feature_columns
    )
    metrics_lgbm = calculate_metrics(df_test["daily_sales"], lgb_recursive_pred)

    sarimax_pred, metrics_sarimax = train_sarimax_forecast(df_train, df_test)

    results_df = build_results_df(
        df_test, prophet_pred, lgb_recursive_pred, sarimax_pred
    )

    print("\nProphet (Baseline):")
    print(f"  - RMSE: {metrics_prophet['RMSE']:,.2f}")
    print(f"  - MAE:  {metrics_prophet['MAE']:,.2f}")
    print(f"  - MAPE: {metrics_prophet['MAPE (%)']:.2f}%")

    print("\nLightGBM (w/ Feature Engineering):")
    print(f"  - RMSE: {metrics_lgbm['RMSE']:,.2f}")
    print(f"  - MAE:  {metrics_lgbm['MAE']:,.2f}")
    print(f"  - MAPE: {metrics_lgbm['MAPE (%)']:.2f}%")

    print("\nSARIMAX (w/ is_holiday):")
    print(f"  - RMSE: {metrics_sarimax['RMSE']:,.2f}")
    print(f"  - MAE:  {metrics_sarimax['MAE']:,.2f}")
    print(f"  - MAPE: {metrics_sarimax['MAPE (%)']:.2f}%")

    plot_forecasts(results_df)


if __name__ == "__main__":
    main()
