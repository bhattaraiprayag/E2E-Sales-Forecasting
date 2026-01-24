from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from .database import get_connection, read_historical
from . import data_processing as dp


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Metrics:
    rmse: float
    mae: float
    mape: float


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Compute RMSE, MAE, MAPE with small epsilon guard for division."""
    eps = 1e-15
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
    return Metrics(rmse=rmse, mae=mae, mape=mape)


def _fetch_dataframe_from_db() -> pd.DataFrame:
    """Load historical data from SQLite and return a DateIndex-sorted frame.

    Raises RuntimeError if no data exists.
    """
    rows = read_historical()
    if not rows:
        raise RuntimeError("No historical data available. Upload data first.")
    df = pd.DataFrame(
        rows,
        columns=["date", "daily_sales", "marketing_spend", "is_holiday", "day_of_week"],
    )  # type: ignore[list-item]
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime index
    df = df.sort_values("date").set_index("date")
    return df


def _prepare_lgb_splits(
    df: pd.DataFrame, holdout_days: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Create LGBM training/test frames and list of feature columns.

    Uses engineered features and keeps a fixed-size holdout window.
    """
    df_feat = dp.create_gbdt_features(df.copy())
    df_feat = df_feat.dropna()

    target_col = "target_sales_diff_log"
    feature_cols = [
        c
        for c in df_feat.columns
        if c
        not in (
            "daily_sales",
            "marketing_spend",
            "is_holiday",
            target_col,
            "sales_log",
            "day_of_week",
        )
    ]

    split_date = df_feat.index.max() - pd.DateOffset(days=holdout_days - 1)
    train = df_feat[df_feat.index < split_date]
    test = df_feat[df_feat.index >= split_date]
    return (
        train[feature_cols + [target_col]],
        test[feature_cols + [target_col]],
        feature_cols,
    )


def _get_lgb_model() -> lgb.LGBMRegressor:
    """Default LightGBM model configuration (early stopping set at fit time)."""
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


def train_lgb_and_evaluate(
    df: pd.DataFrame,
) -> Tuple[lgb.LGBMRegressor, Metrics, List[str]]:
    """Train LGBM on Δlog(sales) features and evaluate on holdout via recursion."""
    train, test, feature_cols = _prepare_lgb_splits(df)
    X_train, y_train = train[feature_cols], train["target_sales_diff_log"]
    X_test, _y_test = test[feature_cols], test["target_sales_diff_log"]

    # validation tail from train
    val_window = min(60, len(X_train) // 5 if len(X_train) > 10 else len(X_train))
    X_tr_main, y_tr_main = X_train.iloc[:-val_window], y_train.iloc[:-val_window]
    X_val, y_val = X_train.iloc[-val_window:], y_train.iloc[-val_window:]

    model = _get_lgb_model()
    model.fit(
        X_tr_main,
        y_tr_main,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # Reconstruct predictions on holdout using recursive approach
    df_train = df[df.index.isin(X_train.index)]
    df_holdout = df[df.index.isin(X_test.index)]
    preds = recursive_forecast_lgb(model, df_train, df_holdout, feature_cols)
    metrics = _calculate_metrics(
        df_holdout["daily_sales"].values.astype(float), np.array(preds, dtype=float)
    )
    return model, metrics, feature_cols


def recursive_forecast_lgb(
    model_lgb: lgb.LGBMRegressor,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str],
) -> List[float]:
    """Recursive one-step-ahead forecasting on Δlog(sales) with feature rebuilds."""
    history = df_train.copy()
    predictions: List[float] = []
    last_log = float(np.log1p(history["daily_sales"].iloc[-1]))
    for date in df_test.index:
        new_row = df_test.loc[[date]].copy()
        df_for_features = pd.concat([history, new_row], axis=0)
        feat = dp.create_gbdt_features(df_for_features)
        X = feat[feature_columns].iloc[[-1]]
        pred_diff = float(model_lgb.predict(X)[0])
        pred_log = last_log + pred_diff
        pred = float(np.expm1(pred_log))
        predictions.append(pred)
        new_row.loc[date, "daily_sales"] = pred
        history = pd.concat([history, new_row], axis=0)
        last_log = pred_log
    return predictions


def train_sarimax_and_evaluate(
    df: pd.DataFrame,
) -> Tuple[sm.tsa.statespace.sarimax.SARIMAXResultsWrapper, Metrics]:
    """Fit SARIMAX with weekly seasonality and holiday exogenous; evaluate on holdout."""
    holdout_days = 60
    split_date = df.index.max() - pd.DateOffset(days=holdout_days - 1)
    df_train = df[df.index < split_date]
    df_test = df[df.index >= split_date]

    np.random.seed(42)
    endog = df_train["daily_sales"]
    exog_tr = df_train[["is_holiday"]].astype(int)
    exog_te = df_test[["is_holiday"]].astype(int)

    mod = sm.tsa.SARIMAX(
        endog=endog,
        exog=exog_tr,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = mod.fit(disp=False)
    fc = res.get_forecast(steps=len(df_test), exog=exog_te)
    yhat = fc.predicted_mean.values
    metrics = _calculate_metrics(
        df_test["daily_sales"].values.astype(float), yhat.astype(float)
    )
    return res, metrics


def _prepare_prophet_frames(
    df: pd.DataFrame, holdout_days: int = 60
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, pd.Timestamp]:
    """Create Prophet-compatible frames: train/test/log-y, holidays, and split date."""
    split_date = df.index.max() - pd.DateOffset(days=holdout_days - 1)
    df_train = df[df.index < split_date]
    df_test = df[df.index >= split_date]

    train_df = df_train.reset_index().rename(columns={"date": "ds", "daily_sales": "y"})
    train_df["y"] = np.log1p(train_df["y"])  # Prophet on log scale
    test_df = df_test.reset_index().rename(columns={"date": "ds", "daily_sales": "y"})

    holidays_df = (
        df[df["is_holiday"] == 1].reset_index()[["date"]].rename(columns={"date": "ds"})
    )
    if not holidays_df.empty:
        holidays_df["holiday"] = "special_day"

    return train_df, test_df, holidays_df, holdout_days, pd.to_datetime(split_date)


def train_prophet_and_evaluate(df: pd.DataFrame) -> Tuple[Any, Metrics]:
    """Fit Prophet on log(y) with marketing regressor; evaluate on holdout (exp back)."""
    train_df, test_df, holidays_df, holdout_days, split_date = _prepare_prophet_frames(
        df
    )

    np.random.seed(42)
    m = Prophet(holidays=holidays_df if not holidays_df.empty else None)
    m.add_regressor("marketing_spend")
    m.fit(train_df[["ds", "y", "marketing_spend"]])

    future = m.make_future_dataframe(periods=holdout_days, freq="D")
    future_reg = pd.concat(
        [
            train_df[["ds", "marketing_spend"]],
            test_df[["ds", "marketing_spend"]],
        ]
    )
    future = future.merge(future_reg, on="ds", how="left")
    fc = m.predict(future)
    yhat_log = fc.loc[fc["ds"] >= split_date, "yhat"].values
    yhat = np.expm1(yhat_log)
    y_true = test_df["y"].values.astype(float)
    metrics = _calculate_metrics(y_true, yhat.astype(float))
    return m, metrics


def save_model(model: Any, name: str) -> str:
    """Pickle model to /models and upsert registry entry; return saved path."""
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    # update registry
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO models_registry (model_name, model_type, path, trained_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(model_name) DO UPDATE SET path=excluded.path, trained_at=excluded.trained_at
        """,
        (name, type(model).__name__, str(path), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return str(path)


def save_prophet_model(m: Any, name: str) -> str:
    """Persist Prophet model to JSON as recommended by the library."""
    path = MODELS_DIR / f"{name}.json"
    with open(path, "w") as f:
        f.write(model_to_json(m))  # type: ignore[arg-type]
    # update registry
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO models_registry (model_name, model_type, path, trained_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(model_name) DO UPDATE SET path=excluded.path, trained_at=excluded.trained_at
        """,
        (name, "Prophet", str(path), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return str(path)


def record_metrics(name: str, m: Metrics) -> None:
    """Upsert model metrics into SQLite with current timestamp."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_metrics (model_name, rmse, mae, mape, trained_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(model_name) DO UPDATE SET
          rmse=excluded.rmse,
          mae=excluded.mae,
          mape=excluded.mape,
          trained_at=excluded.trained_at
        """,
        (name, m.rmse, m.mae, m.mape, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def train_all_and_persist() -> Dict[str, Metrics]:
    """Train Prophet, LGBM, SARIMAX; save artifacts and metrics; return metrics map."""
    np.random.seed(42)
    df = _fetch_dataframe_from_db()
    out: Dict[str, Metrics] = {}

    m_prophet, p_metrics = train_prophet_and_evaluate(df)
    save_prophet_model(m_prophet, "prophet")
    record_metrics("prophet", p_metrics)
    out["prophet"] = p_metrics

    lgb_model, lgb_metrics, feat_cols = train_lgb_and_evaluate(df)
    save_model({"model": lgb_model, "features": feat_cols}, "lgbm")
    record_metrics("lgbm", lgb_metrics)
    out["lgbm"] = lgb_metrics

    sarimax_res, sar_metrics = train_sarimax_and_evaluate(df)
    save_model(sarimax_res, "sarimax")
    record_metrics("sarimax", sar_metrics)
    out["sarimax"] = sar_metrics

    return out


def load_model(name: str) -> Any:
    """Load a pickled model bundle from path in registry by name."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT path FROM models_registry WHERE model_name = ?", (name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise FileNotFoundError(f"Model '{name}' not found in registry")
    path = row["path"]
    with open(path, "rb") as f:
        return pickle.load(f)


def forecast(name: str, horizon: int) -> pd.DataFrame:
    """Generate per-model forecasts for a given horizon.

    Returns a frame with at least columns ['date','yhat'] and model-specific extras.
    """
    df = _fetch_dataframe_from_db()
    last_date = df.index.max()
    dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    if name == "lgbm":
        bundle = load_model("lgbm")
        model: lgb.LGBMRegressor = bundle["model"]
        feat_cols: List[str] = bundle["features"]
        # build a future frame template
        future = pd.DataFrame(index=dates)
        future["daily_sales"] = np.nan
        future["marketing_spend"] = (
            df["marketing_spend"].iloc[-7:].mean()
        )  # naive carry average
        future["is_holiday"] = 0
        future["day_of_week"] = dates.day_name()
        preds = recursive_forecast_lgb(model, df, future, feat_cols)
        return pd.DataFrame({"date": dates, "yhat": preds})

    if name == "sarimax":
        res: sm.tsa.statespace.sarimax.SARIMAXResultsWrapper = load_model("sarimax")
        exog_future = pd.DataFrame(
            index=dates, data={"is_holiday": 0}
        )  # assume non-holiday
        fc = res.get_forecast(steps=horizon, exog=exog_future)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        out = pd.DataFrame(
            {
                "date": dates,
                "yhat": mean.values,
                "yhat_lower": ci.iloc[:, 0].values,
                "yhat_upper": ci.iloc[:, 1].values,
            }
        )
        return out

    if name == "prophet":
        # Load path from registry
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT path FROM models_registry WHERE model_name = ?", ("prophet",)
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            raise FileNotFoundError("Model 'prophet' not found in registry")
        with open(row["path"], "r") as f:
            m = model_from_json(f.read())  # type: ignore[arg-type]
        future = pd.DataFrame({"ds": dates})
        future["marketing_spend"] = float(df["marketing_spend"].iloc[-7:].mean())
        fc = m.predict(future)
        return pd.DataFrame({"date": dates, "yhat": np.expm1(fc["yhat"].values)})

    raise ValueError("Unsupported model name")
