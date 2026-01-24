from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend import data_processing as dp
from backend import model as mdl


def _make_synthetic_df(n_days: int = 180, start: date | None = None) -> pd.DataFrame:
    """Create a small synthetic dataset with required schema for tests.

    Columns: date, daily_sales, marketing_spend, is_holiday, day_of_week
    """
    start = start or date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    # Trend + weekly seasonality + noise
    t = np.arange(n_days)
    weekly = (np.sin(2 * np.pi * (t % 7) / 7.0) + 1.0) * 100.0
    sales = (
        1000.0 + 0.8 * t + weekly + np.random.RandomState(42).normal(0, 20, size=n_days)
    )
    marketing = 50.0 + 0.1 * t + np.random.RandomState(0).normal(0, 2.0, size=n_days)
    is_holiday = np.array(
        [(1 if (i % 30 == 0) else 0) for i in range(n_days)], dtype=int
    )
    day_names = [pd.Timestamp(d).day_name() for d in dates]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "daily_sales": sales.astype(float),
            "marketing_spend": marketing.astype(float),
            "is_holiday": is_holiday,
            "day_of_week": day_names,
        }
    )
    return df


def test_validate_and_clean_handles_dupes_and_gaps():
    # Build a tiny dirty frame with duplicates and a gap
    base = _make_synthetic_df(10)
    dirty = pd.concat(
        [base.iloc[:5], base.iloc[4:7], base.iloc[9:10]], ignore_index=True
    )
    # Introduce some NaNs that should be interpolated/forward-filled
    dirty.loc[2, "daily_sales"] = np.nan
    dirty.loc[3, "marketing_spend"] = np.nan
    dirty.loc[4, "day_of_week"] = None
    dirty.loc[5, "is_holiday"] = None

    cleaned = dp.validate_and_clean(dirty)

    # Expect continuous daily range from min to max date
    start, end = cleaned["date"].min(), cleaned["date"].max()
    expected_len = (end - start).days + 1
    assert len(cleaned) == expected_len

    # Duplicates removed, schema preserved, no missing required fields
    required = ["date", "daily_sales", "marketing_spend", "day_of_week", "is_holiday"]
    assert all(col in cleaned.columns for col in required)
    assert cleaned[required].isnull().sum().sum() == 0


def test_create_gbdt_features_adds_expected_columns():
    df = _make_synthetic_df(90)
    feat = dp.create_gbdt_features(df)

    # Basic existence checks
    expected_cols = [
        "sales_log",
        "target_sales_diff_log",
        "day_of_week_num_sin",
        "day_of_week_num_cos",
        "lag_target_7",
        "rolling_mean_target_7",
    ]
    for c in expected_cols:
        assert c in feat.columns

    # Plausibility: cyclical encodings bounded in [-1, 1]
    assert (feat["day_of_week_num_sin"].abs() <= 1.0 + 1e-9).all()
    assert (feat["day_of_week_num_cos"].abs() <= 1.0 + 1e-9).all()

    # Lagged target should introduce NaNs near the start
    assert feat["lag_target_7"].isna().head(7).all()


def test_encode_cyclical_bounds():
    df = pd.DataFrame({"m": list(range(12))})
    out = dp.encode_cyclical(df.copy(), "m", 12)
    assert "m_sin" in out.columns and "m_cos" in out.columns
    assert (out["m_sin"].abs() <= 1.0 + 1e-9).all()
    assert (out["m_cos"].abs() <= 1.0 + 1e-9).all()


def test_train_sarimax_and_evaluate_runs():
    # Keep small but ensure >= holdout window
    df = _make_synthetic_df(140)
    df = df.set_index("date").sort_index()
    res, m = mdl.train_sarimax_and_evaluate(df)
    # Sanity: metrics are finite numbers
    assert math.isfinite(m.rmse) and math.isfinite(m.mae) and math.isfinite(m.mape)


def test_train_lgb_and_evaluate_runs_fast(monkeypatch):
    # Reduce training time by monkeypatching LightGBM config
    def _small_model():
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            random_state=42,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=15,
            objective="regression_l1",
            n_jobs=-1,
            verbose=-1,
            colsample_bytree=0.7,
            subsample=0.7,
        )

    monkeypatch.setattr(mdl, "_get_lgb_model", _small_model)

    df = _make_synthetic_df(160)
    df = df.set_index("date").sort_index()
    model, metrics, features = mdl.train_lgb_and_evaluate(df)

    assert hasattr(model, "predict")
    assert isinstance(features, list) and len(features) > 5
    assert (
        math.isfinite(metrics.rmse)
        and math.isfinite(metrics.mae)
        and math.isfinite(metrics.mape)
    )
