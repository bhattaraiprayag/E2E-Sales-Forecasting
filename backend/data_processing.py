from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_local_dataset() -> pd.DataFrame:
    """Load dataset from ./data (xlsx preferred, fallback to csv).

    Returns a DataFrame with at least the required schema columns. Raises
    FileNotFoundError if neither file exists.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    xlsx_path = data_dir / "ecommerce_sales_data.xlsx"
    csv_path = data_dir / "ecommerce_sales_data.csv"
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path, parse_dates=["date"])  # type: ignore[arg-type]
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"])  # type: ignore[arg-type]
    else:
        raise FileNotFoundError("No dataset found in ./data (xlsx or csv)")
    return df


def load_from_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load uploaded CSV/XLSX content into a DataFrame with parsed dates.

    Accepts .xlsx/.xls/.csv. Raises ValueError for unsupported types.
    """
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(file_bytes), parse_dates=["date"])  # type: ignore[arg-type]
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes), parse_dates=["date"])  # type: ignore[arg-type]
    raise ValueError("Unsupported file type. Use .csv or .xlsx")


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and produce a clean, daily-frequency frame.

    - Ensures required columns exist
    - Sorts by date and removes duplicate dates
    - Sets daily frequency and interpolates numeric columns
    - Forward-fills categorical columns
    - Drops constant `product_category` if present
    """
    required = {"date", "daily_sales", "marketing_spend", "day_of_week", "is_holiday"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"])  # unique dates

    df = df.set_index("date").asfreq("D")
    df["daily_sales"] = df["daily_sales"].interpolate("linear")
    df["marketing_spend"] = df["marketing_spend"].interpolate("linear")
    df["day_of_week"] = df["day_of_week"].ffill()
    df["is_holiday"] = df["is_holiday"].ffill()
    df = df.reset_index().rename(columns={"index": "date"})

    # drop zero-variance category if present
    if "product_category" in df.columns and df["product_category"].nunique() <= 1:
        df = df.drop(columns=["product_category"])  # type: ignore[arg-type]
    return df


def to_db_rows(df: pd.DataFrame) -> List[Tuple[str, float, float, int, str]]:
    """Convert a clean DataFrame into tuples suitable for SQLite insertion."""
    rows: List[Tuple[str, float, float, int, str]] = []
    for _, r in df.iterrows():
        rows.append(
            (
                pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                float(r["daily_sales"]),
                float(r["marketing_spend"]),
                int(r["is_holiday"]),
                str(r["day_of_week"]),
            )
        )
    return rows


def encode_cyclical(df: pd.DataFrame, col: str, max_val: int) -> pd.DataFrame:
    """Add sin/cos cyclical encodings for an integer-valued periodic column."""
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
    return df


def create_gbdt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature set for gradient boosting on Δlog(sales).

    Adds:
    - Log/differenced targets
    - Calendar/cyclical encodings
    - Lagged targets and rolling stats
    - Marketing lags, rolling mean, and interactions
    """
    df_feat = df.copy()
    if not isinstance(df_feat.index, pd.DatetimeIndex):
        if "date" in df_feat.columns:
            df_feat = df_feat.set_index(pd.to_datetime(df_feat["date"]))
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'date' column")

    df_feat["sales_log"] = np.log1p(df_feat["daily_sales"])
    df_feat["target_sales_diff_log"] = df_feat["sales_log"].diff(1)

    idx = df_feat.index
    assert isinstance(idx, pd.DatetimeIndex)
    df_feat["month_of_year"] = idx.month
    df_feat["day_of_month"] = idx.day
    df_feat["day_of_week_num"] = idx.dayofweek
    df_feat["week_of_year"] = idx.isocalendar().week.astype(int)
    df_feat["quarter"] = idx.quarter
    df_feat["is_month_start"] = idx.is_month_start.astype(int)
    df_feat["is_month_end"] = idx.is_month_end.astype(int)
    df_feat["is_quarter_start"] = idx.is_quarter_start.astype(int)
    df_feat["is_quarter_end"] = idx.is_quarter_end.astype(int)

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
