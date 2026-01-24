# Feature Engineering

This document describes the feature engineering used across models and the rationale behind each choice. The implementation lives in `backend/data_processing.py` and is consumed by `backend/model.py`.

## Common Inputs
The raw schema expected by the system is:
- `date` (daily, unique)
- `daily_sales` (float)
- `marketing_spend` (float)
- `is_holiday` (int 0/1)
- `day_of_week` (string)

Cleaning (`validate_and_clean`) enforces daily frequency, sorts by date, removes duplicates, and interpolates numeric gaps.

## LightGBM (Gradient Boosting)
LightGBM operates on a rich, tabular feature set capturing temporal dynamics and exogenous signals. The target is the first difference of the log sales:
- `sales_log = log1p(daily_sales)`
- `target_sales_diff_log = sales_log.diff(1)`

### Calendar & Cyclical Encodings
- `day_of_week_num` (0-6), `week_of_year`, `month_of_year`, `day_of_month`, `quarter`
- Cyclical encodings using sin/cos for: `day_of_week_num`, `week_of_year`, `month_of_year`, `day_of_month`
  - Rationale: preserve cyclical geometry (e.g., Sunday is closer to both Monday and previous Saturday; in comparison, one-hot encoding (or any other approach) would not preserve that semantic information).

### Lagged Targets (autoregression on Δlog(y))
- `lag_target_{1,2,3,7,14,28}`
  - Rationale: model short- and medium-term autocorrelation, including weekly seasonality.

### Rolling Statistics (on lagged Δlog(y))
- `rolling_mean_target_{7,14,30}`
- `rolling_std_target_{7,14,30}`
  - Rationale: capture local trend/volatility.

### Marketing Effects
- `marketing_lag_{1,3,7}`
- `marketing_rolling_mean_7`
- `marketing_x_holiday` (interaction)
  - Rationale: allow delayed/aggregated lift from marketing and holiday interactions.

### Columns Dropped Prior to Modeling
- Non-numeric or redundant identifiers: `day_of_week`, `month_of_year`, `day_of_month`, `day_of_week_num`, `week_of_year`, `quarter` (raw) are dropped after encodings are added.

### Forecasting Strategy
Recursive forecasting on Δlog(y): we iteratively roll the history forward, predicting next-step differences, accumulating them onto the last log(sales), and exponentiating back to the original scale via `expm1`.

## SARIMAX (Classical)
- Endogenous: `daily_sales`
- Seasonal period: 7 (weekly)
- Order: (1,1,1) with seasonal (1,1,1,7)
- Exogenous: `is_holiday` (binary)
- Rationale: strong weekly seasonality and holiday effect captured in a well-known baseline.

## Prophet (Baseline)
- Trained on `log1p(daily_sales)`
- Regressor: `marketing_spend`
- Holidays: rows where `is_holiday==1` become Prophet holiday calendar (`holiday="special_day"`)
- Rationale: robust baseline for trend/seasonality with simple regressor support.

## Assumptions for Future Exogenous Variables
For future periods when exogenous variables are unknown:
- `marketing_spend`: use last 7-day mean
- `is_holiday`: assume 0
- `day_of_week`: derived from date

These assumptions are documented in the Forecasts tab and can be extended to scenario planning in future iterations.
