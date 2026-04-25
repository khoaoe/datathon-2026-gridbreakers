# Modeling Research Plan: Breaking the Recursive Snowball Effect

**Goal:** Bridge the MAE gap from 795k to the 620k Leaderboard target. We have isolated the root cause of our model's downward bias and formulated a specific architectural plan to fix it.

## The Breakthrough: The Recursive Snowball Effect
In the EX-45 through EX-48 series, we discovered exactly why the model chronically underpredicts the test set (predicting ~3.7M instead of the required ~4.1M+):

1. **The Model is NOT Biased:** On out-of-sample direct prediction (when true historical lags are known), the model achieves an excellent MAE of 562k with virtually no downward bias.
2. **Recursive Degradation:** When forced into recursive 548-day prediction, errors compound. The model naturally undershoots January/February (due to Q4 seasonality crashes). These low predictions become the `lag28` and `lag60` features fed into the spring surge, causing the model to massively underpredict March/April. This drags the entire forecast down by ~10% (the "Snowball Effect").
3. **Stateless CV Success:** In EX-48, we proved that stripping ALL recursive features (lags, rolling means) completely cured this in Cross Validation, yielding a stellar CV MAE of 794k (beating our best baseline of 845k).

## The EX-48 Problem: Linear Extrapolation over a Regime Shift
While EX-48 (Stateless Hybrid) had incredible CV scores, its test set prediction was disastrously low (~2.9M). Why? 
- It used a Ridge Regression on `time_index` (2013-2024) to capture the trend.
- Revenue dropped from ~5M (2013-2018) to ~3M (2019-2022). 
- The linear model extrapolated this 10-year downward trend into 2023-2024, missing the post-2021 recovery completely.

## Next Agent Instructions: The Direct Multi-Step / Corrected Hybrid Architecture

To reach 620k, you must solve the EX-48 trend extrapolation issue or build a Direct Multi-Step pipeline. Choose one of the following paths:

### Path A: The "Recent Trend" Stateless Hybrid (Fixing EX-48)
1. **Truncated Trend Training:** Train the linear trend model (Ridge or simple LinearRegression) ONLY on recent data (e.g., `Date >= '2021-01-01'`) where the recovery trajectory is positive (+12% YoY).
2. **Full Residual Training:** Subtract the recent-trend predictions from actuals to get residuals. Train the LightGBM seasonality model (with NO recursive lags, using Calendar/Profiles only) on the full historical dataset to learn robust seasonality patterns.
3. **Combine:** Predict `Recent_Trend(future) + LGBM_Residual(future)`. This is stateless, has no snowball effect, and extrapolates the correct upward trajectory.

### Path B: Direct Multi-Step Forecasting (No Recursion)
If keeping lag features is strictly necessary for accuracy, abandon recursive loops:
1. Train separate models for different horizons. E.g., train one model to predict `Revenue_{t+30}`, another for `Revenue_{t+90}`, etc.
2. In practice, you can build a model that uses `lag365` as its primary feature. Since the first year of the test set is 2023, its `lag365` values come directly from actual 2022 data—meaning **zero recursion** for the first 365 days of the test set.

### Path C: Growth / AOV Modeling
As discovered in exploratory analysis, Order Counts are flat, but Average Order Value (AOV) grew 47% from 2013-2022. 
- Try forecasting `AOV` and `Orders` separately.
- AOV follows a strong monotonic upward trend that might be easier to extrapolate with a linear model than volatile raw revenue. Multiply the forecasts: `Revenue = AOV_pred * Orders_pred`.

**Immediate Priority:** Start with **Path A (Recent Trend Stateless Hybrid)** as the code from EX-48 is 90% ready and only requires truncating the trend model's training window.
