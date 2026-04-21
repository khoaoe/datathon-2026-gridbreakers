# Next Agent Brief: Holiday and Special Events Feature Engineering

## Mission
- Focus on extending the successful `ex_22` deep feature engineering (which achieved **796,018** MAE on the leaderboard) by building robust temporal features using standard libraries.
- Keep work leakage-safe and reproducible for our recursive forecasting setup.

## Current Anchor
- **Best public score:** `796,018.49022` (`ex_22_deep_fe_holidays_dual_ensemble.csv`)
- **Current active FE pipeline:** `modeling/feature_engineering.py`

## Next Approach & Recommended Research Direction
1. **Use the `holidays` Python Library:** 
   - Refactor and extend the current hardcoded holiday logic to use the official Python `holidays` package.
   - You **must** include Vietnamese holidays (`holidays.VN()`) since the e-commerce data shows major cyclical anomalies around Tet and other national holidays in Vietnam.
2. **Double Dates for E-commerce:** 
   - Explicitly add "Double Date" indicator features (e.g., 1/1, 2/2, ..., 9/9, 10/10, 11/11, 12/12).
   - This is a fashion e-commerce dataset in Southeast Asia where double-date sale campaigns (like Shopee and Lazada mega sales) drive massive revenue spikes. Marking the days leading up to and the day of these sales is critical.
3. **Model Integration:**
   - Add these new features cleanly into `build_calendar_features()` in `feature_engineering.py`.
   - Create a new experiment script (e.g., `ex_24`) based on `ex_22`'s dual-ensemble structure to validate and submit these new features.

## Mandatory First Step
- Read `approaches.md` and `dataset_summary.md` to understand context.
- Check `experiments.md` and `output/tracking/lb_scores.csv` to avoid repeating failures.

## Hard Constraints
- **Recursive Safety:** All new date features must be based on the calendar date, which is known in advance. Do not use lags of exogenous variables that are unavailable during the 548-day test period.
- **Log Everything:** Log every LB outcome to `output/tracking/lb_scores.csv` and `experiments.md`.

## Note on EX_23
- `ex_23` (XGBoost + LightGBM ensemble) was removed due to excessive runtime and overfitting. Stick to the `ex_22` LightGBM ensemble architecture for the next few tests.
