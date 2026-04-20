# Feature Engineering Deepwork Log

## Branch
feature-engineering

## Evaluation Setup
- Model: HistGradientBoostingRegressor
- Folds: 5 expanding-window time-series folds (val_days=365, gap_days=0)
- Minimum training window: 730 days
- Metric priority: MAE (lower is better)

## Best Method
- method: baseline_plus_orders_payments_lags
- mean_mae: 630954.32
- mean_delta_vs_baseline: -3625.37

## Worst Method
- method: baseline_plus_promo_std_profiles
- mean_mae: 674944.24
- mean_delta_vs_baseline: +40364.55

## Files
- method_log.csv: fold-level metrics per method
- method_summary.csv: mean ranking across folds