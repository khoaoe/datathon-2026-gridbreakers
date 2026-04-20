# EX_08 Deeper Feature Engineering Research

## Context
- Provided context: XGBoost(v3) public score ~971k, Ensemble public score ~861k.
- Goal: go deeper on feature engineering with time-series K-fold research.

## Evaluation Setup
- Split: expanding time-series K-fold (5 folds, each 365-day validation)
- Metric: MAE (lower is better)
- Model: LightGBM fallback HistGradientBoostingRegressor

## Best Method
- method: core_plus_promo_interactions
- mean_mae: 614770.73
- mean_delta_vs_baseline: -14299.60
- fold_wins: 2

## Runner Up
- method: core_plus_selected_aux_lags
- mean_mae: 628098.27
- mean_delta_vs_baseline: -972.06

## Suggested Next Production Tests
- Wire top 1-2 FE bundles into EX_03 and EX_04 train scripts.
- Keep leakage-safe profile source (pre-val rows only during local eval).
- Re-run weighted ensemble with new submissions.

## Files
- fold_log.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research/fold_log.csv
- method_summary.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research/method_summary.csv
- feature_importance_by_method.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research/feature_importance_by_method.csv
- best_method_top_features.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research/best_method_top_features.csv