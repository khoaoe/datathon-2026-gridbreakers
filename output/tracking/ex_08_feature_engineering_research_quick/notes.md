# EX_08 Deeper Feature Engineering Research

## Context
- Provided context: XGBoost(v3) public score ~971k, Ensemble public score ~861k.
- Goal: go deeper on feature engineering with time-series K-fold research.

## Evaluation Setup
- Run mode: quick
- Split: expanding time-series K-fold (2 folds, 180-day validation)
- Fold policy: min train history = 2,555 days
- Estimators/iters: LightGBM=250, HistGBR=250
- Metric: MAE (lower is better)
- Model: LightGBM fallback HistGradientBoostingRegressor

## Best Method
- method: core_plus_selected_aux_lags
- family: aux_lags
- mean_mae: 565394.21
- mean_delta_vs_baseline: -3251.39
- fold_wins: 1
- improve_ratio: 0.50
- worst_fold_delta: +277.90

## Runner Up
- method: core_plus_promo_interactions
- family: promo
- mean_mae: 567042.14
- mean_delta_vs_baseline: -1603.46

## Stable Delta Shortlist
- core_plus_selected_aux_lags (aux_lags): mean_delta=-3251.39, improve_ratio=0.50, worst_fold_delta=+277.90
- core_plus_promo_interactions (promo): mean_delta=-1603.46, improve_ratio=0.50, worst_fold_delta=+526.64
- core_plus_regime_features (regime): mean_delta=+243.99, improve_ratio=0.50, worst_fold_delta=+2871.94

## Suggested Next Production Tests
- Wire top 1-2 FE bundles into EX_03 and EX_04 train scripts.
- Keep leakage-safe profile source (pre-val rows only during local eval).
- Re-run weighted ensemble with new submissions.

## Files
- fold_log.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research_quick/fold_log.csv
- method_summary.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research_quick/method_summary.csv
- feature_importance_by_method.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research_quick/feature_importance_by_method.csv
- best_method_top_features.csv: /home/pineapple/Desktop/projects/datathon-2026-gridbreakers/output/tracking/ex_08_feature_engineering_research_quick/best_method_top_features.csv