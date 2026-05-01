# EX_26 Clean Continuous Calendar Dual Ensemble

## Goal
- Start from EX_25 architecture, but remove ALL rigid binary calendar windows.
- Rely exclusively on continuous distance features (`days_to_tet`, `days_to_mega_double`, `days_to_vn_holiday`).
- Allow tree to organically learn shapes instead of forcing step functions.

## Components
- core_v3_like
- aligned_keep_avg
- aligned_no_profiles
- aligned_recency_profiles
- naive_lag365

## Validation Setup
- Folds: 2020, 2021, 2022 yearly recursive holdouts.
- Global robust objective: weighted-mean MAE + std-penalty + L2 shrink.
- Fold recency weights: {'fold_2020': 0.2, 'fold_2021': 0.3, 'fold_2022': 0.5}
- Robust std penalty: 0.08
- Weight shrink L2: 0.01
- Recency profile decay: 0.003

## Global Revenue Weights
- core_v3_like: 0.0477
- aligned_keep_avg: 0.0026
- aligned_no_profiles: 0.6793
- aligned_recency_profiles: 0.0988
- naive_lag365: 0.1715

## Global COGS Weights
- core_v3_like: 0.2191
- aligned_keep_avg: 0.4292
- aligned_no_profiles: 0.0033
- aligned_recency_profiles: 0.0075
- naive_lag365: 0.3409

## Outputs
- Candidate: output/submissions/ex_26_clean_continuous_calendar_dual_ensemble.csv
- fold_component_scores.csv
- fold_weight_search.csv
- fold_global_metrics.csv
- global_weight_search.csv
- ensemble_summary.csv
- bridge_summary.csv
