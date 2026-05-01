# EX_24 Deep FE Recency-Profile Dual Ensemble

## Goal
- Start from EX_22 production winner with controlled FE delta.
- Add holidays.VN + double-date calendar features to deep FE pipeline.
- Use separate Revenue/COGS global weights from recursive OOF folds.

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
- core_v3_like: 0.1324
- aligned_keep_avg: 0.0616
- aligned_no_profiles: 0.4760
- aligned_recency_profiles: 0.1678
- naive_lag365: 0.1621

## Global COGS Weights
- core_v3_like: 0.0251
- aligned_keep_avg: 0.2042
- aligned_no_profiles: 0.5911
- aligned_recency_profiles: 0.0000
- naive_lag365: 0.1796

## Outputs
- Candidate: output/submissions/ex_24_deep_fe_holidays_double_dates_dual_ensemble.csv
- fold_component_scores.csv
- fold_weight_search.csv
- fold_global_metrics.csv
- global_weight_search.csv
- ensemble_summary.csv
- bridge_summary.csv
