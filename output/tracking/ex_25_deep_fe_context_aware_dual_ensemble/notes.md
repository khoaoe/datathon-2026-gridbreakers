# EX_25 Deep FE Context-Aware Dual Ensemble

## Goal
- Start from EX_24 architecture, but fix the misguided FE assumptions.
- Replaced naive Tet spikes with Tet slump windows.
- Restrict double dates to 2018+ and mega months.
- Add pre-April30 and Summer Kickoff vacation spikes.

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
- core_v3_like: 0.2110
- aligned_keep_avg: 0.0000
- aligned_no_profiles: 0.1640
- aligned_recency_profiles: 0.4909
- naive_lag365: 0.1341

## Global COGS Weights
- core_v3_like: 0.1628
- aligned_keep_avg: 0.4222
- aligned_no_profiles: 0.0518
- aligned_recency_profiles: 0.0029
- naive_lag365: 0.3603

## Outputs
- Candidate: output/submissions/ex_25_deep_fe_context_aware_dual_ensemble.csv
- fold_component_scores.csv
- fold_weight_search.csv
- fold_global_metrics.csv
- global_weight_search.csv
- ensemble_summary.csv
- bridge_summary.csv
