# EX_27 EX_22 Baseline + Mega Doubles

## Goal
- Revert to EX_22's exact calendar logic (no holidays.VN) to restore 796k anchor performance.
- Keep our single winning insight from deep research: Q4 mega_doubles cause drops.

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
- core_v3_like: 0.0859
- aligned_keep_avg: 0.2159
- aligned_no_profiles: 0.2623
- aligned_recency_profiles: 0.2233
- naive_lag365: 0.2126

## Global COGS Weights
- core_v3_like: 0.1389
- aligned_keep_avg: 0.5097
- aligned_no_profiles: 0.0619
- aligned_recency_profiles: 0.0000
- naive_lag365: 0.2895

## Outputs
- Candidate: output/submissions/ex_27_ex22_baseline_plus_mega_doubles.csv
- fold_component_scores.csv
- fold_weight_search.csv
- fold_global_metrics.csv
- global_weight_search.csv
- ensemble_summary.csv
- bridge_summary.csv
