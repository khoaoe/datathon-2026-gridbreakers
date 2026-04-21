# EX_19 Robust Dual-Weight Ensemble

## Goal
- Deepen post-EX_18 ensemble stage with stronger robustness constraints.
- Use separate Revenue/COGS global weights from recursive OOF folds.

## Components
- core_v3_like
- aligned_keep_avg
- aligned_drop_avg
- aligned_no_profiles
- naive_lag365
- naive_lag365_lag7_blend

## Validation Setup
- Folds: 2020, 2021, 2022 yearly recursive holdouts.
- Global robust objective: mean MAE + std-penalty + L2 shrink to uniform.
- Robust std penalty: 0.1
- Weight shrink L2: 0.02

## Global Revenue Weights
- core_v3_like: 0.2718
- aligned_keep_avg: 0.3818
- aligned_drop_avg: 0.0080
- aligned_no_profiles: 0.1367
- naive_lag365: 0.1869
- naive_lag365_lag7_blend: 0.0148

## Global COGS Weights
- core_v3_like: 0.3146
- aligned_keep_avg: 0.4654
- aligned_drop_avg: 0.0052
- aligned_no_profiles: 0.0177
- naive_lag365: 0.1924
- naive_lag365_lag7_blend: 0.0047

## Outputs
- Candidate: output/submissions/ex_19_robust_dual_weight_ensemble.csv
- fold_component_scores.csv
- fold_weight_search.csv
- fold_global_metrics.csv
- global_weight_search.csv
- ensemble_summary.csv
- bridge_summary.csv

## Public LB Outcomes (2026-04-21)
- ex_19_robust_dual_weight_ensemble.csv: 834,681.84856 (IMPROVED, new best)
- ex_19_bridge_w01.csv: 859,529.71007 (IMPROVED vs EX_18 anchor)
- ex_19_bridge_w02.csv: skipped after strong direct-candidate win
- ex_19_bridge_w03.csv: skipped after strong direct-candidate win
- ex_19_bridge_w04.csv: skipped after strong direct-candidate win

## Decision
- Promote ex_19_robust_dual_weight_ensemble.csv as production anchor.
- Stop EX_19 bridge ladder.
