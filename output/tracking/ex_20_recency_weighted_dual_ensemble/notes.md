# EX_20 Recency-Weighted Dual Ensemble

## Goal
- Start from EX_19 production anchor and optimize for recent regimes.
- Use separate Revenue/COGS global weights from recursive OOF folds.

## Components
- core_v3_like
- aligned_keep_avg
- aligned_no_profiles
- naive_lag365

## Validation Setup
- Folds: 2021 and 2022 yearly recursive holdouts.
- Global robust objective: weighted-mean MAE + std-penalty + L2 shrink.
- Fold recency weights: {'fold_2021': 0.4, 'fold_2022': 0.6}
- Robust std penalty: 0.08
- Weight shrink L2: 0.01

## Global Revenue Weights
- core_v3_like: 0.2326
- aligned_keep_avg: 0.3154
- aligned_no_profiles: 0.3780
- naive_lag365: 0.0740

## Global COGS Weights
- core_v3_like: 0.0617
- aligned_keep_avg: 0.6511
- aligned_no_profiles: 0.0688
- naive_lag365: 0.2185

## Outputs
- Candidate: output/submissions/ex_20_recency_weighted_dual_ensemble.csv
- fold_component_scores.csv
- fold_weight_search.csv
- fold_global_metrics.csv
- global_weight_search.csv
- ensemble_summary.csv
- bridge_summary.csv

## Public LB Outcomes (2026-04-21)
- ex_20_recency_weighted_dual_ensemble.csv: 830,584.73892 (IMPROVED, new best)
- ex_20_bridge_w01.csv: 834,398.88891 (IMPROVED vs EX_19 anchor)
- ex_20_bridge_w02.csv: skipped after strong direct-candidate win
- ex_20_bridge_w03.csv: skipped after strong direct-candidate win
- ex_20_bridge_w04.csv: skipped after strong direct-candidate win

## Decision
- Promote ex_20_recency_weighted_dual_ensemble.csv as production anchor.
- Stop EX_20 bridge ladder.
