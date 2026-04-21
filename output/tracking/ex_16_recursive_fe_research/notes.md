# EX_16 Recursive FE Research

## Goal
- Rank target-aligned FE variants using recursive holdout metrics.

## Methods
- aligned_drop_avg
- aligned_keep_avg
- aligned_drop_avg_selected_aux
- aligned_keep_avg_selected_aux

## Best Method
- aligned_keep_avg
- Recursive Revenue MAE: 615,249.60
- Recursive COGS MAE: 542,611.54

## Outputs
- Candidate: output/submissions/ex_16_aligned_keep_avg.csv
- Bridge summary: output/tracking/ex_16_recursive_fe_research/bridge_summary.csv

## Public LB Outcomes (2026-04-20)
- ex_16_bridge_w01.csv: 933,137.41901
- ex_16_bridge_w04.csv: 930,099.17893
- ex_16_bridge_w02.csv: not submitted (stopped after early regression)
- ex_16_bridge_w03.csv: not submitted (stopped after early regression)

## Decision
- Submitted EX_16 bridges regressed strongly versus anchor 861,132.08456.
- Stop EX_16 bridge family and keep anchor as production baseline.
