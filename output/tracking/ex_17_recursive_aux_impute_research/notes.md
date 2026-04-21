# EX_17 Recursive Aux-Impute FE Research

## Goal
- Test whether lagged auxiliary features help when future aux values are unknown.

## Methods
- baseline_keep_avg
- keep_avg_exo_zero
- keep_avg_exo_profile

## Validation Setup
- Recursive holdout folds: 2021 and 2022.
- Score = Revenue MAE + 0.4 * COGS MAE.

## Best Method
- baseline_keep_avg
- Mean recursive Revenue MAE: 607,888.67
- Mean recursive COGS MAE: 516,705.00

## Outputs
- Candidate: output/submissions/ex_17_baseline_keep_avg.csv
- fold_results.csv
- method_summary.csv
- bridge_summary.csv

## Suggested Submit Ladder
- Start with ex_17_bridge_w01.csv (lowest drift).
- If early LB signal regresses, stop this family.
- Only continue to ex_17_bridge_w02.csv then ex_17_bridge_w03.csv if signal improves.

## Public LB Outcomes (2026-04-20)
- ex_17_bridge_w01.csv: 932,607 (FAILED, user-reported)
- ex_17_bridge_w02.csv: not submitted (stopped after early regression)
- ex_17_bridge_w03.csv: not submitted (stopped after early regression)
- ex_17_bridge_w04.csv: not submitted (stopped after early regression)

## Decision
- Stop EX_17 bridge family.
- Keep ex_06_ensemble_weighted.csv (861,132.08456) as production anchor.
