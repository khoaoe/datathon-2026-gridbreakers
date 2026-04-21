# EX_18 Ensemble-Step Deep Research

## Goal
- Reintroduce ensemble stage before anchor bridging.
- Use fold-level recursive optimization, not single-candidate drift.

## Components
- core_v3_like
- aligned_keep_avg
- aligned_drop_avg
- naive_lag365

## Validation Setup
- Folds: 2021 and 2022 yearly recursive holdouts.
- Objective: Revenue MAE + 0.4 * COGS MAE.

## Final Weights
- core_v3_like: 0.1956
- aligned_keep_avg: 0.6598
- aligned_drop_avg: 0.0000
- naive_lag365: 0.1446

## Outputs
- Candidate: output/submissions/ex_18_ensemble_step.csv
- fold_component_scores.csv
- fold_weight_search.csv
- ensemble_summary.csv
- bridge_summary.csv

## Public LB Outcomes (2026-04-20)
- ex_18_bridge_w01.csv: 932,735.22259 (FAILED)
- ex_18_ensemble_step.csv: 859,853.23873 (IMPROVED, new best)
- ex_18_bridge_w02.csv: skipped
- ex_18_bridge_w03.csv: skipped
- ex_18_bridge_w04.csv: skipped

## Decision
- Promote `ex_18_ensemble_step.csv` as production anchor.
- Stop EX_18 bridge ladder after w01 regression.
