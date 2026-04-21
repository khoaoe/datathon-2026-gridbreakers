# EX_15 Anchor Bridge

## Objective
- Build very low-drift bridges from 861k anchor to EX_15 hybrid candidate.
- Keep exactly 4 candidates for daily submit cap.

## Inputs
- anchor: output/submissions/ex_06_ensemble_weighted.csv
- candidate: output/submissions/ex_15_hybrid_rev12_cogs14.csv

## Submit Order (lowest drift -> highest drift)
1. ex_15_bridge_w01.csv
2. ex_15_bridge_w02.csv
3. ex_15_bridge_w03.csv
4. ex_15_bridge_w04.csv

## Public LB Outcomes (2026-04-20)
- ex_15_bridge_w01.csv: 932,395.28814
- ex_15_bridge_w04.csv: 927,027.43521
- ex_15_bridge_w02.csv: not submitted (stopped after early regression)
- ex_15_bridge_w03.csv: not submitted (stopped after early regression)

## Decision
- Submitted EX_15 bridges are far worse than anchor 861,132.08456.
- Stop this bridge family and keep anchor as production baseline.
