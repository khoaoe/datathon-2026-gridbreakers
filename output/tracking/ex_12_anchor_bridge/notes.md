# EX_12 Anchor Bridge

## Objective
- Build low-drift bridges from 861k anchor to EX_12 selected-delta model.
- Keep exactly 4 candidates for daily submit cap.

## Inputs
- anchor: output/submissions/ex_06_ensemble_weighted.csv
- candidate: output/submissions/ex_12_lgbm_selected_deltas.csv

## Submit Order (lowest drift -> highest drift)
1. ex_12_bridge_w02.csv
2. ex_12_bridge_w04.csv
3. ex_12_bridge_w06.csv
4. ex_12_bridge_w08.csv

## Public LB Outcomes (2026-04-20)
- ex_12_bridge_w02.csv: 930,846.50105
- ex_12_bridge_w04.csv: 927,547.44210
- ex_12_bridge_w08.csv: 921,204.48695
- ex_12_bridge_w06.csv: skipped by decision (quota + low expected value)

## Decision
- All submitted EX_12 bridges regressed versus anchor 861,132.08456.
- Keep anchor as production baseline and do not commit EX_12 as an improvement.
- Do not submit w06; treat it as deprecated for this candidate family.
