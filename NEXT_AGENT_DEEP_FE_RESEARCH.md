# Next Agent Brief: Deep Feature Engineering Research

## Mission
- Do deeper feature engineering research focused on improving public leaderboard MAE below anchor 861132.08456.
- Keep work leakage-safe and reproducible.

## Mandatory First Step (No Exceptions)
- Read every markdown document in repo before proposing new experiment.

### Required Reading List
- approaches.md
- dataset_summary.md
- README.md
- experiments.md
- output/tracking/ex_08_feature_engineering_research/notes.md
- output/tracking/ex_11_anchor_bridge/notes.md

## Hard Constraints
- Do not overwrite anchor submission: output/submissions/ex_06_ensemble_weighted.csv.
- Log every leaderboard outcome to output/tracking/lb_scores.csv.
- If experiment fails on LB, mark it as FAILED in experiments.md notes and lb_scores.csv notes.
- Prefer small controlled deltas over large blend/model jumps.

## Failed Experiments To Avoid Repeating

### Failed Recovery Family
- ex_06_recovery_v1_lgbm_up.csv -> 927804.15516 (FAILED)
- ex_06_recovery_v2_lgbm_up_more.csv -> 926067.29049 (FAILED)

### Failed Full-Power Microblend Family
- ex_10_fp_ex06opt_w01.csv -> 932988.67676 (FAILED)
- ex_10_fp_med4_w01.csv -> 933214.06121 (FAILED)
- ex_10_fp_mean4_w01.csv -> 933222.02871 (FAILED)
- ex_10_fp_final_w005.csv -> 933297.12358 (FAILED)

### Failed FE Refresh Weighted Ensemble
- ex_06_ensemble_weighted_fe_refresh.csv -> ~879000 (FAILED, user-reported)

## Current Useful Assets
- modeling/feature_engineering.py (active FE pipeline)
- modeling/ex_03_lgbm.py (leakage-safe LGBM training/inference)
- modeling/ex_08_feature_engineering_research.py (method ranking framework)
- modeling/ex_11_anchor_bridge_blends.py (small bridge blend generator)

## Recommended Research Direction
1. Re-run ex_08_feature_engineering_research with stricter time-split validation and per-family ablation.
2. Keep only top 1-2 FE deltas that beat baseline_v3_core consistently across folds.
3. Train ex_03_lgbm with selected FE deltas only.
4. Build minimal bridge variants from anchor (small weights only).
5. Select at most 4 submissions for daily cap, ordered lowest drift to highest drift.

## Deliverables
- Updated tracker files:
  - output/tracking/lb_scores.csv
  - experiments.md
- One short note in output/tracking/ with:
  - What changed in FE
  - Why it should generalize
  - Which failed families were intentionally avoided

## Quick Sanity Checklist Before Submit
- Features available at prediction time only.
- No validation leakage from future periods.
- Submission has 548 rows and no NaN.
- Anchor file unchanged.
- Failed families above not reintroduced.
