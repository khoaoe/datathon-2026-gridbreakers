# Deep FE Round 3 Notes (2026-04-21)

## What Changed in FE
- Removed failed exploratory artifact families (ex13/ex14/ex15 files) to reduce workspace noise.
- Ran recursive-aware FE ablation in EX_16 with target-aligned feature constraints:
  - aligned_drop_avg
  - aligned_keep_avg
  - aligned_drop_avg_selected_aux
  - aligned_keep_avg_selected_aux
- Selected `aligned_keep_avg` as best by recursive score (Revenue-prioritized blended objective).

## Why This Should Generalize Better
- Evaluation emphasized recursive holdout behavior, not only teacher-forced validation.
- Target alignment prevents opposite-target lag leakage at inference time.
- Keeping avg_* profile features improved COGS recursion substantially versus drop_avg variants.
- Candidate is deployed through very low-drift anchor bridges (w01..w04) to control leaderboard risk.

## Failed Families Intentionally Avoided
- Recovery family not reused:
  - ex_06_recovery_v1_lgbm_up.csv
  - ex_06_recovery_v2_lgbm_up_more.csv
- Full-power microblend family not reused:
  - ex_10_fp_ex06opt_w01.csv
  - ex_10_fp_med4_w01.csv
  - ex_10_fp_mean4_w01.csv
  - ex_10_fp_final_w005.csv
- FE refresh weighted ensemble not reused:
  - ex_06_ensemble_weighted_fe_refresh.csv
- Stopped ex_15 bridge family after early regression signal.

## Outputs
- Candidate: output/submissions/ex_16_aligned_keep_avg.csv
- Bridges: output/submissions/ex_16_bridge_w01.csv .. ex_16_bridge_w04.csv
- Tracking: output/tracking/ex_16_recursive_fe_research/

## EX_16 Public LB Outcomes
- ex_16_bridge_w01.csv: 933,137.41901 (FAILED)
- ex_16_bridge_w04.csv: 930,099.17893 (FAILED)
- ex_16_bridge_w02.csv: skipped after early regression signal
- ex_16_bridge_w03.csv: skipped after early regression signal

## Decision
- Stop EX_16 bridge family.
- Keep ex_06_ensemble_weighted.csv (861,132.08456) as production anchor.

## EX_17 Follow-up (Aux-Impute Deep Dive)
- Added future-unknown exogenous lag handling with two strategies:
  - keep_avg_exo_zero
  - keep_avg_exo_profile
- Evaluated on recursive holdout folds (2021 and 2022).
- Result: both exogenous variants regressed versus baseline recursive stability.
- Best method stayed baseline_keep_avg (mean recursive):
  - Revenue MAE: 607,888.67
  - COGS MAE: 516,705.00
- Generated files:
  - Candidate: output/submissions/ex_17_baseline_keep_avg.csv
  - Bridges: output/submissions/ex_17_bridge_w01.csv .. ex_17_bridge_w04.csv
  - Tracking: output/tracking/ex_17_recursive_aux_impute_research/

## EX_17 Public LB Outcomes
- ex_17_bridge_w01.csv: 932,607 (FAILED, user-reported)
- ex_17_bridge_w02.csv: skipped after early regression signal
- ex_17_bridge_w03.csv: skipped after early regression signal
- ex_17_bridge_w04.csv: skipped after early regression signal

## Decision Update
- Stop EX_17 bridge family.
- Keep ex_06_ensemble_weighted.csv (861,132.08456) as production anchor.

## EX_18 Ensemble-Step Deep Research
- Reintroduced explicit ensemble stage before anchor bridge generation.
- Components evaluated recursively on 2021 and 2022 folds:
  - core_v3_like
  - aligned_keep_avg
  - aligned_drop_avg
  - naive_lag365
- Fold-optimized weights (mean):
  - core_v3_like: 0.1956
  - aligned_keep_avg: 0.6598
  - aligned_drop_avg: ~0.0000
  - naive_lag365: 0.1446
- Local recursive ensemble means:
  - Revenue MAE: 570,635.64
  - COGS MAE: 504,370.34
  - Score (Rev + 0.4*COGS): 772,383.78
- Generated files:
  - Candidate: output/submissions/ex_18_ensemble_step.csv
  - Bridges: output/submissions/ex_18_bridge_w01.csv .. ex_18_bridge_w04.csv
  - Tracking: output/tracking/ex_18_ensemble_step_research/

## EX_18 Submit Ladder
- Recommended order: ex_18_bridge_w01.csv -> w02 -> w03 -> w04.
- Stop immediately on first regression signal.

## EX_18 Public LB Outcomes
- ex_18_bridge_w01.csv: 932,735.22259 (FAILED)
- ex_18_ensemble_step.csv: 859,853.23873 (IMPROVED, new best)
- ex_18_bridge_w02.csv: skipped after w01 regression and direct candidate improvement
- ex_18_bridge_w03.csv: skipped after w01 regression and direct candidate improvement
- ex_18_bridge_w04.csv: skipped after w01 regression and direct candidate improvement

## Decision Update
- Promote ex_18_ensemble_step.csv as new production anchor.
- Stop EX_18 bridge ladder (no further bridge submissions needed).

## EX_19 Robust Dual-Weight Ensemble (Deep Pass)
- Built a deeper ensemble stage on top of EX_18 anchor workflow.
- Validation setup:
  - 3 yearly recursive folds: 2020, 2021, 2022
  - Component pool:
    - core_v3_like
    - aligned_keep_avg
    - aligned_drop_avg
    - aligned_no_profiles
    - naive_lag365
    - naive_lag365_lag7_blend
  - Target-specific global weights (separate Revenue and COGS vectors)
  - Robust objective: mean MAE + fold-std penalty + L2 shrink to uniform
- Global weights (Revenue):
  - core_v3_like: 0.2718
  - aligned_keep_avg: 0.3818
  - aligned_drop_avg: 0.0080
  - aligned_no_profiles: 0.1367
  - naive_lag365: 0.1869
  - naive_lag365_lag7_blend: 0.0148
- Global weights (COGS):
  - core_v3_like: 0.3146
  - aligned_keep_avg: 0.4654
  - aligned_drop_avg: 0.0052
  - aligned_no_profiles: 0.0177
  - naive_lag365: 0.1924
  - naive_lag365_lag7_blend: 0.0047
- Local recursive means (global weights):
  - Revenue MAE: 599,677.18
  - COGS MAE: 522,071.36
  - Score (Rev + 0.4*COGS): 808,505.72
- Generated files:
  - Candidate: output/submissions/ex_19_robust_dual_weight_ensemble.csv
  - Bridges: output/submissions/ex_19_bridge_w01.csv .. ex_19_bridge_w04.csv
  - Tracking: output/tracking/ex_19_robust_dual_weight_ensemble/

## EX_19 Public LB Outcomes
- ex_19_bridge_w01.csv: 859,529.71007 (IMPROVED vs EX_18 anchor)
- ex_19_robust_dual_weight_ensemble.csv: 834,681.84856 (IMPROVED, new best)
- ex_19_bridge_w02.csv: skipped after strong direct-candidate win
- ex_19_bridge_w03.csv: skipped after strong direct-candidate win
- ex_19_bridge_w04.csv: skipped after strong direct-candidate win

## Decision Update
- Promote ex_19_robust_dual_weight_ensemble.csv as new production anchor.
- Stop EX_19 bridge ladder.
