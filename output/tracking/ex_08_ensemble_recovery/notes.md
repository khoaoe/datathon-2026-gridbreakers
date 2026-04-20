# Ensemble Recovery Notes

## Goal
Recover from 934k toward previous 861k baseline via controlled weight sweep.

## Candidate Files
- v2_lgbm_up_more: output/submissions/ex_06_recovery_v2_lgbm_up_more.csv
- v1_lgbm_up: output/submissions/ex_06_recovery_v1_lgbm_up.csv
- v3_lgbm_up_soft: output/submissions/ex_06_recovery_v3_lgbm_up_soft.csv
- baseline_861style: output/submissions/ex_06_recovery_baseline_861style.csv

## Submit Order (best CV first)
- 1. v2_lgbm_up_more | mean_mae_overall=825656.05 | tree_mix(lgb=0.722, xgb=0.278)
- 2. v1_lgbm_up | mean_mae_overall=831908.07 | tree_mix(lgb=0.686, xgb=0.314)
- 3. v3_lgbm_up_soft | mean_mae_overall=839959.79 | tree_mix(lgb=0.643, xgb=0.357)
- 4. baseline_861style | mean_mae_overall=854588.21 | tree_mix(lgb=0.571, xgb=0.429)