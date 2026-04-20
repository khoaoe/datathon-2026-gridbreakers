# Experiment Tracker

## Scoreboard

| # | Experiment | Status | Val MAE | Val RMSE | Val R² | Kaggle MAE | Time | Notes |
|---|---|---|---|---|---|---|---|---|
| 01 | Naive Baseline | `[x]` | 837,704 | 1,161,819 | 0.5182 | 1,247,026 | 0.1s | Seasonal naive (best of 3) |
| 02 | Prophet | `[x]` | — | — | — | 1,393,459 | — | Multiplicative seasonality + holidays |
| 03 | LightGBM (v2) | `[x]` | 560,163 | 753,596 | 0.7973 | 973,611 | 99s | v2 features + profiles |
| 04 | XGBoost (GPU) | `[x]` | — | — | — | 1,040,223 | — | v2 features + CUDA |
| 05 | N-HiTS | `[x]` | 557,578 | 765,589 | 0.7570 | 1,287,946 | 50s | Deep learning, neuralforecast |
| 06 | Ensemble | `[x]` | — | — | — | **861,132.085** | — | Weighted avg of best models (Simple Avg: 976,579) |
| 07 | LightGBM (v3) | `[x]` | 556,865 | 753,564 | 0.7973 | 889,940.370 | 180s | promo calendar + leakage-safe profile source |
| 10 | Full-power microblend challengers | `[x]` | — | — | — | **932,988.677** (best of 4) | 2s | FAILED on LB, deprecated |
| 11 | Anchor bridge blends | `[x]` | — | — | — | — | 1s | 4 low-drift bridge files from 861 anchor to FE refresh |

## How to Run

```bash
# From project root
python -m modeling.ex_01_naive_baseline
python -m modeling.ex_02_prophet
python -m modeling.ex_03_lgbm
python -m modeling.ex_04_xgb
python -m modeling.ex_05_nhits
python -m modeling.ex_06_ensemble
python -m modeling.ex_08_feature_engineering_research
python -m modeling.ex_11_anchor_bridge_blends
```

## Submissions

All submission CSVs are saved to `output/submissions/`.

| # | File | Rows | Submitted? |
|---|---|---|---|
| 01 | `ex_01_naive.csv` | 548 | `[x]` |
| 02 | `ex_02_prophet.csv` | 548 | `[ ]` |
| 03 | `ex_03_lgbm.csv` | 548 | `[x]` |
| 04 | `ex_04_xgb.csv` | 548 | `[ ]` |
| 05 | `ex_05_nhits.csv` | 548 | `[ ]` |
| 06 | `ex_06_ensemble_avg.csv` | 548 | `[ ]` |
| 06 | `ex_06_ensemble_weighted.csv` | 548 | `[x]` |

## Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost prophet shap neuralforecast
```

## Notes

- Validation: 2022-01-01 to 2022-12-31 (last year holdout)
- All models retrained on full data before generating test submissions
- EX_03 and EX_04 use recursive prediction (predict day-by-day, updating lags)
- EX_05 uses direct multi-step prediction (N-HiTS predicts all 548 days at once)
- SHAP values saved for EX_03 in `output/models/ex_03_shap_importance.csv`
- Public LB update (2026-04-20): `ex_03_lgbm.csv` scored `889,940.36974`
- Public LB update (2026-04-20): `ex_06_ensemble_weighted.csv` scored `861,132.08456`
- Failed LB (logged): `ex_06_recovery_v1_lgbm_up.csv` = `927,804.15516`
- Failed LB (logged): `ex_06_recovery_v2_lgbm_up_more.csv` = `926,067.29049`
- Failed LB (logged): `ex_10_fp_ex06opt_w01.csv` = `932,988.67676`
- Failed LB (logged): `ex_10_fp_med4_w01.csv` = `933,214.06121`
- Failed LB (logged): `ex_10_fp_mean4_w01.csv` = `933,222.02871`
- Failed LB (logged): `ex_10_fp_final_w005.csv` = `933,297.12358`
- Failed LB (logged): `ex_06_ensemble_weighted_fe_refresh.csv` ≈ `879k` (user-reported)
