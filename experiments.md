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
| 12 | LGBM selected deltas | `[ ]` | 555,893 | 752,198 | 0.7980 | — | 187s | Promo interactions + target-aligned autoreg + drop unstable avg_* aux profiles |
| 13 | EX_12 bridge submissions | `[x]` | — | — | — | 921,204.487 (best of 3) | — | Submitted w02/w04/w08; all failed vs 861k anchor |
| 14 | EX_13 dual-recursive | `[ ]` | 633,947 | 854,743 | 0.7392 | — | 212s | Dual-target recursive inference to preserve cross-target lag consistency |
| 15 | EX_14 target-aligned retained profiles | `[ ]` | 615,250 | 854,900 | 0.7391 | — | 305s | Target-aligned autoreg with avg_* profile features retained |
| 16 | EX_15 hybrid (Rev12 + COGS14) | `[ ]` | — | — | — | — | 1s | Revenue from EX_12 candidate + COGS from EX_14 candidate |
| 17 | EX_15 bridge submissions | `[x]` | — | — | — | 927,027.435 (best of 2) | — | Submitted w01 and w04 only; both failed vs 861k anchor |
| 18 | EX_16 recursive FE research | `[ ]` | 615,250 | 854,900 | 0.7391 | — | 563s | Recursive-aware FE ablation; best method `aligned_keep_avg`; candidate + 4 bridges generated |
| 19 | EX_16 bridge submissions | `[x]` | — | — | — | 930,099.179 (best of 2) | — | Submitted w01 and w04 only; both failed vs 861k anchor |
| 20 | EX_17 recursive aux-impute FE research | `[ ]` | 607,889 | — | — | — | 1,013s | 2-fold recursive study; exogenous lag imputation underperformed; best remained `baseline_keep_avg` |
| 21 | EX_17 bridge submissions | `[x]` | — | — | — | 932,607.000 (best of 1) | — | Submitted w01 only; failed vs 861k anchor; stopped family |
| 22 | EX_18 ensemble-step deep research | `[x]` | 570,636 | — | — | **859,853.239** | 1,976s | 2-fold recursive component ensemble; direct candidate submission improved over 861k anchor |
| 23 | EX_18 bridge submissions | `[x]` | — | — | — | 932,735.223 (best of 1) | — | Submitted w01 only; failed vs anchor; bridge ladder stopped |
| 24 | EX_19 robust dual-weight ensemble | `[x]` | 599,677 | — | — | **834,681.849** | 1,673s | 3-fold recursive OOF (2020-2022) with target-specific robust global weights; direct submission became new best |
| 25 | EX_19 bridge submissions | `[x]` | — | — | — | 859,529.710 (best of 1) | — | Submitted w01; improved vs EX_18 anchor but worse than EX_19 direct candidate, then stopped ladder |
| 26 | EX_20 recency-weighted dual ensemble | `[x]` | 564,951 | — | — | **830,584.739** | 1,009s | 2-fold (2021/2022) recency-weighted global objective, anchored to EX_19 production winner; direct candidate is new best |
| 27 | EX_20 bridge submissions | `[x]` | — | — | — | 834,398.889 (best of 1) | — | Submitted w01; improved vs EX_19 anchor but worse than EX_20 direct candidate; stopped ladder |

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
python -m modeling.ex_12_lgbm_selected_deltas
python -m modeling.ex_12_anchor_bridge_blends
python -m modeling.ex_13_lgbm_dual_recursive
python -m modeling.ex_13_anchor_bridge_blends
python -m modeling.ex_14_lgbm_target_aligned
python -m modeling.ex_14_anchor_bridge_blends
python -m modeling.ex_15_hybrid_rev12_cogs14
python -m modeling.ex_15_anchor_bridge_blends
python -m modeling.ex_16_recursive_fe_research
python -m modeling.ex_17_recursive_aux_impute_research
python -m modeling.ex_18_ensemble_step_research
python -m modeling.ex_19_robust_dual_weight_ensemble
python -m modeling.ex_20_recency_weighted_dual_ensemble
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
| 12 | `ex_12_lgbm_selected_deltas.csv` | 548 | `[ ]` |
| 12 | `ex_12_bridge_w02.csv` | 548 | `[x]` |
| 12 | `ex_12_bridge_w04.csv` | 548 | `[x]` |
| 12 | `ex_12_bridge_w06.csv` | 548 | `[ ]` (skipped) |
| 12 | `ex_12_bridge_w08.csv` | 548 | `[x]` |
| 13 | `ex_13_lgbm_dual_recursive.csv` | 548 | `[ ]` |
| 13 | `ex_13_bridge_w01.csv` | 548 | `[ ]` |
| 13 | `ex_13_bridge_w02.csv` | 548 | `[ ]` |
| 13 | `ex_13_bridge_w03.csv` | 548 | `[ ]` |
| 13 | `ex_13_bridge_w04.csv` | 548 | `[ ]` |
| 14 | `ex_14_lgbm_target_aligned.csv` | 548 | `[ ]` |
| 14 | `ex_14_bridge_w01.csv` | 548 | `[ ]` |
| 14 | `ex_14_bridge_w02.csv` | 548 | `[ ]` |
| 14 | `ex_14_bridge_w03.csv` | 548 | `[ ]` |
| 14 | `ex_14_bridge_w04.csv` | 548 | `[ ]` |
| 15 | `ex_15_hybrid_rev12_cogs14.csv` | 548 | `[ ]` |
| 15 | `ex_15_bridge_w01.csv` | 548 | `[x]` |
| 15 | `ex_15_bridge_w02.csv` | 548 | `[ ]` (skipped) |
| 15 | `ex_15_bridge_w03.csv` | 548 | `[ ]` (skipped) |
| 15 | `ex_15_bridge_w04.csv` | 548 | `[x]` |
| 16 | `ex_16_aligned_keep_avg.csv` | 548 | `[ ]` |
| 16 | `ex_16_bridge_w01.csv` | 548 | `[x]` |
| 16 | `ex_16_bridge_w02.csv` | 548 | `[ ]` (skipped) |
| 16 | `ex_16_bridge_w03.csv` | 548 | `[ ]` (skipped) |
| 16 | `ex_16_bridge_w04.csv` | 548 | `[x]` |
| 17 | `ex_17_baseline_keep_avg.csv` | 548 | `[ ]` |
| 17 | `ex_17_bridge_w01.csv` | 548 | `[x]` |
| 17 | `ex_17_bridge_w02.csv` | 548 | `[ ]` (skipped) |
| 17 | `ex_17_bridge_w03.csv` | 548 | `[ ]` (skipped) |
| 17 | `ex_17_bridge_w04.csv` | 548 | `[ ]` (skipped) |
| 18 | `ex_18_ensemble_step.csv` | 548 | `[x]` |
| 18 | `ex_18_bridge_w01.csv` | 548 | `[x]` |
| 18 | `ex_18_bridge_w02.csv` | 548 | `[ ]` |
| 18 | `ex_18_bridge_w03.csv` | 548 | `[ ]` |
| 18 | `ex_18_bridge_w04.csv` | 548 | `[ ]` |
| 19 | `ex_19_robust_dual_weight_ensemble.csv` | 548 | `[x]` |
| 19 | `ex_19_bridge_w01.csv` | 548 | `[x]` |
| 19 | `ex_19_bridge_w02.csv` | 548 | `[ ]` (skipped) |
| 19 | `ex_19_bridge_w03.csv` | 548 | `[ ]` (skipped) |
| 19 | `ex_19_bridge_w04.csv` | 548 | `[ ]` (skipped) |
| 20 | `ex_20_recency_weighted_dual_ensemble.csv` | 548 | `[x]` |
| 20 | `ex_20_bridge_w01.csv` | 548 | `[x]` |
| 20 | `ex_20_bridge_w02.csv` | 548 | `[ ]` (skipped) |
| 20 | `ex_20_bridge_w03.csv` | 548 | `[ ]` (skipped) |
| 20 | `ex_20_bridge_w04.csv` | 548 | `[ ]` (skipped) |

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
- Failed LB (logged): `ex_12_bridge_w02.csv` = `930,846.50105`
- Failed LB (logged): `ex_12_bridge_w04.csv` = `927,547.44210`
- Failed LB (logged): `ex_12_bridge_w08.csv` = `921,204.48695`
- `ex_12_bridge_w06.csv` intentionally skipped (quota + low expected value after w08 still far from anchor).
- Quick FE ablation (EX_08 quick, 2 folds): `core_plus_selected_aux_lags` ranked #1 with mean MAE `565,394` and mean delta `-3,251` vs fold baseline.
- EX_12 local holdout: Revenue MAE `555,893` (worse than latest EX_03 local check), so no commit yet.
- EX_12 tested bridge sequence: `ex_12_bridge_w02.csv` -> `ex_12_bridge_w04.csv` -> `ex_12_bridge_w08.csv` (w06 skipped).
- EX_12 recursive holdout diagnostic: Revenue MAE `607,588`; COGS MAE `624,120`.
- EX_13 recursive holdout: Revenue MAE `633,947`; COGS MAE `545,961`.
- EX_14 recursive holdout: Revenue MAE `615,250`; COGS MAE `542,612`.
- EX_15 built as hybrid candidate (Revenue from EX_12, COGS from EX_14).
- Failed LB (logged): `ex_15_bridge_w01.csv` = `932,395.28814`
- Failed LB (logged): `ex_15_bridge_w04.csv` = `927,027.43521`
- `ex_15_bridge_w02.csv` and `ex_15_bridge_w03.csv` intentionally not submitted after early regression signal.
- EX_16 recursive FE ablation tested: `aligned_drop_avg`, `aligned_keep_avg`, and selected-aux variants.
- EX_16 best by recursive score: `aligned_keep_avg` (Revenue MAE `615,250`; COGS MAE `542,612`).
- EX_16 selected-aux variants collapsed to same feature counts/metrics as parent methods (no extra signal in current pipeline).
- Failed LB (logged): `ex_16_bridge_w01.csv` = `933,137.41901`
- Failed LB (logged): `ex_16_bridge_w04.csv` = `930,099.17893`
- `ex_16_bridge_w02.csv` and `ex_16_bridge_w03.csv` intentionally not submitted after early regression signal.
- EX_17 2-fold recursive result: `baseline_keep_avg` remained best (Revenue MAE `607,889`; COGS MAE `516,705`).
- EX_17 exogenous lag imputation variants (`keep_avg_exo_zero`, `keep_avg_exo_profile`) degraded recursive stability vs baseline.
- Failed LB (logged): `ex_17_bridge_w01.csv` = `932,607` (user-reported)
- `ex_17_bridge_w02.csv`, `ex_17_bridge_w03.csv`, and `ex_17_bridge_w04.csv` intentionally not submitted after early regression signal.
- EX_18 ensemble-step deep research (2 folds) improved local recursive stability with mean ensemble Revenue MAE `570,636` and COGS MAE `504,370`.
- EX_18 fold-optimized ensemble weights averaged to: `core_v3_like=0.1956`, `aligned_keep_avg=0.6598`, `aligned_drop_avg≈0`, `naive_lag365=0.1446`.
- Public LB update (2026-04-20): `ex_18_bridge_w01.csv` scored `932,735.22259` (FAILED; stopped bridge ladder).
- Public LB update (2026-04-20): `ex_18_ensemble_step.csv` scored `859,853.23873` (IMPROVED; new production anchor).
- EX_19 robust dual-weight local means (3 folds: 2020-2022): Revenue MAE `599,677`; COGS MAE `522,071`; score `808,506`.
- EX_19 global Revenue weights: `core_v3_like=0.2718`, `aligned_keep_avg=0.3818`, `aligned_drop_avg=0.0080`, `aligned_no_profiles=0.1367`, `naive_lag365=0.1869`, `naive_lag365_lag7_blend=0.0148`.
- EX_19 global COGS weights: `core_v3_like=0.3146`, `aligned_keep_avg=0.4654`, `aligned_drop_avg=0.0052`, `aligned_no_profiles=0.0177`, `naive_lag365=0.1924`, `naive_lag365_lag7_blend=0.0047`.
- EX_19 overlapping-fold caution: mean score on 2021+2022 is `779,899.78` vs EX_18 `772,383.78` (EX_19 is locally worse by `7,516.00`).
- Public LB update (2026-04-21): `ex_19_bridge_w01.csv` scored `859,529.71007` (IMPROVED vs EX_18 anchor, but below EX_19 direct score).
- Public LB update (2026-04-21): `ex_19_robust_dual_weight_ensemble.csv` scored `834,681.84856` (IMPROVED; new production anchor).
- `ex_19_bridge_w02.csv`, `ex_19_bridge_w03.csv`, and `ex_19_bridge_w04.csv` intentionally skipped after strong direct-candidate win.
- EX_20 recency-weighted local means (2021/2022): Revenue MAE `564,951`; COGS MAE `504,395`; score `766,709`.
- EX_20 Revenue weights: `core_v3_like=0.2326`, `aligned_keep_avg=0.3154`, `aligned_no_profiles=0.3780`, `naive_lag365=0.0740`.
- EX_20 COGS weights: `core_v3_like=0.0617`, `aligned_keep_avg=0.6511`, `aligned_no_profiles=0.0688`, `naive_lag365=0.2185`.
- EX_20 vs EX_19 on shared folds (2021/2022): `766,709.14` vs `779,899.78` (improved by `13,190.65`).
- Public LB update (2026-04-21): `ex_20_bridge_w01.csv` scored `834,398.88891` (IMPROVED vs EX_19 anchor, but below EX_20 direct score).
- Public LB update (2026-04-21): `ex_20_recency_weighted_dual_ensemble.csv` scored `830,584.73892` (IMPROVED; new production anchor).
- `ex_20_bridge_w02.csv`, `ex_20_bridge_w03.csv`, and `ex_20_bridge_w04.csv` intentionally skipped after strong direct-candidate win.
