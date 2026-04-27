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
| 20 | EX_20 recency-weighted dual ensemble | `[x]` | 564,951 | — | — | **830,584.739** | 1,009s | 2-fold (2021/2022) recency-weighted global objective, anchored to EX_19 production winner; direct candidate is new best |
| 21 | EX_20 bridge submissions | `[x]` | — | — | — | 834,398.889 (best of 1) | — | Submitted w01; improved vs EX_19 anchor but worse than EX_20 direct candidate; stopped ladder |
| 22 | EX_21 deep FE recency dual ensemble | `[x]` | — | — | — | **820,000.000** | — | Deep FE recency-profile dual ensemble |
| 23 | EX_22 deep FE holidays dual ensemble | `[ ]` | 554,142 | — | — | — | 2025s | Added Tet holiday features and volatility regime features; average fold score ~827k |
| 24 | EX_22 bridge submissions | `[ ]` | — | — | — | — | — | 4 bridge submissions from EX_21 anchor |
| 25 | EX_24 deep FE holidays + double-date dual ensemble | `[x]` | 590,550* | — | — | 826,795.666 | — | EX_22 architecture + holidays.VN + double-date campaign windows; direct candidate regressed on LB (`*` quick fold-2022 probe only) |
| 26 | EX_24 bridge submissions | `[x]` | — | — | — | **795,838.944** (best of 1) | — | Submitted w01; improved vs EX_22 direct anchor and became new production anchor |
| 27 | EX_25 context-aware deep FE | `[x]` | 617,583* | — | — | 842,405.185 | 2124s | Fixed naive FE (Tet slump, modern double dates, summer vacation spikes); `*` mean global revenue CV. Direct candidate regressed. |
| 28 | EX_26 clean continuous calendar FE | `[x]` | 613,518* | — | — | — | 1800s | Replaced all rigid binary flags with continuous distance features (`days_to_mega_double`); `*` mean global revenue CV |
| 29 | EX_49 YoY-Growth Stateless Hybrid | `[x]` | 752,846* | — | — | — | 894s | Multi-path stateless (YoY/Quad/AOV/Pure); Path D dominates 90%; test rev_mean=3,712k; `*` mean ens CV |
| 30 | EX_50 Lag-365 Direct Model | `[x]` | 805,133* | — | — | — | 539s | lag365 features, 2-phase (365d direct + 183d recursive); test rev_mean=3,361k; `*` mean direct CV |
| 31 | EX_51 Lag-365 Recursive Ensemble | `[x]` | 769,237* | — | — | — | 2524s | 3-component ensemble (lag365/stateless/full); optimal w=0.0/0.7/0.3; test rev_mean=3,760k; `*` mean ens CV |
| 32 | EX_52 Monthly-Recalibrated Ensemble | `[x]` | 726,699* | — | — | — | 1281s | Fix snowball uneven growth bias using CV error patterns. Best blend (sl=0.7) mean CV=726k. |

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
python -m modeling.ex_21_deep_fe_recency_dual_ensemble
python -m modeling.ex_22_deep_fe_holidays_dual_ensemble
python -m modeling.ex_24_deep_fe_holidays_double_dates_dual_ensemble
python -m modeling.ex_25_deep_fe_context_aware_dual_ensemble
python -m modeling.ex_26_clean_continuous_calendar_dual_ensemble
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
| 21 | `ex_21_deep_fe_recency_dual_ensemble.csv` | 548 | `[x]` |
| 22 | `ex_22_deep_fe_holidays_dual_ensemble.csv` | 548 | `[x]` |
| 22 | `ex_22_bridge_w01.csv` | 548 | `[x]` |
| 22 | `ex_22_bridge_w02.csv` | 548 | `[ ]` (skipped) |
| 22 | `ex_22_bridge_w03.csv` | 548 | `[ ]` (skipped) |
| 22 | `ex_22_bridge_w04.csv` | 548 | `[ ]` (skipped) |
| 24 | `ex_24_deep_fe_holidays_double_dates_dual_ensemble.csv` | 548 | `[x]` |
| 24 | `ex_24_bridge_w01.csv` | 548 | `[x]` |
| 24 | `ex_24_bridge_w02.csv` | 548 | `[ ]` |
| 24 | `ex_24_bridge_w03.csv` | 548 | `[ ]` |
| 24 | `ex_24_bridge_w04.csv` | 548 | `[ ]` |
| 25 | `ex_25_deep_fe_context_aware_dual_ensemble.csv` | 548 | `[x]` |
| 25 | `ex_25_bridge_w01.csv` | 548 | `[x]` |
| 25 | `ex_25_bridge_w02.csv` | 548 | `[ ]` |
| 25 | `ex_25_bridge_w03.csv` | 548 | `[ ]` |
| 25 | `ex_25_bridge_w04.csv` | 548 | `[ ]` |
| 26 | `ex_26_clean_continuous_calendar_dual_ensemble.csv` | 548 | `[x]` |
| 26 | `ex_26_bridge_w01.csv` | 548 | `[x]` |
| 26 | `ex_26_bridge_w02.csv` | 548 | `[ ]` |
| 26 | `ex_26_bridge_w03.csv` | 548 | `[ ]` |
| 26 | `ex_26_bridge_w04.csv` | 548 | `[ ]` |

## Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost prophet shap neuralforecast holidays
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
- Public LB update: `ex_21_deep_fe_recency_dual_ensemble.csv` scored `820,000` (IMPROVED; new production anchor).
- Public LB update: `ex_22_bridge_w01.csv` scored `819,602.36777` (IMPROVED vs EX_21 anchor).
- Public LB update: `ex_22_deep_fe_holidays_dual_ensemble.csv` scored `796,018.49022` (IMPROVED; new production anchor).
- `ex_22_bridge_w02.csv`, `ex_22_bridge_w03.csv`, and `ex_22_bridge_w04.csv` intentionally skipped after strong direct-candidate win.
- 2026-04-23: EX_24 feature research wired to `holidays.VN()` + double-date event windows in `build_calendar_features`; local and LB runs pending.
- 2026-04-23: EX_24 quick recursive fold-2022 probe (datathon env): ensemble Revenue MAE `590,549.54`, COGS MAE `538,369.75`, score `805,897.44`; best single component remained `aligned_keep_avg`.
- Public LB update (2026-04-23): `ex_24_deep_fe_holidays_double_dates_dual_ensemble.csv` scored `826,795.66594` (FAILED vs EX_22 production anchor; direct ensemble is worse).
- Public LB update (2026-04-23): `ex_24_bridge_w01.csv` scored `795,838.94386` (IMPROVED vs EX_22 production anchor; promote as new production anchor).
- Public LB update (2026-04-23): `ex_25_bridge_w01.csv` scored `795,779.93214` (IMPROVED very slightly vs EX_24 anchor).
- Public LB update (2026-04-23): `ex_25_deep_fe_context_aware_dual_ensemble.csv` scored `842,405.18587` (FAILED heavily vs anchor). Binary step functions caused massive overfitting.
- 2026-04-23: EX_26 clean continuous calendar FE. Removed rigid binary flags and relied entirely on continuous distance features (`days_to_tet`, `days_to_vn_holiday`, `days_to_mega_double`). Mean global CV score: Revenue MAE 613,518, COGS MAE 543,590, Score 830,954. Pending LB run for `ex_26_bridge_w01.csv` and `ex_26_clean_continuous_calendar_dual_ensemble.csv`.
- 2026-04-25: EX_49 multi-path stateless hybrid research. 4 paths tested (YoY/Quad/AOV/Pure); Path D (pure stateless) dominated with 90% weight. CV=752,846 but test rev_mean=3,712k (too low).
- 2026-04-25: EX_50 lag-365 direct model. 2-phase (365d direct + 183d recursive). CV=805,133, test rev_mean=3,361k.
- 2026-04-25: EX_51 3-component lag365 recursive ensemble (lag365/stateless/full recursive). Optimal weights: stateless=0.7, full=0.3, lag365=0.0. CV=769,237, test rev_mean=3,760k.
- Public LB update (2026-04-25): `ex_49_yoy_stateless.csv` scored `978,938.26093` (FAILED; pure stateless too low level).
- Public LB update (2026-04-25): `ex_51_lag365_recursive.csv` scored `917,935.48146` (FAILED; direct ensemble too low level).
- Public LB update (2026-04-25): `ex_49_bridge_w10.csv` scored `792,815.11519` (IMPROVED vs 795k anchor).
- Public LB update (2026-04-25): `ex_51_bridge_w10.csv` scored `792,322.09974` (IMPROVED vs 795k anchor).
- Public LB update (2026-04-25): `ex_49_bridge_w20.csv` scored `794,528.44560` (IMPROVED vs 795k anchor).
- Public LB update (2026-04-25): **`ex_51_bridge_w20.csv` scored `792,254.33748`** (IMPROVED; **new production anchor**).
- Public LB update (2026-04-25): `ex_49_bridge_w30.csv` scored `801,501.24626` (FAILED vs 795k anchor; too much stateless weight).
- Public LB update (2026-04-25): `ex_51_bridge_w22.csv` scored `792,756.70420` (IMPROVED vs 795k but worse than w20).
- Public LB update (2026-04-25): `ex_51_bridge_w18.csv` scored `791,908.32236` (IMPROVED vs w20).
- Public LB update (2026-04-25): **`ex_51_bridge_w15.csv` scored `791,763.67199`** (IMPROVED; **new production anchor**).
- 2026-04-25: EX_52 Monthly-Recalibrated Ensemble. Addressed the uneven monthly growth bias of the recursive model by scaling monthly predictions according to error patterns observed during CV. Huge CV improvement: recalibrated blend (726,699) vs raw recursive (820,171). Multiple bridge versions generated for LB testing.
- Public LB update (2026-04-27): `ex_52_recalib_blend.csv` scored `923,416.07451` (FAILED; pure recalib blend too aggressive, level too low).
- Public LB update (2026-04-27): `ex_52_rec_growth.csv` scored `957,781.59438` (FAILED; growth recalibration heavily overfit to CV).
- Public LB update (2026-04-27): `ex_52_blend_bridge_w15.csv` scored `795,568.39916` (FAILED vs 791k anchor; blend bridge worse than EX-51 bridge).
- Public LB update (2026-04-27): `ex_52_recalib_blend_bridge_w15.csv` scored `798,285.42994` (FAILED vs 791k anchor).
- Public LB update (2026-04-27): `ex_52_recalib_blend_bridge_w20.csv` scored `801,617.18770` (FAILED vs 791k anchor).
- Public LB update (2026-04-27): `ex_52_recalib_blend_bridge_w30.csv` scored `809,895.58382` (FAILED vs 791k anchor).
- **Conclusion**: EX-52 monthly recalibration dramatically improved CV (727k vs 820k) but FAILED on LB. The per-month correction factors overfit to the 2-fold CV structure and don't generalize to the 2023-2024 test period. The current best remains **`ex_51_bridge_w15.csv` at 791,764**.
