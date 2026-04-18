# Experiment Tracker

## Scoreboard

| # | Experiment | Status | Val MAE | Val RMSE | Val R² | Kaggle MAE | Time | Notes |
|---|---|---|---|---|---|---|---|---|
| 01 | Naive Baseline | `[x]` | 837,704 | 1,161,819 | 0.5182 | 1,247,026 | 0.1s | Seasonal naive (best of 3) |
| 02 | Prophet | `[x]` | — | — | — | 1,393,459 | — | Multiplicative seasonality + holidays |
| 03 | LightGBM (v2) | `[x]` | 560,163 | 753,596 | 0.7973 | 973,611 | 99s | v2 features + profiles |
| 04 | XGBoost (GPU) | `[x]` | — | — | — | 1,040,223 | — | v2 features + CUDA |
| 05 | N-HiTS | `[x]` | 557,578 | 765,589 | 0.7570 | 1,287,946 | 50s | Deep learning, neuralforecast |
| 07 | LightGBM (Exogenous) | `[x]` | 558,668 | 756,951 | 0.7955 | — | 145s | Historical Exogenous Traffic & Promos |
| 06 | Ensemble | `[x]` | — | — | — | **929,534** | — | Weighted avg of best models (Pure LGBM) |

## How to Run

```bash
# From project root
python -m modeling.ex_01_naive_baseline
python -m modeling.ex_02_prophet
python -m modeling.ex_03_lgbm
python -m modeling.ex_04_xgb
python -m modeling.ex_05_nhits
python -m modeling.ex_06_ensemble
```

## Submissions

All submission CSVs are saved to `output/submissions/`.

| # | File | Rows | Submitted? |
|---|---|---|---|
| 01 | `ex_01_naive.csv` | 548 | `[x]` |
| 02 | `ex_02_prophet.csv` | 548 | `[x]` |
| 03 | `ex_03_lgbm.csv` | 548 | `[x]` |
| 04 | `ex_04_xgb.csv` | 548 | `[x]` |
| 05 | `ex_05_nhits.csv` | 548 | `[x]` |
| 06 | `ex_06_ensemble_avg.csv` | 548 | `[ ]` |
| 06 | `ex_06_ensemble_weighted.csv` | 548 | `[x]` |
| 07 | `ex_07_lgbm_exogenous.csv` | 548 | `[x]` |

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
