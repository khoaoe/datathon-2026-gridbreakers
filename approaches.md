# Modeling Approaches for Revenue Forecasting

Task: Predict daily `Revenue` (and `COGS`) for 548 days (Jan 2023 – Jul 2024).
Train: 3,833 days (Jul 2012 – Dec 2022). Univariate target + rich multi-table covariates.

---

## 1. Feature Engineering (Critical for All Approaches)

Build features from the 12+ auxiliary tables. Aggregate everything to daily grain to match `sales.csv`.

### Time/Calendar Features
- Day of week, month, quarter, year
- Is weekend, is month start/end
- Vietnamese holidays (Tet, National Day, etc.) — derive from `promotions.csv` date patterns
- Day of year (cyclical sin/cos encoding)

### Lag & Rolling Window Features
- Revenue lags: t-1, t-7, t-14, t-28, t-365
- Rolling mean/std/min/max over 7, 14, 28, 90, 365-day windows
- Year-over-year growth rate
- Expanding mean

### From orders.csv (aggregate daily)
- Daily order count, avg order value
- Order status distribution (% delivered, % cancelled, % returned)
- Device type mix, payment method mix, order source mix

### From order_items.csv (aggregate daily)
- Daily total quantity, avg unit price
- Promo usage rate (% items with promo_id not null)
- Avg discount amount per item

### From payments.csv (aggregate daily)
- Daily total payment value
- Avg installment count
- Payment method distribution

### From shipments.csv (aggregate daily)
- Daily shipment count, avg shipping fee
- Avg delivery lead time (delivery_date - ship_date)

### From returns.csv (aggregate daily)
- Daily return count, total refund amount
- Return rate (returns / orders)
- Return reason distribution

### From reviews.csv (aggregate daily)
- Daily review count, avg rating

### From inventory.csv (monthly, forward-fill to daily)
- Total stock on hand, total stockout days
- Avg fill rate, avg sell-through rate
- Stockout flag count across products

### From web_traffic.csv (daily, starts Jan 2013)
- sessions, unique_visitors, page_views
- bounce_rate, avg_session_duration_sec
- Backfill or zero-fill for Jul 2012 – Dec 2012

### From promotions.csv
- Binary flag: is any promo active today?
- Count of active promos on any given day
- Promo type active (percentage vs fixed)

**Leakage Warning:** Never use future revenue/COGS as features. Time-based splits only.

---

## 2. Approach A — Gradient Boosted Trees (Recommended Primary)

**Why:** Kaggle competitions for tabular time series are consistently dominated by LightGBM/XGBoost. They handle mixed feature types, missing values, and non-linear interactions natively. Strong with engineered lag/rolling features.

### Models
- **LightGBM** — fast, memory-efficient, handles categoricals natively
- **XGBoost** — robust, slightly better regularization
- **CatBoost** — handles categoricals without encoding

### Key Papers
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017)
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (KDD 2016)

### Validation Strategy
- **TimeSeriesSplit** or expanding window CV. Never random split.
- Train on 2012–2020, validate on 2021, test-mimic on 2022.
- Or use 3–5 fold forward-chaining splits.

### Why It Wins
- Lim et al. (2021) showed tree-based models outperform deep learning on many tabular datasets.
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023) showed even simple linear models beat complex Transformers on standard benchmarks — tree-based models with good features are even stronger.
- Kaggle M5 Forecasting (2020) and Store Sales (2022) competitions were won by LightGBM ensembles.

---

## 3. Approach B — Statistical Baselines

**Why:** Fast to implement, interpretable, good baselines to beat.

### Models
- **Prophet** (Meta, 2017) — handles trend, multiple seasonalities, holidays. Good for daily business data.
  - Taylor & Letham, "Forecasting at Scale" (PeerJ Preprints, 2017)
- **SARIMA / SARIMAX** — classical. Add exogenous regressors for covariates.
- **Exponential Smoothing (ETS)** — Holt-Winters with multiplicative seasonality.

### Use Cases
- Prophet as a quick baseline for trend + seasonality decomposition.
- Residual stacking: fit Prophet first, then train LightGBM on the residuals.

---

## 4. Approach C — Deep Learning Models

**Why:** Can model complex temporal dependencies and multivariate interactions end-to-end. Useful if feature engineering alone isn't enough.

### Models (ranked by relevance)

#### Temporal Fusion Transformer (TFT)
- Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (Int. J. Forecasting, 2021)
- Built for multi-horizon forecasting with static + dynamic covariates.
- Has Variable Selection Networks (built-in feature importance → good for the explainability requirement).
- Directly supports known future inputs (holidays, promos) and observed past inputs.

#### N-BEATS / N-HiTS
- Oreshkin et al., "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (ICLR 2020)
- Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023)
- Pure univariate. Good if covariates don't help much.
- N-HiTS handles multi-scale patterns better.

#### PatchTST
- Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
- Patches time series into subseries tokens. Channel-independent.
- Strong on long-horizon univariate benchmarks.

#### iTransformer
- Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (ICLR 2024)
- Inverts the standard Transformer: treats each variate as a token.
- Better at multivariate correlation modeling than PatchTST.

### Libraries
- `darts` (Unit8) — unified API for all above models
- `neuralforecast` (Nixtla) — N-BEATS, N-HiTS, TFT, PatchTST
- `pytorch-forecasting` — TFT implementation

---

## 5. Approach D — Foundation Models (Zero-Shot)

**Why:** Pre-trained on billions of time points. Can generate forecasts without training. Good for sanity-checking or ensembling.

### Models

#### Chronos (Amazon, 2024)
- Ansari et al., "Chronos: Learning the Language of Time Series" (ICML 2024)
- T5-based, tokenizes time series values. Probabilistic forecasts.
- Chronos-Bolt variant is faster.
- `pip install chronos-forecasting`

#### TimesFM (Google, 2024–2025)
- Das et al., "A Decoder-Only Foundation Model for Time-Series Forecasting" (ICML 2024)
- Decoder-only transformer, patch-based. 200M params.
- TimesFM 2.0 supports covariates (XReg).
- `pip install timesfm`

### Use Cases
- Zero-shot baseline: no training needed, just feed in sales.csv.
- Ensemble with trained models for diversity.

---

## 6. Approach E — Hybrid / Ensemble (Recommended Final)

**Why:** Best Kaggle solutions almost always ensemble diverse models.

### Strategy
1. **Base models:**
   - LightGBM with full feature set (primary)
   - XGBoost with full feature set
   - Prophet (trend/seasonality baseline)
   - TFT or N-HiTS (deep learning)
   - Chronos or TimesFM (zero-shot)
2. **Meta-learner:**
   - Simple weighted average (weights from CV performance)
   - Or train a Ridge/Linear regression on out-of-fold predictions
3. **Residual stacking:**
   - Prophet captures trend/seasonality → LightGBM models the residuals

### Key Paper
- Makridakis et al., "M5 accuracy competition: Results, findings, and conclusions" (Int. J. Forecasting, 2022) — tree-based ensembles dominated.

---

## 7. Recommended Pipeline

```
Step 1: EDA + Data QA
  └─ Validate joins, check date coverage, spot anomalies

Step 2: Feature Engineering
  └─ Build daily feature table from all 12+ source tables
  └─ Lag features, rolling stats, calendar, promo flags

Step 3: Baseline Models
  └─ Prophet baseline
  └─ Chronos/TimesFM zero-shot baseline
  └─ Naive seasonal baseline (same day last year)

Step 4: Primary Model
  └─ LightGBM with TimeSeriesSplit CV
  └─ Hyperparameter tuning (Optuna)
  └─ SHAP values for explainability (competition requirement!)

Step 5: Deep Learning (if time permits)
  └─ TFT with covariates
  └─ N-HiTS univariate

Step 6: Ensemble
  └─ Weighted average of best 2–3 models
  └─ Validate on holdout 2022

Step 7: Submission
  └─ Generate submission.csv with Date, Revenue, COGS
  └─ Verify row order matches sample_submission.csv
```

---

## 8. Key References

| Paper | Venue | Relevance |
|---|---|---|
| Ke et al., "LightGBM" | NeurIPS 2017 | Primary model |
| Chen & Guestrin, "XGBoost" | KDD 2016 | Primary model |
| Taylor & Letham, "Prophet" | PeerJ 2017 | Statistical baseline |
| Lim et al., "TFT" | Int. J. Forecasting 2021 | Deep learning + interpretability |
| Oreshkin et al., "N-BEATS" | ICLR 2020 | Deep learning univariate |
| Challu et al., "N-HiTS" | AAAI 2023 | Multi-scale deep learning |
| Nie et al., "PatchTST" | ICLR 2023 | Transformer baseline |
| Liu et al., "iTransformer" | ICLR 2024 | Multivariate transformer |
| Zeng et al., "Are Transformers Effective?" | AAAI 2023 | Simple > complex |
| Ansari et al., "Chronos" | ICML 2024 | Foundation model |
| Das et al., "TimesFM" | ICML 2024 | Foundation model |
| Makridakis et al., "M5 Results" | IJF 2022 | Ensemble strategy |
