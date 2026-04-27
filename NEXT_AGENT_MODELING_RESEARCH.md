# Modeling Research Plan: Post EX-52 Session Summary

**Goal:** Bridge the MAE gap from 792k to the 620k Leaderboard target.

## Current Best

**`ex_51_bridge_w15.csv` — LB score: 791,764** (85% recursive anchor + 15% EX-51 stateless/recursive ensemble)

## What We Now Know (April 25-27 Research)

### The Core Dilemma
1. **Recursive models** (current anchor) predict test rev_mean=3.99M — likely close to true level. But lag snowball adds ~175k MAE from day-to-day noise amplification.
2. **Stateless models** (EX-49) achieve excellent CV (753k) but predict test rev_mean=3.66M — ~300k too low. Without lags, LGBM can't distinguish 2023 levels from 2019-2020 levels.
3. **Bridge blends** at 10-20% stateless weight consistently improve LB by ~4k. Sweet spot is w=0.15.
4. **Monthly recalibration (EX-52)** dramatically improved CV (727k vs 820k raw recursive) but **completely failed on LB** (all variants >795k). The per-month correction factors overfit to the 2-fold CV and don't generalize.

### Key Lessons Learned
- **CV-LB divergence is the #1 problem.** EX-52 proved that methods improving CV can catastrophically fail on LB if they encode fold-specific patterns.
- **Bridge blending is reliable but incremental.** It gives ~4k improvement (795k→792k) but won't close the 172k gap.
- **The snowball signature**: Anchor implies Jan=+5% but Nov=+52% YoY growth over 2022 — wildly uneven. Real growth should be ~12% uniform.
- **Stateless models consistently underpredict level** by 7-15% across all experiments (EX-49: 3.71M, EX-50: 3.36M, EX-51: 3.76M vs anchor 3.99M).

## Experiments Completed This Session

| EX | Description | CV Score | Best LB | Result |
|----|------------|----------|---------|--------|
| 49 | 4-path stateless hybrid | 753k | 792,815 (w10 bridge) | Bridge helps, pure fails (979k) |
| 50 | Lag-365 direct model | 805k | — | Level too low (3.36M) |
| 51 | 3-component ensemble (lag365/stateless/full) | 769k | **791,764** (w15 bridge) | **NEW BEST** |
| 52 | Monthly-recalibrated ensemble | **727k** | 795,568 (blend w15 bridge) | CV overfit, LB FAILED |

## What Has NOT Been Tried Yet

### High-Priority Ideas
1. **Teacher Forcing with Noise** — During training, inject random noise into lag features to simulate recursive error. Makes the model robust to its own prediction errors during inference. This is the most theoretically sound fix for snowball.
2. **Multi-Horizon Direct Models** — Train separate LGBM models for different forecast horizons (h=1-30, h=31-90, h=91-180, h=181-365, h=366-548). Short horizons use lags, long horizons use calendar only. No recursion needed.
3. **Quantile/Conformalized Recursion** — Use prediction intervals to dampen recursive lags toward their mean when uncertainty is high, preventing snowball amplification.
4. **Recursive with Lag Dampening** — Instead of feeding raw predictions as future lags, blend them toward historical monthly means: `lag_used = α * pred + (1-α) * monthly_profile`. Dampens snowball without losing level.
5. **Feature Engineering on the Anchor Itself** — Use the existing best anchor predictions as a "pseudo ground truth" for 2023 and train a correction model on the residual patterns.

### Medium-Priority Ideas
6. **COGS-specific optimization** — COGS contributes 0.4× to score. COGS recursive model may have different optimal blend weights or correction strategies.
7. **DOW-specific blend weights** — Different days of week may benefit from different stateless/recursive ratios.
8. **Exponential smoothing of recursive lags** — Apply EMA to predictions before using them as lags to filter high-frequency noise.

## Architecture Matrix (Updated)

| Approach | Level | Seasonality | Snowball | LB Score |
|----------|-------|-------------|----------|----------|
| Recursive anchor (EX-24) | ✅ 3.99M | ⚠️ Noisy | ❌ Full | 795,839 |
| EX-51 w15 bridge | ✅ 3.96M | ✅ Corrected | ⚠️ 85% | **791,764** |
| EX-52 recalib blend | ⚠️ 3.76M | ✅ Clean | ✅ Fixed | 923,416 (overfit) |
| Pure stateless (EX-49) | ❌ 3.71M | ✅ Clean | ✅ None | 978,938 |

## Files Reference
- `modeling/ex_49_yoy_stateless.py` — Multi-path stateless hybrid
- `modeling/ex_50_lag365_direct.py` — Lag-365 direct model
- `modeling/ex_51_lag365_recursive.py` — 3-component recursive ensemble
- `modeling/ex_52_recalibrated.py` — Monthly-recalibrated ensemble (CV-overfit, not for production)
- All submissions in `output/submissions/`

---

## Deep Research Update (April 27, 2026)

### 1) Additional Diagnostics on Existing Submissions

Using `output/tracking/deep_research/submission_diagnostics.csv`:

- Best LB (`ex_51_bridge_w15.csv`, 791,764) has:
  - `rev_mean=3.956M`
  - `rev_month_growth_span=0.450`
  - `rev_jan_growth=1.186`, `rev_nov_growth=1.487`
- Pure stateless family (`ex_49_yoy_stateless.csv`) is **very smooth** (`span=0.232`) but **too low level** (`rev_mean=3.712M`) -> bad LB.
- Strong anchors (`ex_24_bridge_w01.csv`, `ex_25_bridge_w01.csv`) keep level (`~3.99M`) but are **too drifted** (`span~0.477`, Nov growth ~1.52).
- EX-52 recalib variants move toward smooth growth (`span~0.231-0.254`) but lower level too much (`rev_mean~3.70-3.76M`) -> LB collapse.

**Empirical takeaway:** the winning zone appears to be a narrow manifold:
- Level must stay around **3.93M-3.97M**
- Growth span likely needs to improve from ~0.45 toward ~0.38-0.42
- Any method that fixes span by pulling mean below ~3.85M is likely to fail LB.

### 2) What Was Already Tried Before EX-49 (and Why It Matters)

From tracking/meta + fold artifacts:

- `ex_33_horizon_adaptive.py`: drift-correction alpha by horizon; limited success.
- `ex_42_growth_hypothesis.py`: time trend + recency weighting improved CV to ~850k but not enough.
- `ex_45_quantile_stable.py`: quantile recursion very unstable (mean CV >1.0M).
- `ex_46_yoy_ratio.py`: ratio-target idea underperformed strongly in fold scores.
- `ex_47_recalibrate.py`: monthly recalibration family already explored (same failure mode later in EX-52).
- `ex_48_stateless_hybrid.py`: no-recursion architecture has level bias (mean CV ~919k).

So the likely path is **not** pure trend/stateless, quantile-only, or monthly post-hoc recalibration.

### 3) Updated Priority: Highest-EV Next Experiments

#### Priority A — Rectify-Style Residual Correction (Most Promising)
Build a **recursive base** (level-preserving) and train a **direct horizon-residual corrector**:

1. Generate recursive predictions on train folds.
2. For each horizon bucket `h` (1-30, 31-90, 91-180, 181-365, 366-548), train a direct model for residual:
	- `residual_h = y_true_h - y_recursive_h`
	- features: calendar + profiles + horizon index + month + dayofweek + lag365 profile features
3. Final inference:
	- `y_final_h = y_recursive_h + residual_hat_h`

This is a "rectify" strategy (direct + recursive hybrid) designed to reduce recursive drift while preserving level.

#### Priority B — Horizon-Adaptive Blend Weights (but constrained)
Instead of one global bridge weight, use piecewise weights by horizon:

- Early horizon (1-120): keep recursive-heavy (e.g., 90-95%)
- Mid horizon (121-365): modest stateless correction
- Late horizon (366-548): stronger correction

Constrain global level to avoid under-shoot:
- enforce `test_rev_mean >= 3.90M` during candidate selection.

#### Priority C — COGS-specific Policy
COGS carries 0.4 weight in metric and often behaves differently.

- Keep Revenue conservative (preserve level).
- Allow stronger smoothing/recalibration on COGS only.

### 4) Candidate EX-53 Spec (Recommended)

**Name:** `ex_53_rectify_horizon_residual.py`

Core design:
- Base path: EX-51-like recursive anchor.
- Correction path: horizon-bucket direct residual model (LightGBM).
- Optional small bridge to `ex_51_bridge_w15` for risk control.

Selection criteria (not CV alone):
1. Fold score (Revenue + 0.4*COGS)
2. Predicted `rev_mean` in [3.90M, 3.98M]
3. `rev_month_growth_span <= 0.43`
4. No month with growth > 1.48x 2022 mean

### 5) Runtime / Environment Reminder

Run all future experiments in the requested environment:

```bash
PYTHONPATH=. /home/pineapple/miniconda3/bin/conda run -n datathon python -m modeling.<experiment_module>
```

This avoids accidental base-env execution and import-path drift.
