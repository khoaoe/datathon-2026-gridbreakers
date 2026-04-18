# autoresearch (datathon-2026 fork)

Fork of the karpathy/autoresearch pattern adapted to this revenue-forecasting
datathon. Instead of training an LLM on WebText, the agent iterates on a
daily-revenue forecaster with a **fixed** validation harness.

## Setup

Work with the user to initialize a new run:

1. **Agree on a run tag** — propose something like `apr18`, `apr18-v2`, etc.
   The branch `autoresearch/<tag>` must not exist yet.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `autoresearch/prepare.py` — FROZEN harness (do not modify).
   - `autoresearch/train.py` — the only file you edit.
   - `modeling/feature_engineering.py` — v3 feature builder you may call.
   - `modeling/config.py` — time splits / LGBM defaults.
   - `approaches.md`, `dataset_summary.md` — domain context.
4. **Verify the env**: run `./autoresearch/run_experiment.sh --dry-run` (or
   manually: `conda run -n datathon python -m autoresearch.prepare`) to make
   sure the harness loads 3,741 train rows, 92 val rows, 548 test rows.
5. **Initialize results.tsv** (already seeded with the header on first run).
6. Confirm setup and go.

## Experimentation loop

Each experiment:

```bash
conda run -n datathon python -m autoresearch.train > run.log 2>&1
grep -E "^(val|ext)_mae_rev:|^(val|ext)_mae_cogs:|^val_rmse_rev:" run.log
```

1. Look at git state (branch / commit).
2. Edit `autoresearch/train.py` with one idea. Update `EXPERIMENT_DESC`.
3. `git commit -am "autoresearch: <short desc>"`.
4. Run `python -m autoresearch.train > run.log 2>&1`.
5. Parse `val_mae_rev` from the log. Empty grep = crash — `tail -n 50 run.log`.
6. Results are auto-appended to `autoresearch/results.tsv` by `append_result`.
   If you want a different status (`discard` vs `keep`), edit the last line.
7. If `val_mae_rev` dropped vs best-on-branch → keep commit. Otherwise
   `git reset --hard HEAD~1` and try the next idea.

## What you CAN do

- Edit `autoresearch/train.py` freely: feature selection, new features, model
  choice, objective, optimizer, CV, blend, calibration, transform, anything.
- Call any function from the project's `modeling.*` package.
- Introduce new experiments/scripts under `autoresearch/` as needed for
  infrastructure, but keep `train.py` as the canonical entry point.

## What you CANNOT do

- Modify `autoresearch/prepare.py`. It defines the split + metric. Touching it
  invalidates comparisons. Ground truth.
- Look at `sample_submission.csv`'s Revenue/COGS columns as labels. They are a
  template, not truth.
- Train on the validation window (2022-10-01 .. 2022-12-31). The final
  submission retrain may include those days; validation must not.

## Primary metric

**`val_mae_rev`** — MAE of predicted Revenue on the held-out Q4 2022 window.

**`ext_mae_rev`** — MAE on 2021-2022 forecast produced by a model trained
only on data ≤ 2020-12-31. This is a much stricter honesty check because
the 548-day test horizon requires the model to extrapolate trend far into
the future, which Q4-2022 val does NOT exercise.

Rank order preference:

1. Prefer changes that lower **both** `val_mae_rev` and `ext_mae_rev`.
2. If val_mae_rev drops but `ext_mae_rev` rises ≥ 5 %, the change is most
   likely overfitting to near-term patterns — `discard`.
3. If ext_mae_rev drops while val_mae_rev stays flat or rises < 2 %, that's
   a keep (Kaggle cares about extrapolation, Q4-2022 val does not).

Secondary tracked: `val_mae_cogs`, `val_rmse_rev`, `val_mape_rev`,
`val_r2_rev`, `ext_mae_cogs`, `ext_rmse_rev`, `ext_r2_rev`.

### Why this two-metric rule exists

Real Kaggle results from the local submissions in `output/submissions/`:

| model                      | Q4-2022 val MAE | Kaggle MAE | gap |
| -------------------------- | --------------- | ---------- | --- |
| ex_08 (Prophet + residual) | 353 k           | **890 k**  | 2.5 × |
| ex_07 (LGBM v3)            | 366 k           | 1.21 M     | 3.3 × |
| ex_09 (direct-horizon)     | 441 k           | 1.24 M     | 2.8 × |
| autoresearch tree-only     | 580 k           | 2.08 M     | 3.6 × |

Tree models alone **cannot extrapolate** rising trend. Prophet can. That is
why the current baseline is Prophet + LGBM residual, and why any idea that
kills the Prophet layer will almost certainly regress on `ext_mae_rev`.

## Simplicity criterion

All else equal, simpler wins. A 0.1% MAE improvement that adds 200 lines of
gnarly code is probably a discard. A 0.1% improvement that *removes* a
component is a keep. Removing a component with neutral effect is a great win.

## Output format

After each run, `train.py` prints a block like:

```
---
val_mae_rev:   573481.02
val_mae_cogs:  412877.15
val_rmse_rev:  748912.88
...
```

`grep "^val_mae_rev:" run.log` extracts the key metric.

## results.tsv schema

Tab-separated. Columns:

```
commit  val_mae_rev  val_mae_cogs  val_rmse_rev  status  description
```

- `commit` — short git hash (7 chars)
- `val_mae_rev` — Revenue MAE on val (use 0.00 for crashes)
- `val_mae_cogs` — COGS MAE on val (use 0.00 for crashes)
- `val_rmse_rev` — Revenue RMSE on val
- `status` — `baseline`, `keep`, `discard`, `crash`
- `description` — short free-text of what the experiment tried

Example:

```
commit  val_mae_rev  val_mae_cogs  val_rmse_rev  status  description
a1b2c3d 573481.02    412877.15     748912.88     baseline  v3+log LGBM
b2c3d4e 551203.44    408114.30     719881.22     keep      Tweedie objective
c3d4e5f 602004.81    430001.10     801234.55     discard   add 5y SARIMA residuals
d4e5f6g 0.00         0.00          0.00          crash     doubled num_leaves (OOM)
```

## Environment

Runs are executed inside the `datathon` conda env (has lightgbm, prophet,
neuralforecast, xgboost, shap, optuna, chronos-forecasting, holidays). Use:

```bash
conda run -n datathon python -m autoresearch.train
```

or activate first with `conda activate datathon`.

## Idea menu (start here)

A seed list of high-ROI experiments, ordered roughly by expected value.
Most toggles already have hyperparameters near the top of `train.py`.

1. **Log-target Prophet** — set `LOG_PROPHET = True`. Prophet fits `log1p(y)`,
   inverse on predict. Multiplicative trend becomes additive on log scale,
   usually helps heavy-tailed retail revenue that grows by ~% per year.
2. **Prophet regressors** — set `USE_PROPHET_REGRESSORS = True`. Adds
   `is_promo`, `is_vn_holiday`, `is_tet_week`, `is_black_friday_week`,
   `is_xmas_week`, `covid_flag` as exogenous signals known for any future
   date. Expected big lift on event-driven days.
3. **Changepoint prior** — raise `changepoint_prior_scale` 0.05 → 0.1 → 0.2.
   Gives Prophet's trend more flexibility. Watch for overfitting: if val
   improves but ext regresses, back off.
4. **Drop lag features from residual LGBM** — `DROP_LAG_FEATURES = True`.
   Rationale: at long horizons, lag features built from predicted Revenue
   compound error. Profile + calendar features alone may generalize better.
5. **Wider LGBM** — bump `num_leaves` 63 → 127 or `learning_rate` 0.03 → 0.05
   in `LGBM_KW`. Only keep if ext_mae_rev also improves.
6. **Tweedie objective on residual** — change `LGBM_KW["objective"] =
   "tweedie"` + add `"tweedie_variance_power": 1.5`. Heavy-tailed residuals.
7. **Log-trend + LGBM on ratio** — predict `actual / prophet_yhat` instead
   of `actual - prophet_yhat`. Multiplicative correction, bounded target.
8. **Weekly-only seasonality + Fourier yearly** — replace Prophet's yearly
   seasonality with explicit Fourier features in LGBM (already present in
   v3 features via `fourier_sin_k` / `fourier_cos_k`). See if double-counting
   yearly seasonality hurts.
9. **COGS as ratio of Revenue** — predict COGS via a per-day ratio model
   (LGBM on `COGS/Revenue`) then multiply. Stable target, narrower range.
10. **Stacked with Chronos zero-shot** — use `chronos-forecasting` (installed
    in the env) to produce zero-shot Revenue forecasts, blend with the
    Prophet + LGBM-residual stack.

## NEVER STOP

Once the experiment loop is running, do not pause to ask the human "should I
continue?". The user will interrupt manually. If you run out of ideas:

- Work down the Idea Menu above.
- Re-read `approaches.md` for untried modeling families.
- Combine previous near-misses (e.g. best-so-far FE × different objective).
- Check feature importance: drop zero-importance features, try new ones
  inspired by top-importance ones.

Crashes: if trivial (typo, missing import), fix and re-run. If the idea is
fundamentally broken, log as `crash`, revert, move on.

Timeout: if a run exceeds 15 minutes, kill it and treat as `crash`.
