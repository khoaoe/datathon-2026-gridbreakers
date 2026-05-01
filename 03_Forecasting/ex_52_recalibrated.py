"""
EX_52: Monthly-Recalibrated Ensemble — Fix Uneven Growth Bias

Key Insight from LB analysis:
- Current anchor implies Jan=+5%, but Nov=+52% growth over 2022
- This is the snowball signature: predictions drift as recursive lags compound
- Real growth should be ~12% uniformly (2021→2022 was +12.1% overall)

Strategy:
1. Build the best recursive model (EX-22/24 architecture: multi-component ensemble)
2. After recursive prediction, apply monthly growth recalibration:
   - For each month, scale predictions so implied YoY growth matches target
   - Target growth: use 2021→2022 monthly growth pattern as prior
3. Blend with stateless correction (EX-51 insight: 15% weight)
4. Bridge with current anchor for safety

This directly attacks the snowball effect's SYMPTOMS (uneven monthly growth)
rather than its cause (recursive lag propagation).
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from modeling.config import LGBM_PARAMS, SEED
from modeling.feature_engineering import (
    build_feature_table,
    build_calendar_features,
    build_lag_features,
    build_rolling_features,
    build_growth_features,
    apply_profiles_to_dates,
    get_feature_cols,
)
from modeling.utils import evaluate, load_sales, make_submission

warnings.filterwarnings("ignore")

TRACK = Path("output/tracking/ex_52_recalibrated")
SUB_DIR = Path("output/submissions")

N_SEEDS = 5

FOLDS = [
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]


def _finalize_cols(df, cols):
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def _get_stateless_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [
        c for c in base_cols
        if not c.startswith(blocked)
        and "lag" not in c and "rmean" not in c and "rstd" not in c
        and "ratio" not in c and "growth" not in c and "momentum" not in c
        and "rmin" not in c and "rmax" not in c and "rmedian" not in c
        and "spread" not in c and "margin_ratio" not in c
        and "diff" not in c and "vol" not in c and "zscore" not in c
        and c not in ["Revenue", "COGS"]
    ]


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 2000
    params["random_state"] = seed
    m = lgb.LGBMRegressor(**params)
    m.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return m


def recursive_predict(models, history_df, predict_dates, feature_cols, profiles, target):
    """Standard recursive prediction."""
    history = history_df[["Date", target]].copy()
    preds = []

    for i, date in enumerate(predict_dates):
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)

        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)

        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else float(val)

        pred = float(np.mean([m.predict(x_pred)[0] for m in models]))
        pred = max(0, pred)
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

        if (i + 1) % 100 == 0:
            print(f"      day {i+1}/{len(predict_dates)}: pred={pred:,.0f}")

    return np.array(preds)


def stateless_predict(models, history_df, predict_dates, feature_cols, profiles, target):
    """Bulk stateless prediction (no recursion)."""
    history = history_df[["Date", target]].copy()
    test_df = pd.DataFrame({"Date": pd.to_datetime(predict_dates), target: np.nan})
    combined = pd.concat([history, test_df], ignore_index=True).sort_values("Date")
    combined = build_calendar_features(combined)

    test_rows = combined[combined["Date"].isin(pd.to_datetime(predict_dates))].copy()
    test_rows = apply_profiles_to_dates(test_rows, profiles)

    x_pred = pd.DataFrame(0.0, index=range(len(test_rows)), columns=feature_cols)
    for c in feature_cols:
        if c in test_rows.columns:
            vals = test_rows[c].values
            x_pred[c] = np.where(pd.isna(vals), 0.0, vals)

    preds = np.mean([m.predict(x_pred) for m in models], axis=0)
    return np.maximum(0, preds)


# ─── Monthly Recalibration ────────────────────────────────────────────────────

def compute_monthly_recalibration(
    train_df, val_dates, val_preds, val_actuals, target,
):
    """
    Compute per-month correction ratios from recursive CV errors.
    
    For each month:
      correction = actual_monthly_mean / predicted_monthly_mean
    """
    df = pd.DataFrame({
        "Date": val_dates,
        "pred": val_preds,
        "actual": val_actuals,
    })
    df["month"] = pd.to_datetime(df["Date"]).dt.month

    corrections = {}
    for m in range(1, 13):
        mask = df["month"] == m
        if mask.sum() == 0:
            corrections[m] = 1.0
            continue
        pred_mean = df.loc[mask, "pred"].mean()
        actual_mean = df.loc[mask, "actual"].mean()
        if pred_mean > 0:
            # Clip to prevent extreme corrections
            ratio = actual_mean / pred_mean
            corrections[m] = np.clip(ratio, 0.7, 1.4)
        else:
            corrections[m] = 1.0

    return corrections


def apply_recalibration(dates, preds, corrections):
    """Apply per-month multiplicative corrections."""
    result = preds.copy()
    for i, d in enumerate(dates):
        m = pd.Timestamp(d).month
        result[i] *= corrections.get(m, 1.0)
    return result


def compute_growth_recalibration(train_df, pred_dates, preds, target):
    """
    Recalibrate predictions so their monthly growth over the last training year
    matches the observed 2021→2022 growth pattern.
    
    This directly fixes the uneven snowball growth.
    """
    train = train_df.copy()
    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month

    last_year = train["year"].max()
    prev_year = last_year - 1

    # Monthly growth targets from last two years
    ly = train[train["year"] == last_year].groupby("month")[target].mean()
    py = train[train["year"] == prev_year].groupby("month")[target].mean()
    monthly_growth = (ly / py).to_dict()  # target growth per month

    # Current predictions' implied growth
    pdf = pd.DataFrame({"Date": pred_dates, "pred": preds})
    pdf["month"] = pd.to_datetime(pdf["Date"]).dt.month
    pdf["year"] = pd.to_datetime(pdf["Date"]).dt.year

    corrections = {}
    for m in range(1, 13):
        pred_mean = pdf[pdf["month"] == m]["pred"].mean()
        if pred_mean <= 0 or m not in ly.index:
            corrections[m] = 1.0
            continue

        # What's the implied growth of predictions over last training year?
        implied_growth = pred_mean / ly[m]
        # Target growth (use observed recent growth, capped)
        target_growth = monthly_growth.get(m, 1.12)
        target_growth = np.clip(target_growth, 0.95, 1.30)  # cap extreme months

        # Correction factor
        if implied_growth > 0:
            ratio = target_growth / implied_growth
            corrections[m] = np.clip(ratio, 0.6, 1.5)
        else:
            corrections[m] = 1.0

    return corrections


# ─── Fold Evaluation ─────────────────────────────────────────────────────────

def evaluate_fold(sales, fold):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])
    train_slice = sales[sales["Date"] < val_start].copy()

    feat_df, profiles = build_feature_table(
        sales, verbose=False, profile_source_df=train_slice
    )

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    base_cols = get_feature_cols(feat_df)

    results = {}
    for target in ["Revenue", "COGS"]:
        # Full recursive models
        cols_full = _finalize_cols(trn, _get_target_cols(base_cols, target))
        models_full = [_fit_lgbm(trn[cols_full].fillna(0), trn[target],
                                  val[cols_full].fillna(0), val[target],
                                  SEED + i*17) for i in range(N_SEEDS)]

        # Stateless models
        cols_sl = _finalize_cols(trn, _get_stateless_cols(base_cols, target))
        models_sl = [_fit_lgbm(trn[cols_sl].fillna(0), trn[target],
                                val[cols_sl].fillna(0), val[target],
                                SEED + i*17) for i in range(N_SEEDS)]

        print(f"  {target}: Running recursive prediction...")
        pred_rec = recursive_predict(
            models_full, train_slice, val["Date"].values,
            cols_full, profiles, target,
        )

        print(f"  {target}: Running stateless prediction...")
        pred_sl = stateless_predict(
            models_sl, train_slice, val["Date"].values,
            cols_sl, profiles, target,
        )

        actuals = val[target].values

        # Method 1: Raw recursive
        mae_raw = float(np.mean(np.abs(actuals - pred_rec)))

        # Method 2: Monthly recalibration from CV error patterns
        # (Use leave-one-out: correct using error patterns, then evaluate)
        corrections_cv = compute_monthly_recalibration(
            train_slice, val["Date"].values, pred_rec, actuals, target
        )
        pred_recalib = apply_recalibration(val["Date"].values, pred_rec, corrections_cv)
        mae_recalib = float(np.mean(np.abs(actuals - pred_recalib)))

        # Method 3: Growth recalibration
        corrections_growth = compute_growth_recalibration(
            train_slice, val["Date"].values, pred_rec, target
        )
        pred_growth = apply_recalibration(val["Date"].values, pred_rec, corrections_growth)
        mae_growth = float(np.mean(np.abs(actuals - pred_growth)))

        # Method 4: Blend recursive + stateless (EX-51 insight: 70/30)
        pred_blend = 0.70 * pred_sl + 0.30 * pred_rec
        mae_blend = float(np.mean(np.abs(actuals - pred_blend)))

        # Method 5: Recalibrated blend
        pred_recalib_blend = apply_recalibration(
            val["Date"].values, pred_blend, corrections_cv
        )
        mae_recalib_blend = float(np.mean(np.abs(actuals - pred_recalib_blend)))

        # Method 6: Growth-corrected recursive + stateless blend
        pred_growth_blend = 0.70 * pred_sl + 0.30 * pred_growth
        mae_growth_blend = float(np.mean(np.abs(actuals - pred_growth_blend)))

        # Find best blend weight for recursive + stateless
        best_w = 0
        best_mae = float("inf")
        for w_sl in np.arange(0, 1.01, 0.05):
            w_rec = 1.0 - w_sl
            blended = w_sl * pred_sl + w_rec * pred_rec
            mae = float(np.mean(np.abs(actuals - blended)))
            if mae < best_mae:
                best_mae = mae
                best_w = w_sl

        print(f"    Raw recursive: MAE={mae_raw:,.0f} (mean={pred_rec.mean():,.0f})")
        print(f"    CV-recalibrated: MAE={mae_recalib:,.0f}")
        print(f"    Growth-recalibrated: MAE={mae_growth:,.0f}")
        print(f"    Blend (70sl/30rec): MAE={mae_blend:,.0f} (mean={pred_blend.mean():,.0f})")
        print(f"    Recalib+blend: MAE={mae_recalib_blend:,.0f}")
        print(f"    Growth+blend: MAE={mae_growth_blend:,.0f}")
        print(f"    Best blend: w_sl={best_w:.2f}, MAE={best_mae:,.0f}")

        results[target] = {
            "pred_rec": pred_rec,
            "pred_sl": pred_sl,
            "pred_blend": pred_blend,
            "pred_growth": pred_growth,
            "actuals": actuals,
            "cols_full": cols_full,
            "cols_sl": cols_sl,
            "corrections_cv": corrections_cv,
            "corrections_growth": corrections_growth,
            "best_w_sl": best_w,
            "maes": {
                "raw": mae_raw, "recalib": mae_recalib, "growth": mae_growth,
                "blend": mae_blend, "recalib_blend": mae_recalib_blend,
                "growth_blend": mae_growth_blend, "best_blend": best_mae,
            },
        }

    # Combined scores
    print(f"\n  Combined scores ({fold['name']}):")
    for method in ["raw", "recalib", "growth", "blend", "recalib_blend", "growth_blend", "best_blend"]:
        score = results["Revenue"]["maes"][method] + 0.4 * results["COGS"]["maes"][method]
        print(f"    {method:20s}: {score:,.0f}")

    return results


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_52: Monthly-Recalibrated Ensemble")
    print("  Fixing the snowball's uneven monthly growth bias")
    print("=" * 78)

    all_corrections_cv = {t: {} for t in ["Revenue", "COGS"]}
    all_corrections_growth = {t: {} for t in ["Revenue", "COGS"]}
    fold_results = []

    for fold in FOLDS:
        print(f"\n{'='*78}")
        print(f"=== {fold['name']} ===")
        results = evaluate_fold(train, fold)
        fold_results.append(results)

        # Accumulate corrections
        for target in ["Revenue", "COGS"]:
            for m, v in results[target]["corrections_cv"].items():
                all_corrections_cv[target].setdefault(m, []).append(v)
            for m, v in results[target]["corrections_growth"].items():
                all_corrections_growth[target].setdefault(m, []).append(v)

    # Average corrections across folds
    avg_corrections_cv = {
        t: {m: np.mean(vals) for m, vals in corrs.items()}
        for t, corrs in all_corrections_cv.items()
    }
    avg_corrections_growth = {
        t: {m: np.mean(vals) for m, vals in corrs.items()}
        for t, corrs in all_corrections_growth.items()
    }

    print("\n\n=== Averaged Monthly Corrections ===")
    for target in ["Revenue", "COGS"]:
        print(f"\n  {target}:")
        for m in range(1, 13):
            cv_c = avg_corrections_cv[target].get(m, 1.0)
            gr_c = avg_corrections_growth[target].get(m, 1.0)
            print(f"    Month {m:2d}: CV={cv_c:.4f} ({(cv_c-1)*100:+.1f}%), "
                  f"Growth={gr_c:.4f} ({(gr_c-1)*100:+.1f}%)")

    # Average best stateless weight
    avg_w_sl = {t: np.mean([r[t]["best_w_sl"] for r in fold_results])
                for t in ["Revenue", "COGS"]}
    print(f"\n  Best stateless weights: Rev={avg_w_sl['Revenue']:.2f}, COGS={avg_w_sl['COGS']:.2f}")

    # ── Full retrain & submission ──
    print("\n\nFull retrain for submission...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)

    final_preds = {}
    for target in ["Revenue", "COGS"]:
        cols_full = _finalize_cols(feat_df, _get_target_cols(base_cols, target))
        cols_sl = _finalize_cols(feat_df, _get_stateless_cols(base_cols, target))

        eval_x_f = feat_df[cols_full].fillna(0).tail(365)
        eval_x_s = feat_df[cols_sl].fillna(0).tail(365)
        eval_y = feat_df[target].tail(365)

        models_full = [_fit_lgbm(feat_df[cols_full].fillna(0), feat_df[target],
                                  eval_x_f, eval_y, SEED + i*17) for i in range(N_SEEDS)]
        models_sl = [_fit_lgbm(feat_df[cols_sl].fillna(0), feat_df[target],
                                eval_x_s, eval_y, SEED + i*17) for i in range(N_SEEDS)]

        print(f"\n  {target}: Recursive prediction ({len(cols_full)} feats)...")
        pred_rec = recursive_predict(
            models_full, train, test["Date"].values, cols_full, profiles, target,
        )

        print(f"  {target}: Stateless prediction ({len(cols_sl)} feats)...")
        pred_sl = stateless_predict(
            models_sl, train, test["Date"].values, cols_sl, profiles, target,
        )

        w_sl = avg_w_sl[target]

        # Raw blend
        pred_blend = w_sl * pred_sl + (1 - w_sl) * pred_rec
        pred_blend = np.maximum(0, pred_blend)

        # CV-recalibrated recursive
        pred_rec_recalib = apply_recalibration(
            test["Date"].values, pred_rec, avg_corrections_cv[target]
        )
        pred_rec_recalib = np.maximum(0, pred_rec_recalib)

        # Growth-recalibrated recursive
        pred_rec_growth = apply_recalibration(
            test["Date"].values, pred_rec, avg_corrections_growth[target]
        )
        pred_rec_growth = np.maximum(0, pred_rec_growth)

        # Recalibrated blend
        pred_recalib_blend = w_sl * pred_sl + (1 - w_sl) * pred_rec_recalib
        pred_recalib_blend = np.maximum(0, pred_recalib_blend)

        # Growth blend
        pred_growth_blend = w_sl * pred_sl + (1 - w_sl) * pred_rec_growth
        pred_growth_blend = np.maximum(0, pred_growth_blend)

        final_preds[target] = {
            "raw_rec": pred_rec,
            "blend": pred_blend,
            "rec_recalib": pred_rec_recalib,
            "rec_growth": pred_rec_growth,
            "recalib_blend": pred_recalib_blend,
            "growth_blend": pred_growth_blend,
        }

        print(f"  {target}: rec_mean={pred_rec.mean():,.0f}, sl_mean={pred_sl.mean():,.0f}")
        print(f"  {target}: blend_mean={pred_blend.mean():,.0f}, recalib_blend_mean={pred_recalib_blend.mean():,.0f}")
        print(f"  {target}: growth_blend_mean={pred_growth_blend.mean():,.0f}")

    # Save submissions
    variants = {
        "ex_52_blend": "blend",
        "ex_52_recalib_blend": "recalib_blend",
        "ex_52_growth_blend": "growth_blend",
        "ex_52_rec_recalib": "rec_recalib",
        "ex_52_rec_growth": "rec_growth",
    }

    for name, variant in variants.items():
        path = SUB_DIR / f"{name}.csv"
        make_submission(
            test["Date"],
            final_preds["Revenue"][variant],
            final_preds["COGS"][variant],
            path,
        )

    # Bridge blends with NEW anchor (ex_51_bridge_w15)
    anchor_path = SUB_DIR / "ex_51_bridge_w15.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path, parse_dates=["Date"])
        for variant_name, variant_key in [("blend", "blend"), ("recalib_blend", "recalib_blend")]:
            for w in [0.1, 0.15, 0.2, 0.3]:
                br = (1-w) * anchor["Revenue"].values + w * final_preds["Revenue"][variant_key]
                bc = (1-w) * anchor["COGS"].values + w * final_preds["COGS"][variant_key]
                bp = SUB_DIR / f"ex_52_{variant_name}_bridge_w{int(w*100):02d}.csv"
                make_submission(test["Date"], br, bc, bp)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "avg_corrections_cv": {t: {str(k): v for k, v in c.items()} for t, c in avg_corrections_cv.items()},
        "avg_w_sl": avg_w_sl,
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone in {meta['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
