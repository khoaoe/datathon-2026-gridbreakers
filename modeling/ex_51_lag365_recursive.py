"""
EX_51: Lag-365-Only Recursive Model — Best of Both Worlds

Key Insight from EX-49/EX-50 research:
- Recursive models have CORRECT LEVEL (3.99M) but noisy seasonality (snowball)
- Stateless models have CLEAN SEASONALITY but wrong level (3.66M)
- The snowball comes from short-term lags (lag1/7/28/60) compounding errors

Solution: Keep lag365 (real signal, 1-step recursion) but DROP all short lags.
- For 2023: lag365 = actual 2022 data → zero recursion, zero snowball
- For 2024: lag365 = predicted 2023 → only 1 "generation" of error, not 548

This is the recursive architecture with the MINIMUM possible snowball exposure.

Multi-component ensemble (same as EX-22 production winner architecture):
- core_lag365: lag365 features + calendar + profiles
- stateless: no lags at all (pure calendar/profiles)
- Weighted ensemble optimized on 2-fold recursive CV
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize

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

TRACK = Path("output/tracking/ex_51_lag365_recursive")
SUB_DIR = Path("output/submissions")

N_SEEDS = 3

FOLDS = [
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]


def _finalize_cols(df, cols):
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.30]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def _get_stateless_cols(base_cols, target):
    """Remove ALL recursive features."""
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


def _get_lag365_cols(base_cols, target):
    """Keep lag365-family features + calendar/profiles. Drop short-term lags."""
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    keep = []
    for c in base_cols:
        if c.startswith(blocked):
            continue
        if c in ["Revenue", "COGS"]:
            continue
        # Keep lag365-family features
        if "lag365" in c or "lag364" in c or "lag366" in c or "lag730" in c:
            keep.append(c)
            continue
        # Keep YoY ratio (uses lag365)
        if "yoy" in c:
            keep.append(c)
            continue
        # Drop ALL other lag/rolling/growth features
        if any(k in c for k in ["lag", "rmean", "rstd", "rmin", "rmax",
                                  "rmedian", "ratio", "growth", "momentum",
                                  "spread", "margin_ratio", "diff", "vol",
                                  "zscore"]):
            continue
        keep.append(c)
    return keep


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


def _predict_ensemble(models, x):
    return np.mean([m.predict(x) for m in models], axis=0)


# ─── Recursive Prediction (lag365-only) ──────────────────────────────────────

def recursive_predict_lag365(
    models, history_df, predict_dates, feature_cols, profiles, target,
):
    """
    Recursive prediction using ONLY lag365 features.
    
    Phase 1 (first 365 days): lag365 comes from real training data → zero error propagation
    Phase 2 (remaining days): lag365 comes from Phase 1 predictions → 1 generation of error
    """
    history = history_df[["Date", target]].copy()
    preds = []

    for i, date in enumerate(predict_dates):
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        # Build features
        combined = build_calendar_features(combined)

        # Lag-365 features only
        combined[f"{target}_lag365"] = combined[target].shift(365)
        combined[f"{target}_lag364"] = combined[target].shift(364)
        combined[f"{target}_lag366"] = combined[target].shift(366)

        shifted_365 = combined[target].shift(365)
        combined[f"{target}_lag365_smooth7"] = shifted_365.rolling(7, min_periods=1, center=True).mean()
        combined[f"{target}_lag365_rmean28"] = shifted_365.rolling(28, min_periods=7).mean()
        combined[f"{target}_lag365_rstd28"] = shifted_365.rolling(28, min_periods=7).std()

        shifted_730 = combined[target].shift(730)
        combined[f"{target}_yoy_365_730"] = shifted_365 / shifted_730.replace(0, np.nan)
        combined[f"{target}_lag365_rmean30"] = shifted_365.rolling(30, min_periods=7).mean()

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


def recursive_predict_stateless(
    models, history_df, predict_dates, feature_cols, profiles, target,
):
    """Stateless prediction — no recursion needed, bulk predict."""
    history = history_df[["Date", target]].copy()

    # Build all dates at once
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

    preds = _predict_ensemble(models, x_pred)
    return np.maximum(0, preds)


# ─── Fold Evaluation ─────────────────────────────────────────────────────────

def evaluate_fold(sales, fold):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(
        sales, verbose=False, profile_source_df=train_slice
    )

    # Add lag365-specific features
    for target in ["Revenue", "COGS"]:
        shifted_365 = feat_df[target].shift(365)
        feat_df[f"{target}_lag365_smooth7"] = shifted_365.rolling(7, min_periods=1, center=True).mean()
        feat_df[f"{target}_lag365_rmean28"] = shifted_365.rolling(28, min_periods=7).mean()
        feat_df[f"{target}_lag365_rstd28"] = shifted_365.rolling(28, min_periods=7).std()
        shifted_730 = feat_df[target].shift(730)
        feat_df[f"{target}_yoy_365_730"] = shifted_365 / shifted_730.replace(0, np.nan)
        feat_df[f"{target}_lag365_rmean30"] = shifted_365.rolling(30, min_periods=7).mean()

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    base_cols = get_feature_cols(feat_df)

    component_preds = {}

    for target in ["Revenue", "COGS"]:
        print(f"\n  --- {target} ---")

        # Component 1: lag365-only
        cols_lag365 = _finalize_cols(trn, _get_lag365_cols(base_cols, target))
        print(f"    lag365 component: {len(cols_lag365)} features")

        models_lag365 = []
        for i in range(N_SEEDS):
            models_lag365.append(_fit_lgbm(
                trn[cols_lag365].fillna(0), trn[target],
                val[cols_lag365].fillna(0), val[target],
                seed=SEED + i * 17,
            ))

        # Component 2: stateless
        cols_stateless = _finalize_cols(trn, _get_stateless_cols(base_cols, target))
        print(f"    stateless component: {len(cols_stateless)} features")

        models_stateless = []
        for i in range(N_SEEDS):
            models_stateless.append(_fit_lgbm(
                trn[cols_stateless].fillna(0), trn[target],
                val[cols_stateless].fillna(0), val[target],
                seed=SEED + i * 17,
            ))

        # Component 3: full recursive (lag1/7/28/60 + lag365 + all)
        cols_full = _finalize_cols(trn, _get_target_cols(base_cols, target))
        print(f"    full recursive component: {len(cols_full)} features")

        models_full = []
        for i in range(N_SEEDS):
            models_full.append(_fit_lgbm(
                trn[cols_full].fillna(0), trn[target],
                val[cols_full].fillna(0), val[target],
                seed=SEED + i * 17,
            ))

        # Recursive predictions
        print(f"    Running lag365 recursive prediction...")
        pred_lag365 = recursive_predict_lag365(
            models_lag365, train_slice, val["Date"].values,
            cols_lag365, profiles, target,
        )

        print(f"    Running stateless prediction...")
        pred_stateless = recursive_predict_stateless(
            models_stateless, train_slice, val["Date"].values,
            cols_stateless, profiles, target,
        )

        print(f"    Running full recursive prediction...")
        pred_full = recursive_predict_full(
            models_full, train_slice, val["Date"].values,
            cols_full, profiles, target,
        )

        actuals = val[target].values

        mae_lag365 = float(np.mean(np.abs(actuals - pred_lag365)))
        mae_stateless = float(np.mean(np.abs(actuals - pred_stateless)))
        mae_full = float(np.mean(np.abs(actuals - pred_full)))

        print(f"    lag365 recursive MAE: {mae_lag365:,.0f} (mean={pred_lag365.mean():,.0f})")
        print(f"    stateless MAE:        {mae_stateless:,.0f} (mean={pred_stateless.mean():,.0f})")
        print(f"    full recursive MAE:   {mae_full:,.0f} (mean={pred_full.mean():,.0f})")

        component_preds[target] = {
            "lag365": pred_lag365,
            "stateless": pred_stateless,
            "full": pred_full,
            "actuals": actuals,
            "cols_lag365": cols_lag365,
            "cols_stateless": cols_stateless,
            "cols_full": cols_full,
            "mae_lag365": mae_lag365,
            "mae_stateless": mae_stateless,
            "mae_full": mae_full,
        }

    # Optimize ensemble weights
    print(f"\n  Optimizing ensemble weights...")
    best_score = float("inf")
    best_w = None

    for w1 in np.arange(0, 1.05, 0.1):
        for w2 in np.arange(0, 1.05 - w1, 0.1):
            w3 = round(1.0 - w1 - w2, 1)
            if w3 < -0.01:
                continue
            w3 = max(0, w3)

            score = 0
            for target in ["Revenue", "COGS"]:
                cp = component_preds[target]
                ens = w1 * cp["lag365"] + w2 * cp["stateless"] + w3 * cp["full"]
                ens = np.maximum(0, ens)
                mae = float(np.mean(np.abs(cp["actuals"] - ens)))
                weight = 1.0 if target == "Revenue" else 0.4
                score += mae * weight

            if score < best_score:
                best_score = score
                best_w = (w1, w2, w3)

    print(f"  Best ensemble: score={best_score:,.0f}")
    print(f"  Weights: lag365={best_w[0]:.1f}, stateless={best_w[1]:.1f}, full={best_w[2]:.1f}")

    # Individual scores
    for name, idx in [("lag365", 0), ("stateless", 1), ("full", 2)]:
        w = [0, 0, 0]
        w[idx] = 1.0
        score = 0
        for target in ["Revenue", "COGS"]:
            cp = component_preds[target]
            preds = [cp["lag365"], cp["stateless"], cp["full"]]
            mae = float(np.mean(np.abs(cp["actuals"] - preds[idx])))
            weight = 1.0 if target == "Revenue" else 0.4
            score += mae * weight
        print(f"    {name} solo: {score:,.0f}")

    return {
        "fold": fold["name"],
        "best_score": best_score,
        "best_weights": best_w,
        "component_preds": component_preds,
    }


def recursive_predict_full(
    models, history_df, predict_dates, feature_cols, profiles, target,
):
    """Standard full recursive prediction (all lags)."""
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_51: Lag-365-Only Recursive Model")
    print("  Component 1: lag365 + calendar + profiles (minimal snowball)")
    print("  Component 2: stateless (zero snowball)")
    print("  Component 3: full recursive (baseline)")
    print("  Ensemble weights optimized on 2-fold recursive CV")
    print("=" * 78)

    fold_results = []
    for fold in FOLDS:
        print(f"\n{'='*78}")
        print(f"=== {fold['name']} ===")
        result = evaluate_fold(train, fold)
        fold_results.append(result)

    print("\n\n=== FOLD SUMMARY ===")
    for r in fold_results:
        print(f"  {r['fold']}: best_ens={r['best_score']:,.0f}, "
              f"weights=lag365:{r['best_weights'][0]:.1f}/stateless:{r['best_weights'][1]:.1f}/full:{r['best_weights'][2]:.1f}")

    mean_score = np.mean([r["best_score"] for r in fold_results])
    avg_weights = np.mean([r["best_weights"] for r in fold_results], axis=0)
    print(f"\n  Mean ensemble score: {mean_score:,.0f}")
    print(f"  Avg weights: lag365={avg_weights[0]:.2f}, stateless={avg_weights[1]:.2f}, full={avg_weights[2]:.2f}")

    # Full retrain
    print("\n\nFull retrain for submission...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)

    for target in ["Revenue", "COGS"]:
        shifted_365 = feat_df[target].shift(365)
        feat_df[f"{target}_lag365_smooth7"] = shifted_365.rolling(7, min_periods=1, center=True).mean()
        feat_df[f"{target}_lag365_rmean28"] = shifted_365.rolling(28, min_periods=7).mean()
        feat_df[f"{target}_lag365_rstd28"] = shifted_365.rolling(28, min_periods=7).std()
        shifted_730 = feat_df[target].shift(730)
        feat_df[f"{target}_yoy_365_730"] = shifted_365 / shifted_730.replace(0, np.nan)
        feat_df[f"{target}_lag365_rmean30"] = shifted_365.rolling(30, min_periods=7).mean()

    base_cols = get_feature_cols(feat_df)

    final_preds = {}
    for target in ["Revenue", "COGS"]:
        cols_lag365 = _finalize_cols(feat_df, _get_lag365_cols(base_cols, target))
        cols_stateless = _finalize_cols(feat_df, _get_stateless_cols(base_cols, target))
        cols_full = _finalize_cols(feat_df, _get_target_cols(base_cols, target))

        eval_x_l = feat_df[cols_lag365].fillna(0).tail(365)
        eval_x_s = feat_df[cols_stateless].fillna(0).tail(365)
        eval_x_f = feat_df[cols_full].fillna(0).tail(365)
        eval_y = feat_df[target].tail(365)

        models_lag365 = [_fit_lgbm(feat_df[cols_lag365].fillna(0), feat_df[target],
                                    eval_x_l, eval_y, SEED + i*17) for i in range(N_SEEDS)]
        models_stateless = [_fit_lgbm(feat_df[cols_stateless].fillna(0), feat_df[target],
                                       eval_x_s, eval_y, SEED + i*17) for i in range(N_SEEDS)]
        models_full = [_fit_lgbm(feat_df[cols_full].fillna(0), feat_df[target],
                                  eval_x_f, eval_y, SEED + i*17) for i in range(N_SEEDS)]

        print(f"\n  {target}: Running lag365 recursive ({len(cols_lag365)} feats)...")
        pred_lag365 = recursive_predict_lag365(
            models_lag365, train, test["Date"].values, cols_lag365, profiles, target,
        )
        print(f"  {target}: Running stateless ({len(cols_stateless)} feats)...")
        pred_stateless = recursive_predict_stateless(
            models_stateless, train, test["Date"].values, cols_stateless, profiles, target,
        )
        print(f"  {target}: Running full recursive ({len(cols_full)} feats)...")
        pred_full = recursive_predict_full(
            models_full, train, test["Date"].values, cols_full, profiles, target,
        )

        w1, w2, w3 = avg_weights
        final = w1 * pred_lag365 + w2 * pred_stateless + w3 * pred_full
        final = np.maximum(0, final)
        final_preds[target] = final

        print(f"  {target}: lag365_mean={pred_lag365.mean():,.0f}, "
              f"stateless_mean={pred_stateless.mean():,.0f}, "
              f"full_mean={pred_full.mean():,.0f}, "
              f"ensemble_mean={final.mean():,.0f}")

    # Submissions
    path = SUB_DIR / "ex_51_lag365_recursive.csv"
    make_submission(test["Date"], final_preds["Revenue"], final_preds["COGS"], path)

    # Individual component submissions
    # (already have pred_lag365 etc. from last target loop, but need both targets)
    # Save lag365-only submission separately
    # Re-run for both targets to get individual subs... skip for now, ensemble is primary

    # Bridge blends
    anchor_path = SUB_DIR / "ex_24_bridge_w01.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path, parse_dates=["Date"])
        for w in [0.1, 0.2, 0.3, 0.5]:
            br = (1-w) * anchor["Revenue"].values + w * final_preds["Revenue"]
            bc = (1-w) * anchor["COGS"].values + w * final_preds["COGS"]
            bp = SUB_DIR / f"ex_51_bridge_w{int(w*100):02d}.csv"
            make_submission(test["Date"], br, bc, bp)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "mean_score": float(mean_score),
        "avg_weights": list(avg_weights),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone in {meta['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
