"""
EX_50: Lag-365 Direct Model — Zero Recursion for Year 1

Key Insight from EX-49:
- Pure stateless LGBM has great CV (753k) but underpredicts test level by ~300k
- The model NEEDS some lag signal to know "where revenue is now"
- lag365 for Jan-Dec 2023 = actual Jan-Dec 2022 data (ZERO recursion!)
- Only the last 183 days (Jan-Jul 2024) need lag365 from 2023 predictions

Architecture:
1. Features: Calendar + Profiles + lag365 + lag365-derived rolling features
2. Phase 1 (Jan 2023 - Dec 2023): Direct prediction using actual 2022 as lag365
3. Phase 2 (Jan 2024 - Jul 2024): Use Phase 1 predictions as lag365 (minimal recursion)
4. Multi-seed LGBM ensemble
5. Bridge blends with production anchor

This eliminates 90%+ of the recursive snowball since only Phase 2 uses predictions.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from modeling.config import LGBM_PARAMS, SEED, FILES
from modeling.feature_engineering import (
    build_feature_table,
    build_calendar_features,
    compute_historical_profiles,
    compute_aux_profiles,
    compute_order_based_profiles,
    merge_profiles,
    _load_promotions_table,
    _compute_promo_features_for_dates,
    add_promo_interaction_features,
    get_feature_cols,
)
from modeling.utils import evaluate, load_sales, make_submission

warnings.filterwarnings("ignore")

TRACK = Path("output/tracking/ex_50_lag365_direct")
SUB_DIR = Path("output/submissions")

N_SEEDS = 5

FOLDS = [
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]


# ─── Feature Construction ────────────────────────────────────────────────────

def build_lag365_features(df, target="Revenue"):
    """Build lag-365-based features only (no short-term lags)."""
    new = {}
    shifted_365 = df[target].shift(365)
    shifted_366 = df[target].shift(366)
    shifted_364 = df[target].shift(364)

    new[f"{target}_lag365"] = shifted_365
    new[f"{target}_lag364"] = shifted_364
    new[f"{target}_lag366"] = shifted_366

    # Smoothed lag365 (7-day window around same-DOY last year)
    new[f"{target}_lag365_smooth7"] = shifted_365.rolling(7, min_periods=1, center=True).mean()

    # Last year's rolling stats (28-day window centered on lag365)
    new[f"{target}_lag365_rmean28"] = shifted_365.rolling(28, min_periods=7).mean()
    new[f"{target}_lag365_rstd28"] = shifted_365.rolling(28, min_periods=7).std()

    # YoY ratio: how did lag365 compare to lag730?
    shifted_730 = df[target].shift(730)
    new[f"{target}_yoy_365_730"] = shifted_365 / shifted_730.replace(0, np.nan)

    # Last year's monthly mean (use lag365 rolling 30)
    new[f"{target}_lag365_rmean30"] = shifted_365.rolling(30, min_periods=7).mean()

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def build_full_features(sales_df, profile_source_df=None):
    """Build features with calendar, profiles, promos, and lag365."""
    df = sales_df.copy().sort_values("Date").reset_index(drop=True)
    profile_src = profile_source_df if profile_source_df is not None else df

    # Calendar
    df = build_calendar_features(df)
    df["time_index"] = (df["Date"] - pd.Timestamp("2013-01-01")).dt.days

    # Promos
    promos = _load_promotions_table()
    promo_feats = _compute_promo_features_for_dates(df["Date"], promos)
    df = df.merge(promo_feats, on="Date", how="left")
    df = add_promo_interaction_features(df)

    # Lag-365 features (the key differentiator)
    df = build_lag365_features(df, "Revenue")
    df = build_lag365_features(df, "COGS")

    # Also add lag28 and lag60 — but ONLY for training (will be NaN for test Phase 1)
    # This teaches the model to use lag365 as primary and tolerate NaN short lags
    for target in ["Revenue", "COGS"]:
        df[f"{target}_lag28"] = df[target].shift(28)
        df[f"{target}_lag60"] = df[target].shift(60)

    # Profiles
    profiles = compute_historical_profiles(profile_src)
    for key in ["dow", "month", "woy", "dom", "quarter", "month_dow"]:
        if key in profiles:
            merge_key = {
                "dow": "dayofweek", "month": "month", "woy": "weekofyear",
                "dom": "dayofmonth", "quarter": "quarter",
                "month_dow": ["month", "dayofweek"],
            }[key]
            df = merge_profiles(df, profiles[key], merge_key)

    aux_profiles = compute_aux_profiles()
    for name, prof in aux_profiles.items():
        key_col = [c for c in prof.columns if c in df.columns]
        if key_col:
            df = merge_profiles(df, prof, key_col)

    order_profiles = compute_order_based_profiles()
    for name, prof in order_profiles.items():
        key_col = [c for c in prof.columns if c in df.columns]
        if key_col:
            df = merge_profiles(df, prof, key_col)

    return df, profiles


def _get_feature_cols(df, target):
    """Get feature columns, excluding other target's lag features."""
    exclude = {"Date", "Revenue", "COGS"}
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    cols = [c for c in df.columns if c not in exclude and not c.startswith(blocked)]
    # Keep lag365 features but filter
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.30]  # Allow more NaN for lag features
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _fit_models(trn_x, trn_y, val_x, val_y, n_seeds=N_SEEDS):
    models = []
    for i in range(n_seeds):
        params = LGBM_PARAMS.copy()
        params["n_estimators"] = 2000
        params["random_state"] = SEED + i * 17
        m = lgb.LGBMRegressor(**params)
        m.fit(
            trn_x, trn_y,
            eval_set=[(val_x, val_y)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        models.append(m)
    return models


def _predict_ensemble(models, x):
    return np.mean([m.predict(x) for m in models], axis=0)


# ─── Two-Phase Prediction ────────────────────────────────────────────────────

def two_phase_predict(models, full_history, test_dates, feature_cols, target,
                      profiles=None):
    """
    Phase 1: Direct prediction (lag365 = real data)
    Phase 2: Semi-recursive (lag365 = Phase 1 predictions)
    """
    history = full_history.copy().sort_values("Date").reset_index(drop=True)
    last_train_date = history["Date"].max()

    # Phase 1: dates where lag365 comes from actual training data
    phase1_end = last_train_date + pd.DateOffset(days=365)
    phase1_dates = [d for d in test_dates if pd.Timestamp(d) <= phase1_end]
    phase2_dates = [d for d in test_dates if pd.Timestamp(d) > phase1_end]

    print(f"    Phase 1: {len(phase1_dates)} days (direct, real lag365)")
    print(f"    Phase 2: {len(phase2_dates)} days (semi-recursive)")

    all_preds = []

    # ── Phase 1: Bulk direct prediction ──
    if phase1_dates:
        # Build feature rows for all Phase 1 dates at once
        phase1_df = pd.DataFrame({
            "Date": pd.to_datetime(phase1_dates),
            target: np.nan,
        })
        # Add other target column
        other = "COGS" if target == "Revenue" else "Revenue"
        phase1_df[other] = np.nan

        combined = pd.concat([history, phase1_df], ignore_index=True).sort_values("Date")
        combined = build_calendar_features(combined)
        combined["time_index"] = (combined["Date"] - pd.Timestamp("2013-01-01")).dt.days
        combined = build_lag365_features(combined, target)

        # Short lags will be NaN for test rows — that's OK, model learned to handle it
        combined[f"{target}_lag28"] = combined[target].shift(28)
        combined[f"{target}_lag60"] = combined[target].shift(60)

        # Add profiles
        promos = _load_promotions_table()
        promo_feats = _compute_promo_features_for_dates(combined["Date"], promos)
        combined = combined.merge(promo_feats, on="Date", how="left")
        combined = add_promo_interaction_features(combined)

        rev_profiles = compute_historical_profiles(history)
        for key in ["dow", "month", "woy", "dom", "quarter", "month_dow"]:
            if key in rev_profiles:
                merge_key = {
                    "dow": "dayofweek", "month": "month", "woy": "weekofyear",
                    "dom": "dayofmonth", "quarter": "quarter",
                    "month_dow": ["month", "dayofweek"],
                }[key]
                combined = merge_profiles(combined, rev_profiles[key], merge_key)

        aux_profiles = compute_aux_profiles()
        for name, prof in aux_profiles.items():
            key_col = [c for c in prof.columns if c in combined.columns]
            if key_col:
                combined = merge_profiles(combined, prof, key_col)

        order_profiles = compute_order_based_profiles()
        for name, prof in order_profiles.items():
            key_col = [c for c in prof.columns if c in combined.columns]
            if key_col:
                combined = merge_profiles(combined, prof, key_col)

        # Extract Phase 1 rows
        p1_rows = combined[combined["Date"].isin(pd.to_datetime(phase1_dates))].copy()
        p1_x = pd.DataFrame(0.0, index=range(len(p1_rows)), columns=feature_cols)
        for c in feature_cols:
            if c in p1_rows.columns:
                vals = p1_rows[c].values
                p1_x[c] = np.where(np.isnan(vals.astype(float)), 0.0, vals)

        p1_preds = _predict_ensemble(models, p1_x)
        p1_preds = np.maximum(0, p1_preds)
        all_preds.extend(p1_preds.tolist())

        # Update history with Phase 1 predictions
        p1_history = pd.DataFrame({
            "Date": pd.to_datetime(phase1_dates),
            target: p1_preds,
        })
        # Carry forward other columns
        for col in history.columns:
            if col not in p1_history.columns:
                p1_history[col] = np.nan
        history = pd.concat([history, p1_history], ignore_index=True).sort_values("Date")

    # ── Phase 2: Day-by-day recursive ──
    for date in phase2_dates:
        ts = pd.Timestamp(date)
        new_row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        for col in history.columns:
            if col not in new_row.columns:
                new_row[col] = np.nan

        combined = pd.concat([history, new_row], ignore_index=True).sort_values("Date")
        combined = build_calendar_features(combined)
        combined["time_index"] = (combined["Date"] - pd.Timestamp("2013-01-01")).dt.days
        combined = build_lag365_features(combined, target)
        combined[f"{target}_lag28"] = combined[target].shift(28)
        combined[f"{target}_lag60"] = combined[target].shift(60)

        promos = _load_promotions_table()
        promo_feats = _compute_promo_features_for_dates(combined.iloc[-1:]["Date"], promos)
        last_row = combined.iloc[-1:].copy()
        last_row = last_row.merge(promo_feats, on="Date", how="left")
        last_row = add_promo_interaction_features(last_row)

        # Add profiles
        rev_profiles = compute_historical_profiles(
            history[history[target].notna()]
        )
        for key in ["dow", "month", "woy", "dom", "quarter", "month_dow"]:
            if key in rev_profiles:
                merge_key = {
                    "dow": "dayofweek", "month": "month", "woy": "weekofyear",
                    "dom": "dayofmonth", "quarter": "quarter",
                    "month_dow": ["month", "dayofweek"],
                }[key]
                last_row = merge_profiles(last_row, rev_profiles[key], merge_key)

        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else float(val)

        pred = float(np.mean([m.predict(x_pred)[0] for m in models]))
        pred = max(0, pred)
        all_preds.append(pred)

        new_hist = pd.DataFrame({"Date": [ts], target: [pred]})
        for col in history.columns:
            if col not in new_hist.columns:
                new_hist[col] = np.nan
        history = pd.concat([history, new_hist], ignore_index=True)

    return np.array(all_preds)


# ─── CV Evaluation ────────────────────────────────────────────────────────────

def evaluate_fold(sales, fold):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()

    # Build features on full history (profiles from train only)
    feat_df, profiles = build_full_features(sales, profile_source_df=train_slice)

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    results = {}
    for target in ["Revenue", "COGS"]:
        cols = _get_feature_cols(trn, target)
        trn_x = trn[cols].fillna(0)
        val_x = val[cols].fillna(0)

        # Direct evaluation (uses true lag365 since val is within 365 of train)
        models = _fit_models(trn_x, trn[target], val_x, val[target], n_seeds=3)
        direct_preds = np.maximum(0, _predict_ensemble(models, val_x))
        direct_mae = float(np.mean(np.abs(val[target].values - direct_preds)))

        print(f"  {target} direct MAE: {direct_mae:,.0f}, "
              f"pred_mean={direct_preds.mean():,.0f}, actual_mean={val[target].mean():,.0f}")

        # Two-phase recursive evaluation
        phase_preds = two_phase_predict(
            models, train_slice, val["Date"].values, cols, target
        )
        phase_mae = float(np.mean(np.abs(val[target].values - phase_preds)))
        print(f"  {target} 2-phase MAE: {phase_mae:,.0f}, pred_mean={phase_preds.mean():,.0f}")

        results[target] = {
            "direct_mae": direct_mae,
            "phase_mae": phase_mae,
            "direct_preds": direct_preds,
            "phase_preds": phase_preds,
            "cols": cols,
        }

    # Scores
    direct_score = results["Revenue"]["direct_mae"] + 0.4 * results["COGS"]["direct_mae"]
    phase_score = results["Revenue"]["phase_mae"] + 0.4 * results["COGS"]["phase_mae"]

    print(f"\n  {fold['name']} Direct score: {direct_score:,.0f}")
    print(f"  {fold['name']} 2-Phase score: {phase_score:,.0f}")

    return {
        "fold": fold["name"],
        "direct_score": direct_score,
        "phase_score": phase_score,
        "results": results,
    }


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_50: Lag-365 Direct Model")
    print("  Phase 1: Direct prediction (365 days, real lag365 from 2022)")
    print("  Phase 2: Semi-recursive (183 days, lag365 from Phase 1)")
    print("  Eliminates 67% of recursive snowball")
    print("=" * 78)

    # CV
    fold_results = []
    for fold in FOLDS:
        print(f"\n{'='*78}")
        print(f"=== {fold['name']} ===")
        result = evaluate_fold(train, fold)
        fold_results.append(result)

    print("\n\n=== FOLD SUMMARY ===")
    for r in fold_results:
        print(f"  {r['fold']}: direct={r['direct_score']:,.0f}, 2-phase={r['phase_score']:,.0f}")

    mean_direct = np.mean([r["direct_score"] for r in fold_results])
    mean_phase = np.mean([r["phase_score"] for r in fold_results])
    print(f"\n  Mean direct: {mean_direct:,.0f}")
    print(f"  Mean 2-phase: {mean_phase:,.0f}")

    # Full retrain
    print("\n\nFull retrain for submission...")
    feat_df, profiles = build_full_features(train, profile_source_df=train)

    for target in ["Revenue", "COGS"]:
        cols = _get_feature_cols(feat_df, target)
        trn_x = feat_df[cols].fillna(0)
        eval_x = trn_x.tail(365)
        eval_y = feat_df[target].tail(365)

        print(f"\n  {target}: {len(cols)} features")
        models = _fit_models(trn_x, feat_df[target], eval_x, eval_y, n_seeds=N_SEEDS)

        # Two-phase test prediction
        preds = two_phase_predict(models, train, test["Date"].values, cols, target)

        if target == "Revenue":
            final_rev = preds
        else:
            final_cogs = preds

    print(f"\n  Revenue mean: {final_rev.mean():,.0f}")
    print(f"  COGS mean: {final_cogs.mean():,.0f}")

    # Save
    path = SUB_DIR / "ex_50_lag365_direct.csv"
    make_submission(test["Date"], final_rev, final_cogs, path)

    # Bridge blends with anchor
    anchor_path = SUB_DIR / "ex_24_bridge_w01.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path, parse_dates=["Date"])
        for w in [0.1, 0.2, 0.3, 0.5]:
            br = (1 - w) * anchor["Revenue"].values + w * final_rev
            bc = (1 - w) * anchor["COGS"].values + w * final_cogs
            bp = SUB_DIR / f"ex_50_bridge_w{int(w*100):02d}.csv"
            make_submission(test["Date"], br, bc, bp)
            print(f"  Bridge w={w}: rev_mean={br.mean():,.0f}")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "mean_direct_score": float(mean_direct),
        "mean_phase_score": float(mean_phase),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone in {meta['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
