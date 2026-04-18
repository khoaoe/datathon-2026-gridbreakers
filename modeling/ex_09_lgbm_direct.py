"""
EX_09: Direct per-horizon LightGBM.

Instead of recursive prediction (where each day's output feeds back as the
next day's lag and errors compound across 548 days), we train ONE model per
horizon bucket. Each model only uses lags that are at least as old as the
bucket, so prediction at test-day-h uses genuine training-time features, no
recursion. Gives much stronger long-horizon accuracy.

Buckets chosen so every day in the 548-day horizon has a model whose minimum
lag fits:
    h_7      lags ∈ [7, 14, 21, 28, 60, 90, 180, 365]
    h_30     lags ∈ [30, 60, 90, 180, 365]
    h_90     lags ∈ [90, 180, 365]
    h_180    lags ∈ [180, 365]
    h_365    lags ∈ [365]

Test day k (k days past train_end) uses the largest bucket h s.t. h ≤ k.
"""
from __future__ import annotations

import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.config import (
    LGBM_PARAMS, MODEL_DIR, SUBMISSION_DIR, VAL_START, VAL_END, TRAIN_END,
)
from modeling.utils import (
    evaluate, load_sales, make_submission,
    horizon_stratified_metrics, save_val_predictions,
)
from modeling.feature_engineering import (
    build_calendar_features, build_lag_features, build_rolling_features,
    compute_historical_profiles, compute_aux_profiles, build_promo_calendar,
    build_inventory_daily, apply_profiles_to_dates, get_feature_cols,
)
from modeling.tracker import ExperimentTracker


HORIZON_BUCKETS = {
    "h_7":   [7, 14, 21, 28, 60, 90, 180, 365],
    "h_30":  [30, 60, 90, 180, 365],
    "h_90":  [90, 180, 365],
    "h_180": [180, 365],
    "h_365": [365],
}
BUCKET_MIN = {"h_7": 1, "h_30": 8, "h_90": 31, "h_180": 91, "h_365": 181}
BUCKET_MAX = {"h_7": 7, "h_30": 30, "h_90": 90, "h_180": 180, "h_365": 548}


def pick_bucket(offset_days: int) -> str:
    """Select the smallest-lag bucket that still respects the offset."""
    for name in ["h_7", "h_30", "h_90", "h_180", "h_365"]:
        if BUCKET_MIN[name] <= offset_days <= BUCKET_MAX[name]:
            return name
    return "h_365"


def build_direct_features(
    df: pd.DataFrame,
    lags,
    promo_daily: pd.DataFrame,
    inv_daily: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df = build_calendar_features(df)
    df = build_lag_features(df, "Revenue", lags=lags)
    df = build_lag_features(df, "COGS", lags=lags)
    # Rolling features using a window that starts at min(lags) to avoid leakage
    min_lag = min(lags)
    shifted = df["Revenue"].shift(min_lag)
    for w in (7, 28, 90):
        df[f"Revenue_rmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"Revenue_rstd_{w}"] = shifted.rolling(w, min_periods=1).std()
    df = df.merge(promo_daily, on="Date", how="left")
    df = df.merge(inv_daily, on="Date", how="left")
    for col in list(promo_daily.columns) + list(inv_daily.columns):
        if col != "Date" and col in df.columns:
            df[col] = df[col].fillna(0)
    df = apply_profiles_to_dates(df, bundle)
    return df


def main():
    start = time.time()
    import lightgbm as lgb

    train, test = load_sales()
    tracker = ExperimentTracker("ex_09_lgbm_direct")

    print("=" * 70)
    print("EX_09: DIRECT PER-HORIZON LGBM")
    print("=" * 70)

    # Date-exact calendars + profiles
    full_dates = pd.date_range(train["Date"].min(), test["Date"].max(), freq="D")
    promo_daily = build_promo_calendar(full_dates)
    inv_daily = build_inventory_daily(full_dates)
    rev_profiles = compute_historical_profiles(train)
    aux_profiles = compute_aux_profiles()
    bundle = {**rev_profiles, **aux_profiles}

    # Concatenate train + placeholder test so we can build all features in one pass
    combined = pd.concat(
        [train[["Date", "Revenue", "COGS"]],
         test.assign(Revenue=np.nan, COGS=np.nan)[["Date", "Revenue", "COGS"]]],
        ignore_index=True,
    )

    val_mask = (combined["Date"] >= VAL_START) & (combined["Date"] <= VAL_END)
    trn_mask = (combined["Date"] < VAL_START) & combined["Revenue"].notna()
    test_mask = combined["Date"] > pd.Timestamp(TRAIN_END)

    models = {}
    val_preds_rev_by_bucket = {}
    val_preds_cogs_by_bucket = {}
    test_preds_rev_by_bucket = {}
    test_preds_cogs_by_bucket = {}

    for name, lags in HORIZON_BUCKETS.items():
        print(f"\n── Bucket {name} lags={lags} ──")
        feat = build_direct_features(combined.copy(), lags, promo_daily,
                                     inv_daily, bundle)
        feature_cols = get_feature_cols(feat)
        feature_cols = [c for c in feature_cols
                        if feat.loc[trn_mask, c].notna().mean() > 0.5
                        and feat.loc[trn_mask, c].nunique(dropna=True) > 1]
        X_trn = feat.loc[trn_mask, feature_cols].fillna(0)
        y_trn_rev = feat.loc[trn_mask, "Revenue"]
        y_trn_cogs = feat.loc[trn_mask, "COGS"]

        X_val = feat.loc[val_mask, feature_cols].fillna(0)
        y_val_rev = feat.loc[val_mask, "Revenue"]
        y_val_cogs = feat.loc[val_mask, "COGS"]

        m_rev = lgb.LGBMRegressor(**LGBM_PARAMS)
        m_rev.fit(
            X_trn, y_trn_rev,
            eval_set=[(X_val, y_val_rev)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)],
        )
        m_cogs = lgb.LGBMRegressor(**LGBM_PARAMS)
        m_cogs.fit(
            X_trn, y_trn_cogs,
            eval_set=[(X_val, y_val_cogs)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)],
        )
        models[name] = (m_rev, m_cogs, feature_cols)

        val_preds_rev_by_bucket[name] = m_rev.predict(X_val)
        val_preds_cogs_by_bucket[name] = m_cogs.predict(X_val)

        X_test = feat.loc[test_mask, feature_cols].fillna(0)
        test_preds_rev_by_bucket[name] = np.clip(m_rev.predict(X_test), 0, None)
        test_preds_cogs_by_bucket[name] = np.clip(m_cogs.predict(X_test), 0, None)

    # ── Validation stitching: pick bucket per val date ──
    val_dates = combined.loc[val_mask, "Date"].reset_index(drop=True)
    train_end_ts = pd.Timestamp(TRAIN_END)
    val_offsets = (val_dates - train_end_ts).dt.days.clip(lower=1).values
    val_rev_true = combined.loc[val_mask, "Revenue"].values
    val_cogs_true = combined.loc[val_mask, "COGS"].values

    val_pred_rev = np.zeros(len(val_dates))
    val_pred_cogs = np.zeros(len(val_dates))
    # Val offsets are negative (val is before TRAIN_END). Use horizon = TRAIN_END - date.
    val_offsets_from_end = (train_end_ts - val_dates).dt.days.values
    for i, off in enumerate(val_offsets_from_end):
        bucket = pick_bucket(max(off, 1))
        val_pred_rev[i] = val_preds_rev_by_bucket[bucket][i]
        val_pred_cogs[i] = val_preds_cogs_by_bucket[bucket][i]
    val_pred_rev = np.clip(val_pred_rev, 0, None)
    val_pred_cogs = np.clip(val_pred_cogs, 0, None)

    print("\n── Validation (stitched across buckets) ──")
    res_rev = evaluate(val_rev_true, val_pred_rev, "Revenue")
    res_cogs = evaluate(val_cogs_true, val_pred_cogs, "COGS")

    print("\n  Per-bucket val MAE (diagnostic):")
    for name in HORIZON_BUCKETS:
        mae = float(np.mean(np.abs(val_rev_true - np.clip(val_preds_rev_by_bucket[name], 0, None))))
        print(f"    {name}: {mae:,.0f}")

    save_val_predictions(val_dates, val_pred_rev, val_pred_cogs, "ex_09_lgbm_direct")

    # ── Test stitching: pick bucket per test date ──
    test_dates = combined.loc[test_mask, "Date"].reset_index(drop=True)
    test_offsets = (test_dates - train_end_ts).dt.days.values
    test_rev = np.zeros(len(test_dates))
    test_cogs = np.zeros(len(test_dates))
    for i, off in enumerate(test_offsets):
        bucket = pick_bucket(off)
        test_rev[i] = test_preds_rev_by_bucket[bucket][i]
        test_cogs[i] = test_preds_cogs_by_bucket[bucket][i]
    test_rev = np.clip(test_rev, 0, None)
    test_cogs = np.clip(test_cogs, 0, None)

    make_submission(test_dates, test_rev, test_cogs,
                    SUBMISSION_DIR / "ex_09_lgbm_direct.csv")

    with open(MODEL_DIR / "ex_09_models.pkl", "wb") as f:
        pickle.dump(models, f)

    tracker.log_final(res_rev)
    tracker.add_note(
        f"COGS — MAE={res_cogs['mae']:,.0f} RMSE={res_cogs['rmse']:,.0f} "
        f"R²={res_cogs['r2']:.4f}"
    )
    tracker.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return res_rev


if __name__ == "__main__":
    main()
