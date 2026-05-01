"""
EX_48: Stateless Hybrid Architecture
- Trend Model: Ridge Regression (can extrapolate trend linearly)
- Seasonality Model: LightGBM on residuals (Calendar, Promo, Profiles)
- NO LAGS -> NO RECURSION -> NO SNOWBALL EFFECT
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from modeling.config import LGBM_PARAMS, SEED
from modeling.feature_engineering import (
    build_feature_table,
    get_feature_cols,
)
from modeling.utils import evaluate, load_sales, make_submission

warnings.filterwarnings("ignore")

TRACK = Path("output/tracking/ex_48_stateless_hybrid")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]


def build_stateless_features(df, profile_df=None):
    """Build features and add time index."""
    feat_df, profiles = build_feature_table(
        df, verbose=False, profile_source_df=profile_df
    )

    # Add time index for trend model
    feat_df["time_index"] = (feat_df["Date"] - pd.Timestamp("2013-01-01")).dt.days

    return feat_df, profiles


def _get_stateless_cols(base_cols, target):
    """Filter out ALL recursive features (lags, rolling, growth)."""
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    cols = [c for c in base_cols if not c.startswith(blocked)]
    # Strict filter: remove any feature that depends on past target values
    cols = [
        c
        for c in cols
        if "lag" not in c
        and "rmean" not in c
        and "rstd" not in c
        and "ratio" not in c
        and "growth" not in c
        and c not in ["Revenue", "COGS"]
    ]
    return cols


def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def evaluate_fold(sales: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, _ = build_stateless_features(sales, profile_df=train_slice)

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _get_stateless_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_stateless_cols(base_cols, "COGS"))

    # 1. Trend Model (Ridge)
    trend_features = ["time_index"]

    # Use only recent data for trend to avoid 10-year regime shifts
    # For CV, use the last 365 days of training if recent data is empty
    recent_trn = trn[trn["Date"] >= "2021-01-01"].copy()
    if len(recent_trn) < 180:
        recent_trn = trn.tail(365).copy()

    scaler_rev = StandardScaler()
    x_trn_trend_rev_full = scaler_rev.fit_transform(trn[trend_features])
    x_trn_trend_rev_recent = scaler_rev.transform(recent_trn[trend_features])
    x_val_trend_rev = scaler_rev.transform(val[trend_features])

    trend_model_rev = Ridge(alpha=10.0)
    trend_model_rev.fit(x_trn_trend_rev_recent, recent_trn["Revenue"])

    scaler_cogs = StandardScaler()
    x_trn_trend_cogs_full = scaler_cogs.fit_transform(trn[trend_features])
    x_trn_trend_cogs_recent = scaler_cogs.transform(recent_trn[trend_features])
    x_val_trend_cogs = scaler_cogs.transform(val[trend_features])

    trend_model_cogs = Ridge(alpha=10.0)
    trend_model_cogs.fit(x_trn_trend_cogs_recent, recent_trn["COGS"])

    # Get trend predictions for FULL training set to compute residuals
    trn_trend_pred_rev = trend_model_rev.predict(x_trn_trend_rev_full)
    val_trend_pred_rev = trend_model_rev.predict(x_val_trend_rev)

    trn_trend_pred_cogs = trend_model_cogs.predict(x_trn_trend_cogs_full)
    val_trend_pred_cogs = trend_model_cogs.predict(x_val_trend_cogs)

    # 2. Compute Residuals
    trn_resid_rev = trn["Revenue"] - trn_trend_pred_rev
    trn_resid_cogs = trn["COGS"] - trn_trend_pred_cogs

    # 3. Seasonality Model (LGBM)
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500
    params["random_state"] = SEED

    lgb_rev = lgb.LGBMRegressor(**params)
    lgb_rev.fit(
        trn[cols_rev].fillna(0),
        trn_resid_rev,
        eval_set=[(val[cols_rev].fillna(0), val["Revenue"] - val_trend_pred_rev)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    lgb_cogs = lgb.LGBMRegressor(**params)
    lgb_cogs.fit(
        trn[cols_cogs].fillna(0),
        trn_resid_cogs,
        eval_set=[(val[cols_cogs].fillna(0), val["COGS"] - val_trend_pred_cogs)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    # 4. Final Predictions (Direct - NO RECURSION)
    val_resid_pred_rev = lgb_rev.predict(val[cols_rev].fillna(0))
    val_resid_pred_cogs = lgb_cogs.predict(val[cols_cogs].fillna(0))

    pred_rev = np.maximum(0, val_trend_pred_rev + val_resid_pred_rev)
    pred_cogs = np.maximum(0, val_trend_pred_cogs + val_resid_pred_cogs)

    res_rev = evaluate(val["Revenue"].values, pred_rev, f"{fold['name']} Rev")
    res_cogs = evaluate(val["COGS"].values, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    print(f"  Rev Trend  Mean: {val_trend_pred_rev.mean():,.0f}")
    print(f"  Rev Resid  Mean: {val_resid_pred_rev.mean():,.0f}")
    print(f"  Rev Final  Mean: {pred_rev.mean():,.0f}")
    print(f"  Rev Actual Mean: {val['Revenue'].mean():,.0f}")

    return pd.DataFrame(
        [
            {
                "fold": fold["name"],
                "revenue_mae": float(res_rev["mae"]),
                "cogs_mae": float(res_cogs["mae"]),
                "revenue_pred_mean": float(pred_rev.mean()),
                "score": score,
            }
        ]
    )


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_48: Stateless Hybrid (Ridge Trend + LGBM Residuals)")
    print("=" * 78)

    fold_results = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ===")
        s_df = evaluate_fold(train, fold)
        fold_results.append(s_df)

    fold_scores = pd.concat(fold_results, ignore_index=True)

    print("\nFold Scores:")
    print(fold_scores.to_string(index=False))
    print(f"\nMean score: {fold_scores['score'].mean():,.0f}")

    # Full retrain & submission
    print("\nTraining on full data for submission...")

    # For full submission, we need to build features for train+test together
    # to ensure all dates are processed statelessly
    test_dummy = test.copy()
    test_dummy["Revenue"] = np.nan
    test_dummy["COGS"] = np.nan
    full_df = pd.concat([train, test_dummy], ignore_index=True).sort_values("Date")

    feat_df, profiles = build_stateless_features(full_df, profile_df=train)

    trn = feat_df[feat_df["Date"].isin(train["Date"])].copy()
    tst = feat_df[feat_df["Date"].isin(test["Date"])].copy()

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _get_stateless_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_stateless_cols(base_cols, "COGS"))

    print(f"Features used: {len(cols_rev)} (ALL LAGS REMOVED)")

    # 1. Trend Model (Recent Data Only)
    trend_features = ["time_index"]
    recent_trn = trn[trn["Date"] >= "2021-01-01"].copy()

    scaler_rev = StandardScaler()
    x_trn_trend_rev_full = scaler_rev.fit_transform(trn[trend_features])
    x_trn_trend_rev_recent = scaler_rev.transform(recent_trn[trend_features])
    x_tst_trend_rev = scaler_rev.transform(tst[trend_features])
    trend_model_rev = Ridge(alpha=10.0).fit(x_trn_trend_rev_recent, recent_trn["Revenue"])

    scaler_cogs = StandardScaler()
    x_trn_trend_cogs_full = scaler_cogs.fit_transform(trn[trend_features])
    x_trn_trend_cogs_recent = scaler_cogs.transform(recent_trn[trend_features])
    x_tst_trend_cogs = scaler_cogs.transform(tst[trend_features])
    trend_model_cogs = Ridge(alpha=10.0).fit(x_trn_trend_cogs_recent, recent_trn["COGS"])

    # 2. Seasonality Model
    trn_trend_pred_rev = trend_model_rev.predict(x_trn_trend_rev_full)
    trn_resid_rev = trn["Revenue"] - trn_trend_pred_rev

    trn_trend_pred_cogs = trend_model_cogs.predict(x_trn_trend_cogs_full)
    trn_resid_cogs = trn["COGS"] - trn_trend_pred_cogs

    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500

    lgb_rev = lgb.LGBMRegressor(**params)
    lgb_rev.fit(trn[cols_rev].fillna(0), trn_resid_rev)

    lgb_cogs = lgb.LGBMRegressor(**params)
    lgb_cogs.fit(trn[cols_cogs].fillna(0), trn_resid_cogs)

    # 3. Direct Prediction on Test Set (NO RECURSION)
    tst_trend_pred_rev = trend_model_rev.predict(x_tst_trend_rev)
    tst_resid_pred_rev = lgb_rev.predict(tst[cols_rev].fillna(0))
    final_rev = np.maximum(0, tst_trend_pred_rev + tst_resid_pred_rev)

    tst_trend_pred_cogs = trend_model_cogs.predict(x_tst_trend_cogs)
    tst_resid_pred_cogs = lgb_cogs.predict(tst[cols_cogs].fillna(0))
    final_cogs = np.maximum(0, tst_trend_pred_cogs + tst_resid_pred_cogs)

    print(f"\nRev Trend  Mean: {tst_trend_pred_rev.mean():,.0f}")
    print(f"Rev Resid  Mean: {tst_resid_pred_rev.mean():,.0f}")
    print(f"Rev Final  Mean: {final_rev.mean():,.0f}")

    path = SUB_DIR / "ex_48_stateless_hybrid.csv"
    make_submission(test["Date"], final_rev, final_cogs, path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "mean_cv_score": float(fold_scores["score"].mean()),
        "mean_rev_pred": float(final_rev.mean()),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
