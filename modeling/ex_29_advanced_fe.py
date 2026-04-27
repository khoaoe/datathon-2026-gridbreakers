"""
EX_29: Advanced Feature Engineering Pipeline v3 (Vietnamese E-Commerce Context)

Purpose:
- Trains LightGBM with advanced Vietnamese calendar features (Tet curves, Payday tracking, Double Dates).
- Fresh start: NO anchors, NO bridges, NO blending. Pure feature-driven modeling.

v3 Improvements:
- COGS growth features + full COGS rolling windows
- Multi-seed ensemble (3 seeds) for variance reduction
- Stabilized hyperparameters (lower complexity for drift resistance)
- Recursive drift damping (mean-reversion to monthly profiles)
- Revenue/COGS ratio post-processing
- NEW: Revenue-COGS spread/margin features (margin dynamics)
- NEW: Weekend×month profiles, margin profiles, category mix profiles
- NEW: Order-count profiles by (month, dow)
- NEW: Dual-recursive prediction (both targets updated together for spread features)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from modeling.config import LGBM_PARAMS, SEED
from modeling.feature_engineering import (
    apply_profiles_to_dates,
    build_calendar_features,
    build_feature_table,
    build_growth_features,
    build_lag_features,
    build_rolling_features,
    get_feature_cols,
)
from modeling.utils import evaluate, load_sales, make_submission

TRACK = Path("output/tracking/ex_29_advanced_fe")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

# Stabilized params: lower complexity to resist recursive drift
STABLE_LGBM = LGBM_PARAMS.copy()
STABLE_LGBM.update({
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 7,
    "num_leaves": 40,
    "min_child_samples": 40,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
})

N_SEEDS = 3  # Multi-seed ensemble


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED):
    params = STABLE_LGBM.copy()
    params["random_state"] = seed

    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_trn,
        y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols: list[str], target: str) -> list[str]:
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def _compute_monthly_bounds(train_df: pd.DataFrame) -> dict:
    """Compute mean ± 2.5*std for each month for drift damping."""
    train_df = train_df.copy()
    train_df["month"] = train_df["Date"].dt.month
    bounds = {}
    for target in ["Revenue", "COGS"]:
        monthly = train_df.groupby("month")[target].agg(["mean", "std"])
        target_bounds = {}
        for m in range(1, 13):
            if m in monthly.index:
                mn = monthly.loc[m, "mean"]
                sd = monthly.loc[m, "std"]
                target_bounds[m] = {
                    "mean": mn,
                    "lower": max(0, mn - 2.5 * sd),
                    "upper": mn + 2.5 * sd,
                }
        bounds[target] = target_bounds
    return bounds


def _build_spread_features(combined: pd.DataFrame) -> pd.DataFrame:
    """Build Revenue-COGS spread/margin features from dual history."""
    if "Revenue" in combined.columns and "COGS" in combined.columns:
        spread = combined["Revenue"].shift(1) - combined["COGS"].shift(1)
        combined["spread_lag1"] = spread
        combined["spread_rmean_7"] = spread.rolling(7, min_periods=1).mean()
        combined["spread_rmean_28"] = spread.rolling(28, min_periods=1).mean()
        ratio = combined["Revenue"].shift(1) / combined["COGS"].shift(1).replace(0, np.nan)
        combined["margin_ratio_lag1"] = ratio
        combined["margin_ratio_rmean_7"] = ratio.rolling(7, min_periods=1).mean()
        combined["margin_ratio_rmean_28"] = ratio.rolling(28, min_periods=1).mean()
    return combined


def dual_recursive_predict(
    models_rev: list,
    models_cogs: list,
    history_df: pd.DataFrame,
    predict_dates,
    cols_rev: list[str],
    cols_cogs: list[str],
    profiles,
    monthly_bounds: dict | None = None,
    damping_alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-recursive prediction: predict Revenue and COGS together so spread features stay consistent.
    
    This ensures that margin_ratio_lag1, spread_lag1, etc. use BOTH targets' predictions
    rather than only seeing one target's history.
    """
    # Keep full dual-target history
    history = history_df[["Date", "Revenue", "COGS"]].copy()
    preds_rev: list[float] = []
    preds_cogs: list[float] = []

    rev_bounds = monthly_bounds.get("Revenue") if monthly_bounds else None
    cogs_bounds = monthly_bounds.get("COGS") if monthly_bounds else None

    for date in predict_dates:
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], "Revenue": [np.nan], "COGS": [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        # Build calendar features (shared)
        combined = build_calendar_features(combined)
        
        # Build lag/rolling/growth for BOTH targets
        combined = build_lag_features(combined, "Revenue")
        combined = build_rolling_features(combined, "Revenue")
        combined = build_growth_features(combined, "Revenue")
        
        combined = build_lag_features(combined, "COGS")
        combined = build_rolling_features(combined, "COGS", windows=[7, 14, 28, 60, 90, 180, 365])
        combined = build_growth_features(combined, "COGS")
        
        # Build spread/margin features (need both Revenue and COGS history)
        combined = _build_spread_features(combined)

        # Apply profiles to last row
        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)

        # --- Predict Revenue ---
        x_rev = pd.DataFrame(0.0, index=[0], columns=cols_rev)
        for c in cols_rev:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_rev[c] = 0.0 if pd.isna(val) else val
        
        raw_rev = [float(m.predict(x_rev)[0]) for m in models_rev]
        pred_rev = float(np.mean(raw_rev))
        pred_rev = max(0, pred_rev)
        
        # Drift damping for Revenue
        if rev_bounds is not None:
            month = ts.month
            if month in rev_bounds:
                mb = rev_bounds[month]
                if pred_rev > mb["upper"]:
                    pred_rev = mb["upper"] + damping_alpha * (pred_rev - mb["upper"])
                elif pred_rev < mb["lower"]:
                    pred_rev = mb["lower"] - damping_alpha * (mb["lower"] - pred_rev)

        # --- Predict COGS ---
        x_cogs = pd.DataFrame(0.0, index=[0], columns=cols_cogs)
        for c in cols_cogs:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_cogs[c] = 0.0 if pd.isna(val) else val
        
        raw_cogs = [float(m.predict(x_cogs)[0]) for m in models_cogs]
        pred_cogs = float(np.mean(raw_cogs))
        pred_cogs = max(0, pred_cogs)
        
        # Drift damping for COGS
        if cogs_bounds is not None:
            month = ts.month
            if month in cogs_bounds:
                mb = cogs_bounds[month]
                if pred_cogs > mb["upper"]:
                    pred_cogs = mb["upper"] + damping_alpha * (pred_cogs - mb["upper"])
                elif pred_cogs < mb["lower"]:
                    pred_cogs = mb["lower"] - damping_alpha * (mb["lower"] - pred_cogs)

        preds_rev.append(pred_rev)
        preds_cogs.append(pred_cogs)

        # Update history with BOTH predictions (so spread features work next step)
        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], "Revenue": [pred_rev], "COGS": [pred_cogs]})],
            ignore_index=True,
        )

    return np.array(preds_rev), np.array(preds_cogs)


def _postprocess_ratio(revenue: np.ndarray, cogs: np.ndarray, 
                       min_ratio: float = 0.90, max_ratio: float = 1.40) -> np.ndarray:
    """Enforce Revenue/COGS ratio within historical bounds."""
    corrected_cogs = cogs.copy()
    for i in range(len(revenue)):
        if revenue[i] > 0 and corrected_cogs[i] > 0:
            ratio = revenue[i] / corrected_cogs[i]
            if ratio < min_ratio:
                corrected_cogs[i] = revenue[i] / min_ratio
            elif ratio > max_ratio:
                corrected_cogs[i] = revenue[i] / max_ratio
    return corrected_cogs


def evaluate_fold(sales: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(
        sales, verbose=False, profile_source_df=train_slice
    )

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    y_val_rev = val["Revenue"].values
    y_val_cogs = val["COGS"].values

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_target_cols(base_cols, "COGS"))

    # Monthly bounds for drift damping
    monthly_bounds = _compute_monthly_bounds(train_slice)

    # Multi-seed ensemble
    models_rev = []
    models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), val["Revenue"], seed=seed
        ))
        models_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), val["COGS"], seed=seed
        ))

    # Dual-recursive prediction (both targets together for spread features)
    pred_rev, pred_cogs = dual_recursive_predict(
        models_rev,
        models_cogs,
        trn[["Date", "Revenue", "COGS"]],
        val["Date"].values,
        cols_rev,
        cols_cogs,
        profiles,
        monthly_bounds,
        damping_alpha=0.3,
    )

    # Post-process: enforce Revenue/COGS ratio
    pred_cogs = _postprocess_ratio(pred_rev, pred_cogs, min_ratio=0.85, max_ratio=1.40)

    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    return pd.DataFrame(
        [
            {
                "fold": fold["name"],
                "n_features_rev": len(cols_rev),
                "n_features_cogs": len(cols_cogs),
                "revenue_mae": float(res_rev["mae"]),
                "cogs_mae": float(res_cogs["mae"]),
                "score": score,
            }
        ]
    )


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_29: Advanced Feature Engineering (v3 - deep FE + dual-recursive)")
    print("=" * 78)

    fold_score_parts = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df = evaluate_fold(train, fold)
        fold_score_parts.append(s_df)

    fold_scores = pd.concat(fold_score_parts, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\nFold Scores:")
    print(fold_scores)
    print(f"\nMean score: {fold_scores['score'].mean():,.0f}")

    print("\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(
        train, verbose=True, profile_source_df=train
    )
    base_cols = get_feature_cols(feat_df)

    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")

    # Monthly bounds from full training data
    monthly_bounds = _compute_monthly_bounds(train)

    # Multi-seed ensemble for final models
    final_models_rev = []
    final_models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_models_rev.append(_fit_lgbm(
            feat_df[cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            seed=seed,
        ))
        final_models_cogs.append(_fit_lgbm(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed,
        ))

    print("Running dual-recursive inference on test set...")
    final_rev, final_cogs = dual_recursive_predict(
        final_models_rev,
        final_models_cogs,
        train[["Date", "Revenue", "COGS"]],
        test["Date"].values,
        cols_rev,
        cols_cogs,
        profiles,
        monthly_bounds,
        damping_alpha=0.3,
    )

    # Post-process: enforce Revenue/COGS ratio
    final_cogs = _postprocess_ratio(final_rev, final_cogs, min_ratio=0.85, max_ratio=1.40)

    # Diagnostics
    ratios = final_rev / np.clip(final_cogs, 1, None)
    print(f"\nRev/COGS ratio: mean={ratios.mean():.3f}  min={ratios.min():.3f}  max={ratios.max():.3f}")
    print(f"Revenue: mean={final_rev.mean():,.0f}  std={final_rev.std():,.0f}")
    print(f"COGS: mean={final_cogs.mean():,.0f}  std={final_cogs.std():,.0f}")

    candidate_path = SUB_DIR / "ex_29_advanced_fe.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
        "n_seeds": N_SEEDS,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
