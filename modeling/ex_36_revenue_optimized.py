"""
EX_36: Revenue-Optimized Pipeline

Key learnings from Kaggle submissions:
- EX-31 (raw 3-seed LGB, no post-processing) scored 845,163 = BEST
- Drift correction HURTS because the 2023-2024 values are HIGHER than training data
  (the model's upward extrapolation was correct, not "drift")
- sample_submission.csv values are NOT ground truth
  
Strategy:
1. Optimize for MAE directly (mae objective in LightGBM)
2. Try MAE + Huber loss for robustness
3. Multi-seed ensemble (3 seeds) — same as EX-31
4. NO drift correction
5. Try mild upward scaling to account for continued growth
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

TRACK = Path("output/tracking/ex_36_revenue_optimized")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED, objective="regression"):
    """LightGBM with configurable objective."""
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 2000
    params["random_state"] = seed
    params["objective"] = objective
    
    # For MAE objective, use mae metric
    if objective == "mae":
        params["metric"] = "mae"
    elif objective == "huber":
        params["metric"] = "huber"
        params["alpha"] = 0.9  # huber delta parameter
    
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


def recursive_predict(
    models: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
) -> np.ndarray:
    history = history_df[["Date", target]].copy()
    preds: list[float] = []

    for date in predict_dates:
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
                x_pred[c] = 0.0 if pd.isna(val) else val

        raw_preds = [float(m.predict(x_pred)[0]) for m in models]
        pred = float(np.mean(raw_preds))
        pred = max(0, pred)
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


def evaluate_fold(sales: pd.DataFrame, fold: dict, objective: str = "regression"):
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

    # Multi-seed ensemble
    models_rev = []
    models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), val["Revenue"], seed=seed,
            objective=objective,
        ))
        models_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), val["COGS"], seed=seed,
            objective=objective,
        ))

    pred_rev = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    pred_cogs = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS")
    
    # Score: focus on Revenue MAE primarily
    score_rev_only = float(res_rev["mae"])
    score_combined = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    return pd.DataFrame([{
        "fold": fold["name"],
        "objective": objective,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "revenue_mae": float(res_rev["mae"]),
        "revenue_rmse": float(res_rev["rmse"]),
        "cogs_mae": float(res_cogs["mae"]),
        "cogs_rmse": float(res_cogs["rmse"]),
        "score_rev_only": score_rev_only,
        "score_combined": score_combined,
    }])


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_36: Revenue-Optimized (no drift correction)")
    print("=" * 78)

    # Test different objectives
    objectives = ["regression", "mae", "huber"]
    all_results = []

    for obj in objectives:
        print(f"\n{'='*40}")
        print(f"Objective: {obj}")
        print(f"{'='*40}")
        
        fold_parts = []
        for fold in FOLDS:
            print(f"\n--- {fold['name']} ---")
            s_df = evaluate_fold(train, fold, objective=obj)
            fold_parts.append(s_df)
        
        fold_df = pd.concat(fold_parts, ignore_index=True)
        all_results.append(fold_df)
        
        mean_rev_mae = fold_df["revenue_mae"].mean()
        mean_cogs_mae = fold_df["cogs_mae"].mean()
        mean_combined = fold_df["score_combined"].mean()
        print(f"\n  {obj}: Mean Rev MAE={mean_rev_mae:,.0f}  COGS MAE={mean_cogs_mae:,.0f}  Combined={mean_combined:,.0f}")

    # Combine all results
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(TRACK / "fold_scores.csv", index=False)

    # Find best objective
    obj_means = all_df.groupby("objective")["score_combined"].mean()
    best_obj = obj_means.idxmin()
    print(f"\n\nBest objective: {best_obj} (combined score = {obj_means[best_obj]:,.0f})")
    
    print(f"\nObjective comparison:")
    for obj in objectives:
        mask = all_df["objective"] == obj
        rev_mae = all_df.loc[mask, "revenue_mae"].mean()
        cogs_mae = all_df.loc[mask, "cogs_mae"].mean()
        combined = all_df.loc[mask, "score_combined"].mean()
        print(f"  {obj:15s}: Rev MAE={rev_mae:>10,.0f}  COGS MAE={cogs_mae:>10,.0f}  Combined={combined:>10,.0f}")

    # Train final model with best objective
    print(f"\nTraining final model with objective='{best_obj}'...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)

    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")

    final_models_rev = []
    final_models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_models_rev.append(_fit_lgbm(
            feat_df[cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            seed=seed, objective=best_obj,
        ))
        final_models_cogs.append(_fit_lgbm(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed, objective=best_obj,
        ))

    print("Running recursive inference on test set...")
    final_rev = recursive_predict(
        final_models_rev, train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    final_cogs = recursive_predict(
        final_models_cogs, train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # Diagnostics
    ratios = final_rev / np.clip(final_cogs, 1, None)
    print(f"\nRevenue: mean={final_rev.mean():,.0f}  std={final_rev.std():,.0f}")
    print(f"COGS: mean={final_cogs.mean():,.0f}  std={final_cogs.std():,.0f}")
    print(f"Rev/COGS ratio: mean={ratios.mean():.3f}  min={ratios.min():.3f}  max={ratios.max():.3f}")

    candidate_path = SUB_DIR / "ex_36_revenue_optimized.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
        "n_seeds": N_SEEDS,
        "best_objective": best_obj,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "objectives_tested": objectives,
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
