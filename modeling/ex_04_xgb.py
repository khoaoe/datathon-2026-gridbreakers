"""
EX_04: XGBoost with v3 Feature Engineering (GPU accelerated)
- Same v3 features as EX_03 (all available at prediction time)
- XGBoost with GPU (tree_method=hist, device=cuda)
"""

import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from modeling.config import XGB_PARAMS, MODEL_DIR, SUBMISSION_DIR, VAL_START, SEED
from modeling.utils import evaluate, load_sales, make_submission
from modeling.feature_engineering import (
    build_feature_table,
    get_feature_cols,
)
from modeling.ex_03_lgbm import recursive_predict
from modeling.tracker import ExperimentTracker


def main():
    start = time.time()

    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed. Run: pip install xgboost")
        return None

    train, test = load_sales()
    tracker = ExperimentTracker("ex_04_xgb")

    print("=" * 60)
    print("EX_04: XGBoost + GPU + v3 FEATURES")
    print("=" * 60)

    # ── Build feature table ──────────────────────────────────────────
    print("\n[1/4] Building features...")
    profile_source = train[train["Date"] < pd.Timestamp(VAL_START)].copy()
    feat_df, profiles = build_feature_table(
        train,
        verbose=True,
        profile_source_df=profile_source,
    )

    # ── Train/Val split ──────────────────────────────────────────────
    print("\n[2/4] Splitting train/val...")
    val_mask = feat_df["Date"] >= VAL_START
    trn = feat_df[~val_mask].copy()
    val = feat_df[val_mask].copy()

    feature_cols = get_feature_cols(feat_df)
    feature_cols = [c for c in feature_cols if trn[c].notna().mean() > 0.5]

    print(f"  Features: {len(feature_cols)}")

    X_trn, y_trn = trn[feature_cols].fillna(0), trn["Revenue"]
    X_val, y_val = val[feature_cols].fillna(0), val["Revenue"]

    # ── Check GPU availability ───────────────────────────────────────
    params = XGB_PARAMS.copy()
    try:
        test_model = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
        test_model.fit(X_trn.head(10), y_trn.head(10))
        print("  GPU: CUDA enabled")
    except Exception:
        params["device"] = "cpu"
        print("  GPU: Not available, falling back to CPU")

    tracker.log_params(params)
    tracker.log_params({"n_features": len(feature_cols)})

    # ── Train Revenue model ──────────────────────────────────────────
    print("\n[3/4] Training Revenue model...")
    model_rev = xgb.XGBRegressor(**params)
    model_rev.fit(
        X_trn,
        y_trn,
        eval_set=[(X_trn, y_trn), (X_val, y_val)],
        verbose=200,
    )
    # Log evals
    if hasattr(model_rev, "evals_result_"):
        results = model_rev.evals_result()
        for ds_name in results:
            for metric_name in results[ds_name]:
                vals = results[ds_name][metric_name]
                for i in range(0, len(vals), 50):
                    tracker.log_step(i, {f"{ds_name}_{metric_name}": vals[i]})

    val_pred_rev = model_rev.predict(X_val)
    res_rev = evaluate(y_val, val_pred_rev, "Revenue")

    # ── Train COGS model ─────────────────────────────────────────────
    print("\nTraining COGS model...")
    X_trn_cogs, y_trn_cogs = trn[feature_cols].fillna(0), trn["COGS"]
    X_val_cogs, y_val_cogs = val[feature_cols].fillna(0), val["COGS"]

    model_cogs = xgb.XGBRegressor(**params)
    model_cogs.fit(
        X_trn_cogs,
        y_trn_cogs,
        eval_set=[(X_val_cogs, y_val_cogs)],
        verbose=200,
    )

    val_pred_cogs = model_cogs.predict(X_val_cogs)
    res_cogs = evaluate(y_val_cogs, val_pred_cogs, "COGS")

    # ── Save models ──────────────────────────────────────────────────
    with open(MODEL_DIR / "ex_04_xgb_rev.pkl", "wb") as f:
        pickle.dump(model_rev, f)
    with open(MODEL_DIR / "ex_04_xgb_cogs.pkl", "wb") as f:
        pickle.dump(model_cogs, f)
    print("Models saved.")

    # ── Submission ───────────────────────────────────────────────────
    print("\n[4/4] Generating submission...")
    full_df, full_profiles = build_feature_table(
        train,
        verbose=False,
        profile_source_df=train,
    )
    full_feature_cols = [c for c in feature_cols if c in full_df.columns]
    X_full = full_df[full_feature_cols].fillna(0)

    model_rev_full = xgb.XGBRegressor(**params)
    model_rev_full.fit(X_full, full_df["Revenue"])

    model_cogs_full = xgb.XGBRegressor(**params)
    model_cogs_full.fit(X_full, full_df["COGS"])

    rev_preds = recursive_predict(
        model_rev_full,
        train,
        test["Date"].values,
        full_feature_cols,
        full_profiles,
        "Revenue",
    )
    cogs_preds = recursive_predict(
        model_cogs_full,
        train,
        test["Date"].values,
        full_feature_cols,
        full_profiles,
        "COGS",
    )

    make_submission(
        test["Date"], rev_preds, cogs_preds, SUBMISSION_DIR / "ex_04_xgb.csv"
    )

    # ── Save tracking ────────────────────────────────────────────────
    tracker.log_final(res_rev)
    tracker.add_note(
        f"COGS — MAE={res_cogs['mae']:,.0f} RMSE={res_cogs['rmse']:,.0f} R²={res_cogs['r2']:.4f}"
    )
    tracker.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return res_rev


if __name__ == "__main__":
    main()
