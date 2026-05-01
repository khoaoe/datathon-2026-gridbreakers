"""
EX_03: LightGBM with leakage-safe v3 Feature Engineering
- Calendar + Fourier + historical profiles
- Target lags + rolling stats (recursive prediction)
- Promotion calendar + promo interaction features
- SHAP explainability
"""

import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from modeling.config import LGBM_PARAMS, MODEL_DIR, SUBMISSION_DIR, VAL_START
from modeling.utils import evaluate, load_sales, make_submission
from modeling.feature_engineering import (
    build_feature_table,
    build_calendar_features,
    build_lag_features,
    build_rolling_features,
    build_growth_features,
    apply_profiles_to_dates,
    get_feature_cols,
)
from modeling.tracker import ExperimentTracker, LGBMCallback


def recursive_predict(
    model, train_df, test_dates, feature_cols, profiles, target="Revenue"
):
    """
    Predict one day at a time, updating lags with previous predictions.
    Profile features are applied from precomputed historical patterns.
    """
    history = train_df[["Date", target]].copy()
    predictions = []

    for i, date in enumerate(test_dates):
        row = pd.DataFrame({"Date": [pd.Timestamp(date)], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        # Build features available at prediction time
        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)

        # Get last row and apply historical profiles
        last_row = combined.iloc[-1:].copy()
        last_row = apply_profiles_to_dates(last_row, profiles)

        # Build prediction frame with ALL expected feature columns
        X_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for col in feature_cols:
            if col in last_row.columns:
                val = last_row[col].values[0]
                X_pred[col] = 0.0 if pd.isna(val) else val

        pred = model.predict(X_pred)[0]
        predictions.append(pred)

        # Update history
        history = pd.concat(
            [history, pd.DataFrame({"Date": [pd.Timestamp(date)], target: [pred]})],
            ignore_index=True,
        )

        if (i + 1) % 100 == 0:
            print(f"  Predicted {i + 1}/{len(test_dates)} days...")

    return np.array(predictions)


def main():
    start = time.time()

    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Run: pip install lightgbm")
        return None

    train, test = load_sales()
    tracker = ExperimentTracker("ex_03_lgbm")

    print("=" * 60)
    print("EX_03: LightGBM + leakage-safe v3 FEATURES")
    print("=" * 60)

    # ── Build feature table ──────────────────────────────────────────
    print("\n[1/5] Building features...")
    profile_source = train[train["Date"] < pd.Timestamp(VAL_START)].copy()
    feat_df, profiles = build_feature_table(
        train,
        verbose=True,
        profile_source_df=profile_source,
    )

    # ── Train/Val split ──────────────────────────────────────────────
    print("\n[2/5] Splitting train/val...")
    val_mask = feat_df["Date"] >= pd.Timestamp(VAL_START)
    trn = feat_df[~val_mask].copy()
    val = feat_df[val_mask].copy()

    feature_cols = get_feature_cols(feat_df)
    feature_cols = [c for c in feature_cols if trn[c].notna().mean() > 0.5]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(trn[c])]
    feature_cols = [c for c in feature_cols if trn[c].nunique(dropna=True) > 1]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(trn)} rows, Val: {len(val)} rows")

    X_trn, y_trn = trn[feature_cols].fillna(0), trn["Revenue"]
    X_val, y_val = val[feature_cols].fillna(0), val["Revenue"]

    # ── Train Revenue model ──────────────────────────────────────────
    print("\n[3/5] Training Revenue model...")
    params = LGBM_PARAMS.copy()
    tracker.log_params(params)
    tracker.log_params(
        {"n_features": len(feature_cols), "n_train": len(trn), "n_val": len(val)}
    )

    lgbm_cb = LGBMCallback(tracker, log_every=50)
    model_rev = lgb.LGBMRegressor(**params)
    model_rev.fit(
        X_trn,
        y_trn,
        eval_set=[(X_trn, y_trn), (X_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(200),
            lgbm_cb,
        ],
    )

    val_pred_rev = np.clip(model_rev.predict(X_val), 0, None)
    res_rev = evaluate(y_val, val_pred_rev, "Revenue")

    # ── Train COGS model ─────────────────────────────────────────────
    print("\n[4/5] Training COGS model...")
    X_trn_cogs, y_trn_cogs = trn[feature_cols].fillna(0), trn["COGS"]
    X_val_cogs, y_val_cogs = val[feature_cols].fillna(0), val["COGS"]

    tracker_cogs = ExperimentTracker("ex_03_lgbm_cogs")
    lgbm_cb_cogs = LGBMCallback(tracker_cogs, log_every=50)
    model_cogs = lgb.LGBMRegressor(**params)
    model_cogs.fit(
        X_trn_cogs,
        y_trn_cogs,
        eval_set=[(X_trn_cogs, y_trn_cogs), (X_val_cogs, y_val_cogs)],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(200),
            lgbm_cb_cogs,
        ],
    )

    val_pred_cogs = np.clip(model_cogs.predict(X_val_cogs), 0, None)
    res_cogs = evaluate(y_val_cogs, val_pred_cogs, "COGS")

    # ── Save models ──────────────────────────────────────────────────
    with open(MODEL_DIR / "ex_03_lgbm_rev.pkl", "wb") as f:
        pickle.dump(model_rev, f)
    with open(MODEL_DIR / "ex_03_lgbm_cogs.pkl", "wb") as f:
        pickle.dump(model_cogs, f)
    with open(MODEL_DIR / "ex_03_features.pkl", "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "profiles": profiles}, f)
    print("Models saved.")

    # ── SHAP explainability ──────────────────────────────────────────
    print("\n[5/5] Computing SHAP values...")
    try:
        import shap

        explainer = shap.TreeExplainer(model_rev)
        shap_values = explainer.shap_values(X_val.head(500))
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "shap_mean_abs": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("shap_mean_abs", ascending=False)
        importance.to_csv(MODEL_DIR / "ex_03_shap_importance.csv", index=False)
        print("Top 10 features by SHAP:")
        print(importance.head(10).to_string(index=False))
    except ImportError:
        print("SHAP not installed, using LightGBM feature importance.")
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model_rev.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        print("Top 10 features by gain:")
        print(importance.head(10).to_string(index=False))

    # ── Submission: retrain on full data + recursive predict ─────────
    print("\nRetraining on full data...")
    full_df, full_profiles = build_feature_table(
        train,
        verbose=False,
        profile_source_df=train,
    )
    full_feature_cols = [c for c in feature_cols if c in full_df.columns]
    X_full = full_df[full_feature_cols].fillna(0)

    model_rev_full = lgb.LGBMRegressor(**params)
    model_rev_full.fit(X_full, full_df["Revenue"])

    model_cogs_full = lgb.LGBMRegressor(**params)
    model_cogs_full.fit(X_full, full_df["COGS"])

    print("Generating submission (recursive prediction)...")
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
        test["Date"], rev_preds, cogs_preds, SUBMISSION_DIR / "ex_03_lgbm.csv"
    )

    # ── Save tracking ────────────────────────────────────────────────
    tracker.log_final(res_rev)
    tracker.add_note(
        f"COGS — MAE={res_cogs['mae']:,.0f} RMSE={res_cogs['rmse']:,.0f} R²={res_cogs['r2']:.4f}"
    )
    tracker.save()
    tracker_cogs.log_final(res_cogs)
    tracker_cogs.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return res_rev


if __name__ == "__main__":
    main()
