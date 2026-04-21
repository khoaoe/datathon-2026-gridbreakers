"""
EX_12: LightGBM with selected FE deltas from strict EX_08 research.

Selected deltas:
- Keep promo interaction family (stable in strict folds)
- Drop unstable auxiliary profile family (avg_* features)
- Enforce target-aligned autoregressive features to match recursive inference
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
    model,
    train_df,
    test_dates,
    feature_cols,
    profiles,
    target="Revenue",
):
    """Predict recursively while using only target-aligned features."""
    history = train_df[["Date", target]].copy()
    predictions = []

    for i, date in enumerate(test_dates):
        row = pd.DataFrame({"Date": [pd.Timestamp(date)], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)

        last_row = combined.iloc[-1:].copy()
        last_row = apply_profiles_to_dates(last_row, profiles)

        X_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for col in feature_cols:
            if col in last_row.columns:
                val = last_row[col].values[0]
                X_pred[col] = 0.0 if pd.isna(val) else val

        pred = model.predict(X_pred)[0]
        predictions.append(pred)

        history = pd.concat(
            [
                history,
                pd.DataFrame({"Date": [pd.Timestamp(date)], target: [pred]}),
            ],
            ignore_index=True,
        )

        if (i + 1) % 100 == 0:
            print(f"  Predicted {i + 1}/{len(test_dates)} days for {target}...")

    return np.array(predictions)


def _drop_opposite_target_autoreg(cols, target):
    """Remove features unavailable at recursive inference for given target."""
    if target == "Revenue":
        blocked_prefixes = ("COGS_",)
    else:
        blocked_prefixes = ("Revenue_",)

    out = []
    for col in cols:
        if col.startswith(blocked_prefixes):
            continue
        out.append(col)
    return out


def _drop_unstable_aux_profiles(cols):
    """Drop unstable auxiliary profile family from strict EX_08 research."""
    return [c for c in cols if not c.startswith("avg_")]


def select_feature_cols(train_df, all_cols, target):
    cols = [c for c in all_cols if c in train_df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]
    cols = [c for c in cols if train_df[c].notna().mean() > 0.5]
    cols = [c for c in cols if train_df[c].nunique(dropna=True) > 1]

    cols = _drop_opposite_target_autoreg(cols, target)
    cols = _drop_unstable_aux_profiles(cols)
    return cols


def main():
    start = time.time()

    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Run: pip install lightgbm")
        return None

    train, test = load_sales()
    tracker = ExperimentTracker("ex_12_lgbm_selected_deltas")

    print("=" * 64)
    print("EX_12: LightGBM + selected FE deltas (strict EX_08)")
    print("=" * 64)

    print("\n[1/4] Building leakage-safe feature table...")
    profile_source = train[train["Date"] < pd.Timestamp(VAL_START)].copy()
    feat_df, profiles = build_feature_table(
        train,
        verbose=True,
        profile_source_df=profile_source,
    )

    print("\n[2/4] Splitting train/val and selecting target-aligned features...")
    val_mask = feat_df["Date"] >= pd.Timestamp(VAL_START)
    trn = feat_df[~val_mask].copy()
    val = feat_df[val_mask].copy()

    base_feature_cols = get_feature_cols(feat_df)
    feature_cols_rev = select_feature_cols(trn, base_feature_cols, target="Revenue")
    feature_cols_cogs = select_feature_cols(trn, base_feature_cols, target="COGS")

    print(f"  Revenue features: {len(feature_cols_rev)}")
    print(f"  COGS features:    {len(feature_cols_cogs)}")
    print(f"  Train: {len(trn)} rows, Val: {len(val)} rows")

    X_trn_rev = trn[feature_cols_rev].fillna(0)
    y_trn_rev = trn["Revenue"]
    X_val_rev = val[feature_cols_rev].fillna(0)
    y_val_rev = val["Revenue"]

    X_trn_cogs = trn[feature_cols_cogs].fillna(0)
    y_trn_cogs = trn["COGS"]
    X_val_cogs = val[feature_cols_cogs].fillna(0)
    y_val_cogs = val["COGS"]

    params = LGBM_PARAMS.copy()
    tracker.log_params(params)
    tracker.log_params(
        {
            "n_features_revenue": len(feature_cols_rev),
            "n_features_cogs": len(feature_cols_cogs),
            "n_train": len(trn),
            "n_val": len(val),
            "selected_deltas": [
                "keep_promo_interactions",
                "drop_unstable_aux_profiles",
                "target_aligned_autoreg",
            ],
        }
    )

    print("\n[3/4] Training Revenue and COGS models...")
    cb_rev = LGBMCallback(tracker, log_every=50)
    model_rev = lgb.LGBMRegressor(**params)
    model_rev.fit(
        X_trn_rev,
        y_trn_rev,
        eval_set=[(X_trn_rev, y_trn_rev), (X_val_rev, y_val_rev)],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(200),
            cb_rev,
        ],
    )
    val_pred_rev = np.clip(model_rev.predict(X_val_rev), 0, None)
    res_rev = evaluate(y_val_rev, val_pred_rev, "Revenue")

    tracker_cogs = ExperimentTracker("ex_12_lgbm_selected_deltas_cogs")
    cb_cogs = LGBMCallback(tracker_cogs, log_every=50)
    model_cogs = lgb.LGBMRegressor(**params)
    model_cogs.fit(
        X_trn_cogs,
        y_trn_cogs,
        eval_set=[(X_trn_cogs, y_trn_cogs), (X_val_cogs, y_val_cogs)],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(200),
            cb_cogs,
        ],
    )
    val_pred_cogs = np.clip(model_cogs.predict(X_val_cogs), 0, None)
    res_cogs = evaluate(y_val_cogs, val_pred_cogs, "COGS")

    with open(MODEL_DIR / "ex_12_lgbm_selected_deltas_rev.pkl", "wb") as f:
        pickle.dump(model_rev, f)
    with open(MODEL_DIR / "ex_12_lgbm_selected_deltas_cogs.pkl", "wb") as f:
        pickle.dump(model_cogs, f)
    with open(MODEL_DIR / "ex_12_lgbm_selected_deltas_features.pkl", "wb") as f:
        pickle.dump(
            {
                "feature_cols_revenue": feature_cols_rev,
                "feature_cols_cogs": feature_cols_cogs,
                "profiles": profiles,
            },
            f,
        )

    print("\n[4/4] Retraining on full data and generating submission...")
    full_df, full_profiles = build_feature_table(
        train,
        verbose=False,
        profile_source_df=train,
    )

    full_cols_rev = [c for c in feature_cols_rev if c in full_df.columns]
    full_cols_cogs = [c for c in feature_cols_cogs if c in full_df.columns]

    model_rev_full = lgb.LGBMRegressor(**params)
    model_rev_full.fit(full_df[full_cols_rev].fillna(0), full_df["Revenue"])

    model_cogs_full = lgb.LGBMRegressor(**params)
    model_cogs_full.fit(full_df[full_cols_cogs].fillna(0), full_df["COGS"])

    rev_preds = recursive_predict(
        model_rev_full,
        train,
        test["Date"].values,
        full_cols_rev,
        full_profiles,
        target="Revenue",
    )
    cogs_preds = recursive_predict(
        model_cogs_full,
        train,
        test["Date"].values,
        full_cols_cogs,
        full_profiles,
        target="COGS",
    )

    out_path = SUBMISSION_DIR / "ex_12_lgbm_selected_deltas.csv"
    make_submission(test["Date"], rev_preds, cogs_preds, out_path)

    tracker.log_final(res_rev)
    tracker.add_note(
        f"COGS — MAE={res_cogs['mae']:,.0f} RMSE={res_cogs['rmse']:,.0f} R²={res_cogs['r2']:.4f}"
    )
    tracker.add_note(
        "Selected FE deltas: promo interactions + target alignment + drop avg_* aux profiles"
    )
    tracker.save()

    tracker_cogs.log_final(res_cogs)
    tracker_cogs.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Submission: {out_path}")
    return res_rev


if __name__ == "__main__":
    main()
