"""
EX_28: Clean LGBM Pipeline

Purpose:
- Normal feature engineering + LightGBM on the raw data.
- NO anchor. NO bridges. NO complex component ensembles.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from modeling.config import LGBM_PARAMS
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

TRACK = Path("output/tracking/ex_28_clean_lgbm")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

def _fit_lgbm(x_trn, y_trn, x_val, y_val):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500
    
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
    model,
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

        pred = float(np.clip(model.predict(x_pred)[0], 0, None))
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)

def evaluate_fold(sales: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(sales, verbose=False, profile_source_df=train_slice)
    
    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()
    
    y_val_rev = val["Revenue"].values
    y_val_cogs = val["COGS"].values

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_target_cols(base_cols, "COGS"))

    model_rev = _fit_lgbm(trn[cols_rev].fillna(0), trn["Revenue"], val[cols_rev].fillna(0), val["Revenue"])
    model_cogs = _fit_lgbm(trn[cols_cogs].fillna(0), trn["COGS"], val[cols_cogs].fillna(0), val["COGS"])

    pred_rev = recursive_predict(model_rev, trn[["Date", "Revenue"]], val["Date"].values, cols_rev, profiles, "Revenue")
    pred_cogs = recursive_predict(model_cogs, trn[["Date", "COGS"]], val["Date"].values, cols_cogs, profiles, "COGS")

    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    return pd.DataFrame([{
        "fold": fold["name"],
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "revenue_mae": float(res_rev["mae"]),
        "cogs_mae": float(res_cogs["mae"]),
        "score": score,
    }])

def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_28: Clean LGBM Pipeline")
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

    print("\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(train, verbose=False, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)
    
    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    model_rev = _fit_lgbm(
        feat_df[cols_rev].fillna(0), feat_df["Revenue"],
        feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365)
    )
    model_cogs = _fit_lgbm(
        feat_df[cols_cogs].fillna(0), feat_df["COGS"],
        feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365)
    )

    print("Running recursive inference on test set...")
    final_rev = recursive_predict(model_rev, train[["Date", "Revenue"]], test["Date"].values, cols_rev, profiles, "Revenue")
    final_cogs = recursive_predict(model_cogs, train[["Date", "COGS"]], test["Date"].values, cols_cogs, profiles, "COGS")

    candidate_path = SUB_DIR / "ex_28_clean_lgbm.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")

if __name__ == "__main__":
    main()
