"""
EX_37: Log-Transformed Revenue Pipeline

Key insight: Revenue follows multiplicative dynamics (growth rates, seasonal
multipliers). Training on log(Revenue) makes the model learn relative changes
instead of absolute ones, which:
1. Naturally handles the ~5% systematic underprediction we saw
2. Stabilizes variance (high-value days have proportionally more noise)
3. Guarantees positive predictions via exp()
4. MSE on log(y) ≈ MSLE, which penalizes relative errors equally

Architecture:
- Target: log1p(Revenue), log1p(COGS)
- Features: same as EX-31 (lag/rolling features computed on raw values,
  but model learns in log-space)
- Prediction: exp(model output) - 1  (inverse of log1p)
- 3-seed ensemble, no drift correction
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

TRACK = Path("output/tracking/ex_37_log_transform")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500
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


def recursive_predict(
    models: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
) -> np.ndarray:
    """Recursive prediction in log-space.
    
    The model predicts log1p(target). We:
    1. Build features from raw (untransformed) history
    2. Predict in log-space
    3. Inverse-transform with expm1() to get raw prediction
    4. Feed raw prediction back into history for next step
    """
    history = history_df[["Date", target]].copy()
    preds: list[float] = []

    for date in predict_dates:
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        # Build features on RAW values (not log-transformed)
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

        # Ensemble: average predictions in log-space, then inverse transform
        log_preds = [float(m.predict(x_pred)[0]) for m in models]
        log_mean = float(np.mean(log_preds))
        
        # Inverse transform: expm1(log prediction)
        pred = float(np.expm1(log_mean))
        pred = max(0, pred)
        preds.append(pred)

        # Feed RAW prediction back into history
        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


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

    # Log-transform targets for training
    y_trn_rev_log = np.log1p(trn["Revenue"].values)
    y_trn_cogs_log = np.log1p(trn["COGS"].values)
    y_val_rev_log = np.log1p(val["Revenue"].values)
    y_val_cogs_log = np.log1p(val["COGS"].values)

    # Multi-seed ensemble trained on log targets
    models_rev = []
    models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), y_trn_rev_log,
            val[cols_rev].fillna(0), y_val_rev_log, seed=seed,
        ))
        models_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), y_trn_cogs_log,
            val[cols_cogs].fillna(0), y_val_cogs_log, seed=seed,
        ))

    # Recursive predict (handles log-space internally)
    pred_rev = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    pred_cogs = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # Evaluate on raw scale
    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    # Also compute log-scale metrics
    log_rev_mae = np.mean(np.abs(np.log1p(pred_rev) - np.log1p(y_val_rev)))
    log_cogs_mae = np.mean(np.abs(np.log1p(pred_cogs) - np.log1p(y_val_cogs)))

    print(f"  Log-scale MAE: Rev={log_rev_mae:.4f}  COGS={log_cogs_mae:.4f}")
    print(f"  Pred means: Rev={pred_rev.mean():,.0f}  COGS={pred_cogs.mean():,.0f}")
    print(f"  True means: Rev={y_val_rev.mean():,.0f}  COGS={y_val_cogs.mean():,.0f}")
    bias_rev = pred_rev.mean() / y_val_rev.mean() - 1
    bias_cogs = pred_cogs.mean() / y_val_cogs.mean() - 1
    print(f"  Bias: Rev={bias_rev:+.1%}  COGS={bias_cogs:+.1%}")

    return pd.DataFrame([{
        "fold": fold["name"],
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "revenue_mae": float(res_rev["mae"]),
        "revenue_rmse": float(res_rev["rmse"]),
        "cogs_mae": float(res_cogs["mae"]),
        "cogs_rmse": float(res_cogs["rmse"]),
        "score": score,
        "log_rev_mae": float(log_rev_mae),
        "log_cogs_mae": float(log_cogs_mae),
        "bias_rev_pct": float(bias_rev * 100),
        "bias_cogs_pct": float(bias_cogs * 100),
    }])


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_37: Log-Transformed Revenue Pipeline")
    print("=" * 78)

    fold_score_parts = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df = evaluate_fold(train, fold)
        fold_score_parts.append(s_df)

    fold_scores = pd.concat(fold_score_parts, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\nFold Scores:")
    print(fold_scores[["fold", "score", "revenue_mae", "cogs_mae", "bias_rev_pct", "bias_cogs_pct"]])
    print(f"\nMean score: {fold_scores['score'].mean():,.0f}")
    print(f"Mean bias: Rev={fold_scores['bias_rev_pct'].mean():+.1f}%  COGS={fold_scores['bias_cogs_pct'].mean():+.1f}%")

    # Train final models on full data
    print("\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)

    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")

    # Log-transform targets
    y_full_rev_log = np.log1p(feat_df["Revenue"].values)
    y_full_cogs_log = np.log1p(feat_df["COGS"].values)
    y_last365_rev_log = np.log1p(feat_df["Revenue"].tail(365).values)
    y_last365_cogs_log = np.log1p(feat_df["COGS"].tail(365).values)

    final_models_rev = []
    final_models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_models_rev.append(_fit_lgbm(
            feat_df[cols_rev].fillna(0), y_full_rev_log,
            feat_df[cols_rev].fillna(0).tail(365), y_last365_rev_log,
            seed=seed,
        ))
        final_models_cogs.append(_fit_lgbm(
            feat_df[cols_cogs].fillna(0), y_full_cogs_log,
            feat_df[cols_cogs].fillna(0).tail(365), y_last365_cogs_log,
            seed=seed,
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

    # Compare to EX-31
    try:
        sub31 = pd.read_csv(SUB_DIR / "ex_31_refined_ensemble.csv")
        rev_diff = final_rev.mean() / sub31["Revenue"].mean() - 1
        cogs_diff = final_cogs.mean() / sub31["COGS"].mean() - 1
        print(f"\nvs EX-31: Rev {rev_diff:+.1%}  COGS {cogs_diff:+.1%}")
    except Exception:
        pass

    candidate_path = SUB_DIR / "ex_37_log_transform.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
        "n_seeds": N_SEEDS,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "mean_score": float(fold_scores["score"].mean()),
        "mean_bias_rev_pct": float(fold_scores["bias_rev_pct"].mean()),
        "mean_bias_cogs_pct": float(fold_scores["bias_cogs_pct"].mean()),
        "transform": "log1p",
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
