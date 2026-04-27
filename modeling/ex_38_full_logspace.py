"""
EX_38: Full Log-Space Pipeline

Approach: log-transform Revenue & COGS as a PREPROCESSING step, then run the
entire pipeline (feature engineering, training, recursive prediction) in
log-space. Only reverse the transform (expm1) when writing the final submission.

Why this works:
- Lags, rolling means, diffs — all computed on log values
- Rolling mean in log-space ≈ geometric mean in raw space (natural for growth)
- Model learns additive relationships on log scale = multiplicative on raw scale
- Eliminates the systematic ~5% underprediction from MSE on raw values
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

TRACK = Path("output/tracking/ex_38_full_logspace")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


# ── Preprocessing ─────────────────────────────────────────────────────────────

def log_transform_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform Revenue and COGS columns. Returns a copy."""
    out = df.copy()
    out["Revenue"] = np.log1p(out["Revenue"].clip(lower=0))
    out["COGS"] = np.log1p(out["COGS"].clip(lower=0))
    return out


def inverse_transform(values: np.ndarray) -> np.ndarray:
    """Reverse the log1p transform."""
    return np.expm1(values)


# ── Model fitting ─────────────────────────────────────────────────────────────

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


# ── Recursive prediction (all in log-space) ──────────────────────────────────

def recursive_predict(
    models: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
) -> np.ndarray:
    """Recursive prediction entirely in log-space.
    
    history_df contains log-transformed values.
    Returns log-space predictions (caller must inverse_transform).
    """
    history = history_df[["Date", target]].copy()
    preds: list[float] = []

    for date in predict_dates:
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        # Features are computed on log-transformed values
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

        # Ensemble prediction (in log-space)
        raw_preds = [float(m.predict(x_pred)[0]) for m in models]
        pred = float(np.mean(raw_preds))
        preds.append(pred)

        # Feed log-space prediction back into history
        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


# ── Fold evaluation ───────────────────────────────────────────────────────────

def evaluate_fold(sales_log: pd.DataFrame, sales_raw: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice_log = sales_log[sales_log["Date"] < val_start].copy()

    # Build features on log-transformed data
    feat_df, profiles = build_feature_table(
        sales_log, verbose=False, profile_source_df=train_slice_log
    )

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    # Targets are already log-transformed
    y_trn_rev = trn["Revenue"].values  # log-space
    y_trn_cogs = trn["COGS"].values    # log-space
    y_val_rev = val["Revenue"].values   # log-space
    y_val_cogs = val["COGS"].values     # log-space

    # Ground truth in raw scale for evaluation
    val_raw = sales_raw[
        (sales_raw["Date"] >= val_start) & (sales_raw["Date"] <= val_end)
    ]
    y_val_rev_raw = val_raw["Revenue"].values
    y_val_cogs_raw = val_raw["COGS"].values

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_target_cols(base_cols, "COGS"))

    # Train on log-space targets
    models_rev = []
    models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), y_trn_rev,
            val[cols_rev].fillna(0), y_val_rev, seed=seed,
        ))
        models_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), y_trn_cogs,
            val[cols_cogs].fillna(0), y_val_cogs, seed=seed,
        ))

    # Recursive predict in log-space
    pred_rev_log = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val_raw["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    pred_cogs_log = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val_raw["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # Inverse transform to raw scale for evaluation
    pred_rev = inverse_transform(pred_rev_log)
    pred_cogs = inverse_transform(pred_cogs_log)

    res_rev = evaluate(y_val_rev_raw, pred_rev, f"{fold['name']} Revenue")
    res_cogs = evaluate(y_val_cogs_raw, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    bias_rev = pred_rev.mean() / y_val_rev_raw.mean() - 1
    bias_cogs = pred_cogs.mean() / y_val_cogs_raw.mean() - 1
    print(f"  Pred means: Rev={pred_rev.mean():,.0f}  COGS={pred_cogs.mean():,.0f}")
    print(f"  True means: Rev={y_val_rev_raw.mean():,.0f}  COGS={y_val_cogs_raw.mean():,.0f}")
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
        "bias_rev_pct": float(bias_rev * 100),
        "bias_cogs_pct": float(bias_cogs * 100),
    }])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test = load_sales()

    # === PREPROCESSING: log-transform ===
    train_log = log_transform_sales(train_raw)
    print("Preprocessing: log1p transform applied to Revenue & COGS")
    print(f"  Raw Revenue range: [{train_raw['Revenue'].min():,.0f}, {train_raw['Revenue'].max():,.0f}]")
    print(f"  Log Revenue range: [{train_log['Revenue'].min():.3f}, {train_log['Revenue'].max():.3f}]")

    print("\n" + "=" * 78)
    print("EX_38: Full Log-Space Pipeline")
    print("=" * 78)

    fold_score_parts = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df = evaluate_fold(train_log, train_raw, fold)
        fold_score_parts.append(s_df)

    fold_scores = pd.concat(fold_score_parts, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\n" + "=" * 78)
    print("FOLD SUMMARY")
    print("=" * 78)
    print(fold_scores[["fold", "score", "revenue_mae", "cogs_mae", "bias_rev_pct", "bias_cogs_pct"]])
    print(f"\nMean score: {fold_scores['score'].mean():,.0f}")
    print(f"Mean bias: Rev={fold_scores['bias_rev_pct'].mean():+.1f}%  COGS={fold_scores['bias_cogs_pct'].mean():+.1f}%")

    # Compare to EX-31 (raw-space baseline)
    try:
        ex31 = pd.read_csv("output/tracking/ex_31_refined_ensemble/fold_scores.csv")
        print(f"\nvs EX-31 (raw-space): mean score {ex31['score'].mean():,.0f}")
        improvement = (ex31['score'].mean() - fold_scores['score'].mean()) / ex31['score'].mean() * 100
        print(f"  Improvement: {improvement:+.1f}%")
    except Exception:
        pass

    # Train final model on full data
    print("\nTraining final model on full log-transformed data...")
    feat_df, profiles = build_feature_table(
        train_log, verbose=True, profile_source_df=train_log
    )
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
            seed=seed,
        ))
        final_models_cogs.append(_fit_lgbm(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed,
        ))

    # Recursive inference in log-space
    print("Running recursive inference on test set (log-space)...")
    pred_rev_log = recursive_predict(
        final_models_rev, train_log[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    pred_cogs_log = recursive_predict(
        final_models_cogs, train_log[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # === REVERSE TRANSFORM ===
    final_rev = inverse_transform(pred_rev_log)
    final_cogs = inverse_transform(pred_cogs_log)

    # Diagnostics
    ratios = final_rev / np.clip(final_cogs, 1, None)
    print(f"\nLog-space predictions: Rev mean={pred_rev_log.mean():.4f}  COGS mean={pred_cogs_log.mean():.4f}")
    print(f"Raw-space after expm1:")
    print(f"  Revenue: mean={final_rev.mean():,.0f}  std={final_rev.std():,.0f}")
    print(f"  COGS: mean={final_cogs.mean():,.0f}  std={final_cogs.std():,.0f}")
    print(f"  Rev/COGS ratio: mean={ratios.mean():.3f}  min={ratios.min():.3f}  max={ratios.max():.3f}")

    # Compare to EX-31
    try:
        sub31 = pd.read_csv(SUB_DIR / "ex_31_refined_ensemble.csv")
        rev_diff = final_rev.mean() / sub31["Revenue"].mean() - 1
        cogs_diff = final_cogs.mean() / sub31["COGS"].mean() - 1
        print(f"\nvs EX-31 means: Rev {rev_diff:+.1%}  COGS {cogs_diff:+.1%}")
    except Exception:
        pass

    candidate_path = SUB_DIR / "ex_38_full_logspace.csv"
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
        "transform": "full_log1p_preprocess",
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
