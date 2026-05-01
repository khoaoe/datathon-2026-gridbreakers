"""
EX_35: Final Consolidated Pipeline

Combines all winning techniques:
1. LightGBM + XGBoost ensemble (model diversity)
2. 3-seed ensemble per model type (variance reduction)
3. Monthly drift correction with horizon-adaptive α
4. Per-target alpha tuning (COGS drifts more than Revenue)

Architecture:
  Base models: 3 LightGBM seeds + 3 XGBoost seeds = 6 models per target
  Post-processing: monthly profile drift correction
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

from modeling.config import LGBM_PARAMS, XGB_PARAMS, SEED
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

# Check if XGBoost is available
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not available, using LightGBM only")

TRACK = Path("output/tracking/ex_35_final_pipeline")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


# ── Model fitting ────────────────────────────────────────────────────────────

def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500
    params["random_state"] = seed
    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _fit_xgb(x_trn, y_trn, x_val, y_val, seed=SEED):
    if not HAS_XGB:
        return _fit_lgbm(x_trn, y_trn, x_val, y_val, seed)
    
    params = XGB_PARAMS.copy()
    params["n_estimators"] = 1500
    params["random_state"] = seed
    params["early_stopping_rounds"] = 100
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def _fit_ensemble(x_trn, y_trn, x_val, y_val, n_seeds=N_SEEDS):
    """Fit LightGBM + XGBoost ensemble with multiple seeds."""
    models = []
    for i in range(n_seeds):
        seed = SEED + i * 17
        models.append(_fit_lgbm(x_trn, y_trn, x_val, y_val, seed=seed))
        if HAS_XGB:
            models.append(_fit_xgb(x_trn, y_trn, x_val, y_val, seed=seed))
    return models


# ── Feature selection ─────────────────────────────────────────────────────────

def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols: list[str], target: str) -> list[str]:
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


# ── Recursive prediction ─────────────────────────────────────────────────────

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


# ── Drift correction ─────────────────────────────────────────────────────────

def _compute_monthly_means(df: pd.DataFrame, years_back: int = 3) -> dict:
    max_year = df["Date"].dt.year.max()
    recent = df[df["Date"].dt.year > max_year - years_back]
    result = {}
    for target in ["Revenue", "COGS"]:
        result[target] = recent.groupby(recent["Date"].dt.month)[target].mean().to_dict()
    return result


def _apply_drift_correction(
    preds: np.ndarray,
    dates,
    monthly_means: dict,
    target: str,
    alpha: float,
) -> np.ndarray:
    corrected = preds.copy()
    months = pd.to_datetime(pd.Series(dates)).dt.month.values

    pred_monthly = {}
    for m in range(1, 13):
        mask = months == m
        if mask.sum() > 0:
            pred_monthly[m] = preds[mask].mean()

    for m in range(1, 13):
        mask = months == m
        if mask.sum() == 0 or m not in monthly_means[target]:
            continue

        ref_mean = monthly_means[target][m]
        pred_mean = pred_monthly.get(m, ref_mean)

        if pred_mean > 0:
            scale = ref_mean / pred_mean
            blend_scale = (1 - alpha) + alpha * scale
            corrected[mask] = preds[mask] * blend_scale

    return np.clip(corrected, 0, None)


def _tune_alpha_per_target(
    preds_rev, preds_cogs, dates,
    y_true_rev, y_true_cogs, monthly_means,
) -> tuple[float, float]:
    best_alpha_rev = 0.0
    best_alpha_cogs = 0.0
    best_score = float("inf")

    for alpha_rev in np.arange(0.0, 1.05, 0.05):
        corr_rev = _apply_drift_correction(preds_rev, dates, monthly_means, "Revenue", alpha_rev)
        rev_mae = mean_absolute_error(y_true_rev, corr_rev)

        for alpha_cogs in np.arange(0.0, 1.05, 0.05):
            corr_cogs = _apply_drift_correction(preds_cogs, dates, monthly_means, "COGS", alpha_cogs)
            cogs_mae = mean_absolute_error(y_true_cogs, corr_cogs)
            score = rev_mae + 0.4 * cogs_mae

            if score < best_score:
                best_score = score
                best_alpha_rev = alpha_rev
                best_alpha_cogs = alpha_cogs

    return best_alpha_rev, best_alpha_cogs


# ── Fold evaluation ───────────────────────────────────────────────────────────

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

    # Fit ensemble
    models_rev = _fit_ensemble(trn[cols_rev].fillna(0), trn["Revenue"],
                               val[cols_rev].fillna(0), val["Revenue"])
    models_cogs = _fit_ensemble(trn[cols_cogs].fillna(0), trn["COGS"],
                                val[cols_cogs].fillna(0), val["COGS"])

    print(f"  Models per target: {len(models_rev)} ({N_SEEDS} LGB" +
          (f" + {N_SEEDS} XGB" if HAS_XGB else "") + ")")

    # Recursive predict
    raw_pred_rev = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    raw_pred_cogs = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # Drift correction
    monthly_means = _compute_monthly_means(train_slice, years_back=3)
    opt_alpha_rev, opt_alpha_cogs = _tune_alpha_per_target(
        raw_pred_rev, raw_pred_cogs, val["Date"],
        y_val_rev, y_val_cogs, monthly_means
    )

    pred_rev = _apply_drift_correction(raw_pred_rev, val["Date"], monthly_means, "Revenue", opt_alpha_rev)
    pred_cogs = _apply_drift_correction(raw_pred_cogs, val["Date"], monthly_means, "COGS", opt_alpha_cogs)

    # Evaluate
    raw_res_rev = evaluate(y_val_rev, raw_pred_rev, f"{fold['name']} Revenue (raw)")
    raw_res_cogs = evaluate(y_val_cogs, raw_pred_cogs, f"{fold['name']} COGS (raw)")
    raw_score = float(raw_res_rev["mae"] + 0.4 * raw_res_cogs["mae"])

    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue (corrected)")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS (corrected)")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    horizon = (val_end - val_start).days + 1
    print(f"  Raw: {raw_score:,.0f} → Corrected: {score:,.0f} (α_rev={opt_alpha_rev:.2f}, α_cogs={opt_alpha_cogs:.2f})")

    return pd.DataFrame([{
        "fold": fold["name"],
        "horizon_days": horizon,
        "n_models": len(models_rev),
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "revenue_mae_raw": float(raw_res_rev["mae"]),
        "cogs_mae_raw": float(raw_res_cogs["mae"]),
        "score_raw": raw_score,
        "revenue_mae": float(res_rev["mae"]),
        "cogs_mae": float(res_cogs["mae"]),
        "score": score,
        "alpha_rev": opt_alpha_rev,
        "alpha_cogs": opt_alpha_cogs,
    }])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_35: Final Pipeline (LGB+XGB ensemble + drift correction)")
    print(f"  XGBoost available: {HAS_XGB}")
    print("=" * 78)

    fold_score_parts = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df = evaluate_fold(train, fold)
        fold_score_parts.append(s_df)

    fold_scores = pd.concat(fold_score_parts, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\nFold Scores:")
    print(fold_scores[["fold", "score_raw", "score", "alpha_rev", "alpha_cogs"]])
    print(f"\nMean raw score: {fold_scores['score_raw'].mean():,.0f}")
    print(f"Mean corrected score: {fold_scores['score'].mean():,.0f}")

    # Determine test alphas using horizon-adaptive heuristic
    test_horizon = (pd.Timestamp(test["Date"].max()) - pd.Timestamp(test["Date"].min())).days + 1
    fold_2022 = fold_scores[fold_scores["fold"] == "fold_2022"].iloc[0]
    cv_horizon = fold_2022["horizon_days"]

    # Scale alpha with power-law extrapolation
    # α_test = min(α_cv * (test_horizon/cv_horizon)^1.3, 1.0)
    scale_power = 1.3
    alpha_rev_test = min(float(fold_2022["alpha_rev"]) * (test_horizon / cv_horizon) ** scale_power, 1.0)
    alpha_cogs_test = min(float(fold_2022["alpha_cogs"]) * (test_horizon / cv_horizon) ** scale_power, 1.0)

    # Floor: at least use the fold_2022 value
    alpha_rev_test = max(alpha_rev_test, float(fold_2022["alpha_rev"]))
    alpha_cogs_test = max(alpha_cogs_test, float(fold_2022["alpha_cogs"]))

    print(f"\nTest horizon: {test_horizon} days (CV: {cv_horizon} days)")
    print(f"fold_2022 alphas: α_rev={fold_2022['alpha_rev']:.2f}, α_cogs={fold_2022['alpha_cogs']:.2f}")
    print(f"Final test alphas: α_rev={alpha_rev_test:.2f}, α_cogs={alpha_cogs_test:.2f}")

    # Train final models
    print("\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)

    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")

    final_models_rev = _fit_ensemble(
        feat_df[cols_rev].fillna(0), feat_df["Revenue"],
        feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
    )
    final_models_cogs = _fit_ensemble(
        feat_df[cols_cogs].fillna(0), feat_df["COGS"],
        feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
    )

    print(f"  Final ensemble: {len(final_models_rev)} models per target")

    # Recursive inference
    print("Running recursive inference on test set...")
    raw_rev = recursive_predict(
        final_models_rev, train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    raw_cogs = recursive_predict(
        final_models_cogs, train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # Apply drift correction
    monthly_means = _compute_monthly_means(train, years_back=3)
    final_rev = _apply_drift_correction(raw_rev, test["Date"], monthly_means, "Revenue", alpha_rev_test)
    final_cogs = _apply_drift_correction(raw_cogs, test["Date"], monthly_means, "COGS", alpha_cogs_test)

    # Diagnostics
    print(f"\nDrift correction: α_rev={alpha_rev_test:.2f}, α_cogs={alpha_cogs_test:.2f}")
    print(f"Raw Revenue: mean={raw_rev.mean():,.0f} → Corrected: mean={final_rev.mean():,.0f}")
    print(f"Raw COGS: mean={raw_cogs.mean():,.0f} → Corrected: mean={final_cogs.mean():,.0f}")

    ratios = final_rev / np.clip(final_cogs, 1, None)
    print(f"Rev/COGS ratio: mean={ratios.mean():.3f}  min={ratios.min():.3f}  max={ratios.max():.3f}")

    candidate_path = SUB_DIR / "ex_35_final_pipeline.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
        "has_xgb": HAS_XGB,
        "n_models_per_target": len(final_models_rev),
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "alpha_rev": float(alpha_rev_test),
        "alpha_cogs": float(alpha_cogs_test),
        "mean_score_raw": float(fold_scores["score_raw"].mean()),
        "mean_score_corrected": float(fold_scores["score"].mean()),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
