"""
EX_45: Quantile Regression (α=0.60) with Stable Recursive Prediction

Evidence:
- Direct validation (fold_2022): α=0.60 achieves 520k MAE vs 562k regression
- α=0.60 predicts 3,161k mean vs actual 3,205k (within 1.4%)
- Standard regression predicts 3,083k (-3.8% bias)
- The test period (2023-2024) has even higher revenue → α=0.60 is better calibrated

Key design decisions:
1. Use quantile α=0.60 for REVENUE only (targets upper half of distribution)
2. Use standard regression for COGS (no evidence of COGS bias direction)
3. Clamp recursive predictions within ±2σ of rolling history to prevent drift
4. Multi-seed ensemble (3 seeds, same as EX-31)
5. Same feature engineering as EX-31 (proven best)

Hypothesis: Quantile regression with α=0.60 will produce higher, more accurate
revenue predictions for the test period without arbitrary scaling.
"""

from __future__ import annotations

import json
import time
import warnings
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

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

TRACK = Path("output/tracking/ex_45_quantile_stable")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3
QUANTILE_ALPHA = 0.60  # Targets 60th percentile → naturally higher predictions


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_quantile(x_trn, y_trn, x_val, y_val, alpha=QUANTILE_ALPHA, seed=SEED):
    """Train quantile regression model."""
    params = LGBM_PARAMS.copy()
    params["objective"] = "quantile"
    params["alpha"] = alpha
    params["metric"] = "mae"
    params["n_estimators"] = 1500
    params["random_state"] = seed

    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _fit_regression(x_trn, y_trn, x_val, y_val, seed=SEED):
    """Train standard regression model (for COGS)."""
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


def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols: list[str], target: str) -> list[str]:
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


# ─────────────────────────────────────────────────────────────────────────────
# Recursive prediction with stability clamp
# ─────────────────────────────────────────────────────────────────────────────

def recursive_predict(
    models: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
    clamp_sigma: float = 2.5,
) -> np.ndarray:
    """
    Recursive prediction with multi-model ensemble.
    
    Stability mechanism: clamp predictions to ±clamp_sigma standard deviations
    from the rolling 90-day mean. This prevents exponential drift that quantile
    regression can cause in recursive settings.
    """
    history = history_df[["Date", target]].copy()
    preds: list[float] = []

    # Pre-compute clamp boundaries from recent history
    recent_vals = history[target].dropna().tail(365).values
    hist_mean = float(np.mean(recent_vals))
    hist_std = float(np.std(recent_vals))
    clamp_low = max(0, hist_mean - clamp_sigma * hist_std)
    clamp_high = hist_mean + clamp_sigma * hist_std

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

        # Ensemble: average predictions from all models
        raw_preds = [float(m.predict(x_pred)[0]) for m in models]
        pred = float(np.mean(raw_preds))

        # Stability clamp: prevent recursive drift
        pred = np.clip(pred, clamp_low, clamp_high)
        pred = max(0, pred)
        preds.append(pred)

        # Update history with clamped prediction
        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

        # Slowly adapt clamp boundaries using rolling stats
        recent = history[target].dropna().tail(365).values
        rolling_mean = float(np.mean(recent))
        rolling_std = float(np.std(recent))
        clamp_low = max(0, rolling_mean - clamp_sigma * rolling_std)
        clamp_high = rolling_mean + clamp_sigma * rolling_std

    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fold(sales: pd.DataFrame, fold: dict, alpha: float = QUANTILE_ALPHA):
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
        # Revenue: quantile regression
        models_rev.append(_fit_quantile(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), val["Revenue"],
            alpha=alpha, seed=seed,
        ))
        # COGS: standard regression
        models_cogs.append(_fit_regression(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), val["COGS"], seed=seed,
        ))

    pred_rev = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue",
    )
    pred_cogs = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS",
    )

    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    return pd.DataFrame([{
        "fold": fold["name"],
        "alpha": alpha,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "revenue_mae": float(res_rev["mae"]),
        "cogs_mae": float(res_cogs["mae"]),
        "revenue_pred_mean": float(pred_rev.mean()),
        "cogs_pred_mean": float(pred_cogs.mean()),
        "score": score,
    }])


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_45: Quantile Regression (α=0.60) + Stable Recursive Prediction")
    print(f"  Revenue: quantile α={QUANTILE_ALPHA} (targets 60th percentile)")
    print("  COGS: standard regression (unbiased)")
    print("  Recursive clamp: ±2.5σ from rolling 365d mean")
    print("=" * 78)

    # ── Cross-validation ──
    fold_results = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df = evaluate_fold(train, fold)
        fold_results.append(s_df)

    fold_scores = pd.concat(fold_results, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\nFold Scores:")
    print(fold_scores[["fold", "revenue_mae", "cogs_mae", "revenue_pred_mean", "score"]].to_string(index=False))
    print(f"\nMean score: {fold_scores['score'].mean():,.0f}")

    # Compare with EX-31 baseline
    ex31_mean_score = 896_733  # from previous run
    improvement = ex31_mean_score - fold_scores["score"].mean()
    print(f"vs EX-31 baseline ({ex31_mean_score:,}): {'+' if improvement > 0 else ''}{improvement:,.0f}")

    # ── Full retrain + submission ──
    print("\n\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(
        train, verbose=True, profile_source_df=train
    )
    base_cols = get_feature_cols(feat_df)

    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")

    # Multi-seed ensemble
    final_models_rev = []
    final_models_cogs = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_models_rev.append(_fit_quantile(
            feat_df[cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            alpha=QUANTILE_ALPHA, seed=seed,
        ))
        final_models_cogs.append(_fit_regression(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed,
        ))

    print("\nRunning recursive inference on test set...")
    final_rev = recursive_predict(
        final_models_rev, train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue",
    )
    final_cogs = recursive_predict(
        final_models_cogs, train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS",
    )

    # Diagnostics
    ratios = final_rev / np.clip(final_cogs, 1, None)
    print(f"\nRev/COGS ratio: mean={ratios.mean():.3f}")
    print(f"Revenue: mean={final_rev.mean():,.0f}  std={final_rev.std():,.0f}")
    print(f"COGS: mean={final_cogs.mean():,.0f}  std={final_cogs.std():,.0f}")

    candidate_path = SUB_DIR / "ex_45_quantile_stable.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
        "quantile_alpha": QUANTILE_ALPHA,
        "n_seeds": N_SEEDS,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "mean_cv_score": float(fold_scores["score"].mean()),
        "mean_rev_pred": float(final_rev.mean()),
        "mean_cogs_pred": float(final_cogs.mean()),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
