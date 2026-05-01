"""
EX_47: Monthly Recalibration + LGB/XGB Ensemble

Evidence-based approach:
1. Direct prediction MAE=562k but recursive MAE=682k (21% degradation)
2. Recursive errors are MONTH-SPECIFIC and SYSTEMATIC:
   - March: -992k bias (spring surge → lag snowball)
   - August: -997k bias (complex seasonality)
   - Jan: -382k bias (big seasonal drop)
3. These monthly errors are stable across folds → learnable
4. LGB/XGB blend provides ~2% direct MAE improvement (557k→555k)

Strategy:
1. Run recursive prediction on each CV fold
2. Compute per-MONTH correction ratios: actual_mean / predicted_mean
3. Apply these corrections to test predictions (per-month recalibration)
4. Use LGB+XGB ensemble for base predictions
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

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

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

TRACK = Path("output/tracking/ex_47_recalibrate")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

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
    params = XGB_PARAMS.copy()
    params["n_estimators"] = 1500
    params["random_state"] = seed
    params["device"] = "cpu"
    model = xgb.XGBRegressor(**params)
    model.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        verbose=False,
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
# Recursive prediction
# ─────────────────────────────────────────────────────────────────────────────

def recursive_predict(
    models_lgb: list,
    models_xgb: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
    xgb_weight: float = 0.3,
) -> np.ndarray:
    """Recursive prediction with LGB/XGB ensemble."""
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

        # LGB ensemble
        lgb_preds = [float(m.predict(x_pred)[0]) for m in models_lgb]
        lgb_pred = float(np.mean(lgb_preds))

        # XGB ensemble
        xgb_preds = [float(m.predict(x_pred)[0]) for m in models_xgb]
        xgb_pred = float(np.mean(xgb_preds))

        # Blend
        pred = (1 - xgb_weight) * lgb_pred + xgb_weight * xgb_pred
        pred = max(0, pred)
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Monthly recalibration
# ─────────────────────────────────────────────────────────────────────────────

def compute_monthly_corrections(
    fold_results: list[dict],
) -> dict[int, float]:
    """
    From CV fold results, compute per-month correction ratios.
    
    Returns dict: month -> correction_ratio (multiply predicted by this)
    """
    all_records = []
    for fold_result in fold_results:
        for target in ["Revenue", "COGS"]:
            dates = fold_result["dates"]
            actuals = fold_result[f"actual_{target}"]
            preds = fold_result[f"pred_{target}"]
            
            for d, a, p in zip(dates, actuals, preds):
                all_records.append({
                    "month": d.month,
                    "target": target,
                    "actual": a,
                    "pred": p,
                })
    
    df = pd.DataFrame(all_records)
    corrections = {}
    
    for target in ["Revenue", "COGS"]:
        target_df = df[df["target"] == target]
        monthly = target_df.groupby("month").agg(
            actual_mean=("actual", "mean"),
            pred_mean=("pred", "mean"),
        ).reset_index()
        monthly["ratio"] = monthly["actual_mean"] / monthly["pred_mean"]
        corrections[target] = dict(zip(monthly["month"], monthly["ratio"]))
    
    return corrections


def apply_monthly_corrections(
    dates: np.ndarray, predictions: np.ndarray, corrections: dict[int, float]
) -> np.ndarray:
    """Apply per-month correction ratios to predictions."""
    corrected = predictions.copy()
    for i, date in enumerate(dates):
        month = pd.Timestamp(date).month
        if month in corrections:
            corrected[i] *= corrections[month]
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Fold evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fold(sales: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(
        sales, verbose=False, profile_source_df=train_slice
    )

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_target_cols(base_cols, "COGS"))

    # Train models
    models_lgb_rev, models_lgb_cogs = [], []
    models_xgb_rev, models_xgb_cogs = [], []

    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_lgb_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), val["Revenue"], seed=seed,
        ))
        models_lgb_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), val["COGS"], seed=seed,
        ))
        models_xgb_rev.append(_fit_xgb(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), val["Revenue"], seed=seed,
        ))
        models_xgb_cogs.append(_fit_xgb(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), val["COGS"], seed=seed,
        ))

    # Recursive prediction
    pred_rev = recursive_predict(
        models_lgb_rev, models_xgb_rev,
        trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue",
    )
    pred_cogs = recursive_predict(
        models_lgb_cogs, models_xgb_cogs,
        trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS",
    )

    # Evaluate uncorrected
    res_rev = evaluate(val["Revenue"].values, pred_rev, f"{fold['name']} Rev")
    res_cogs = evaluate(val["COGS"].values, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])
    print(f"  Uncorrected score: {score:,.0f}  rev_mean={pred_rev.mean():,.0f}")

    return {
        "fold": fold["name"],
        "dates": [pd.Timestamp(d) for d in val["Date"].values],
        "actual_Revenue": val["Revenue"].values,
        "pred_Revenue": pred_rev,
        "actual_COGS": val["COGS"].values,
        "pred_COGS": pred_cogs,
        "score_uncorrected": score,
        "cols_rev": cols_rev,
        "cols_cogs": cols_cogs,
    }


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_47: Monthly Recalibration + LGB/XGB Ensemble")
    print("  Step 1: Run recursive prediction on all CV folds")
    print("  Step 2: Compute per-month correction factors from CV errors")
    print("  Step 3: Apply corrections to test predictions")
    print("=" * 78)

    # ── Step 1: Get fold predictions ──
    fold_results = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        result = evaluate_fold(train, fold)
        fold_results.append(result)

    # ── Step 2: Compute monthly corrections ──
    corrections = compute_monthly_corrections(fold_results)
    
    print("\n\n=== Monthly Correction Ratios ===")
    for target in ["Revenue", "COGS"]:
        print(f"\n  {target}:")
        for month in sorted(corrections[target].keys()):
            ratio = corrections[target][month]
            print(f"    Month {month:2d}: ×{ratio:.4f} ({'+' if ratio > 1 else ''}{(ratio-1)*100:.1f}%)")

    # ── Step 2.5: Leave-one-fold-out validation ──
    # Apply corrections from OTHER folds, evaluate on held-out fold
    print("\n\n=== Leave-One-Out Validation ===")
    corrected_scores = []
    for i, fold in enumerate(FOLDS):
        other_folds = [fold_results[j] for j in range(len(FOLDS)) if j != i]
        loo_corrections = compute_monthly_corrections(other_folds)
        
        result = fold_results[i]
        dates = result["dates"]
        
        # Apply corrections
        corrected_rev = apply_monthly_corrections(
            np.array(dates), result["pred_Revenue"], loo_corrections["Revenue"]
        )
        corrected_cogs = apply_monthly_corrections(
            np.array(dates), result["pred_COGS"], loo_corrections["COGS"]
        )
        
        res_rev = evaluate(result["actual_Revenue"], corrected_rev, f"{fold['name']} Rev CORRECTED")
        res_cogs = evaluate(result["actual_COGS"], corrected_cogs, f"{fold['name']} COGS CORRECTED")
        score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])
        
        print(f"  {fold['name']}: uncorrected={result['score_uncorrected']:,.0f} → corrected={score:,.0f} "
              f"({'+' if score > result['score_uncorrected'] else ''}{score - result['score_uncorrected']:,.0f})")
        corrected_scores.append(score)
    
    print(f"\n  Mean uncorrected: {np.mean([r['score_uncorrected'] for r in fold_results]):,.0f}")
    print(f"  Mean corrected: {np.mean(corrected_scores):,.0f}")

    # ── Step 3: Full retrain + submission ──
    print("\n\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")

    final_lgb_rev, final_lgb_cogs = [], []
    final_xgb_rev, final_xgb_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_lgb_rev.append(_fit_lgbm(
            feat_df[cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            seed=seed,
        ))
        final_lgb_cogs.append(_fit_lgbm(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed,
        ))
        final_xgb_rev.append(_fit_xgb(
            feat_df[cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            seed=seed,
        ))
        final_xgb_cogs.append(_fit_xgb(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed,
        ))

    print("\nRunning recursive inference on test set...")
    test_rev = recursive_predict(
        final_lgb_rev, final_xgb_rev,
        train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue",
    )
    test_cogs = recursive_predict(
        final_lgb_cogs, final_xgb_cogs,
        train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS",
    )

    # Save uncorrected
    path_raw = SUB_DIR / "ex_47_raw.csv"
    make_submission(test["Date"], test_rev, test_cogs, path_raw)
    print(f"  Raw: rev_mean={test_rev.mean():,.0f}, cogs_mean={test_cogs.mean():,.0f}")

    # Apply corrections from ALL folds
    test_dates = test["Date"].values
    corrected_rev = apply_monthly_corrections(test_dates, test_rev, corrections["Revenue"])
    corrected_cogs = apply_monthly_corrections(test_dates, test_cogs, corrections["COGS"])

    path_corrected = SUB_DIR / "ex_47_recalibrated.csv"
    make_submission(test["Date"], corrected_rev, corrected_cogs, path_corrected)
    print(f"  Corrected: rev_mean={corrected_rev.mean():,.0f}, cogs_mean={corrected_cogs.mean():,.0f}")

    # Also save scaled versions for comparison
    for scale in [1.05, 1.08, 1.10]:
        path_scaled = SUB_DIR / f"ex_47_scale{int(scale*100)}.csv"
        make_submission(test["Date"], test_rev * scale, test_cogs * scale, path_scaled)
        print(f"  Scale ×{scale}: rev_mean={test_rev.mean() * scale:,.0f}")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "corrections": {t: {str(k): v for k, v in c.items()} for t, c in corrections.items()},
        "mean_uncorrected_score": float(np.mean([r["score_uncorrected"] for r in fold_results])),
        "mean_corrected_score": float(np.mean(corrected_scores)) if corrected_scores else None,
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
