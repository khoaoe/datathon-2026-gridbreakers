"""
EX_46: YoY Ratio Model — Predict Growth Rate, Not Level

Key Insight:
- The model struggles because trees can't extrapolate absolute revenue levels
- But YoY ratios (revenue_t / revenue_{t-365}) are WITHIN training range:
  * 2022 YoY mean = 1.24, test period inferred = 1.15-1.25 (70th-79th pctl)
- Predicting the RATIO instead of the LEVEL makes the problem much easier
  because trees only need to interpolate, not extrapolate

How it works:
1. Target = Revenue_t / Revenue_{t-365} (YoY ratio)
2. Train model on ratio features (calendar, rolling ratio stats, etc.)
3. At prediction time: pred_ratio × revenue_{t-365} = predicted_revenue
4. For test: revenue_{t-365} comes from actual 2022 training data

Benefits:
- Scale-invariant: model learns patterns, not levels
- Tree-friendly: no extrapolation needed
- Naturally adapts: if 2022 was higher, predictions are higher
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

TRACK = Path("output/tracking/ex_46_yoy_ratio")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Ratio target construction
# ─────────────────────────────────────────────────────────────────────────────

def add_ratio_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add YoY ratio targets: revenue_t / revenue_{t-365}."""
    out = df.copy()
    out["Revenue_base"] = out["Revenue"].shift(365)
    out["COGS_base"] = out["COGS"].shift(365)
    out["Revenue_ratio"] = out["Revenue"] / out["Revenue_base"].replace(0, np.nan)
    out["COGS_ratio"] = out["COGS"] / out["COGS_base"].replace(0, np.nan)
    return out


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features specific to the ratio model."""
    out = df.copy()
    new_cols = {}

    for target in ["Revenue_ratio", "COGS_ratio"]:
        if target not in out.columns:
            continue
        prefix = target.replace("_ratio", "").lower()[:3]  # 'rev' or 'cog'
        shifted = out[target].shift(1)

        new_cols[f"{prefix}_ratio_lag1"] = shifted
        new_cols[f"{prefix}_ratio_lag7"] = out[target].shift(7)
        new_cols[f"{prefix}_ratio_lag28"] = out[target].shift(28)
        new_cols[f"{prefix}_ratio_lag90"] = out[target].shift(90)
        new_cols[f"{prefix}_ratio_rmean7"] = shifted.rolling(7, min_periods=1).mean()
        new_cols[f"{prefix}_ratio_rmean28"] = shifted.rolling(28, min_periods=1).mean()
        new_cols[f"{prefix}_ratio_rmean90"] = shifted.rolling(90, min_periods=7).mean()
        new_cols[f"{prefix}_ratio_rstd28"] = shifted.rolling(28, min_periods=7).std()

        # Momentum
        short = shifted.rolling(7, min_periods=1).mean()
        long = shifted.rolling(28, min_periods=1).mean()
        new_cols[f"{prefix}_ratio_momentum"] = short / long.replace(0, np.nan)

    # Base value features (the denominator — prior year revenue)
    for base_col in ["Revenue_base", "COGS_base"]:
        if base_col not in out.columns:
            continue
        prefix = base_col.replace("_base", "").lower()[:3]
        new_cols[f"{prefix}_base_rmean7"] = out[base_col].rolling(7, min_periods=1).mean()
        new_cols[f"{prefix}_base_rmean28"] = out[base_col].rolling(28, min_periods=1).mean()

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out


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


def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols: list[str], target: str) -> list[str]:
    """Block opposite target and ratio/base columns from features."""
    if "Revenue" in target:
        blocked = ("COGS_", "cogs_", "cog_ratio", "cog_base")
    else:
        blocked = ("Revenue_", "rev_", "rev_ratio", "rev_base")
    # Also block raw ratio targets from being features
    blocked_exact = {"Revenue_ratio", "COGS_ratio", "Revenue_base", "COGS_base"}
    return [c for c in base_cols if not c.startswith(blocked) and c not in blocked_exact]


# ─────────────────────────────────────────────────────────────────────────────
# Recursive prediction for ratio model
# ─────────────────────────────────────────────────────────────────────────────

def recursive_predict_ratio(
    models: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,  # "Revenue" or "COGS"
) -> np.ndarray:
    """
    Recursive prediction via YoY ratio model.

    For each date t:
    1. Model predicts ratio = revenue_t / revenue_{t-365}
    2. Final prediction = ratio × revenue_{t-365} (from actual training data)
    """
    ratio_target = f"{target}_ratio"
    base_col = f"{target}_base"

    # History needs both actual values AND ratio/base columns
    history = history_df[["Date", target]].copy()

    # Add base (t-365) values
    history[base_col] = history[target].shift(365)
    history[ratio_target] = history[target] / history[base_col].replace(0, np.nan)

    preds: list[float] = []

    for date in predict_dates:
        ts = pd.Timestamp(date)

        # Find the base value: actual revenue from exactly 365 days ago
        base_date = ts - pd.Timedelta(days=365)
        base_matches = history[history["Date"] == base_date]
        if len(base_matches) > 0:
            base_val = float(base_matches[target].values[0])
        else:
            # Interpolate: find closest date within ±3 days
            nearby = history[
                (history["Date"] >= base_date - pd.Timedelta(days=3))
                & (history["Date"] <= base_date + pd.Timedelta(days=3))
            ]
            base_val = float(nearby[target].mean()) if len(nearby) > 0 else float(history[target].tail(365).mean())

        # Create row with unknown ratio
        row = pd.DataFrame({
            "Date": [ts],
            target: [np.nan],
            base_col: [base_val],
            ratio_target: [np.nan],
        })
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)

        # Add ratio-specific features
        combined = add_ratio_features(combined)

        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)

        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else val

        # Ensemble: predict the ratio
        raw_ratios = [float(m.predict(x_pred)[0]) for m in models]
        pred_ratio = float(np.mean(raw_ratios))

        # Clamp ratio to reasonable range (0.2 to 5.0)
        pred_ratio = np.clip(pred_ratio, 0.2, 5.0)

        # Convert ratio to absolute value
        pred_value = pred_ratio * base_val
        pred_value = max(0, pred_value)
        preds.append(pred_value)

        # Update history
        new_row = pd.DataFrame({
            "Date": [ts],
            target: [pred_value],
            base_col: [base_val],
            ratio_target: [pred_ratio],
        })
        history = pd.concat([history, new_row], ignore_index=True)

    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Also run the standard model for comparison / blending
# ─────────────────────────────────────────────────────────────────────────────

def recursive_predict_standard(
    models: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
) -> np.ndarray:
    """Standard recursive prediction (same as EX-31)."""
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

    # Add ratio targets and features
    feat_df = add_ratio_targets(feat_df)
    feat_df = add_ratio_features(feat_df)

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    # Only use rows with valid ratio (need 365d history)
    trn_ratio = trn.dropna(subset=["Revenue_ratio", "COGS_ratio"]).copy()
    print(f"  Ratio training rows: {len(trn_ratio)} / {len(trn)}")

    y_val_rev = val["Revenue"].values
    y_val_cogs = val["COGS"].values

    base_cols = get_feature_cols(feat_df)
    # Add ratio-specific feature columns
    ratio_feat_cols = [c for c in feat_df.columns if any(
        c.startswith(p) for p in ["rev_ratio", "cog_ratio", "rev_base", "cog_base"]
    )]
    all_cols = list(set(base_cols + ratio_feat_cols))

    cols_rev = _finalize_cols(trn_ratio, _get_target_cols(all_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn_ratio, _get_target_cols(all_cols, "COGS"))

    print(f"  Features: Rev={len(cols_rev)}, COGS={len(cols_cogs)}")

    # Train RATIO models
    models_rev_ratio = []
    models_cogs_ratio = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev_ratio.append(_fit_lgbm(
            trn_ratio[cols_rev].fillna(0), trn_ratio["Revenue_ratio"],
            trn_ratio[cols_rev].fillna(0).tail(365), trn_ratio["Revenue_ratio"].tail(365),
            seed=seed,
        ))
        models_cogs_ratio.append(_fit_lgbm(
            trn_ratio[cols_cogs].fillna(0), trn_ratio["COGS_ratio"],
            trn_ratio[cols_cogs].fillna(0).tail(365), trn_ratio["COGS_ratio"].tail(365),
            seed=seed,
        ))

    # Also train STANDARD models for blending
    std_cols_rev = _finalize_cols(trn, _get_target_cols(base_cols, "Revenue"))
    std_cols_cogs = _finalize_cols(trn, _get_target_cols(base_cols, "COGS"))

    models_rev_std = []
    models_cogs_std = []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev_std.append(_fit_lgbm(
            trn[std_cols_rev].fillna(0), trn["Revenue"],
            val[std_cols_rev].fillna(0), val["Revenue"], seed=seed,
        ))
        models_cogs_std.append(_fit_lgbm(
            trn[std_cols_cogs].fillna(0), trn["COGS"],
            val[std_cols_cogs].fillna(0), val["COGS"], seed=seed,
        ))

    # Predict with RATIO model
    pred_rev_ratio = recursive_predict_ratio(
        models_rev_ratio, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue",
    )
    pred_cogs_ratio = recursive_predict_ratio(
        models_cogs_ratio, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS",
    )

    # Predict with STANDARD model
    pred_rev_std = recursive_predict_standard(
        models_rev_std, trn[["Date", "Revenue"]], val["Date"].values,
        std_cols_rev, profiles, "Revenue",
    )
    pred_cogs_std = recursive_predict_standard(
        models_cogs_std, trn[["Date", "COGS"]], val["Date"].values,
        std_cols_cogs, profiles, "COGS",
    )

    # Evaluate different blends
    results = []

    # Pure ratio
    res_rev = evaluate(y_val_rev, pred_rev_ratio, f"{fold['name']} Rev RATIO")
    res_cogs = evaluate(y_val_cogs, pred_cogs_ratio, f"{fold['name']} COGS RATIO")
    score_ratio = float(res_rev["mae"] + 0.4 * res_cogs["mae"])
    results.append({"method": "ratio_pure", "score": score_ratio,
                     "rev_mae": res_rev["mae"], "rev_mean": pred_rev_ratio.mean()})

    # Pure standard
    res_rev = evaluate(y_val_rev, pred_rev_std, f"{fold['name']} Rev STD")
    res_cogs = evaluate(y_val_cogs, pred_cogs_std, f"{fold['name']} COGS STD")
    score_std = float(res_rev["mae"] + 0.4 * res_cogs["mae"])
    results.append({"method": "standard", "score": score_std,
                     "rev_mae": res_rev["mae"], "rev_mean": pred_rev_std.mean()})

    # Blended (50/50)
    for w_ratio in [0.3, 0.5, 0.7]:
        blend_rev = w_ratio * pred_rev_ratio + (1 - w_ratio) * pred_rev_std
        blend_cogs = w_ratio * pred_cogs_ratio + (1 - w_ratio) * pred_cogs_std
        res_rev = evaluate(y_val_rev, blend_rev, f"{fold['name']} Rev BLEND_{int(w_ratio*100)}")
        res_cogs = evaluate(y_val_cogs, blend_cogs, f"{fold['name']} COGS BLEND_{int(w_ratio*100)}")
        score_blend = float(res_rev["mae"] + 0.4 * res_cogs["mae"])
        results.append({"method": f"blend_{int(w_ratio*100)}", "score": score_blend,
                         "rev_mae": res_rev["mae"], "rev_mean": blend_rev.mean()})

    print(f"\n  {fold['name']} Summary:")
    for r in sorted(results, key=lambda x: x["score"]):
        print(f"    {r['method']:15s}  score={r['score']:,.0f}  rev_mae={r['rev_mae']:,.0f}  rev_mean={r['rev_mean']:,.0f}")

    best = min(results, key=lambda x: x["score"])
    return pd.DataFrame([{
        "fold": fold["name"],
        "best_method": best["method"],
        "ratio_score": score_ratio,
        "standard_score": score_std,
        "best_score": best["score"],
        "ratio_rev_mean": float(pred_rev_ratio.mean()),
        "std_rev_mean": float(pred_rev_std.mean()),
    }])


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_46: YoY Ratio Model — Predict Growth Rate, Not Level")
    print("  Target = revenue_t / revenue_{t-365}")
    print("  Trees interpolate growth ratios (within range) instead of")
    print("  extrapolating absolute levels (out of range)")
    print("=" * 78)

    fold_results = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df = evaluate_fold(train, fold)
        fold_results.append(s_df)

    fold_scores = pd.concat(fold_results, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\n\nFold Scores Summary:")
    print(fold_scores.to_string(index=False))
    print(f"\nMean ratio score: {fold_scores['ratio_score'].mean():,.0f}")
    print(f"Mean standard score: {fold_scores['standard_score'].mean():,.0f}")
    print(f"Mean best score: {fold_scores['best_score'].mean():,.0f}")

    # ── Submission with best method ──
    print("\n\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    feat_df = add_ratio_targets(feat_df)
    feat_df = add_ratio_features(feat_df)

    trn_ratio = feat_df.dropna(subset=["Revenue_ratio", "COGS_ratio"]).copy()

    base_cols = get_feature_cols(feat_df)
    ratio_feat_cols = [c for c in feat_df.columns if any(
        c.startswith(p) for p in ["rev_ratio", "cog_ratio", "rev_base", "cog_base"]
    )]
    all_cols = list(set(base_cols + ratio_feat_cols))

    cols_rev = _finalize_cols(trn_ratio, _get_target_cols(all_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn_ratio, _get_target_cols(all_cols, "COGS"))
    std_cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    std_cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    # Train both ratio and standard models
    final_ratio_rev, final_ratio_cogs = [], []
    final_std_rev, final_std_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_ratio_rev.append(_fit_lgbm(
            trn_ratio[cols_rev].fillna(0), trn_ratio["Revenue_ratio"],
            trn_ratio[cols_rev].fillna(0).tail(365), trn_ratio["Revenue_ratio"].tail(365),
            seed=seed,
        ))
        final_ratio_cogs.append(_fit_lgbm(
            trn_ratio[cols_cogs].fillna(0), trn_ratio["COGS_ratio"],
            trn_ratio[cols_cogs].fillna(0).tail(365), trn_ratio["COGS_ratio"].tail(365),
            seed=seed,
        ))
        final_std_rev.append(_fit_lgbm(
            feat_df[std_cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[std_cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            seed=seed,
        ))
        final_std_cogs.append(_fit_lgbm(
            feat_df[std_cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[std_cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed,
        ))

    print("\nRunning ratio inference on test set...")
    ratio_rev = recursive_predict_ratio(
        final_ratio_rev, train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue",
    )
    ratio_cogs = recursive_predict_ratio(
        final_ratio_cogs, train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS",
    )

    print("Running standard inference on test set...")
    std_rev = recursive_predict_standard(
        final_std_rev, train[["Date", "Revenue"]], test["Date"].values,
        std_cols_rev, profiles, "Revenue",
    )
    std_cogs = recursive_predict_standard(
        final_std_cogs, train[["Date", "COGS"]], test["Date"].values,
        std_cols_cogs, profiles, "COGS",
    )

    # Save all variants
    for name, rev, cogs in [
        ("ratio_pure", ratio_rev, ratio_cogs),
        ("standard", std_rev, std_cogs),
        ("blend_50", 0.5 * ratio_rev + 0.5 * std_rev, 0.5 * ratio_cogs + 0.5 * std_cogs),
        ("blend_30", 0.3 * ratio_rev + 0.7 * std_rev, 0.3 * ratio_cogs + 0.7 * std_cogs),
    ]:
        path = SUB_DIR / f"ex_46_{name}.csv"
        make_submission(test["Date"], rev, cogs, path)
        print(f"  {name}: rev_mean={rev.mean():,.0f}, cogs_mean={cogs.mean():,.0f}")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "mean_ratio_score": float(fold_scores["ratio_score"].mean()),
        "mean_standard_score": float(fold_scores["standard_score"].mean()),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
