"""
EX_39: Log-Transform with Smearing (Bias Correction)

The naive log-transform underpredicts because of Jensen's inequality:
  exp(E[log(X)]) < E[X]
  
Fix: "Duan's smearing estimator" — multiply expm1(prediction) by a
correction factor estimated from the training residuals:
  correction = mean(exp(residuals))
  
This restores the correct expected value on the original scale.

Pipeline:
1. Preprocess: log1p(Revenue), log1p(COGS)
2. Build all features on log-transformed data  
3. Train LightGBM on log-space targets
4. Recursive predict in log-space
5. Reverse: expm1(prediction) × smearing_factor
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

TRACK = Path("output/tracking/ex_39_log_smearing")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


def log_transform_sales(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Revenue"] = np.log1p(out["Revenue"].clip(lower=0))
    out["COGS"] = np.log1p(out["COGS"].clip(lower=0))
    return out


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
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def _compute_smearing_factor(models, X_trn, y_trn_log):
    """Duan's smearing estimator: mean(exp(residuals)) on training data.
    
    This corrects for Jensen's inequality when back-transforming from log-space.
    """
    # Average log-space residuals across ensemble
    residuals_all = []
    for model in models:
        pred_log = model.predict(X_trn)
        residuals = y_trn_log - pred_log  # log-space residuals
        residuals_all.append(residuals)
    
    mean_residuals = np.mean(residuals_all, axis=0)
    smearing = np.mean(np.exp(mean_residuals))
    return smearing


def recursive_predict(
    models, history_df, predict_dates, feature_cols, profiles, target,
):
    """Recursive prediction in log-space."""
    history = history_df[["Date", target]].copy()
    preds = []

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
        preds.append(pred)

        # Feed log-space prediction back
        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


def evaluate_fold(sales_log, sales_raw, fold):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice_log = sales_log[sales_log["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(
        sales_log, verbose=False, profile_source_df=train_slice_log
    )

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    val_raw = sales_raw[
        (sales_raw["Date"] >= val_start) & (sales_raw["Date"] <= val_end)
    ]
    y_val_rev_raw = val_raw["Revenue"].values
    y_val_cogs_raw = val_raw["COGS"].values

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_target_cols(base_cols, "COGS"))

    X_trn_rev = trn[cols_rev].fillna(0)
    X_trn_cogs = trn[cols_cogs].fillna(0)

    models_rev, models_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            X_trn_rev, trn["Revenue"], val[cols_rev].fillna(0), val["Revenue"], seed=seed,
        ))
        models_cogs.append(_fit_lgbm(
            X_trn_cogs, trn["COGS"], val[cols_cogs].fillna(0), val["COGS"], seed=seed,
        ))

    # Compute smearing factor from training residuals
    smear_rev = _compute_smearing_factor(models_rev, X_trn_rev, trn["Revenue"].values)
    smear_cogs = _compute_smearing_factor(models_cogs, X_trn_cogs, trn["COGS"].values)
    print(f"  Smearing factors: Rev={smear_rev:.4f}  COGS={smear_cogs:.4f}")

    # Recursive predict in log-space
    pred_rev_log = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val_raw["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    pred_cogs_log = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val_raw["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # Inverse transform WITH smearing correction
    pred_rev = np.expm1(pred_rev_log) * smear_rev
    pred_cogs = np.expm1(pred_cogs_log) * smear_cogs
    pred_rev = np.clip(pred_rev, 0, None)
    pred_cogs = np.clip(pred_cogs, 0, None)

    # Also compute without smearing for comparison
    pred_rev_naive = np.expm1(pred_rev_log)
    pred_cogs_naive = np.expm1(pred_cogs_log)

    res_rev = evaluate(y_val_rev_raw, pred_rev, f"{fold['name']} Revenue (smeared)")
    res_cogs = evaluate(y_val_cogs_raw, pred_cogs, f"{fold['name']} COGS (smeared)")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    res_rev_naive = evaluate(y_val_rev_raw, pred_rev_naive, f"{fold['name']} Revenue (naive)")
    res_cogs_naive = evaluate(y_val_cogs_raw, pred_cogs_naive, f"{fold['name']} COGS (naive)")
    score_naive = float(res_rev_naive["mae"] + 0.4 * res_cogs_naive["mae"])

    bias_rev = pred_rev.mean() / y_val_rev_raw.mean() - 1
    bias_cogs = pred_cogs.mean() / y_val_cogs_raw.mean() - 1
    print(f"  Score: smeared={score:,.0f}  naive={score_naive:,.0f}")
    print(f"  Bias: Rev={bias_rev:+.1%}  COGS={bias_cogs:+.1%}")

    return pd.DataFrame([{
        "fold": fold["name"],
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "smear_rev": smear_rev,
        "smear_cogs": smear_cogs,
        "revenue_mae": float(res_rev["mae"]),
        "cogs_mae": float(res_cogs["mae"]),
        "score": score,
        "revenue_mae_naive": float(res_rev_naive["mae"]),
        "cogs_mae_naive": float(res_cogs_naive["mae"]),
        "score_naive": score_naive,
        "bias_rev_pct": float(bias_rev * 100),
        "bias_cogs_pct": float(bias_cogs * 100),
    }]), smear_rev, smear_cogs


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test = load_sales()
    train_log = log_transform_sales(train_raw)

    print("Preprocessing: log1p transform applied")
    print(f"  Raw Revenue range: [{train_raw['Revenue'].min():,.0f}, {train_raw['Revenue'].max():,.0f}]")
    print(f"  Log Revenue range: [{train_log['Revenue'].min():.3f}, {train_log['Revenue'].max():.3f}]")

    print("\n" + "=" * 78)
    print("EX_39: Log-Transform with Smearing Correction")
    print("=" * 78)

    fold_parts = []
    smear_revs, smear_cogss = [], []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ===")
        s_df, sr, sc = evaluate_fold(train_log, train_raw, fold)
        fold_parts.append(s_df)
        smear_revs.append(sr)
        smear_cogss.append(sc)

    fold_scores = pd.concat(fold_parts, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(fold_scores[["fold", "score", "score_naive", "smear_rev", "smear_cogs", "bias_rev_pct"]].to_string(index=False))
    print(f"\nMean score: {fold_scores['score'].mean():,.0f} (naive: {fold_scores['score_naive'].mean():,.0f})")

    # Use average smearing factors for test
    avg_smear_rev = np.mean(smear_revs)
    avg_smear_cogs = np.mean(smear_cogss)
    print(f"\nAvg smearing factors: Rev={avg_smear_rev:.4f}  COGS={avg_smear_cogs:.4f}")

    # Train final model
    print("\nTraining final model...")
    feat_df, profiles = build_feature_table(train_log, verbose=True, profile_source_df=train_log)
    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    X_full_rev = feat_df[cols_rev].fillna(0)
    X_full_cogs = feat_df[cols_cogs].fillna(0)

    final_models_rev, final_models_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_models_rev.append(_fit_lgbm(
            X_full_rev, feat_df["Revenue"],
            X_full_rev.tail(365), feat_df["Revenue"].tail(365), seed=seed,
        ))
        final_models_cogs.append(_fit_lgbm(
            X_full_cogs, feat_df["COGS"],
            X_full_cogs.tail(365), feat_df["COGS"].tail(365), seed=seed,
        ))

    # Final smearing from full training data
    smear_rev_final = _compute_smearing_factor(final_models_rev, X_full_rev, feat_df["Revenue"].values)
    smear_cogs_final = _compute_smearing_factor(final_models_cogs, X_full_cogs, feat_df["COGS"].values)
    print(f"  Final smearing factors: Rev={smear_rev_final:.4f}  COGS={smear_cogs_final:.4f}")

    # Recursive inference
    print("Running recursive inference (log-space)...")
    pred_rev_log = recursive_predict(
        final_models_rev, train_log[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    pred_cogs_log = recursive_predict(
        final_models_cogs, train_log[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    # Reverse transform with smearing
    final_rev = np.expm1(pred_rev_log) * smear_rev_final
    final_cogs = np.expm1(pred_cogs_log) * smear_cogs_final
    final_rev = np.clip(final_rev, 0, None)
    final_cogs = np.clip(final_cogs, 0, None)

    print(f"\nRaw-space predictions:")
    print(f"  Revenue: mean={final_rev.mean():,.0f}")
    print(f"  COGS: mean={final_cogs.mean():,.0f}")

    try:
        sub31 = pd.read_csv(SUB_DIR / "ex_31_refined_ensemble.csv")
        print(f"\nvs EX-31: Rev {final_rev.mean() / sub31['Revenue'].mean() - 1:+.1%}  COGS {final_cogs.mean() / sub31['COGS'].mean() - 1:+.1%}")
    except Exception:
        pass

    candidate_path = SUB_DIR / "ex_39_log_smearing.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
        "n_seeds": N_SEEDS,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "mean_score": float(fold_scores["score"].mean()),
        "smear_rev": float(smear_rev_final),
        "smear_cogs": float(smear_cogs_final),
        "transform": "log1p_with_duan_smearing",
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
