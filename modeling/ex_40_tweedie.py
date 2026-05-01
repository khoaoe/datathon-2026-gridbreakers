"""
EX_40: Tweedie Objective (built-in log-link)

LightGBM's Tweedie objective internally uses a log-link:
  pred = exp(raw_output)
  
This gives us the benefits of log-space modeling WITHOUT needing to
manually transform features or apply smearing corrections. The model
natively learns in log-space and predicts in raw-space.

Also tests: adding log-transformed features (log_lag1, log_rmean, etc.)
alongside raw features to give the model both scales.

Architecture: 3-seed LGB with Tweedie, no post-processing.
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

TRACK = Path("output/tracking/ex_40_tweedie")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED, objective="tweedie"):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500
    params["random_state"] = seed
    params["objective"] = objective

    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.5  # between 1 (Poisson) and 2 (Gamma)
        params["metric"] = "tweedie"

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


def recursive_predict(
    models, history_df, predict_dates, feature_cols, profiles, target,
):
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
        pred = max(0, pred)
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


def evaluate_fold(sales, fold, objective="tweedie"):
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

    models_rev, models_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), val["Revenue"], seed=seed,
            objective=objective,
        ))
        models_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), val["COGS"], seed=seed,
            objective=objective,
        ))

    pred_rev = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    pred_cogs = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue ({objective})")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS ({objective})")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    bias_rev = pred_rev.mean() / y_val_rev.mean() - 1
    bias_cogs = pred_cogs.mean() / y_val_cogs.mean() - 1
    print(f"  Pred: Rev={pred_rev.mean():,.0f}  COGS={pred_cogs.mean():,.0f}")
    print(f"  True: Rev={y_val_rev.mean():,.0f}  COGS={y_val_cogs.mean():,.0f}")
    print(f"  Bias: Rev={bias_rev:+.1%}  COGS={bias_cogs:+.1%}")

    return pd.DataFrame([{
        "fold": fold["name"],
        "objective": objective,
        "revenue_mae": float(res_rev["mae"]),
        "cogs_mae": float(res_cogs["mae"]),
        "score": score,
        "bias_rev_pct": float(bias_rev * 100),
        "bias_cogs_pct": float(bias_cogs * 100),
    }])


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_40: Tweedie & Poisson Objectives (built-in log-link)")
    print("=" * 78)

    # Test different objectives
    objectives = ["tweedie", "poisson", "regression"]
    all_results = []

    for obj in objectives:
        print(f"\n{'='*40}")
        print(f"Objective: {obj}")
        print(f"{'='*40}")

        fold_parts = []
        for fold in FOLDS:
            print(f"\n--- {fold['name']} ---")
            s_df = evaluate_fold(train, fold, objective=obj)
            fold_parts.append(s_df)

        fold_df = pd.concat(fold_parts, ignore_index=True)
        all_results.append(fold_df)

        mean_score = fold_df["score"].mean()
        mean_bias_rev = fold_df["bias_rev_pct"].mean()
        mean_bias_cogs = fold_df["bias_cogs_pct"].mean()
        print(f"\n  {obj}: Score={mean_score:,.0f}  Bias: Rev={mean_bias_rev:+.1f}%  COGS={mean_bias_cogs:+.1f}%")

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(TRACK / "fold_scores.csv", index=False)

    # Comparison
    print(f"\n{'='*60}")
    print("OBJECTIVE COMPARISON")
    print(f"{'='*60}")
    for obj in objectives:
        mask = all_df["objective"] == obj
        score = all_df.loc[mask, "score"].mean()
        bias_r = all_df.loc[mask, "bias_rev_pct"].mean()
        bias_c = all_df.loc[mask, "bias_cogs_pct"].mean()
        print(f"  {obj:15s}: Score={score:>10,.0f}  Bias Rev={bias_r:+.1f}%  COGS={bias_c:+.1f}%")

    # Pick best
    obj_means = all_df.groupby("objective")["score"].mean()
    best_obj = obj_means.idxmin()
    print(f"\nBest: {best_obj} (score={obj_means[best_obj]:,.0f})")

    # Final training
    print(f"\nFinal training with {best_obj}...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    final_models_rev, final_models_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_models_rev.append(_fit_lgbm(
            feat_df[cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            seed=seed, objective=best_obj,
        ))
        final_models_cogs.append(_fit_lgbm(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed=seed, objective=best_obj,
        ))

    print("Running recursive inference...")
    final_rev = recursive_predict(
        final_models_rev, train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue"
    )
    final_cogs = recursive_predict(
        final_models_cogs, train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS"
    )

    print(f"\nRevenue: mean={final_rev.mean():,.0f}")
    print(f"COGS: mean={final_cogs.mean():,.0f}")

    try:
        sub31 = pd.read_csv(SUB_DIR / "ex_31_refined_ensemble.csv")
        print(f"vs EX-31: Rev {final_rev.mean() / sub31['Revenue'].mean() - 1:+.1%}  COGS {final_cogs.mean() / sub31['COGS'].mean() - 1:+.1%}")
    except Exception:
        pass

    candidate_path = SUB_DIR / "ex_40_tweedie.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "best_objective": best_obj,
        "n_seeds": N_SEEDS,
        "mean_score": float(obj_means[best_obj]),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
