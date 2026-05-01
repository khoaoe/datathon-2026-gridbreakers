"""
EX_41: Dedicated Tweedie + Regression Blend

EX-40 selected regression because it narrowly beat Tweedie on CV (896,733 vs 953,832).
But Tweedie has a log-link that captures multiplicative growth differently.

Strategy:
1. Tune Tweedie variance_power (1.0=Poisson, 1.5=default, 2.0=Gamma)
2. Blend Tweedie + regression predictions (model diversity)
3. The blend should capture both additive (regression) and multiplicative (Tweedie) patterns
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

TRACK = Path("output/tracking/ex_41_tweedie_blend")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED, objective="regression",
              tweedie_power=1.5):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500
    params["random_state"] = seed
    params["objective"] = objective

    if objective == "tweedie":
        params["tweedie_variance_power"] = tweedie_power
        params["metric"] = "tweedie"

    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _finalize_cols(df, cols):
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def recursive_predict(models, history_df, predict_dates, feature_cols, profiles, target):
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
        pred = max(0, float(np.mean(raw_preds)))
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


def recursive_predict_blend(
    models_reg, models_tweedie, history_df, predict_dates,
    feature_cols, profiles, target, blend_weight=0.5,
):
    """Recursive prediction blending regression and Tweedie models.
    
    At each step, both model types predict, then we blend:
      pred = (1 - w) * regression_pred + w * tweedie_pred
    and feed the blended prediction back into history.
    """
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

        reg_preds = [float(m.predict(x_pred)[0]) for m in models_reg]
        tw_preds = [float(m.predict(x_pred)[0]) for m in models_tweedie]
        
        reg_mean = float(np.mean(reg_preds))
        tw_mean = float(np.mean(tw_preds))
        
        pred = (1 - blend_weight) * reg_mean + blend_weight * tw_mean
        pred = max(0, pred)
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


def evaluate_fold(sales, fold, tweedie_power=1.5):
    """Evaluate regression, tweedie, and blends on one fold."""
    from sklearn.metrics import mean_absolute_error
    
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

    X_trn_rev = trn[cols_rev].fillna(0)
    X_val_rev = val[cols_rev].fillna(0)
    X_trn_cogs = trn[cols_cogs].fillna(0)
    X_val_cogs = val[cols_cogs].fillna(0)

    # Train regression models
    reg_rev, reg_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        reg_rev.append(_fit_lgbm(X_trn_rev, trn["Revenue"], X_val_rev, y_val_rev, seed, "regression"))
        reg_cogs.append(_fit_lgbm(X_trn_cogs, trn["COGS"], X_val_cogs, y_val_cogs, seed, "regression"))

    # Train Tweedie models
    tw_rev, tw_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        tw_rev.append(_fit_lgbm(X_trn_rev, trn["Revenue"], X_val_rev, y_val_rev, seed, "tweedie", tweedie_power))
        tw_cogs.append(_fit_lgbm(X_trn_cogs, trn["COGS"], X_val_cogs, y_val_cogs, seed, "tweedie", tweedie_power))

    results = []

    # Pure regression
    pred_rev_reg = recursive_predict(reg_rev, trn[["Date","Revenue"]], val["Date"].values, cols_rev, profiles, "Revenue")
    pred_cogs_reg = recursive_predict(reg_cogs, trn[["Date","COGS"]], val["Date"].values, cols_cogs, profiles, "COGS")
    
    r_rev = evaluate(y_val_rev, pred_rev_reg, f"{fold['name']} Revenue (reg)")
    r_cogs = evaluate(y_val_cogs, pred_cogs_reg, f"{fold['name']} COGS (reg)")
    score_reg = float(r_rev["mae"] + 0.4 * r_cogs["mae"])
    bias_reg = pred_rev_reg.mean() / y_val_rev.mean() - 1
    results.append({"type": "regression", "score": score_reg, "bias_rev": bias_reg,
                     "rev_mean": pred_rev_reg.mean(), "cogs_mean": pred_cogs_reg.mean()})

    # Pure Tweedie
    pred_rev_tw = recursive_predict(tw_rev, trn[["Date","Revenue"]], val["Date"].values, cols_rev, profiles, "Revenue")
    pred_cogs_tw = recursive_predict(tw_cogs, trn[["Date","COGS"]], val["Date"].values, cols_cogs, profiles, "COGS")
    
    r_rev = evaluate(y_val_rev, pred_rev_tw, f"{fold['name']} Revenue (tw)")
    r_cogs = evaluate(y_val_cogs, pred_cogs_tw, f"{fold['name']} COGS (tw)")
    score_tw = float(r_rev["mae"] + 0.4 * r_cogs["mae"])
    bias_tw = pred_rev_tw.mean() / y_val_rev.mean() - 1
    results.append({"type": f"tweedie_p{tweedie_power}", "score": score_tw, "bias_rev": bias_tw,
                     "rev_mean": pred_rev_tw.mean(), "cogs_mean": pred_cogs_tw.mean()})

    # Blends at different weights
    for w in [0.2, 0.3, 0.4, 0.5]:
        pred_rev_blend = recursive_predict_blend(
            reg_rev, tw_rev, trn[["Date","Revenue"]], val["Date"].values,
            cols_rev, profiles, "Revenue", w
        )
        pred_cogs_blend = recursive_predict_blend(
            reg_cogs, tw_cogs, trn[["Date","COGS"]], val["Date"].values,
            cols_cogs, profiles, "COGS", w
        )
        
        r_rev = evaluate(y_val_rev, pred_rev_blend, f"{fold['name']} Revenue (blend w={w})")
        r_cogs = evaluate(y_val_cogs, pred_cogs_blend, f"{fold['name']} COGS (blend w={w})")
        score_blend = float(r_rev["mae"] + 0.4 * r_cogs["mae"])
        bias_blend = pred_rev_blend.mean() / y_val_rev.mean() - 1
        results.append({"type": f"blend_w{w}", "score": score_blend, "bias_rev": bias_blend,
                         "rev_mean": pred_rev_blend.mean(), "cogs_mean": pred_cogs_blend.mean()})

    # Print summary
    print(f"\n  {'Type':20s}  {'Score':>10s}  {'Rev Bias':>10s}  {'Rev Mean':>12s}")
    for r in results:
        print(f"  {r['type']:20s}  {r['score']:>10,.0f}  {r['bias_rev']:>+10.1%}  {r['rev_mean']:>12,.0f}")

    return pd.DataFrame([{
        "fold": fold["name"],
        **{f"{r['type']}_score": r["score"] for r in results},
        **{f"{r['type']}_bias": r["bias_rev"] for r in results},
    }]), results


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_41: Tweedie + Regression Blend")
    print("=" * 78)

    # Test different Tweedie variance powers
    best_power = 1.5
    best_score = float("inf")

    for power in [1.1, 1.3, 1.5, 1.7, 1.9]:
        print(f"\n{'='*60}")
        print(f"Tweedie variance_power = {power}")
        print(f"{'='*60}")

        fold_results_all = []
        for fold in FOLDS:
            print(f"\n--- {fold['name']} ---")
            _, results = evaluate_fold(train, fold, tweedie_power=power)
            fold_results_all.append(results)

        # Average scores across folds
        type_scores = {}
        for fold_results in fold_results_all:
            for r in fold_results:
                if r["type"] not in type_scores:
                    type_scores[r["type"]] = []
                type_scores[r["type"]].append(r["score"])

        print(f"\n  Mean scores for power={power}:")
        for t, scores in type_scores.items():
            mean_s = np.mean(scores)
            print(f"    {t:20s}: {mean_s:>10,.0f}")
            if "tweedie" in t and mean_s < best_score:
                best_score = mean_s
                best_power = power

    print(f"\nBest Tweedie power: {best_power} (score={best_score:,.0f})")

    # Now find the best blend weight using the best power
    print(f"\n{'='*60}")
    print(f"Final evaluation with power={best_power}")
    print(f"{'='*60}")

    all_fold_results = []
    for fold in FOLDS:
        print(f"\n--- {fold['name']} ---")
        _, results = evaluate_fold(train, fold, tweedie_power=best_power)
        all_fold_results.append(results)

    # Find best overall configuration
    type_scores = {}
    type_biases = {}
    for fold_results in all_fold_results:
        for r in fold_results:
            if r["type"] not in type_scores:
                type_scores[r["type"]] = []
                type_biases[r["type"]] = []
            type_scores[r["type"]].append(r["score"])
            type_biases[r["type"]].append(r["bias_rev"])

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    best_type = None
    best_overall = float("inf")
    for t in type_scores:
        mean_s = np.mean(type_scores[t])
        mean_b = np.mean(type_biases[t])
        print(f"  {t:20s}: Score={mean_s:>10,.0f}  Rev Bias={mean_b:+.1%}")
        if mean_s < best_overall:
            best_overall = mean_s
            best_type = t

    print(f"\nBest: {best_type} (score={best_overall:,.0f})")

    # Determine final config
    if "blend" in best_type:
        blend_w = float(best_type.split("w")[1])
    else:
        blend_w = 0.0 if best_type == "regression" else 1.0

    # Train final model
    print(f"\nFinal training (blend_weight={blend_w}, tweedie_power={best_power})...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    X_full_rev = feat_df[cols_rev].fillna(0)
    X_full_cogs = feat_df[cols_cogs].fillna(0)
    X_last = 365

    final_reg_rev, final_reg_cogs = [], []
    final_tw_rev, final_tw_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_reg_rev.append(_fit_lgbm(
            X_full_rev, feat_df["Revenue"], X_full_rev.tail(X_last),
            feat_df["Revenue"].tail(X_last), seed, "regression"))
        final_reg_cogs.append(_fit_lgbm(
            X_full_cogs, feat_df["COGS"], X_full_cogs.tail(X_last),
            feat_df["COGS"].tail(X_last), seed, "regression"))
        final_tw_rev.append(_fit_lgbm(
            X_full_rev, feat_df["Revenue"], X_full_rev.tail(X_last),
            feat_df["Revenue"].tail(X_last), seed, "tweedie", best_power))
        final_tw_cogs.append(_fit_lgbm(
            X_full_cogs, feat_df["COGS"], X_full_cogs.tail(X_last),
            feat_df["COGS"].tail(X_last), seed, "tweedie", best_power))

    print("Recursive inference...")
    if blend_w == 0.0:
        final_rev = recursive_predict(final_reg_rev, train[["Date","Revenue"]], test["Date"].values, cols_rev, profiles, "Revenue")
        final_cogs = recursive_predict(final_reg_cogs, train[["Date","COGS"]], test["Date"].values, cols_cogs, profiles, "COGS")
    elif blend_w == 1.0:
        final_rev = recursive_predict(final_tw_rev, train[["Date","Revenue"]], test["Date"].values, cols_rev, profiles, "Revenue")
        final_cogs = recursive_predict(final_tw_cogs, train[["Date","COGS"]], test["Date"].values, cols_cogs, profiles, "COGS")
    else:
        final_rev = recursive_predict_blend(
            final_reg_rev, final_tw_rev, train[["Date","Revenue"]],
            test["Date"].values, cols_rev, profiles, "Revenue", blend_w)
        final_cogs = recursive_predict_blend(
            final_reg_cogs, final_tw_cogs, train[["Date","COGS"]],
            test["Date"].values, cols_cogs, profiles, "COGS", blend_w)

    print(f"\nRevenue: mean={final_rev.mean():,.0f}")
    print(f"COGS: mean={final_cogs.mean():,.0f}")

    try:
        sub31 = pd.read_csv(SUB_DIR / "ex_31_refined_ensemble.csv")
        print(f"vs EX-31: Rev {final_rev.mean()/sub31['Revenue'].mean()-1:+.1%}  COGS {final_cogs.mean()/sub31['COGS'].mean()-1:+.1%}")
    except Exception:
        pass

    candidate_path = SUB_DIR / "ex_41_tweedie_blend.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    # Also save pure tweedie for comparison
    if blend_w != 1.0:
        tw_rev = recursive_predict(final_tw_rev, train[["Date","Revenue"]], test["Date"].values, cols_rev, profiles, "Revenue")
        tw_cogs = recursive_predict(final_tw_cogs, train[["Date","COGS"]], test["Date"].values, cols_cogs, profiles, "COGS")
        make_submission(test["Date"], tw_rev, tw_cogs, SUB_DIR / "ex_41_tweedie_pure.csv")
        print(f"Pure Tweedie: Rev={tw_rev.mean():,.0f}  COGS={tw_cogs.mean():,.0f}")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "best_type": best_type,
        "best_power": best_power,
        "blend_weight": blend_w,
        "mean_score": float(best_overall),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
