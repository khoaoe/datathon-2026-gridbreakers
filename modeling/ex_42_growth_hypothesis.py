"""
EX_42: Hypothesis-Driven Growth Correction

ROOT CAUSE ANALYSIS:
- Training data has TWO regimes: pre-2019 (~5M mean) and post-2019 (~3M mean)
- The model is trained on 62% pre-2019 data, 38% post-2019 data
- The test period (2023-2024) appears to be in the 3.7-4.2M range (recovery)
- The model predicts 3.74M — close but slightly low

HYPOTHESES TESTED:
H1: Training on 2019+ only removes regime confusion → better extrapolation
H2: Adding a linear time trend feature lets the model learn the growth trajectory
H3: Exponential sample weighting focuses the model on the recovery era
H4: Combining H2+H3 gives the best of both worlds
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

TRACK = Path("output/tracking/ex_42_growth_hypothesis")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED, sample_weight=None):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500
    params["random_state"] = seed

    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
        sample_weight=sample_weight,
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


def add_time_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Add linear time trend features.
    
    These let the model learn that revenue is increasing over time,
    rather than anchoring to the static average of all training data.
    """
    out = df.copy()
    # Days since a reference point (start of recovery era)
    ref = pd.Timestamp("2019-01-01")
    days = (out["Date"] - ref).dt.days.astype(float)
    out["time_trend"] = days
    out["time_trend_yr"] = days / 365.25  # in years, easier for model
    out["time_trend_sq"] = (days / 365.25) ** 2  # quadratic for acceleration
    return out


def compute_sample_weights(dates: pd.Series, halflife_days: int = 365) -> np.ndarray:
    """Exponential decay weighting — recent data gets higher weight.
    
    halflife_days: number of days for weight to halve.
    The most recent day gets weight 1.0, older days decay exponentially.
    """
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.astype(float)
    weights = np.exp(-np.log(2) * days_ago / halflife_days)
    return weights.values


def recursive_predict(models, history_df, predict_dates, feature_cols,
                      profiles, target, add_trend=False):
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
        
        if add_trend:
            combined = add_time_trend(combined)

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


def evaluate_hypothesis(sales, fold, hypothesis, add_trend=False,
                        train_from=None, halflife=None):
    """Evaluate a single hypothesis on one fold."""
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    # H1: restrict training data
    if train_from:
        sales_filtered = sales[sales["Date"] >= pd.Timestamp(train_from)].copy()
    else:
        sales_filtered = sales.copy()

    train_slice = sales_filtered[sales_filtered["Date"] < val_start].copy()
    
    # Need enough data for features
    if len(train_slice) < 400:
        print(f"  SKIP: only {len(train_slice)} training rows")
        return None

    feat_df, profiles = build_feature_table(
        sales_filtered, verbose=False, profile_source_df=train_slice
    )

    # H2: add time trend features
    if add_trend:
        feat_df = add_time_trend(feat_df)

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    y_val_rev = val["Revenue"].values
    y_val_cogs = val["COGS"].values

    base_cols = get_feature_cols(feat_df)
    # Include time trend columns if added
    if add_trend:
        base_cols = list(set(base_cols) | {"time_trend", "time_trend_yr", "time_trend_sq"})

    cols_rev = _finalize_cols(trn, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_target_cols(base_cols, "COGS"))

    # H3: sample weights
    weights = None
    if halflife:
        weights = compute_sample_weights(trn["Date"], halflife_days=halflife)

    models_rev, models_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), y_val_rev, seed, sample_weight=weights,
        ))
        models_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), y_val_cogs, seed, sample_weight=weights,
        ))

    pred_rev = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue", add_trend=add_trend,
    )
    pred_cogs = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS", add_trend=add_trend,
    )

    r_rev = evaluate(y_val_rev, pred_rev, f"  {hypothesis} Rev")
    r_cogs = evaluate(y_val_cogs, pred_cogs, f"  {hypothesis} COGS")
    score = float(r_rev["mae"] + 0.4 * r_cogs["mae"])
    bias_rev = pred_rev.mean() / y_val_rev.mean() - 1
    bias_cogs = pred_cogs.mean() / y_val_cogs.mean() - 1

    print(f"  → Score={score:,.0f}  Bias: Rev={bias_rev:+.1%}  COGS={bias_cogs:+.1%}")

    return {
        "fold": fold["name"],
        "hypothesis": hypothesis,
        "score": score,
        "revenue_mae": float(r_rev["mae"]),
        "cogs_mae": float(r_cogs["mae"]),
        "bias_rev_pct": float(bias_rev * 100),
        "bias_cogs_pct": float(bias_cogs * 100),
        "n_train": len(trn),
    }


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_42: Hypothesis-Driven Growth Correction")
    print("=" * 78)

    hypotheses = [
        ("H0_baseline", {"add_trend": False, "train_from": None, "halflife": None}),
        ("H1_train_2017plus", {"add_trend": False, "train_from": "2017-01-01", "halflife": None}),
        ("H2_time_trend", {"add_trend": True, "train_from": None, "halflife": None}),
        ("H3_weight_hl365", {"add_trend": False, "train_from": None, "halflife": 365}),
        ("H3_weight_hl730", {"add_trend": False, "train_from": None, "halflife": 730}),
        ("H4_trend_plus_weight", {"add_trend": True, "train_from": None, "halflife": 730}),
    ]

    all_results = []

    for hyp_name, kwargs in hypotheses:
        print(f"\n{'='*60}")
        print(f"Testing: {hyp_name}")
        print(f"  Config: {kwargs}")
        print(f"{'='*60}")

        for fold in FOLDS:
            print(f"\n--- {fold['name']} ---")
            result = evaluate_hypothesis(train, fold, hyp_name, **kwargs)
            if result:
                all_results.append(result)

    # Summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TRACK / "fold_scores.csv", index=False)

    print(f"\n{'='*78}")
    print("HYPOTHESIS COMPARISON")
    print(f"{'='*78}")

    summary = results_df.groupby("hypothesis").agg({
        "score": "mean",
        "bias_rev_pct": "mean",
        "bias_cogs_pct": "mean",
        "n_train": "mean",
    }).sort_values("score")

    for hyp, row in summary.iterrows():
        print(f"  {hyp:25s}: Score={row['score']:>10,.0f}  Rev Bias={row['bias_rev_pct']:+.1f}%  "
              f"COGS Bias={row['bias_cogs_pct']:+.1f}%  n_train={row['n_train']:.0f}")

    best_hyp = summary.index[0]
    print(f"\nBest hypothesis: {best_hyp} (score={summary.loc[best_hyp, 'score']:,.0f})")

    # Determine config for best hypothesis
    best_config = dict(hypotheses)[best_hyp]
    print(f"Config: {best_config}")

    # Train final model with best hypothesis
    print(f"\nFinal training with {best_hyp}...")

    if best_config.get("train_from"):
        train_filtered = train[train["Date"] >= pd.Timestamp(best_config["train_from"])].copy()
    else:
        train_filtered = train.copy()

    feat_df, profiles = build_feature_table(train_filtered, verbose=True, profile_source_df=train_filtered)

    if best_config.get("add_trend"):
        feat_df = add_time_trend(feat_df)

    base_cols = get_feature_cols(feat_df)
    if best_config.get("add_trend"):
        base_cols = list(set(base_cols) | {"time_trend", "time_trend_yr", "time_trend_sq"})

    cols_rev = _finalize_cols(feat_df, _get_target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(base_cols, "COGS"))

    weights = None
    if best_config.get("halflife"):
        weights = compute_sample_weights(feat_df["Date"], halflife_days=best_config["halflife"])

    final_rev, final_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        final_rev.append(_fit_lgbm(
            feat_df[cols_rev].fillna(0), feat_df["Revenue"],
            feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365),
            seed, sample_weight=weights,
        ))
        final_cogs.append(_fit_lgbm(
            feat_df[cols_cogs].fillna(0), feat_df["COGS"],
            feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365),
            seed, sample_weight=weights,
        ))

    print("Recursive inference...")
    pred_rev = recursive_predict(
        final_rev, train_filtered[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue", add_trend=best_config.get("add_trend", False),
    )
    pred_cogs = recursive_predict(
        final_cogs, train_filtered[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS", add_trend=best_config.get("add_trend", False),
    )

    print(f"\nRevenue: mean={pred_rev.mean():,.0f}")
    print(f"COGS: mean={pred_cogs.mean():,.0f}")

    try:
        sub31 = pd.read_csv(SUB_DIR / "ex_31_refined_ensemble.csv")
        diff = pred_rev.mean() / sub31["Revenue"].mean() - 1
        print(f"vs EX-31: {diff:+.1%}")
    except Exception:
        pass

    candidate_path = SUB_DIR / "ex_42_growth_hypothesis.csv"
    make_submission(test["Date"], pred_rev, pred_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "best_hypothesis": best_hyp,
        "best_config": best_config,
        "mean_score": float(summary.loc[best_hyp, "score"]),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
