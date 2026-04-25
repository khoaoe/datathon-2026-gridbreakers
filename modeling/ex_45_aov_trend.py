"""
EX_45: AOV Trend + Daily Auxiliary Features

Hypothesis:
- Revenue = n_orders × avg_order_value (AOV)
- n_orders dropped 225→99 (2016→2022) but AOV grew 22k→32k (+47%)
- The recovery trajectory is DRIVEN by rising AOV
- By providing AOV trend and daily order/traffic features as lags,
  we give the tree model a usable growth signal

Key design:
1. Merge daily order counts & web traffic directly into training data
2. Build lag features on these auxiliary series (same as revenue lags)
3. For test: recursively predict using lagged aux features from training tail
4. AOV trend feature: computed from rolling revenue/orders ratio
5. Multi-seed ensemble (same as EX-31)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from modeling.config import LGBM_PARAMS, SEED, FILES
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

TRACK = Path("output/tracking/ex_45_aov_trend")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

N_SEEDS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary data: daily order stats + web traffic
# ─────────────────────────────────────────────────────────────────────────────

def load_daily_aux() -> pd.DataFrame:
    """Load daily order counts and web traffic, merge on Date."""
    # Daily orders
    orders = pd.read_csv(
        FILES["orders"],
        parse_dates=["order_date"],
        usecols=["order_id", "order_date"],
    )
    daily_orders = (
        orders.groupby("order_date")
        .agg(n_orders=("order_id", "count"))
        .reset_index()
    )
    daily_orders.columns = ["Date", "n_orders"]

    # Daily web traffic
    wt = pd.read_csv(FILES["web_traffic"], parse_dates=["date"])
    daily_wt = (
        wt.groupby("date")
        .agg(
            wt_sessions=("sessions", "sum"),
            wt_visitors=("unique_visitors", "sum"),
            wt_pageviews=("page_views", "sum"),
        )
        .reset_index()
    )
    daily_wt.columns = ["Date"] + list(daily_wt.columns[1:])

    # Merge
    aux = daily_orders.merge(daily_wt, on="Date", how="outer").sort_values("Date")
    return aux


def add_aux_features(df: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    """Merge daily aux data and build lag/rolling features from it."""
    out = df.merge(aux, on="Date", how="left").copy()

    # Compute AOV (revenue per order) — key growth signal
    out["aov"] = out["Revenue"] / out["n_orders"].replace(0, np.nan)

    # Build features using pd.concat to avoid fragmentation
    new_cols = {}
    for col in ["n_orders", "wt_sessions", "aov"]:
        if col not in out.columns:
            continue
        shifted = out[col].shift(1)
        new_cols[f"{col}_lag1"] = shifted
        new_cols[f"{col}_lag7"] = out[col].shift(7)
        new_cols[f"{col}_lag28"] = out[col].shift(28)
        new_cols[f"{col}_rmean7"] = shifted.rolling(7, min_periods=1).mean()
        new_cols[f"{col}_rmean28"] = shifted.rolling(28, min_periods=1).mean()
        new_cols[f"{col}_rmean90"] = shifted.rolling(90, min_periods=1).mean()

        # Momentum: 7d vs 28d
        short = shifted.rolling(7, min_periods=1).mean()
        long = shifted.rolling(28, min_periods=1).mean()
        new_cols[f"{col}_momentum"] = short / long.replace(0, np.nan)

    # AOV trend: rolling 90d average (captures the monotonic growth)
    if "aov" in out.columns:
        aov_shifted = out["aov"].shift(1)
        new_cols["aov_trend_90"] = aov_shifted.rolling(90, min_periods=30).mean()
        new_cols["aov_trend_365"] = aov_shifted.rolling(365, min_periods=90).mean()
        # YoY AOV growth
        new_cols["aov_yoy"] = aov_shifted / out["aov"].shift(366).replace(0, np.nan)

    # Revenue per session (if both available)
    if "wt_sessions" in out.columns:
        rev_per_session = out["Revenue"] / out["wt_sessions"].replace(0, np.nan)
        new_cols["rev_per_session_lag1"] = rev_per_session.shift(1)
        new_cols["rev_per_session_rmean28"] = (
            rev_per_session.shift(1).rolling(28, min_periods=7).mean()
        )

    # Concat all new columns at once (avoid fragmentation!)
    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED):
    """Same params as EX-28/31 which proved best."""
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
# Recursive prediction — extended with aux features
# ─────────────────────────────────────────────────────────────────────────────

def recursive_predict(
    models: list,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
    aux_history: pd.DataFrame | None = None,
) -> np.ndarray:
    """Recursive prediction with multi-model ensemble + aux features."""
    # Build history with aux columns
    history = history_df[["Date", target]].copy()
    if aux_history is not None:
        aux_cols = [c for c in aux_history.columns if c != "Date"]
        history = history.merge(
            aux_history[["Date"] + aux_cols], on="Date", how="left"
        )

    preds: list[float] = []

    for date in predict_dates:
        ts = pd.Timestamp(date)

        # Create new row: revenue unknown, aux features extrapolated
        new_row = {"Date": [ts], target: [np.nan]}
        if aux_history is not None:
            for col in aux_cols:
                # Use last known value (persistence) for aux
                if col in history.columns:
                    last_val = history[col].dropna().iloc[-1] if history[col].notna().any() else 0
                    new_row[col] = [last_val]

        row = pd.DataFrame(new_row)
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)

        # Build aux lag features if available
        if aux_history is not None:
            aux_new = {}
            for col in ["n_orders", "wt_sessions", "aov"]:
                if col not in combined.columns:
                    continue
                shifted = combined[col].shift(1)
                aux_new[f"{col}_lag1"] = shifted
                aux_new[f"{col}_lag7"] = combined[col].shift(7)
                aux_new[f"{col}_lag28"] = combined[col].shift(28)
                aux_new[f"{col}_rmean7"] = shifted.rolling(7, min_periods=1).mean()
                aux_new[f"{col}_rmean28"] = shifted.rolling(28, min_periods=1).mean()
                aux_new[f"{col}_rmean90"] = shifted.rolling(90, min_periods=1).mean()
                short = shifted.rolling(7, min_periods=1).mean()
                long = shifted.rolling(28, min_periods=1).mean()
                aux_new[f"{col}_momentum"] = short / long.replace(0, np.nan)

            if "aov" in combined.columns:
                aov_shifted = combined["aov"].shift(1)
                aux_new["aov_trend_90"] = aov_shifted.rolling(90, min_periods=30).mean()
                aux_new["aov_trend_365"] = aov_shifted.rolling(365, min_periods=90).mean()
                aux_new["aov_yoy"] = aov_shifted / combined["aov"].shift(366).replace(0, np.nan)

            if "wt_sessions" in combined.columns and target in combined.columns:
                rps = combined[target] / combined["wt_sessions"].replace(0, np.nan)
                aux_new["rev_per_session_lag1"] = rps.shift(1)
                aux_new["rev_per_session_rmean28"] = rps.shift(1).rolling(28, min_periods=7).mean()

            # Concat aux features to last row only
            for k, v in aux_new.items():
                combined[k] = v

        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)

        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else val

        # Ensemble
        raw_preds = [float(m.predict(x_pred)[0]) for m in models]
        pred = float(np.mean(raw_preds))
        pred = max(0, pred)
        preds.append(pred)

        # Update history
        new_hist = {"Date": [ts], target: [pred]}
        if aux_history is not None:
            # Estimate n_orders from predicted revenue using last known AOV
            last_aov = history["aov"].dropna().iloc[-1] if "aov" in history.columns and history["aov"].notna().any() else 30000
            estimated_orders = pred / last_aov if last_aov > 0 else 100
            new_hist["n_orders"] = [estimated_orders]
            new_hist["aov"] = [last_aov]  # AOV persists (it changes slowly)
            # Web traffic: extrapolate last value
            if "wt_sessions" in history.columns:
                last_wt = history["wt_sessions"].dropna().iloc[-1] if history["wt_sessions"].notna().any() else 30000
                new_hist["wt_sessions"] = [last_wt]
            for col in aux_cols:
                if col not in new_hist and col in history.columns:
                    last_v = history[col].dropna().iloc[-1] if history[col].notna().any() else 0
                    new_hist[col] = [last_v]

        history = pd.concat(
            [history, pd.DataFrame(new_hist)],
            ignore_index=True,
        )

    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation fold
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fold(sales: pd.DataFrame, aux: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()

    # Build standard features
    feat_df, profiles = build_feature_table(
        sales, verbose=False, profile_source_df=train_slice
    )

    # Add aux features
    feat_df = add_aux_features(feat_df, aux)

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    y_val_rev = val["Revenue"].values
    y_val_cogs = val["COGS"].values

    # Get all feature columns (original + aux)
    base_cols = get_feature_cols(feat_df)
    # Add aux-derived columns
    aux_feature_cols = [
        c for c in feat_df.columns
        if any(c.startswith(p) for p in [
            "n_orders_", "wt_sessions_", "aov_", "rev_per_session_",
            "wt_visitors_", "wt_pageviews_",
        ])
    ]
    all_cols = list(set(base_cols + aux_feature_cols))

    cols_rev = _finalize_cols(trn, _get_target_cols(all_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _get_target_cols(all_cols, "COGS"))

    print(f"  Features: Rev={len(cols_rev)}, COGS={len(cols_cogs)}")

    # Multi-seed ensemble
    models_rev, models_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        models_rev.append(_fit_lgbm(
            trn[cols_rev].fillna(0), trn["Revenue"],
            val[cols_rev].fillna(0), val["Revenue"], seed=seed,
        ))
        models_cogs.append(_fit_lgbm(
            trn[cols_cogs].fillna(0), trn["COGS"],
            val[cols_cogs].fillna(0), val["COGS"], seed=seed,
        ))

    # Build aux history for recursive prediction
    aux_history = trn[["Date"] + [c for c in ["n_orders", "wt_sessions", "aov", "wt_visitors", "wt_pageviews"]
                                   if c in trn.columns]].copy()

    pred_rev = recursive_predict(
        models_rev, trn[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue", aux_history=aux_history,
    )
    pred_cogs = recursive_predict(
        models_cogs, trn[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS", aux_history=aux_history,
    )

    res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} Revenue")
    res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} COGS")
    score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

    print(f"  {fold['name']} score: {score:,.0f}")
    print(f"  pred Rev mean: {pred_rev.mean():,.0f}")

    return pd.DataFrame([{
        "fold": fold["name"],
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "revenue_mae": float(res_rev["mae"]),
        "cogs_mae": float(res_cogs["mae"]),
        "revenue_pred_mean": float(pred_rev.mean()),
        "score": score,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()
    aux = load_daily_aux()

    print("=" * 78)
    print("EX_45: AOV Trend + Daily Auxiliary Features")
    print("  Hypothesis: AOV (rev/order) grew +47% from 22k→32k and is the key")
    print("  growth signal. By providing AOV lag/trend features, trees can")
    print("  'extrapolate' growth via the AOV trend continuation.")
    print("=" * 78)

    # ── Cross-validation ──
    fold_results = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df = evaluate_fold(train, aux, fold)
        fold_results.append(s_df)

    fold_scores = pd.concat(fold_results, ignore_index=True)
    fold_scores.to_csv(TRACK / "fold_scores.csv", index=False)

    print("\nFold Scores:")
    print(fold_scores.to_string(index=False))
    print(f"\nMean score: {fold_scores['score'].mean():,.0f}")
    print(f"Mean Rev pred: {fold_scores['revenue_pred_mean'].mean():,.0f}")

    # ── Full retrain + submission ──
    print("\nTraining on full data for submission...")
    feat_df, profiles = build_feature_table(
        train, verbose=True, profile_source_df=train
    )
    feat_df = add_aux_features(feat_df, aux)

    base_cols = get_feature_cols(feat_df)
    aux_feature_cols = [
        c for c in feat_df.columns
        if any(c.startswith(p) for p in [
            "n_orders_", "wt_sessions_", "aov_", "rev_per_session_",
            "wt_visitors_", "wt_pageviews_",
        ])
    ]
    all_cols = list(set(base_cols + aux_feature_cols))

    cols_rev = _finalize_cols(feat_df, _get_target_cols(all_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _get_target_cols(all_cols, "COGS"))

    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")

    # Multi-seed ensemble
    final_models_rev, final_models_cogs = [], []
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

    # Aux history for test prediction
    aux_history = feat_df[["Date"] + [c for c in ["n_orders", "wt_sessions", "aov", "wt_visitors", "wt_pageviews"]
                                       if c in feat_df.columns]].copy()

    print("\nRunning recursive inference on test set...")
    final_rev = recursive_predict(
        final_models_rev, train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue", aux_history=aux_history,
    )
    final_cogs = recursive_predict(
        final_models_cogs, train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS", aux_history=aux_history,
    )

    # Diagnostics
    ratios = final_rev / np.clip(final_cogs, 1, None)
    print(f"\nRev/COGS ratio: mean={ratios.mean():.3f}")
    print(f"Revenue: mean={final_rev.mean():,.0f}  std={final_rev.std():,.0f}")
    print(f"COGS: mean={final_cogs.mean():,.0f}  std={final_cogs.std():,.0f}")

    candidate_path = SUB_DIR / "ex_45_aov_trend.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "candidate_path": str(candidate_path),
        "n_seeds": N_SEEDS,
        "n_features_rev": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "mean_score": float(fold_scores["score"].mean()),
        "mean_rev_pred": float(fold_scores["revenue_pred_mean"].mean()),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
