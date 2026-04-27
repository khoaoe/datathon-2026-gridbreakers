"""
EX-44: Hybrid Tree + Trend Model with Daily Auxiliary Features

Two structural improvements over EX-31:

1. DAILY AUXILIARY FEATURES: Instead of just profile averages, compute
   actual daily aggregates from orders/payments/traffic tables, and
   use them as lag features (lag-1 order_count, lag-1 payment_value, etc.).
   These have 0.82-0.86 lagged correlation with Revenue.

2. HYBRID TREND: Combine LightGBM (captures seasonal patterns, profiles)
   with a simple linear trend model that CAN extrapolate beyond training
   range. LightGBM predicts the seasonal component; the linear model
   captures the growth trend.

   Revenue_t = LGB_seasonal(features) + LinearTrend(t)
   
   This is equivalent to detrending: train LGB on residuals after
   removing the linear trend, then add the trend back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

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

TRACK = Path("output/tracking/ex_44_hybrid")
SUB_DIR = Path("output/submissions")
N_SEEDS = 3


# ── 1. Daily Auxiliary Features ───────────────────────────────────────────


def build_daily_auxiliary() -> pd.DataFrame:
    """Build daily-level features from orders, payments, web_traffic.
    
    These are REAL daily signals — not profile averages — and have
    very high correlation with Revenue (0.82-0.99).
    """
    # Orders
    orders = pd.read_csv(FILES["orders"], parse_dates=["order_date"])
    daily_orders = orders.groupby("order_date").agg(
        order_count=("order_id", "nunique"),
        unique_customers=("customer_id", "nunique"),
    ).reset_index().rename(columns={"order_date": "Date"})
    
    # Add order status breakdown (delivered vs cancelled ratio)
    status_daily = orders.groupby(["order_date", "order_status"]).size().unstack(fill_value=0)
    if "delivered" in status_daily.columns and "cancelled" in status_daily.columns:
        status_daily["delivery_rate"] = (
            status_daily["delivered"] / status_daily.sum(axis=1).replace(0, np.nan)
        )
        status_daily["cancel_rate"] = (
            status_daily["cancelled"] / status_daily.sum(axis=1).replace(0, np.nan)
        )
        daily_orders = daily_orders.merge(
            status_daily[["delivery_rate", "cancel_rate"]].reset_index().rename(
                columns={"order_date": "Date"}
            ),
            on="Date", how="left"
        )
    
    # Order Items — quantity, avg price, discounts
    items = pd.read_csv(FILES["order_items"], low_memory=False)
    items_dated = items.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    daily_items = items_dated.groupby("order_date").agg(
        total_quantity=("quantity", "sum"),
        avg_unit_price=("unit_price", "mean"),
        median_unit_price=("unit_price", "median"),
        total_discount=("discount_amount", "sum"),
        unique_products=("product_id", "nunique"),
        items_per_order=("quantity", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})
    
    # Payments — daily payment volume
    payments = pd.read_csv(FILES["payments"])
    pay_dated = payments.merge(orders[["order_id", "order_date"]], on="order_id", how="left")
    daily_payments = pay_dated.groupby("order_date").agg(
        total_payment=("payment_value", "sum"),
        avg_payment=("payment_value", "mean"),
        avg_installments=("installments", "mean"),
    ).reset_index().rename(columns={"order_date": "Date"})
    
    # Web Traffic
    try:
        traffic = pd.read_csv(FILES["web_traffic"], parse_dates=["date"])
        # Aggregate across traffic sources
        daily_traffic = traffic.groupby("date").agg(
            sessions=("sessions", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            page_views=("page_views", "sum"),
            bounce_rate=("bounce_rate", "mean"),
            avg_session_duration=("avg_session_duration_sec", "mean"),
        ).reset_index().rename(columns={"date": "Date"})
    except Exception:
        daily_traffic = pd.DataFrame(columns=["Date"])
    
    # Merge all
    result = daily_orders
    result = result.merge(daily_items, on="Date", how="outer")
    result = result.merge(daily_payments, on="Date", how="outer")
    result = result.merge(daily_traffic, on="Date", how="outer")
    
    return result.sort_values("Date").reset_index(drop=True)


def add_aux_lag_features(df: pd.DataFrame, aux_df: pd.DataFrame) -> pd.DataFrame:
    """Merge auxiliary daily data and create lag features from them.
    
    We use LAGGED versions (shift(1)) because during recursive prediction,
    we only have access to yesterday's auxiliary values.
    """
    out = df.merge(aux_df, on="Date", how="left")
    
    # Core daily features to lag
    aux_cols = [
        "order_count", "unique_customers", "total_quantity",
        "avg_unit_price", "total_discount", "unique_products",
        "items_per_order", "total_payment", "avg_payment",
        "avg_installments", "delivery_rate", "cancel_rate",
        "sessions", "unique_visitors", "page_views",
        "bounce_rate", "avg_session_duration",
    ]
    
    for col in aux_cols:
        if col not in out.columns:
            continue
        
        # Lag-1 (yesterday's value)
        out[f"aux_{col}_lag1"] = out[col].shift(1)
        
        # 7-day rolling average
        out[f"aux_{col}_rmean7"] = out[col].shift(1).rolling(7, min_periods=1).mean()
        
        # 28-day rolling average
        out[f"aux_{col}_rmean28"] = out[col].shift(1).rolling(28, min_periods=1).mean()
        
        # Short/long ratio (momentum)
        short = out[col].shift(1).rolling(7, min_periods=1).mean()
        long = out[col].shift(1).rolling(28, min_periods=7).mean()
        out[f"aux_{col}_momentum"] = short / long.replace(0, np.nan)
    
    # Drop raw (non-lagged) aux columns to prevent leakage
    for col in aux_cols:
        if col in out.columns:
            out.drop(columns=[col], inplace=True)
    
    return out


# ── 2. Hybrid Trend Model ────────────────────────────────────────────────


class HybridTrendModel:
    """Combines LightGBM (seasonal patterns) + Ridge (linear trend).
    
    Strategy:
    1. Fit a Ridge regression on time features → captures linear growth
    2. Compute residuals = Revenue - Ridge_prediction
    3. Train LightGBM on residuals → captures seasonal/cyclical patterns
    4. Final prediction = Ridge(t) + LightGBM(features)
    
    Why this works: Ridge CAN extrapolate linearly beyond training range,
    while LightGBM captures all the complex seasonal patterns it excels at.
    """
    
    def __init__(self, lgb_params, seed=SEED):
        self.lgb_params = lgb_params
        self.seed = seed
        self.trend_model = None
        self.lgb_model = None
        self.trend_features = ["trend", "month_sin", "month_cos",
                               "dayofweek_sin", "dayofweek_cos"]
    
    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):
        # Step 1: Fit trend model
        trend_cols = [c for c in self.trend_features if c in X_train.columns]
        X_trend = X_train[trend_cols].fillna(0)
        X_trend_val = X_val[trend_cols].fillna(0)
        
        self.trend_model = Ridge(alpha=1.0)
        self.trend_model.fit(X_trend, y_train)
        
        # Step 2: Compute residuals
        trend_pred_train = self.trend_model.predict(X_trend)
        trend_pred_val = self.trend_model.predict(X_trend_val)
        residuals_train = y_train - trend_pred_train
        residuals_val = y_val - trend_pred_val
        
        # Step 3: Train LightGBM on residuals
        p = self.lgb_params.copy()
        p["random_state"] = self.seed
        self.lgb_model = lgb.LGBMRegressor(**p)
        self.lgb_model.fit(
            X_train.fillna(0), residuals_train,
            eval_set=[(X_val.fillna(0), residuals_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
            sample_weight=sample_weight,
        )
        
        return self
    
    def predict(self, X):
        trend_cols = [c for c in self.trend_features if c in X.columns]
        X_trend = X[trend_cols].fillna(0)
        
        trend_pred = self.trend_model.predict(X_trend)
        lgb_pred = self.lgb_model.predict(X.fillna(0))
        
        return trend_pred + lgb_pred


# ── 3. Core Pipeline ─────────────────────────────────────────────────────


def _finalize_cols(df, cols):
    return [c for c in cols if c in df.columns
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].notna().mean() > 0.50
            and df[c].nunique(dropna=True) > 1]


def _target_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def recursive_predict(models, history_df, predict_dates, feature_cols,
                      profiles, target, aux_df=None):
    """Recursive prediction with auxiliary data support."""
    history = history_df[["Date", target]].copy()
    
    # Prepare aux data for the history period
    if aux_df is not None:
        aux_history = history.merge(aux_df, on="Date", how="left")
    
    preds = []
    for date in predict_dates:
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")
        
        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)
        
        # Add aux features if available
        if aux_df is not None:
            combined = add_aux_lag_features(combined, aux_df)
        
        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)
        
        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else val
        
        if isinstance(models[0], HybridTrendModel):
            raw_preds = [float(m.predict(x_pred)[0]) for m in models]
        else:
            raw_preds = [float(m.predict(x_pred)[0]) for m in models]
        
        pred = max(0, float(np.mean(raw_preds)))
        preds.append(pred)
        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )
    
    return np.array(preds)


def run_fold(train_df, fold_name, val_start, val_end, aux_df, use_hybrid=False):
    """Run a single fold evaluation."""
    val_start = pd.Timestamp(val_start)
    val_end = pd.Timestamp(val_end)
    
    train_slice = train_df[train_df["Date"] < val_start].copy()
    
    # Build features WITH auxiliary data
    feat_df, profiles = build_feature_table(
        train_df, verbose=False, profile_source_df=train_slice
    )
    feat_df = add_aux_lag_features(feat_df, aux_df)
    
    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()
    
    base_cols = list(set(get_feature_cols(feat_df)) | set(
        [c for c in feat_df.columns if c.startswith("aux_")]
    ))
    cols_rev = _finalize_cols(trn, _target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(trn, _target_cols(base_cols, "COGS"))
    
    y_val_rev = val["Revenue"].values
    y_val_cogs = val["COGS"].values
    
    lgb_params = LGBM_PARAMS.copy()
    lgb_params["n_estimators"] = 1500
    
    models_rev, models_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        
        if use_hybrid:
            m_rev = HybridTrendModel(lgb_params, seed).fit(
                trn[cols_rev], trn["Revenue"], val[cols_rev], y_val_rev)
            m_cogs = HybridTrendModel(lgb_params, seed).fit(
                trn[cols_cogs], trn["COGS"], val[cols_cogs], y_val_cogs)
        else:
            lgb_params["random_state"] = seed
            m_rev = lgb.LGBMRegressor(**lgb_params)
            m_rev.fit(trn[cols_rev].fillna(0), trn["Revenue"],
                     eval_set=[(val[cols_rev].fillna(0), y_val_rev)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
            m_cogs = lgb.LGBMRegressor(**lgb_params)
            m_cogs.fit(trn[cols_cogs].fillna(0), trn["COGS"],
                      eval_set=[(val[cols_cogs].fillna(0), y_val_cogs)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        
        models_rev.append(m_rev)
        models_cogs.append(m_cogs)
    
    # Recursive prediction
    pred_rev = recursive_predict(
        models_rev, train_slice[["Date", "Revenue"]], val["Date"].values,
        cols_rev, profiles, "Revenue", aux_df=aux_df,
    )
    pred_cogs = recursive_predict(
        models_cogs, train_slice[["Date", "COGS"]], val["Date"].values,
        cols_cogs, profiles, "COGS", aux_df=aux_df,
    )
    
    r_rev = evaluate(y_val_rev, pred_rev, f"  {fold_name} Rev")
    r_cogs = evaluate(y_val_cogs, pred_cogs, f"  {fold_name} COGS")
    score = float(r_rev["mae"] + 0.4 * r_cogs["mae"])
    bias_rev = pred_rev.mean() / y_val_rev.mean() - 1
    bias_cogs = pred_cogs.mean() / y_val_cogs.mean() - 1
    
    print(f"  → Score={score:,.0f}  Bias: Rev={bias_rev:+.1%} COGS={bias_cogs:+.1%}")
    
    return {
        "fold": fold_name, "score": score,
        "rev_mae": float(r_rev["mae"]), "cogs_mae": float(r_cogs["mae"]),
        "bias_rev": float(bias_rev), "bias_cogs": float(bias_cogs),
        "rev_mean": float(pred_rev.mean()),
    }


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    
    train, test = load_sales()
    
    print("Building daily auxiliary features...")
    aux_df = build_daily_auxiliary()
    print(f"  Aux table: {aux_df.shape[0]} rows × {aux_df.shape[1]} cols")
    print(f"  Date range: {aux_df['Date'].min()} to {aux_df['Date'].max()}")
    
    FOLDS = [
        ("fold_2020", "2020-01-01", "2020-12-31"),
        ("fold_2021", "2021-01-01", "2021-12-31"),
        ("fold_2022", "2022-01-01", "2022-12-31"),
    ]
    
    # ── Experiment A: Baseline + Daily Aux Features (no hybrid) ──
    print("\n" + "=" * 70)
    print("EXPERIMENT A: LightGBM + Daily Auxiliary Features")
    print("=" * 70)
    results_a = []
    for fn, vs, ve in FOLDS:
        print(f"\n--- {fn} ---")
        r = run_fold(train, fn, vs, ve, aux_df, use_hybrid=False)
        r["experiment"] = "A_aux_features"
        results_a.append(r)
    
    mean_a = np.mean([r["score"] for r in results_a])
    print(f"\nExperiment A mean score: {mean_a:,.0f}")
    
    # ── Experiment B: Hybrid (Ridge trend + LGB residuals) + Daily Aux ──
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Hybrid (Ridge + LGB) + Daily Auxiliary Features")
    print("=" * 70)
    results_b = []
    for fn, vs, ve in FOLDS:
        print(f"\n--- {fn} ---")
        r = run_fold(train, fn, vs, ve, aux_df, use_hybrid=True)
        r["experiment"] = "B_hybrid_aux"
        results_b.append(r)
    
    mean_b = np.mean([r["score"] for r in results_b])
    print(f"\nExperiment B mean score: {mean_b:,.0f}")
    
    # ── Select best and generate submission ──
    all_results = results_a + results_b
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(TRACK / "fold_scores.csv", index=False)
    
    best_exp = "A_aux_features" if mean_a <= mean_b else "B_hybrid_aux"
    use_hybrid = best_exp == "B_hybrid_aux"
    print(f"\n{'='*70}")
    print(f"BEST: {best_exp} (score={min(mean_a, mean_b):,.0f})")
    print(f"{'='*70}")
    
    # Final training on all data
    print(f"\nFinal training ({best_exp})...")
    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)
    feat_df = add_aux_lag_features(feat_df, aux_df)
    
    base_cols = list(set(get_feature_cols(feat_df)) | set(
        [c for c in feat_df.columns if c.startswith("aux_")]
    ))
    cols_rev = _finalize_cols(feat_df, _target_cols(base_cols, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _target_cols(base_cols, "COGS"))
    
    print(f"  Revenue features: {len(cols_rev)}")
    print(f"  COGS features: {len(cols_cogs)}")
    print(f"  Aux features included: {len([c for c in cols_rev if c.startswith('aux_')])}")
    
    lgb_params = LGBM_PARAMS.copy()
    lgb_params["n_estimators"] = 1500
    
    final_rev, final_cogs = [], []
    for i in range(N_SEEDS):
        seed = SEED + i * 17
        if use_hybrid:
            m_r = HybridTrendModel(lgb_params, seed).fit(
                feat_df[cols_rev], feat_df["Revenue"],
                feat_df[cols_rev].tail(365), feat_df["Revenue"].tail(365))
            m_c = HybridTrendModel(lgb_params, seed).fit(
                feat_df[cols_cogs], feat_df["COGS"],
                feat_df[cols_cogs].tail(365), feat_df["COGS"].tail(365))
        else:
            lgb_params["random_state"] = seed
            m_r = lgb.LGBMRegressor(**lgb_params)
            m_r.fit(feat_df[cols_rev].fillna(0), feat_df["Revenue"],
                   eval_set=[(feat_df[cols_rev].fillna(0).tail(365), feat_df["Revenue"].tail(365))],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
            m_c = lgb.LGBMRegressor(**lgb_params)
            m_c.fit(feat_df[cols_cogs].fillna(0), feat_df["COGS"],
                   eval_set=[(feat_df[cols_cogs].fillna(0).tail(365), feat_df["COGS"].tail(365))],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
        final_rev.append(m_r)
        final_cogs.append(m_c)
    
    print("Recursive inference...")
    pred_rev = recursive_predict(
        final_rev, train[["Date", "Revenue"]], test["Date"].values,
        cols_rev, profiles, "Revenue", aux_df=aux_df,
    )
    pred_cogs = recursive_predict(
        final_cogs, train[["Date", "COGS"]], test["Date"].values,
        cols_cogs, profiles, "COGS", aux_df=aux_df,
    )
    
    print(f"\nRevenue: mean={pred_rev.mean():,.0f}")
    print(f"COGS: mean={pred_cogs.mean():,.0f}")
    
    try:
        sub31 = pd.read_csv(SUB_DIR / "ex_31_refined_ensemble.csv")
        print(f"vs EX-31: Rev {pred_rev.mean()/sub31['Revenue'].mean()-1:+.1%}")
    except Exception:
        pass
    
    path = SUB_DIR / "ex_44_hybrid_daily.csv"
    make_submission(test["Date"], pred_rev, pred_cogs, path)
    
    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "best_experiment": best_exp,
        "mean_cv_score": float(min(mean_a, mean_b)),
        "exp_a_score": float(mean_a),
        "exp_b_score": float(mean_b),
        "n_rev_features": len(cols_rev),
        "n_aux_features": len([c for c in cols_rev if c.startswith("aux_")]),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone in {meta['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
