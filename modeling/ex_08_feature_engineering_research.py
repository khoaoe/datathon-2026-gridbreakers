"""
EX_08: Deeper Feature Engineering Research (Time-Series K-Fold)

Goal:
- Run expanding time-series K-fold CV for deeper feature engineering research
- Compare stronger feature bundles under leakage-safe fold logic
- Produce fold logs, method ranking, and feature-importance exports

Outputs:
- output/tracking/ex_08_feature_engineering_research/fold_log.csv
- output/tracking/ex_08_feature_engineering_research/method_summary.csv
- output/tracking/ex_08_feature_engineering_research/feature_importance_by_method.csv
- output/tracking/ex_08_feature_engineering_research/best_method_top_features.csv
- output/tracking/ex_08_feature_engineering_research/notes.md
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling.config import FILES, OUTPUT_DIR, SEED
from modeling.feature_engineering import (
    build_calendar_features,
    build_growth_features,
    build_lag_features,
    build_rolling_features,
)
from modeling.utils import load_sales


TRACK_DIR = OUTPUT_DIR / "tracking" / "ex_08_feature_engineering_research"
TRACK_DIR.mkdir(parents=True, exist_ok=True)

RUN_MODE = os.getenv("EX08_MODE", "strict").strip().lower()

if RUN_MODE == "quick":
    TRACK_DIR = OUTPUT_DIR / "tracking" / "ex_08_feature_engineering_research_quick"
    STRICT_N_SPLITS = 2
    STRICT_VAL_DAYS = 180
    STRICT_MIN_TRAIN_DAYS = 2555
    LGBM_ESTIMATORS = 250
    HGB_MAX_ITER = 250
    STABLE_IMPROVE_RATIO = 0.50
    MAX_STABLE_WORST_DELTA = 12000.0
    QUICK_METHOD_NAMES = {
        "baseline_v3_core",
        "core_plus_promo_interactions",
        "core_plus_selected_aux_lags",
        "core_plus_regime_features",
    }
else:
    TRACK_DIR = OUTPUT_DIR / "tracking" / "ex_08_feature_engineering_research"
    STRICT_N_SPLITS = 4
    STRICT_VAL_DAYS = 365
    STRICT_MIN_TRAIN_DAYS = 2190
    LGBM_ESTIMATORS = 900
    HGB_MAX_ITER = 700
    STABLE_IMPROVE_RATIO = 0.75
    MAX_STABLE_WORST_DELTA = 6000.0
    QUICK_METHOD_NAMES = None

TRACK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Fold:
    name: str
    train_end: str
    val_start: str
    val_end: str
    train_days: int
    val_days: int


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_base_frame(sales: pd.DataFrame) -> pd.DataFrame:
    df = sales.copy().sort_values("Date").reset_index(drop=True)
    df = build_calendar_features(df)
    df = build_lag_features(df, "Revenue")
    df = build_lag_features(df, "COGS")
    df = build_rolling_features(df, "Revenue")
    df = build_rolling_features(df, "COGS", windows=[7, 14, 28, 90])
    df = build_growth_features(df, "Revenue")
    df["rev_over_cogs_lag1"] = _safe_div(df["Revenue"].shift(1), df["COGS"].shift(1))
    df["rev_over_cogs_lag7"] = _safe_div(df["Revenue"].shift(7), df["COGS"].shift(7))
    return df


def build_promo_features(date_index: pd.Series) -> pd.DataFrame:
    promos = pd.read_csv(
        FILES["promotions"],
        parse_dates=["start_date", "end_date"],
        usecols=["promo_type", "discount_value", "start_date", "end_date"],
    )

    out = pd.DataFrame({"Date": pd.to_datetime(date_index).sort_values().unique()})
    out["active_promo_count"] = 0
    out["active_pct_count"] = 0
    out["active_fixed_count"] = 0
    out["active_discount_sum"] = 0.0

    for _, row in promos.iterrows():
        mask = (out["Date"] >= row["start_date"]) & (out["Date"] <= row["end_date"])
        out.loc[mask, "active_promo_count"] += 1
        out.loc[mask, "active_discount_sum"] += float(row.get("discount_value", 0.0))

        ptype = str(row.get("promo_type", "")).strip().lower()
        if ptype == "percentage":
            out.loc[mask, "active_pct_count"] += 1
        elif ptype == "fixed":
            out.loc[mask, "active_fixed_count"] += 1

    starts = np.sort(promos["start_date"].dropna().values.astype("datetime64[ns]"))
    ends = np.sort(promos["end_date"].dropna().values.astype("datetime64[ns]"))
    dvals = out["Date"].values.astype("datetime64[ns]")

    out["days_to_next_promo_start"] = np.nan
    out["days_since_last_promo_end"] = np.nan

    if len(starts) > 0:
        next_idx = np.searchsorted(starts, dvals, side="left")
        has_next = next_idx < len(starts)
        capped = np.minimum(next_idx, len(starts) - 1)
        next_dates = starts[capped]
        out.loc[has_next, "days_to_next_promo_start"] = (
            (next_dates[has_next] - dvals[has_next])
            .astype("timedelta64[D]")
            .astype(float)
        )

    if len(ends) > 0:
        prev_idx = np.searchsorted(ends, dvals, side="right") - 1
        has_prev = prev_idx >= 0
        capped = np.maximum(prev_idx, 0)
        prev_dates = ends[capped]
        out.loc[has_prev, "days_since_last_promo_end"] = (
            (dvals[has_prev] - prev_dates[has_prev])
            .astype("timedelta64[D]")
            .astype(float)
        )

    out["is_promo_active"] = (out["active_promo_count"] > 0).astype(int)
    return out


def build_aux_daily_lag_features(date_index: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"Date": pd.to_datetime(date_index).sort_values().unique()})

    orders = pd.read_csv(
        FILES["orders"],
        parse_dates=["order_date"],
        usecols=["order_id", "order_date", "order_status"],
    )
    payments = pd.read_csv(FILES["payments"], usecols=["order_id", "payment_value"])
    ordp = orders.merge(payments, on="order_id", how="left")
    ord_daily = ordp.groupby("order_date", as_index=False).agg(
        order_count=("order_id", "count"),
        pay_total=("payment_value", "sum"),
        cancel_rate=("order_status", lambda s: (s == "cancelled").mean()),
    )

    returns = pd.read_csv(
        FILES["returns"],
        parse_dates=["return_date"],
        usecols=["return_id", "return_date", "refund_amount", "return_quantity"],
    )
    ret_daily = returns.groupby("return_date", as_index=False).agg(
        return_count=("return_id", "count"),
        refund_total=("refund_amount", "sum"),
        return_qty=("return_quantity", "sum"),
    )

    shipments = pd.read_csv(
        FILES["shipments"],
        parse_dates=["ship_date"],
        usecols=["order_id", "ship_date", "shipping_fee"],
    )
    ship_daily = shipments.groupby("ship_date", as_index=False).agg(
        ship_count=("order_id", "count"),
        ship_fee_total=("shipping_fee", "sum"),
    )

    web = pd.read_csv(FILES["web_traffic"], parse_dates=["date"])
    web_daily = web.groupby("date", as_index=False).agg(
        sessions=("sessions", "sum"),
        visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
    )

    out = out.merge(
        ord_daily.rename(columns={"order_date": "Date"}), on="Date", how="left"
    )
    out = out.merge(
        ret_daily.rename(columns={"return_date": "Date"}), on="Date", how="left"
    )
    out = out.merge(
        ship_daily.rename(columns={"ship_date": "Date"}), on="Date", how="left"
    )
    out = out.merge(web_daily.rename(columns={"date": "Date"}), on="Date", how="left")

    raw_aux = [c for c in out.columns if c != "Date"]
    out[raw_aux] = out[raw_aux].fillna(0)

    for col in raw_aux:
        for lag in [1, 7, 14, 28]:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)

    return out.drop(columns=raw_aux)


def add_deeper_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["promo_discount_per_active"] = _safe_div(
        out["active_discount_sum"], out["active_promo_count"].replace(0, np.nan)
    )
    out["promo_weekend_discount"] = out["active_discount_sum"] * out["is_weekend"]
    out["promo_monthend_discount"] = out["active_discount_sum"] * out["is_month_end"]
    out["promo_dow_pressure"] = out["active_promo_count"] * (out["dayofweek"] + 1)
    out["next_promo_7d"] = out["days_to_next_promo_start"].between(0, 7).astype(int)
    out["after_promo_7d"] = out["days_since_last_promo_end"].between(0, 7).astype(int)

    rev_l1 = out["Revenue"].shift(1)
    rev_mean_28 = rev_l1.rolling(28, min_periods=7).mean()
    rev_std_28 = rev_l1.rolling(28, min_periods=7).std()
    out["rev_vol_14"] = rev_l1.rolling(14, min_periods=4).std()
    out["rev_vol_28"] = rev_std_28
    out["rev_trend_7_28"] = _safe_div(
        rev_l1.rolling(7, min_periods=3).mean(), rev_mean_28
    )
    out["rev_zscore_28"] = _safe_div(rev_l1 - rev_mean_28, rev_std_28)

    cogs_l1 = out["COGS"].shift(1)
    out["cogs_trend_7_28"] = _safe_div(
        cogs_l1.rolling(7, min_periods=3).mean(),
        cogs_l1.rolling(28, min_periods=7).mean(),
    )
    out["rev_cogs_spread_lag1"] = out["Revenue_lag_1"] - out["COGS_lag_1"]
    out["rev_cogs_ratio_lag1"] = _safe_div(out["Revenue_lag_1"], out["COGS_lag_1"])

    out["weekend_monthend"] = out["is_weekend"] * out["is_month_end"]
    out["trend_x_month_sin"] = out["trend"] * out["month_sin"]
    out["trend_x_month_cos"] = out["trend"] * out["month_cos"]
    out["trend_x_dayofyear_sin"] = out["trend"] * out["dayofyear_sin"]

    return out


def make_standard_profiles(train_slice: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = train_slice.copy()
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    return {
        "dow": df.groupby("dayofweek")
        .agg(std_rev_dow_mean=("Revenue", "mean"))
        .reset_index(),
        "month": df.groupby("month")
        .agg(std_rev_month_mean=("Revenue", "mean"))
        .reset_index(),
        "woy": df.groupby("weekofyear")
        .agg(std_rev_woy_mean=("Revenue", "mean"))
        .reset_index(),
        "month_dow": df.groupby(["month", "dayofweek"])
        .agg(std_rev_month_dow_mean=("Revenue", "mean"))
        .reset_index(),
    }


def make_recency_profiles(
    train_slice: pd.DataFrame, decay: float = 0.003
) -> dict[str, pd.DataFrame]:
    df = train_slice.copy()
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    ref = df["Date"].max()
    age_days = (ref - df["Date"]).dt.days
    df["w"] = np.exp(-decay * age_days)
    df["wr"] = df["Revenue"] * df["w"]

    def _weighted_mean(group_cols: list[str], out_col: str) -> pd.DataFrame:
        agg = df.groupby(group_cols, as_index=False).agg(
            w_sum=("w", "sum"), wr_sum=("wr", "sum")
        )
        agg[out_col] = agg["wr_sum"] / agg["w_sum"].replace(0, np.nan)
        return agg[group_cols + [out_col]]

    return {
        "dow": _weighted_mean(["dayofweek"], "rec_rev_dow_mean"),
        "month": _weighted_mean(["month"], "rec_rev_month_mean"),
        "woy": _weighted_mean(["weekofyear"], "rec_rev_woy_mean"),
        "month_dow": _weighted_mean(["month", "dayofweek"], "rec_rev_month_dow_mean"),
    }


def apply_profiles(
    frame: pd.DataFrame, profiles: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    df = frame.copy()
    if "dow" in profiles:
        df = df.merge(profiles["dow"], on=["dayofweek"], how="left")
    if "month" in profiles:
        df = df.merge(profiles["month"], on=["month"], how="left")
    if "woy" in profiles:
        df = df.merge(profiles["woy"], on=["weekofyear"], how="left")
    if "month_dow" in profiles:
        df = df.merge(profiles["month_dow"], on=["month", "dayofweek"], how="left")
    return df


def make_expanding_time_folds(
    date_series: pd.Series,
    n_splits: int = 5,
    val_days: int = 365,
    min_train_days: int = 1460,
) -> list[Fold]:
    unique_dates = pd.Series(pd.to_datetime(date_series).sort_values().unique())
    total_days = len(unique_dates)

    max_splits = (total_days - min_train_days) // val_days
    if max_splits < 2:
        raise ValueError(
            f"Not enough history for time-series K-fold: total_days={total_days}"
        )

    n_splits = int(min(n_splits, max_splits))
    start_train_end_idx = total_days - (n_splits * val_days) - 1

    folds: list[Fold] = []
    for i in range(n_splits):
        train_end_idx = start_train_end_idx + i * val_days
        val_start_idx = train_end_idx + 1
        val_end_idx = min(val_start_idx + val_days - 1, total_days - 1)

        train_end = unique_dates.iloc[train_end_idx]
        val_start = unique_dates.iloc[val_start_idx]
        val_end = unique_dates.iloc[val_end_idx]

        folds.append(
            Fold(
                name=f"tscv_{i + 1}",
                train_end=train_end.strftime("%Y-%m-%d"),
                val_start=val_start.strftime("%Y-%m-%d"),
                val_end=val_end.strftime("%Y-%m-%d"),
                train_days=int(train_end_idx + 1),
                val_days=int(val_end_idx - val_start_idx + 1),
            )
        )
    return folds


def fit_predict_model(
    X_trn: pd.DataFrame,
    y_trn: pd.Series,
    X_val: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            objective="regression",
            metric="mae",
            boosting_type="gbdt",
            n_estimators=LGBM_ESTIMATORS,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_trn, y_trn)
        pred = model.predict(X_val)
        return pred, getattr(model, "feature_importances_", None), "lightgbm"
    except Exception:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_iter=HGB_MAX_ITER,
            l2_regularization=0.1,
            random_state=SEED,
        )
        model.fit(X_trn, y_trn)
        pred = model.predict(X_val)
        return pred, None, "histgbr"


def evaluate_fold(
    df: pd.DataFrame,
    fold: Fold,
    feature_cols: list[str],
) -> tuple[dict, pd.DataFrame | None]:
    tr_end = pd.Timestamp(fold.train_end)
    va_start = pd.Timestamp(fold.val_start)
    va_end = pd.Timestamp(fold.val_end)

    trn = df[df["Date"] <= tr_end].copy()
    val = df[(df["Date"] >= va_start) & (df["Date"] <= va_end)].copy()

    cols = list(dict.fromkeys(feature_cols))
    cols = [c for c in cols if c in trn.columns and c in val.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(trn[c])]
    cols = [c for c in cols if trn[c].notna().mean() >= 0.50]
    cols = [c for c in cols if trn[c].nunique(dropna=True) > 1]

    trn = trn.dropna(subset=["Revenue"])
    val = val.dropna(subset=["Revenue"])

    if len(cols) == 0:
        raise ValueError(f"No usable features for {fold.name}")

    X_trn = trn[cols].replace([np.inf, -np.inf], np.nan)
    X_val = val[cols].replace([np.inf, -np.inf], np.nan)

    fill_values = X_trn.median(numeric_only=True)
    X_trn = X_trn.fillna(fill_values)
    X_val = X_val.fillna(fill_values)

    y_trn = trn["Revenue"]
    y_val = val["Revenue"]

    pred, importances, model_name = fit_predict_model(X_trn, y_trn, X_val)

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    result = {
        "fold": fold.name,
        "train_end": fold.train_end,
        "val_start": fold.val_start,
        "val_end": fold.val_end,
        "train_days": fold.train_days,
        "val_days": fold.val_days,
        "n_features": len(cols),
        "model": model_name,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    if importances is None:
        return result, None

    imp_df = pd.DataFrame({"feature": cols, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return result, imp_df


def main() -> None:
    print("=" * 78)
    print("EX_08: DEEPER FEATURE ENGINEERING RESEARCH (TIME-SERIES K-FOLD)")
    print(f"RUN MODE: {RUN_MODE}")
    print("=" * 78)

    sales, _ = load_sales()

    print("[1/5] Building shared feature frame...")
    base = build_base_frame(sales)
    promo_feat = build_promo_features(base["Date"])
    aux_lag_feat = build_aux_daily_lag_features(base["Date"])

    data = base.merge(promo_feat, on="Date", how="left")
    data = data.merge(aux_lag_feat, on="Date", how="left")
    data = add_deeper_features(data)

    promo_basic_cols = [c for c in promo_feat.columns if c != "Date"]
    aux_cols = [c for c in aux_lag_feat.columns if c != "Date"]

    deep_promo_cols = [
        "promo_discount_per_active",
        "promo_weekend_discount",
        "promo_monthend_discount",
        "promo_dow_pressure",
        "next_promo_7d",
        "after_promo_7d",
    ]
    regime_cols = [
        "rev_vol_14",
        "rev_vol_28",
        "rev_trend_7_28",
        "rev_zscore_28",
        "cogs_trend_7_28",
        "rev_cogs_spread_lag1",
        "rev_cogs_ratio_lag1",
        "weekend_monthend",
        "trend_x_month_sin",
        "trend_x_month_cos",
        "trend_x_dayofyear_sin",
    ]

    selected_aux_candidates = {
        "sessions_lag_7",
        "sessions_lag_14",
        "visitors_lag_7",
        "visitors_lag_14",
        "page_views_lag_7",
        "bounce_rate_lag_7",
        "order_count_lag_7",
        "pay_total_lag_7",
        "cancel_rate_lag_7",
        "return_count_lag_7",
        "refund_total_lag_7",
        "ship_count_lag_7",
    }
    selected_aux_cols = [c for c in aux_cols if c in selected_aux_candidates]

    all_feature_cols = [
        c
        for c in data.columns
        if c not in {"Date", "Revenue", "COGS"}
        and not c.startswith("std_")
        and not c.startswith("rec_")
    ]

    deep_extra_cols = set(deep_promo_cols + regime_cols + aux_cols)
    base_v3_cols = [c for c in all_feature_cols if c not in deep_extra_cols]
    base_v3_cols = [c for c in base_v3_cols if c in data.columns]
    deep_promo_cols = [c for c in deep_promo_cols if c in data.columns]
    regime_cols = [c for c in regime_cols if c in data.columns]

    print("[2/5] Building expanding time-series folds...")
    folds = make_expanding_time_folds(
        data["Date"],
        n_splits=STRICT_N_SPLITS,
        val_days=STRICT_VAL_DAYS,
        min_train_days=STRICT_MIN_TRAIN_DAYS,
    )
    for f in folds:
        print(
            f"  {f.name}: train<= {f.train_end} ({f.train_days}d), "
            f"val={f.val_start}..{f.val_end} ({f.val_days}d)"
        )

    records: list[dict] = []
    importance_records: list[pd.DataFrame] = []

    print("[3/5] Evaluating feature bundles across folds...")
    for fold in folds:
        print(f"\n[{fold.name}]")
        tr_end = pd.Timestamp(fold.train_end)
        train_slice = data[data["Date"] <= tr_end][
            ["Date", "Revenue", "dayofweek", "month", "weekofyear"]
        ].copy()

        std_profiles = make_standard_profiles(train_slice)
        rec_profiles = make_recency_profiles(train_slice)

        fold_df_std = apply_profiles(data, std_profiles)
        fold_df_rec = apply_profiles(data, rec_profiles)
        fold_df_blend = apply_profiles(fold_df_std, rec_profiles)

        std_cols = [c for c in fold_df_std.columns if c.startswith("std_")]
        rec_cols = [c for c in fold_df_rec.columns if c.startswith("rec_")]

        methods: list[tuple[str, pd.DataFrame, list[str], str, str]] = [
            (
                "baseline_v3_core",
                data,
                base_v3_cols,
                "v3 core features + known-future promo basics",
                "baseline",
            ),
            (
                "core_plus_promo_interactions",
                data,
                base_v3_cols + deep_promo_cols,
                "core plus richer promo timing/intensity interactions",
                "promo",
            ),
            (
                "core_plus_regime_features",
                data,
                base_v3_cols + regime_cols,
                "core plus volatility/regime trend interactions",
                "regime",
            ),
            (
                "core_plus_selected_aux_lags",
                data,
                base_v3_cols + selected_aux_cols,
                "core plus selected lagged aux signals (web/orders/returns/ship)",
                "aux_lags",
            ),
            (
                "core_plus_std_profiles",
                fold_df_std,
                base_v3_cols + std_cols,
                "core plus fold-safe standard seasonal profiles",
                "std_profile",
            ),
            (
                "core_plus_recency_profiles",
                fold_df_rec,
                base_v3_cols + rec_cols,
                "core plus fold-safe recency-weighted seasonal profiles",
                "recency_profile",
            ),
            (
                "deep_combo_promo_std",
                fold_df_std,
                base_v3_cols + deep_promo_cols + std_cols,
                "core plus promo interactions plus std profiles",
                "combo",
            ),
            (
                "deep_combo_blend",
                fold_df_blend,
                base_v3_cols
                + deep_promo_cols
                + regime_cols
                + selected_aux_cols
                + std_cols
                + rec_cols,
                "deeper blend: promo+regime+selected aux+std/rec profiles",
                "combo",
            ),
        ]

        if QUICK_METHOD_NAMES is not None:
            methods = [m for m in methods if m[0] in QUICK_METHOD_NAMES]

        baseline_mae = None
        for method_name, method_df, method_cols, note, family in methods:
            result, imp_df = evaluate_fold(method_df, fold, method_cols)
            result["method"] = method_name
            result["note"] = note
            result["family"] = family

            if method_name == "baseline_v3_core":
                baseline_mae = result["mae"]

            result["delta_vs_fold_baseline_mae"] = (
                float(result["mae"] - baseline_mae)
                if baseline_mae is not None
                else np.nan
            )
            records.append(result)

            if imp_df is not None:
                tmp = imp_df.copy()
                tmp["fold"] = fold.name
                tmp["method"] = method_name
                importance_records.append(tmp)

            print(
                f"  {method_name:<30} MAE={result['mae']:,.2f} "
                f"delta={result['delta_vs_fold_baseline_mae']:+,.2f} "
                f"features={result['n_features']}"
            )

    print("[4/5] Writing logs...")
    log_df = pd.DataFrame(records)
    log_df = log_df.sort_values(["fold", "mae", "method"]).reset_index(drop=True)
    log_path = TRACK_DIR / "fold_log.csv"
    log_df.to_csv(log_path, index=False)

    wins = (
        log_df.loc[log_df.groupby("fold")["mae"].idxmin()]["method"]
        .value_counts()
        .rename_axis("method")
        .reset_index(name="fold_wins")
    )

    total_folds = max(int(log_df["fold"].nunique()), 1)
    method_family = log_df.groupby("method", as_index=False).agg(
        family=("family", "first")
    )
    stability = log_df.groupby("method", as_index=False).agg(
        improve_folds=(
            "delta_vs_fold_baseline_mae",
            lambda s: int((s < 0).sum()),
        ),
        worsen_folds=(
            "delta_vs_fold_baseline_mae",
            lambda s: int((s > 0).sum()),
        ),
        neutral_folds=(
            "delta_vs_fold_baseline_mae",
            lambda s: int((s == 0).sum()),
        ),
        best_fold_delta=("delta_vs_fold_baseline_mae", "min"),
        worst_fold_delta=("delta_vs_fold_baseline_mae", "max"),
    )
    stability["improve_ratio"] = stability["improve_folds"] / total_folds
    stability["stable_candidate"] = (
        (stability["improve_ratio"] >= STABLE_IMPROVE_RATIO)
        & (stability["best_fold_delta"] < 0)
        & (stability["worst_fold_delta"] <= MAX_STABLE_WORST_DELTA)
    )

    summary = (
        log_df.groupby("method", as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_delta_vs_baseline=("delta_vs_fold_baseline_mae", "mean"),
            median_features=("n_features", "median"),
            model=("model", lambda x: x.mode().iloc[0]),
        )
        .merge(method_family, on="method", how="left")
        .merge(wins, on="method", how="left")
        .merge(stability, on="method", how="left")
        .fillna({"fold_wins": 0})
        .sort_values(["stable_candidate", "mean_mae"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["rank_mean_mae"] = np.arange(1, len(summary) + 1)

    summary_path = TRACK_DIR / "method_summary.csv"
    summary.to_csv(summary_path, index=False)

    best_method = str(summary.iloc[0]["method"])

    if importance_records:
        imp_all = pd.concat(importance_records, ignore_index=True)
        imp_group = (
            imp_all.groupby(["method", "feature"], as_index=False)
            .agg(mean_importance=("importance", "mean"))
            .sort_values(["method", "mean_importance"], ascending=[True, False])
        )
        imp_path = TRACK_DIR / "feature_importance_by_method.csv"
        imp_group.to_csv(imp_path, index=False)

        best_imp = imp_group[imp_group["method"] == best_method].head(40)
        best_imp_path = TRACK_DIR / "best_method_top_features.csv"
        best_imp.to_csv(best_imp_path, index=False)
    else:
        imp_path = None
        best_imp_path = None

    print("[5/5] Writing research notes...")
    best = summary.iloc[0].to_dict()
    runner_up = summary.iloc[1].to_dict() if len(summary) > 1 else None
    stable_rows = summary[summary["stable_candidate"]].copy()

    notes = [
        "# EX_08 Deeper Feature Engineering Research",
        "",
        "## Context",
        "- Provided context: XGBoost(v3) public score ~971k, Ensemble public score ~861k.",
        "- Goal: go deeper on feature engineering with time-series K-fold research.",
        "",
        "## Evaluation Setup",
        f"- Run mode: {RUN_MODE}",
        f"- Split: expanding time-series K-fold ({STRICT_N_SPLITS} folds, {STRICT_VAL_DAYS}-day validation)",
        f"- Fold policy: min train history = {STRICT_MIN_TRAIN_DAYS:,} days",
        f"- Estimators/iters: LightGBM={LGBM_ESTIMATORS}, HistGBR={HGB_MAX_ITER}",
        "- Metric: MAE (lower is better)",
        "- Model: LightGBM fallback HistGradientBoostingRegressor",
        "",
        "## Best Method",
        f"- method: {best['method']}",
        f"- family: {best.get('family')}",
        f"- mean_mae: {best['mean_mae']:.2f}",
        f"- mean_delta_vs_baseline: {best['mean_delta_vs_baseline']:+.2f}",
        f"- fold_wins: {int(best['fold_wins'])}",
        f"- improve_ratio: {best.get('improve_ratio', 0.0):.2f}",
        f"- worst_fold_delta: {best.get('worst_fold_delta', np.nan):+.2f}",
        "",
    ]

    if runner_up is not None:
        notes.extend(
            [
                "## Runner Up",
                f"- method: {runner_up['method']}",
                f"- family: {runner_up.get('family')}",
                f"- mean_mae: {runner_up['mean_mae']:.2f}",
                f"- mean_delta_vs_baseline: {runner_up['mean_delta_vs_baseline']:+.2f}",
                "",
            ]
        )

    notes.append("## Stable Delta Shortlist")
    if stable_rows.empty:
        notes.append(
            "- No method met stability gate (improve_ratio>=0.75 and worst_fold_delta<=+6000)."
        )
    else:
        for _, row in stable_rows.head(4).iterrows():
            notes.append(
                "- "
                f"{row['method']} ({row.get('family')}): "
                f"mean_delta={row['mean_delta_vs_baseline']:+.2f}, "
                f"improve_ratio={row['improve_ratio']:.2f}, "
                f"worst_fold_delta={row['worst_fold_delta']:+.2f}"
            )
    notes.append("")

    notes.extend(
        [
            "## Suggested Next Production Tests",
            "- Wire top 1-2 FE bundles into EX_03 and EX_04 train scripts.",
            "- Keep leakage-safe profile source (pre-val rows only during local eval).",
            "- Re-run weighted ensemble with new submissions.",
            "",
            "## Files",
            f"- fold_log.csv: {log_path}",
            f"- method_summary.csv: {summary_path}",
            f"- feature_importance_by_method.csv: {imp_path}",
            f"- best_method_top_features.csv: {best_imp_path}",
        ]
    )

    notes_path = TRACK_DIR / "notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    meta = {
        "n_folds": int(len(folds)),
        "n_methods": int(summary.shape[0]),
        "n_records": int(log_df.shape[0]),
        "best_method": best_method,
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "notes_path": str(notes_path),
        "importance_path": str(imp_path) if imp_path is not None else None,
        "best_importance_path": (
            str(best_imp_path) if best_imp_path is not None else None
        ),
    }
    (TRACK_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "-" * 78)
    print(f"Saved fold log:      {log_path}")
    print(f"Saved summary:       {summary_path}")
    print(f"Saved notes:         {notes_path}")
    if imp_path is not None:
        print(f"Saved importances:   {imp_path}")
    if best_imp_path is not None:
        print(f"Saved best features: {best_imp_path}")
    print("-" * 78)
    print("Top methods by mean MAE:")
    print(
        summary[
            [
                "rank_mean_mae",
                "method",
                "family",
                "mean_mae",
                "mean_delta_vs_baseline",
                "improve_ratio",
                "worst_fold_delta",
                "stable_candidate",
                "fold_wins",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
