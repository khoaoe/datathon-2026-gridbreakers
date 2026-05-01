"""
EX_07: Feature Engineering Deepwork

Purpose:
- Evaluate multiple feature-engineering methods with walk-forward validation
- Log fold-level metrics and method summary rankings

Output:
- output/tracking/ex_07_feature_engineering_deepwork/method_log.csv
- output/tracking/ex_07_feature_engineering_deepwork/method_summary.csv
- output/tracking/ex_07_feature_engineering_deepwork/notes.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modeling.config import FILES, OUTPUT_DIR
from modeling.feature_engineering import (
    build_calendar_features,
    build_growth_features,
    build_lag_features,
    build_rolling_features,
)
from modeling.utils import load_sales


TRACK_DIR = OUTPUT_DIR / "tracking" / "ex_07_feature_engineering_deepwork"
TRACK_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Fold:
    name: str
    train_end: str
    val_start: str
    val_end: str


FOLDS = [
    Fold(
        "fold_2020",
        train_end="2019-12-31",
        val_start="2020-01-01",
        val_end="2020-12-31",
    ),
    Fold(
        "fold_2021",
        train_end="2020-12-31",
        val_start="2021-01-01",
        val_end="2021-12-31",
    ),
    Fold(
        "fold_2022",
        train_end="2021-12-31",
        val_start="2022-01-01",
        val_end="2022-12-31",
    ),
]


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

    # Extra stable target transforms
    df["rev_over_cogs_lag1"] = _safe_div(df["Revenue"].shift(1), df["COGS"].shift(1))
    df["rev_over_cogs_lag7"] = _safe_div(df["Revenue"].shift(7), df["COGS"].shift(7))
    return df


def build_promo_features(date_index: pd.Series) -> pd.DataFrame:
    promos = pd.read_csv(FILES["promotions"], parse_dates=["start_date", "end_date"])
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

    starts = np.sort(promos["start_date"].dropna().unique())
    ends = np.sort(promos["end_date"].dropna().unique())

    # Days to next promo start
    next_idx = np.searchsorted(starts, out["Date"].values, side="left")
    has_next = next_idx < len(starts)
    out["days_to_next_promo_start"] = np.where(
        has_next,
        (starts[np.clip(next_idx, 0, max(len(starts) - 1, 0))] - out["Date"].values)
        .astype("timedelta64[D]")
        .astype(float),
        np.nan,
    )

    # Days since last promo end
    prev_idx = np.searchsorted(ends, out["Date"].values, side="right") - 1
    has_prev = prev_idx >= 0
    out["days_since_last_promo_end"] = np.where(
        has_prev,
        (out["Date"].values - ends[np.clip(prev_idx, 0, max(len(ends) - 1, 0))])
        .astype("timedelta64[D]")
        .astype(float),
        np.nan,
    )

    return out


def build_aux_daily_lag_features(date_index: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"Date": pd.to_datetime(date_index).sort_values().unique()})

    # Orders + payments
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

    # Returns
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

    # Shipments
    shipments = pd.read_csv(
        FILES["shipments"],
        parse_dates=["ship_date"],
        usecols=["order_id", "ship_date", "shipping_fee"],
    )
    ship_daily = shipments.groupby("ship_date", as_index=False).agg(
        ship_count=("order_id", "count"),
        ship_fee_total=("shipping_fee", "sum"),
    )

    # Web traffic
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

    drop_raw = [
        "order_count",
        "pay_total",
        "cancel_rate",
        "return_count",
        "refund_total",
        "return_qty",
        "ship_count",
        "ship_fee_total",
        "sessions",
        "visitors",
        "page_views",
        "bounce_rate",
        "avg_session_duration_sec",
    ]
    return out.drop(columns=[c for c in drop_raw if c in out.columns])


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
    train_slice: pd.DataFrame, decay: float = 0.002
) -> dict[str, pd.DataFrame]:
    """Decay-weighted profiles. Higher weight for recent history."""
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


def evaluate_fold(df: pd.DataFrame, fold: Fold, feature_cols: list[str]) -> dict:
    tr_end = pd.Timestamp(fold.train_end)
    va_start = pd.Timestamp(fold.val_start)
    va_end = pd.Timestamp(fold.val_end)

    trn = df[df["Date"] <= tr_end].copy()
    val = df[(df["Date"] >= va_start) & (df["Date"] <= va_end)].copy()

    # Keep robust columns only from training fold
    cols = [c for c in feature_cols if c in trn.columns]
    cols = [c for c in cols if trn[c].notna().mean() > 0.50]

    trn = trn.dropna(subset=["Revenue"])
    val = val.dropna(subset=["Revenue"])
    trn = trn.dropna(
        subset=[
            c
            for c in cols
            if c.startswith("Revenue_lag_365") or c.startswith("COGS_lag_365")
        ],
        how="any",
    )

    X_trn = trn[cols].fillna(0.0)
    y_trn = trn["Revenue"]
    X_val = val[cols].fillna(0.0)
    y_val = val["Revenue"]

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=700,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_trn, y_trn)
    pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    return {
        "fold": fold.name,
        "train_end": fold.train_end,
        "val_start": fold.val_start,
        "val_end": fold.val_end,
        "n_features": len(cols),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def main() -> None:
    print("=" * 72)
    print("EX_07: FEATURE ENGINEERING DEEPWORK")
    print("=" * 72)

    sales, _ = load_sales()
    base = build_base_frame(sales)

    promo_feat = build_promo_features(base["Date"])
    aux_lag_feat = build_aux_daily_lag_features(base["Date"])

    data = base.merge(promo_feat, on="Date", how="left")
    data = data.merge(aux_lag_feat, on="Date", how="left")

    promo_cols = [c for c in promo_feat.columns if c != "Date"]
    aux_cols = [c for c in aux_lag_feat.columns if c != "Date"]

    orders_cols = [
        c for c in aux_cols if c.startswith(("order_count", "pay_total", "cancel_rate"))
    ]
    returns_cols = [
        c
        for c in aux_cols
        if c.startswith(("return_count", "refund_total", "return_qty"))
    ]
    shipments_cols = [
        c for c in aux_cols if c.startswith(("ship_count", "ship_fee_total"))
    ]
    web_cols = [
        c
        for c in aux_cols
        if c.startswith(
            (
                "sessions",
                "visitors",
                "page_views",
                "bounce_rate",
                "avg_session_duration_sec",
            )
        )
    ]

    base_cols = [
        c
        for c in data.columns
        if c
        not in {
            "Date",
            "Revenue",
            "COGS",
        }
        and not c.startswith("std_")
        and not c.startswith("rec_")
    ]
    # Baseline excludes exogenous promo/aux blocks to measure incremental impact
    base_cols = [c for c in base_cols if c not in promo_cols and c not in aux_cols]

    records = []

    for fold in FOLDS:
        print(f"\n[{fold.name}] building fold-specific features...")
        tr_end = pd.Timestamp(fold.train_end)
        train_slice = data[data["Date"] <= tr_end][
            ["Date", "Revenue", "dayofweek", "month", "weekofyear"]
        ].copy()

        std_profiles = make_standard_profiles(train_slice)
        rec_profiles = make_recency_profiles(train_slice)

        fold_df_std = apply_profiles(data, std_profiles)
        fold_df_rec = apply_profiles(data, rec_profiles)

        std_cols = [c for c in fold_df_std.columns if c.startswith("std_")]
        rec_cols = [c for c in fold_df_rec.columns if c.startswith("rec_")]

        methods: list[tuple[str, pd.DataFrame, list[str], str]] = [
            ("baseline_autoreg", data, base_cols, "calendar + lag + rolling + growth"),
            (
                "baseline_plus_promo_known_future",
                data,
                base_cols + promo_cols,
                "adds promo/event features available for all future dates",
            ),
            (
                "baseline_plus_std_profiles",
                fold_df_std,
                base_cols + std_cols,
                "adds train-only standard seasonal profiles",
            ),
            (
                "baseline_plus_recency_profiles",
                fold_df_rec,
                base_cols + rec_cols,
                "adds train-only recency-weighted seasonal profiles",
            ),
            (
                "baseline_plus_orders_payments_lags",
                data,
                base_cols + orders_cols,
                "adds lagged orders and payment aggregates",
            ),
            (
                "baseline_plus_returns_lags",
                data,
                base_cols + returns_cols,
                "adds lagged returns and refunds aggregates",
            ),
            (
                "baseline_plus_shipments_lags",
                data,
                base_cols + shipments_cols,
                "adds lagged shipment aggregates",
            ),
            (
                "baseline_plus_web_lags",
                data,
                base_cols + web_cols,
                "adds lagged web traffic aggregates",
            ),
            (
                "baseline_plus_all_aux_lags",
                data,
                base_cols + aux_cols,
                "adds all lagged auxiliary aggregates",
            ),
            (
                "baseline_plus_promo_std_profiles",
                fold_df_std,
                base_cols + promo_cols + std_cols,
                "promo/event + train-only standard profiles",
            ),
            (
                "kitchen_sink_recency",
                fold_df_rec,
                base_cols + promo_cols + aux_cols + rec_cols,
                "all tested blocks together with recency profiles",
            ),
        ]

        baseline_mae = None
        for method_name, method_df, method_cols, note in methods:
            result = evaluate_fold(method_df, fold, method_cols)
            result["method"] = method_name
            result["note"] = note
            if method_name == "baseline_autoreg":
                baseline_mae = result["mae"]
            result["delta_vs_fold_baseline_mae"] = (
                float(result["mae"] - baseline_mae)
                if baseline_mae is not None
                else np.nan
            )
            records.append(result)
            print(
                f"  {method_name:<38} MAE={result['mae']:,.2f} "
                f"delta={result['delta_vs_fold_baseline_mae']:+,.2f} "
                f"features={result['n_features']}"
            )

    log_df = pd.DataFrame(records)
    log_df = log_df.sort_values(["fold", "mae", "method"]).reset_index(drop=True)
    log_path = TRACK_DIR / "method_log.csv"
    log_df.to_csv(log_path, index=False)

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
        )
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )
    summary_path = TRACK_DIR / "method_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.iloc[0].to_dict()
    worst = summary.iloc[-1].to_dict()

    notes = [
        "# Feature Engineering Deepwork Log",
        "",
        "## Branch",
        "feature-engineering",
        "",
        "## Evaluation Setup",
        "- Model: HistGradientBoostingRegressor",
        "- Folds: 2020, 2021, 2022 walk-forward holdouts",
        "- Metric priority: MAE (lower is better)",
        "",
        "## Best Method",
        f"- method: {best['method']}",
        f"- mean_mae: {best['mean_mae']:.2f}",
        f"- mean_delta_vs_baseline: {best['mean_delta_vs_baseline']:+.2f}",
        "",
        "## Worst Method",
        f"- method: {worst['method']}",
        f"- mean_mae: {worst['mean_mae']:.2f}",
        f"- mean_delta_vs_baseline: {worst['mean_delta_vs_baseline']:+.2f}",
        "",
        "## Files",
        "- method_log.csv: fold-level metrics per method",
        "- method_summary.csv: mean ranking across folds",
    ]
    notes_path = TRACK_DIR / "notes.md"
    notes_path.write_text("\n".join(notes), encoding="utf-8")

    meta = {
        "log_path": str(log_path),
        "summary_path": str(summary_path),
        "notes_path": str(notes_path),
        "n_methods": int(summary.shape[0]),
        "n_rows": int(log_df.shape[0]),
    }
    (TRACK_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "-" * 72)
    print(f"Saved log:     {log_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved notes:   {notes_path}")
    print("-" * 72)
    print("Top methods by mean MAE:")
    print(summary.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
