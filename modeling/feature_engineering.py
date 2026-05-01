"""
Feature engineering v3: all features are computable for ANY date (train or test).
No features that require unknown same-day targets at prediction time.

Four categories:
  1. Calendar + Fourier: known for any date
  2. Target lags + rolling: available via recursive prediction
  3. Historical patterns: static profiles indexed by calendar keys
  4. Promotion calendar features: known in advance from promotions table
"""

import numpy as np
import pandas as pd
from modeling.config import FILES, LAG_DAYS, ROLLING_WINDOWS


def _load_promotions_table():
    """Load normalized promotions table used by date-based promo features."""
    try:
        promos = pd.read_csv(
            FILES["promotions"],
            parse_dates=["start_date", "end_date"],
            usecols=["promo_type", "discount_value", "start_date", "end_date"],
        )
    except Exception:
        return pd.DataFrame(
            columns=["promo_type", "discount_value", "start_date", "end_date"]
        )

    promos = promos.dropna(subset=["start_date", "end_date"]).copy()
    promos["promo_type"] = (
        promos["promo_type"].fillna("").astype(str).str.lower().str.strip()
    )
    promos["discount_value"] = pd.to_numeric(
        promos["discount_value"], errors="coerce"
    ).fillna(0.0)
    return promos.reset_index(drop=True)


def _compute_promo_features_for_dates(date_series, promos):
    """Build promotion features for arbitrary date series."""
    dates = pd.to_datetime(pd.Series(date_series))
    out = pd.DataFrame({"Date": np.sort(dates.unique())})

    if out.empty:
        return out

    out["active_promo_count"] = 0
    out["active_pct_count"] = 0
    out["active_fixed_count"] = 0
    out["active_discount_sum"] = 0.0
    out["days_to_next_promo_start"] = np.nan
    out["days_since_last_promo_end"] = np.nan

    if promos is None or promos.empty:
        out["is_promo_active"] = 0
        return out

    for _, row in promos.iterrows():
        mask = (out["Date"] >= row["start_date"]) & (out["Date"] <= row["end_date"])
        out.loc[mask, "active_promo_count"] += 1
        out.loc[mask, "active_discount_sum"] += float(row.get("discount_value", 0.0))

        ptype = row.get("promo_type", "")
        if ptype == "percentage":
            out.loc[mask, "active_pct_count"] += 1
        elif ptype == "fixed":
            out.loc[mask, "active_fixed_count"] += 1

    date_values = out["Date"].values.astype("datetime64[ns]")
    starts = np.sort(promos["start_date"].dropna().values.astype("datetime64[ns]"))
    ends = np.sort(promos["end_date"].dropna().values.astype("datetime64[ns]"))

    if len(starts) > 0:
        next_idx = np.searchsorted(starts, date_values, side="left")
        has_next = next_idx < len(starts)
        capped_next_idx = np.minimum(next_idx, len(starts) - 1)
        next_dates = starts[capped_next_idx]
        out.loc[has_next, "days_to_next_promo_start"] = (
            (next_dates[has_next] - date_values[has_next])
            .astype("timedelta64[D]")
            .astype(float)
        )

    if len(ends) > 0:
        prev_idx = np.searchsorted(ends, date_values, side="right") - 1
        has_prev = prev_idx >= 0
        capped_prev_idx = np.maximum(prev_idx, 0)
        prev_dates = ends[capped_prev_idx]
        out.loc[has_prev, "days_since_last_promo_end"] = (
            (date_values[has_prev] - prev_dates[has_prev])
            .astype("timedelta64[D]")
            .astype(float)
        )

    out["is_promo_active"] = (out["active_promo_count"] > 0).astype(int)
    return out


def add_promo_interaction_features(df):
    """Add richer promo interaction features that remain known for future dates."""
    out = df.copy()

    required_defaults = {
        "active_promo_count": 0.0,
        "active_discount_sum": 0.0,
        "is_weekend": 0.0,
        "is_month_end": 0.0,
        "dayofweek": 0.0,
        "days_to_next_promo_start": np.nan,
        "days_since_last_promo_end": np.nan,
    }
    for col, default in required_defaults.items():
        if col not in out.columns:
            out[col] = default

    out["promo_discount_per_active"] = out["active_discount_sum"] / out[
        "active_promo_count"
    ].replace(0, np.nan)
    out["promo_weekend_discount"] = out["active_discount_sum"] * out["is_weekend"]
    out["promo_monthend_discount"] = out["active_discount_sum"] * out["is_month_end"]
    out["promo_dow_pressure"] = out["active_promo_count"] * (out["dayofweek"] + 1)
    out["next_promo_7d"] = out["days_to_next_promo_start"].between(0, 7).astype(int)
    out["after_promo_7d"] = out["days_since_last_promo_end"].between(0, 7).astype(int)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. Calendar + Fourier (known for any date)
# ─────────────────────────────────────────────────────────────────────────────


def build_calendar_features(df):
    """Time/calendar features from Date column."""
    d = df["Date"]
    df["dayofweek"] = d.dt.dayofweek
    df["dayofmonth"] = d.dt.day
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)

    # Cyclical encoding
    for col, period in [("dayofweek", 7), ("month", 12), ("dayofyear", 365)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    # Fourier terms for yearly seasonality (capture complex patterns)
    for k in range(1, 6):  # 5 harmonics
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)

    # Trend: days since start (linear trend feature)
    df["trend"] = (d - d.min()).dt.days

    # Holidays
    df["is_national_holiday"] = (
        ((df["month"] == 9) & (df["dayofmonth"] == 2))
        | ((df["month"] == 4) & (df["dayofmonth"] == 30))
        | ((df["month"] == 5) & (df["dayofmonth"] == 1))
    ).astype(int)

    TET_DATES = [
        "2012-01-23",
        "2013-02-10",
        "2014-01-31",
        "2015-02-19",
        "2016-02-08",
        "2017-01-28",
        "2018-02-16",
        "2019-02-05",
        "2020-01-25",
        "2021-02-12",
        "2022-02-01",
        "2023-01-22",
        "2024-02-10",
        "2025-01-29",
    ]
    tet_dates = pd.to_datetime(TET_DATES)

    df["days_to_tet"] = np.nan
    df["days_since_tet"] = np.nan

    dvals = d.values.astype("datetime64[ns]")
    starts = tet_dates.values.astype("datetime64[ns]")

    next_idx = np.searchsorted(starts, dvals, side="left")
    has_next = next_idx < len(starts)
    capped_next_idx = np.minimum(next_idx, len(starts) - 1)
    next_dates = starts[capped_next_idx]
    df.loc[has_next, "days_to_tet"] = (
        (next_dates[has_next] - dvals[has_next]).astype("timedelta64[D]").astype(float)
    )

    prev_idx = np.searchsorted(starts, dvals, side="right") - 1
    has_prev = prev_idx >= 0
    capped_prev_idx = np.maximum(prev_idx, 0)
    prev_dates = starts[capped_prev_idx]
    df.loc[has_prev, "days_since_tet"] = (
        (dvals[has_prev] - prev_dates[has_prev]).astype("timedelta64[D]").astype(float)
    )

    df["is_tet_week"] = ((df["days_to_tet"] <= 3) | (df["days_since_tet"] <= 4)).astype(
        int
    )

    # Tet Ramp Up (peak 10 days before Tet)
    df["tet_ramp_up_intensity"] = np.exp(-0.5 * ((df["days_to_tet"] - 10) / 4) ** 2)
    df["tet_ramp_up_intensity"] = df["tet_ramp_up_intensity"].fillna(0)

    # Tet Hangover (decay starting 2 days after Tet)
    df["tet_hangover_penalty"] = np.exp(-0.5 * ((df["days_since_tet"] - 2) / 3) ** 2)
    df["tet_hangover_penalty"] = df["tet_hangover_penalty"].fillna(0)

    # Mega Double Dates (9.9, 10.10, 11.11, 12.12)
    double_dates = [
        f"{y}-{m:02d}-{m:02d}" for y in range(2012, 2026) for m in [9, 10, 11, 12]
    ]
    double_dates = pd.to_datetime(double_dates)

    dd_starts = double_dates.values.astype("datetime64[ns]")

    dd_next_idx = np.searchsorted(dd_starts, dvals, side="left")
    dd_has_next = dd_next_idx < len(dd_starts)
    dd_capped_next_idx = np.minimum(dd_next_idx, len(dd_starts) - 1)
    dd_next_dates = dd_starts[dd_capped_next_idx]

    df["days_to_next_mega_double"] = np.nan
    df.loc[dd_has_next, "days_to_next_mega_double"] = (
        (dd_next_dates[dd_has_next] - dvals[dd_has_next])
        .astype("timedelta64[D]")
        .astype(float)
    )

    # Pre-sale slump (1-4 days before double date)
    df["is_pre_double_date_slump"] = (
        df["days_to_next_mega_double"].between(1, 4)
    ).astype(int)
    df["is_mega_double_date"] = (df["days_to_next_mega_double"] == 0).astype(int)

    # Payday Rhythms (Vietnamese paydays usually 25th)
    df["is_payday_window"] = (
        (df["dayofmonth"] >= 25) | (df["dayofmonth"] <= 5)
    ).astype(int)

    def _days_since_25th(date):
        if date.day >= 25:
            return date.day - 25
        else:
            prev_month = date - pd.DateOffset(months=1)
            prev_25 = pd.Timestamp(year=prev_month.year, month=prev_month.month, day=25)
            return (date - prev_25).days

    df["days_since_payday"] = d.apply(_days_since_25th)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Target lags + rolling (available via recursive prediction)
# ─────────────────────────────────────────────────────────────────────────────


def build_lag_features(df, col="Revenue", lags=None):
    """Lag features for target column."""
    if lags is None:
        lags = LAG_DAYS
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def build_rolling_features(df, col="Revenue", windows=None):
    """Rolling statistics for target column."""
    if windows is None:
        windows = ROLLING_WINDOWS
    for w in windows:
        shifted = df[col].shift(1)
        df[f"{col}_rmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"{col}_rstd_{w}"] = shifted.rolling(w, min_periods=1).std()
        df[f"{col}_rmin_{w}"] = shifted.rolling(w, min_periods=1).min()
        df[f"{col}_rmax_{w}"] = shifted.rolling(w, min_periods=1).max()
        df[f"{col}_rmedian_{w}"] = shifted.rolling(w, min_periods=1).median()
    return df


def build_growth_features(df, col="Revenue"):
    """Growth rates and momentum."""
    df[f"{col}_yoy_ratio"] = df[col].shift(1) / df[col].shift(366).replace(0, np.nan)
    df[f"{col}_wow_ratio"] = df[col].shift(1) / df[col].shift(8).replace(0, np.nan)
    df[f"{col}_mom_ratio"] = df[col].shift(1) / df[col].shift(31).replace(0, np.nan)

    # Momentum: recent avg vs older avg
    recent = df[col].shift(1).rolling(7, min_periods=1).mean()
    older = df[col].shift(8).rolling(28, min_periods=1).mean()
    df[f"{col}_momentum"] = recent / older.replace(0, np.nan)

    # Diff features
    df[f"{col}_diff_1"] = df[col].shift(1) - df[col].shift(2)
    df[f"{col}_diff_7"] = df[col].shift(1) - df[col].shift(8)

    # Regime / Volatility features
    col_l1 = df[col].shift(1)
    col_mean_28 = col_l1.rolling(28, min_periods=7).mean()
    col_std_28 = col_l1.rolling(28, min_periods=7).std()

    df[f"{col}_vol_14"] = col_l1.rolling(14, min_periods=4).std()
    df[f"{col}_vol_28"] = col_std_28
    df[f"{col}_trend_7_28"] = col_l1.rolling(
        7, min_periods=3
    ).mean() / col_mean_28.replace(0, np.nan)
    df[f"{col}_zscore_28"] = (col_l1 - col_mean_28) / col_std_28.replace(0, np.nan)

    # Cross-target features if applicable
    if col == "Revenue" and "COGS" in df.columns:
        cogs_l1 = df["COGS"].shift(1)
        cogs_mean_28 = cogs_l1.rolling(28, min_periods=7).mean()
        df["cogs_trend_7_28"] = cogs_l1.rolling(
            7, min_periods=3
        ).mean() / cogs_mean_28.replace(0, np.nan)
        df["rev_cogs_spread_lag1"] = col_l1 - cogs_l1
        df["rev_cogs_ratio_lag1"] = col_l1 / cogs_l1.replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Historical patterns from auxiliary tables
#    Computed ONCE from training data, then merged by day-of-week / month / etc.
# ─────────────────────────────────────────────────────────────────────────────


def compute_historical_profiles(train_df):
    """
    Compute static profiles from training data.
    These are averages indexed by calendar features, applicable to any future date.

    Returns dict of DataFrames ready to merge.
    """
    df = train_df.copy()
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dayofmonth"] = df["Date"].dt.day
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    profiles = {}

    # Revenue/COGS by day-of-week
    profiles["dow"] = (
        df.groupby("dayofweek")
        .agg(
            rev_dow_mean=("Revenue", "mean"),
            rev_dow_std=("Revenue", "std"),
            rev_dow_median=("Revenue", "median"),
            cogs_dow_mean=("COGS", "mean"),
        )
        .reset_index()
    )

    # Revenue/COGS by month
    profiles["month"] = (
        df.groupby("month")
        .agg(
            rev_month_mean=("Revenue", "mean"),
            rev_month_std=("Revenue", "std"),
            rev_month_median=("Revenue", "median"),
            cogs_month_mean=("COGS", "mean"),
            cogs_month_std=("COGS", "std"),
        )
        .reset_index()
    )

    # Revenue by week-of-year (captures fine-grained seasonality)
    profiles["woy"] = (
        df.groupby("weekofyear")
        .agg(
            rev_woy_mean=("Revenue", "mean"),
            rev_woy_std=("Revenue", "std"),
            cogs_woy_mean=("COGS", "mean"),
        )
        .reset_index()
    )

    # Revenue by day-of-month
    profiles["dom"] = (
        df.groupby("dayofmonth")
        .agg(
            rev_dom_mean=("Revenue", "mean"),
            cogs_dom_mean=("COGS", "mean"),
        )
        .reset_index()
    )

    # Revenue by quarter
    profiles["quarter"] = (
        df.groupby("quarter")
        .agg(
            rev_qtr_mean=("Revenue", "mean"),
            cogs_qtr_mean=("COGS", "mean"),
        )
        .reset_index()
    )

    # Revenue by (month, dayofweek) — fine-grained seasonal pattern
    profiles["month_dow"] = (
        df.groupby(["month", "dayofweek"])
        .agg(
            rev_month_dow_mean=("Revenue", "mean"),
            cogs_month_dow_mean=("COGS", "mean"),
        )
        .reset_index()
    )

    # Weekend × Month interaction (captures large weekday/weekend gap per month)
    profiles["month_weekend"] = (
        df.groupby(["month", "is_weekend"])
        .agg(
            rev_month_weekend_mean=("Revenue", "mean"),
            cogs_month_weekend_mean=("COGS", "mean"),
        )
        .reset_index()
    )

    # Margin profile by month (Revenue-COGS spread varies hugely: -5% Aug to +20% May)
    df["_margin_pct"] = (df["Revenue"] - df["COGS"]) / df["Revenue"].replace(0, np.nan) * 100
    profiles["margin_month"] = (
        df.groupby("month")
        .agg(
            margin_pct_month_mean=("_margin_pct", "mean"),
            margin_pct_month_std=("_margin_pct", "std"),
        )
        .reset_index()
    )

    return profiles


def compute_order_based_profiles():
    """Compute profile features from the orders and order_items tables.
    
    These capture AOV (average order value), order count patterns,
    and category mix seasonality — all indexed by calendar keys.
    """
    profiles = {}
    
    try:
        orders = pd.read_csv(
            FILES["orders"],
            parse_dates=["order_date"],
            usecols=["order_id", "order_date"],
        )
        orders["dayofweek"] = orders["order_date"].dt.dayofweek
        orders["month"] = orders["order_date"].dt.month
        
        daily_orders = orders.groupby("order_date").agg(
            n_orders=("order_id", "count"),
        ).reset_index()
        daily_orders["dayofweek"] = daily_orders["order_date"].dt.dayofweek
        daily_orders["month"] = daily_orders["order_date"].dt.month
        
        # Order count by (month, dow)
        profiles["orders_month_dow"] = (
            daily_orders.groupby(["month", "dayofweek"])
            .agg(avg_orders_month_dow=("n_orders", "mean"))
            .reset_index()
        )
        
    except Exception as e:
        print(f"  Warning: could not compute order-based profiles — {e}")
    
    # Category mix seasonality (Outdoor share varies 17% summer → 55% December)
    try:
        oi = pd.read_csv(
            FILES["order_items"],
            usecols=["order_id", "product_id", "quantity"],
            low_memory=False,
        )
        products = pd.read_csv(
            FILES["products"],
            usecols=["product_id", "category"],
        )
        oi_prod = oi.merge(products, on="product_id", how="left")
        oi_orders = oi_prod.merge(
            orders[["order_id", "order_date"]], on="order_id"
        )
        oi_orders["month"] = oi_orders["order_date"].dt.month
        
        monthly_cat = oi_orders.groupby(["month", "category"]).agg(
            n_items=("product_id", "count")
        ).reset_index()
        monthly_total = monthly_cat.groupby("month")["n_items"].sum().reset_index(
            name="total_items"
        )
        monthly_cat = monthly_cat.merge(monthly_total, on="month")
        monthly_cat["cat_share"] = monthly_cat["n_items"] / monthly_cat["total_items"]
        
        # Pivot to get category shares as separate columns
        cat_pivot = monthly_cat.pivot_table(
            index="month", columns="category", values="cat_share", fill_value=0
        ).reset_index()
        cat_pivot.columns = [
            f"catshare_{c.lower().replace(' ', '_')}" if c != "month" else c
            for c in cat_pivot.columns
        ]
        profiles["category_month"] = cat_pivot
        
    except Exception as e:
        print(f"  Warning: could not compute category profiles — {e}")
    
    return profiles


def compute_aux_profiles():
    """
    Extract static patterns from auxiliary tables.
    These are indexed by calendar features so they apply to any date.
    """
    profiles = {}

    def _robust_winsorize(s: pd.Series, k: float = 8.0) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").astype(float)
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0:
            return x
        lo = med - k * sigma
        hi = med + k * sigma
        return x.clip(lo, hi)

    def _align_daily_index(df: pd.DataFrame, idx_name: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        full = pd.date_range(start, end, freq="D")
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        out = out.reindex(full)
        out.index.name = idx_name
        return out

    # ── Orders: average order count & cancel rate by day-of-week and month ──
    try:
        orders = pd.read_csv(
            FILES["orders"],
            parse_dates=["order_date"],
            usecols=["order_id", "order_date", "order_status"],
        )
        orders["dayofweek"] = orders["order_date"].dt.dayofweek
        orders["month"] = orders["order_date"].dt.month

        daily_orders = orders.groupby("order_date").agg(
            order_count=("order_id", "count"),
            cancel_rate=("order_status", lambda x: (x == "cancelled").mean()),
        )
        daily_orders = _align_daily_index(
            daily_orders,
            "order_date",
            orders["order_date"].min(),
            orders["order_date"].max(),
        )
        # Keep truly-missing days as NaN (NOT zeros) so profile means don't get biased low
        daily_orders["order_count"] = _robust_winsorize(daily_orders["order_count"], k=8.0)
        daily_orders["cancel_rate"] = pd.to_numeric(daily_orders["cancel_rate"], errors="coerce")
        daily_orders["dayofweek"] = daily_orders.index.dayofweek
        daily_orders["month"] = daily_orders.index.month

        profiles["orders_dow"] = (
            daily_orders.groupby("dayofweek")
            .agg(
                avg_orders_dow=("order_count", "mean"),
                avg_cancel_rate_dow=("cancel_rate", "mean"),
            )
            .reset_index()
        )

        profiles["orders_month"] = (
            daily_orders.groupby("month")
            .agg(
                avg_orders_month=("order_count", "mean"),
                avg_cancel_rate_month=("cancel_rate", "mean"),
            )
            .reset_index()
        )
    except Exception as e:
        print(f"  Warning: could not process orders — {e}")

    # ── Returns: return rate pattern by month ──
    try:
        returns = pd.read_csv(
            FILES["returns"],
            parse_dates=["return_date"],
            usecols=["return_id", "return_date", "refund_amount"],
        )
        daily_returns = returns.groupby("return_date").agg(
            return_count=("return_id", "count"),
            avg_refund=("refund_amount", "mean"),
        )
        daily_returns = _align_daily_index(
            daily_returns,
            "return_date",
            returns["return_date"].min(),
            returns["return_date"].max(),
        )
        # Keep missing days as NaN to avoid biasing monthly averages toward zero
        daily_returns["return_count"] = _robust_winsorize(daily_returns["return_count"], k=8.0)
        daily_returns["avg_refund"] = pd.to_numeric(daily_returns["avg_refund"], errors="coerce")
        daily_returns["month"] = daily_returns.index.month
        profiles["returns_month"] = (
            daily_returns.groupby("month")
            .agg(
                avg_returns_month=("return_count", "mean"),
                avg_refund_month=("avg_refund", "mean"),
            )
            .reset_index()
        )
    except Exception as e:
        print(f"  Warning: could not process returns — {e}")

    # ── Web traffic: session patterns by day-of-week and month ──
    try:
        wt = pd.read_csv(FILES["web_traffic"], parse_dates=["date"])
        daily_wt = wt.groupby("date").agg(
            total_sessions=("sessions", "sum"),
            total_visitors=("unique_visitors", "sum"),
            avg_bounce=("bounce_rate", "mean"),
        )
        daily_wt = _align_daily_index(
            daily_wt,
            "date",
            wt["date"].min(),
            wt["date"].max(),
        )
        daily_wt["total_sessions"] = _robust_winsorize(
            pd.to_numeric(daily_wt["total_sessions"], errors="coerce"),
            k=8.0,
        )
        daily_wt["total_visitors"] = _robust_winsorize(
            pd.to_numeric(daily_wt["total_visitors"], errors="coerce"),
            k=8.0,
        )
        daily_wt["avg_bounce"] = pd.to_numeric(daily_wt["avg_bounce"], errors="coerce")
        daily_wt["dayofweek"] = daily_wt.index.dayofweek
        daily_wt["month"] = daily_wt.index.month

        profiles["traffic_dow"] = (
            daily_wt.groupby("dayofweek")
            .agg(
                avg_sessions_dow=("total_sessions", "mean"),
                avg_visitors_dow=("total_visitors", "mean"),
            )
            .reset_index()
        )

        profiles["traffic_month"] = (
            daily_wt.groupby("month")
            .agg(
                avg_sessions_month=("total_sessions", "mean"),
                avg_bounce_month=("avg_bounce", "mean"),
            )
            .reset_index()
        )
    except Exception as e:
        print(f"  Warning: could not process web_traffic — {e}")

    # ── Promotions: count of active promos by month ──
    try:
        promos = pd.read_csv(
            FILES["promotions"], parse_dates=["start_date", "end_date"]
        )
        promo_months = []
        for m in range(1, 13):
            active = promos[
                (promos["start_date"].dt.month <= m)
                & (promos["end_date"].dt.month >= m)
            ]
            promo_months.append({"month": m, "avg_promos_month": len(active) / 10})
        profiles["promos_month"] = pd.DataFrame(promo_months)
    except Exception as e:
        print(f"  Warning: could not process promotions — {e}")

    # ── Inventory: avg stockout rate by month ──
    try:
        inv = pd.read_csv(
            FILES["inventory"],
            usecols=["month", "stockout_flag", "sell_through_rate", "fill_rate"],
        )
        profiles["inv_month"] = (
            inv.groupby("month")
            .agg(
                avg_stockout_rate_month=("stockout_flag", "mean"),
                avg_sell_through_month=("sell_through_rate", "mean"),
                avg_fill_rate_month=("fill_rate", "mean"),
            )
            .reset_index()
        )
    except Exception as e:
        print(f"  Warning: could not process inventory — {e}")

    return profiles


def merge_profiles(df, profiles, merge_key, prefix=""):
    """Merge a profile DataFrame onto the main DataFrame."""
    if isinstance(merge_key, str):
        merge_key = [merge_key]
    for key in merge_key:
        if key not in df.columns:
            return df
    return df.merge(profiles, on=merge_key, how="left")


def build_feature_table(train_df, verbose=True, profile_source_df=None):
    """
    Build full feature table. ALL features are computable for any date.

    Parameters
    ----------
    train_df : DataFrame with Date, Revenue, COGS (training data).
    verbose : bool, print progress.
    profile_source_df : optional DataFrame used to compute calendar profiles.
        If None, defaults to train_df. For leakage-safe validation, pass only
        pre-validation rows here.

    Returns
    -------
    df : DataFrame with features
    profiles : dict of profile DataFrames (needed for test prediction)
    """
    df = train_df.copy().sort_values("Date").reset_index(drop=True)
    profile_source = train_df if profile_source_df is None else profile_source_df
    profile_source = profile_source.copy().sort_values("Date").reset_index(drop=True)

    if verbose:
        print("  Calendar + Fourier features...")
    df = build_calendar_features(df)

    if verbose:
        print("  Promotion calendar features...")
    promotions = _load_promotions_table()
    promo_features = _compute_promo_features_for_dates(df["Date"], promotions)
    df = df.merge(promo_features, on="Date", how="left")
    df = add_promo_interaction_features(df)

    if verbose:
        print("  Lag features...")
    df = build_lag_features(df, "Revenue")
    df = build_lag_features(df, "COGS")

    if verbose:
        print("  Rolling features...")
    df = build_rolling_features(df, "Revenue")
    df = build_rolling_features(df, "COGS", windows=[7, 14, 28, 60, 90, 180, 365])

    if verbose:
        print("  Growth features...")
    df = build_growth_features(df, "Revenue")
    df = build_growth_features(df, "COGS")

    # Revenue-COGS spread features (margin dynamics)
    if "Revenue" in df.columns and "COGS" in df.columns:
        spread = df["Revenue"].shift(1) - df["COGS"].shift(1)
        df["spread_lag1"] = spread
        df["spread_rmean_7"] = spread.rolling(7, min_periods=1).mean()
        df["spread_rmean_28"] = spread.rolling(28, min_periods=1).mean()
        ratio = df["Revenue"].shift(1) / df["COGS"].shift(1).replace(0, np.nan)
        df["margin_ratio_lag1"] = ratio
        df["margin_ratio_rmean_7"] = ratio.rolling(7, min_periods=1).mean()
        df["margin_ratio_rmean_28"] = ratio.rolling(28, min_periods=1).mean()

    if verbose:
        print("  Historical revenue profiles...")
    rev_profiles = compute_historical_profiles(profile_source)
    df = merge_profiles(df, rev_profiles["dow"], "dayofweek")
    df = merge_profiles(df, rev_profiles["month"], "month")
    df = merge_profiles(df, rev_profiles["woy"], "weekofyear")
    df = merge_profiles(df, rev_profiles["dom"], "dayofmonth")
    df = merge_profiles(df, rev_profiles["quarter"], "quarter")
    df = merge_profiles(df, rev_profiles["month_dow"], ["month", "dayofweek"])

    if verbose:
        print("  Auxiliary table profiles...")
    aux_profiles = compute_aux_profiles()
    for name, prof in aux_profiles.items():
        key_col = [c for c in prof.columns if c in df.columns]
        if key_col:
            df = merge_profiles(df, prof, key_col)

    if verbose:
        print("  Order-based profiles...")
    order_profiles = compute_order_based_profiles()
    for name, prof in order_profiles.items():
        key_col = [c for c in prof.columns if c in df.columns]
        if key_col:
            df = merge_profiles(df, prof, key_col)

    # Combine all profiles for reuse at prediction time
    all_profiles = {
        "__promotions__": promotions,
        **rev_profiles,
        **aux_profiles,
        **order_profiles,
    }

    if verbose:
        print(f"  Feature table: {df.shape[0]} rows × {df.shape[1]} cols")

    return df, all_profiles


def apply_profiles_to_dates(dates_df, profiles):
    """
    Apply precomputed profiles to a DataFrame with calendar features.
    Used during prediction to get the same profile features.
    """
    df = dates_df.copy()

    # Promo features are known in advance and can be recomputed for any date.
    promotions = profiles.get("__promotions__") if isinstance(profiles, dict) else None
    if promotions is not None:
        promo_features = _compute_promo_features_for_dates(df["Date"], promotions)
        df = df.merge(promo_features, on="Date", how="left")
        df = add_promo_interaction_features(df)

    profile_merge_keys = {
        "dow": "dayofweek",
        "month": "month",
        "woy": "weekofyear",
        "rec_dow": "dayofweek",
        "rec_month": "month",
        "rec_woy": "weekofyear",
        "rec_month_dow": ["month", "dayofweek"],
        "dom": "dayofmonth",
        "quarter": "quarter",
        "month_dow": ["month", "dayofweek"],
        "month_weekend": ["month", "is_weekend"],
        "margin_month": "month",
        "orders_dow": "dayofweek",
        "orders_month": "month",
        "orders_month_dow": ["month", "dayofweek"],
        "category_month": "month",
        "returns_month": "month",
        "traffic_dow": "dayofweek",
        "traffic_month": "month",
        "promos_month": "month",
        "inv_month": "month",
    }

    for name, prof in profiles.items():
        if name.startswith("__"):
            continue

        key_cols = profile_merge_keys.get(name)
        if key_cols is None:
            # Fallback: use non-numeric cols as keys.
            numeric_cols = prof.select_dtypes(include=[np.number]).columns.tolist()
            key_cols = [c for c in prof.columns if c not in numeric_cols]

        if isinstance(key_cols, str):
            key_cols = [key_cols]

        key_cols = [c for c in key_cols if c in df.columns and c in prof.columns]
        if key_cols:
            df = df.merge(prof, on=key_cols, how="left")
    return df


def get_feature_cols(df):
    """Return feature column names (everything except target + date)."""
    exclude = {"Date", "Revenue", "COGS"}
    return [c for c in df.columns if c not in exclude]
