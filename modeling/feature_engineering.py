"""
Feature engineering v2: all features are computable for ANY date (train or test).
No features that require same-day auxiliary data (those vanish at prediction time).

Three categories:
  1. Calendar + Fourier: known for any date
  2. Target lags + rolling: available via recursive prediction
  3. Historical patterns: static profiles learned from training data, indexed by
     day-of-week / month / week-of-year so they apply to future dates too
"""
import numpy as np
import pandas as pd
from modeling.config import FILES, LAG_DAYS, ROLLING_WINDOWS


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

    profiles = {}

    # Revenue/COGS by day-of-week
    profiles["dow"] = df.groupby("dayofweek").agg(
        rev_dow_mean=("Revenue", "mean"),
        rev_dow_std=("Revenue", "std"),
        rev_dow_median=("Revenue", "median"),
        cogs_dow_mean=("COGS", "mean"),
    ).reset_index()

    # Revenue/COGS by month
    profiles["month"] = df.groupby("month").agg(
        rev_month_mean=("Revenue", "mean"),
        rev_month_std=("Revenue", "std"),
        rev_month_median=("Revenue", "median"),
        cogs_month_mean=("COGS", "mean"),
    ).reset_index()

    # Revenue by week-of-year (captures fine-grained seasonality)
    profiles["woy"] = df.groupby("weekofyear").agg(
        rev_woy_mean=("Revenue", "mean"),
        rev_woy_std=("Revenue", "std"),
        cogs_woy_mean=("COGS", "mean"),
    ).reset_index()

    # Revenue by day-of-month
    profiles["dom"] = df.groupby("dayofmonth").agg(
        rev_dom_mean=("Revenue", "mean"),
        cogs_dom_mean=("COGS", "mean"),
    ).reset_index()

    # Revenue by quarter
    profiles["quarter"] = df.groupby("quarter").agg(
        rev_qtr_mean=("Revenue", "mean"),
        cogs_qtr_mean=("COGS", "mean"),
    ).reset_index()

    # Revenue by (month, dayofweek) — fine-grained seasonal pattern
    profiles["month_dow"] = df.groupby(["month", "dayofweek"]).agg(
        rev_month_dow_mean=("Revenue", "mean"),
    ).reset_index()

    return profiles


def compute_aux_profiles():
    """
    Extract static patterns from auxiliary tables.
    These are indexed by calendar features so they apply to any date.
    """
    profiles = {}

    # ── Orders: average order count & cancel rate by day-of-week and month ──
    try:
        orders = pd.read_csv(FILES["orders"], parse_dates=["order_date"],
                             usecols=["order_id", "order_date", "order_status"])
        orders["dayofweek"] = orders["order_date"].dt.dayofweek
        orders["month"] = orders["order_date"].dt.month

        daily_orders = orders.groupby("order_date").agg(
            order_count=("order_id", "count"),
            cancel_rate=("order_status", lambda x: (x == "cancelled").mean()),
        )
        daily_orders["dayofweek"] = daily_orders.index.dayofweek
        daily_orders["month"] = daily_orders.index.month

        profiles["orders_dow"] = daily_orders.groupby("dayofweek").agg(
            avg_orders_dow=("order_count", "mean"),
            avg_cancel_rate_dow=("cancel_rate", "mean"),
        ).reset_index()

        profiles["orders_month"] = daily_orders.groupby("month").agg(
            avg_orders_month=("order_count", "mean"),
            avg_cancel_rate_month=("cancel_rate", "mean"),
        ).reset_index()
    except Exception as e:
        print(f"  Warning: could not process orders — {e}")

    # ── Returns: return rate pattern by month ──
    try:
        returns = pd.read_csv(FILES["returns"], parse_dates=["return_date"],
                              usecols=["return_id", "return_date", "refund_amount"])
        daily_returns = returns.groupby("return_date").agg(
            return_count=("return_id", "count"),
            avg_refund=("refund_amount", "mean"),
        )
        daily_returns["month"] = daily_returns.index.month
        profiles["returns_month"] = daily_returns.groupby("month").agg(
            avg_returns_month=("return_count", "mean"),
            avg_refund_month=("avg_refund", "mean"),
        ).reset_index()
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
        daily_wt["dayofweek"] = daily_wt.index.dayofweek
        daily_wt["month"] = daily_wt.index.month

        profiles["traffic_dow"] = daily_wt.groupby("dayofweek").agg(
            avg_sessions_dow=("total_sessions", "mean"),
            avg_visitors_dow=("total_visitors", "mean"),
        ).reset_index()

        profiles["traffic_month"] = daily_wt.groupby("month").agg(
            avg_sessions_month=("total_sessions", "mean"),
            avg_bounce_month=("avg_bounce", "mean"),
        ).reset_index()
    except Exception as e:
        print(f"  Warning: could not process web_traffic — {e}")

    # ── Promotions: count of active promos by month ──
    try:
        promos = pd.read_csv(FILES["promotions"], parse_dates=["start_date", "end_date"])
        promo_months = []
        for m in range(1, 13):
            active = promos[(promos["start_date"].dt.month <= m) &
                            (promos["end_date"].dt.month >= m)]
            promo_months.append({"month": m, "avg_promos_month": len(active) / 10})
        profiles["promos_month"] = pd.DataFrame(promo_months)
    except Exception as e:
        print(f"  Warning: could not process promotions — {e}")

    # ── Inventory: avg stockout rate by month ──
    try:
        inv = pd.read_csv(FILES["inventory"], usecols=["month", "stockout_flag",
                          "sell_through_rate", "fill_rate"])
        profiles["inv_month"] = inv.groupby("month").agg(
            avg_stockout_rate_month=("stockout_flag", "mean"),
            avg_sell_through_month=("sell_through_rate", "mean"),
            avg_fill_rate_month=("fill_rate", "mean"),
        ).reset_index()
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


def build_feature_table(train_df, verbose=True):
    """
    Build full feature table. ALL features are computable for any date.

    Parameters
    ----------
    train_df : DataFrame with Date, Revenue, COGS (training data).
    verbose : bool, print progress.

    Returns
    -------
    df : DataFrame with features
    profiles : dict of profile DataFrames (needed for test prediction)
    """
    df = train_df.copy().sort_values("Date").reset_index(drop=True)

    if verbose:
        print("  Calendar + Fourier features...")
    df = build_calendar_features(df)

    if verbose:
        print("  Lag features...")
    df = build_lag_features(df, "Revenue")
    df = build_lag_features(df, "COGS")

    if verbose:
        print("  Rolling features...")
    df = build_rolling_features(df, "Revenue")
    df = build_rolling_features(df, "COGS", windows=[7, 14, 28, 90])

    if verbose:
        print("  Growth features...")
    df = build_growth_features(df, "Revenue")

    if verbose:
        print("  Historical revenue profiles...")
    rev_profiles = compute_historical_profiles(train_df)
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

    # Combine all profiles for reuse at prediction time
    all_profiles = {**rev_profiles, **aux_profiles}

    if verbose:
        print(f"  Feature table: {df.shape[0]} rows × {df.shape[1]} cols")

    return df, all_profiles


def apply_profiles_to_dates(dates_df, profiles):
    """
    Apply precomputed profiles to a DataFrame with calendar features.
    Used during prediction to get the same profile features.
    """
    df = dates_df.copy()
    for name, prof in profiles.items():
        key_cols = [c for c in prof.columns if c in df.columns and
                    c not in prof.select_dtypes(include=[np.number]).columns.tolist() + [c for c in prof.columns if c.startswith(("rev_", "cogs_", "avg_", "active_"))]]
        # Simpler: find merge keys (non-metric columns)
        metric_cols = [c for c in prof.columns if any(c.startswith(p) for p in
                       ["rev_", "cogs_", "avg_", "active_"])]
        key_cols = [c for c in prof.columns if c not in metric_cols]
        key_cols = [c for c in key_cols if c in df.columns]
        if key_cols:
            df = df.merge(prof, on=key_cols, how="left")
    return df


def get_feature_cols(df):
    """Return feature column names (everything except target + date)."""
    exclude = {"Date", "Revenue", "COGS"}
    return [c for c in df.columns if c not in exclude]
