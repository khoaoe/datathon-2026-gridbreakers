"""
Feature engineering v3.

All features are computable for ANY date (train, val, test), so the recursive
prediction loop only has to refresh lag/rolling state. Three layers:

  1. Calendar + Fourier + VN holidays + covid flag  (known for any date)
  2. Target lags + rolling + growth                 (recursive at predict time)
  3. Date-exact features                            (daily, merged on Date):
        - Promotion calendar (exploded from promotions.csv, with historical
          per-month fallback for dates past the last observed promo)
        - Inventory monthly snapshot, forward-filled
  4. Calendar-keyed profiles                        (static lookups):
        - Revenue/COGS by dow / month / woy / dom / quarter / (month,dow)
        - Detrended Revenue profile (level-adjusted seasonality)
        - Recent-window Revenue profile (last 3y only, recency-weighted)
        - Order-mix profiles (% paid_search, % mobile, % credit_card ...)
        - Category-mix profile (category revenue share by month)
        - Return rate, web-traffic, promo-density, inventory profiles

Both train and test go through the same builder so feature schemas stay aligned.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from modeling.config import (
    FILES,
    LAG_DAYS,
    ROLLING_WINDOWS,
    COVID_START,
    COVID_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Calendar + Fourier + holidays + covid
# ─────────────────────────────────────────────────────────────────────────────

def _safe_vn_holidays(years):
    """Return a set of Vietnam holiday dates. Falls back to empty if pkg missing."""
    try:
        import holidays

        return holidays.country_holidays("VN", years=list(years))
    except Exception:
        return {}


# Hard-coded Tet (lunar new year) start dates 2010–2025 so features work
# without any network dependency. Tet is the biggest retail signal in VN.
TET_DATES: Dict[int, str] = {
    2010: "2010-02-14", 2011: "2011-02-03", 2012: "2012-01-23",
    2013: "2013-02-10", 2014: "2014-01-31", 2015: "2015-02-19",
    2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16",
    2019: "2019-02-05", 2020: "2020-01-25", 2021: "2021-02-12",
    2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10",
    2025: "2025-01-29",
}


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time/calendar + Fourier + VN holiday + covid features."""
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
    df["is_quarter_start"] = d.dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = d.dt.is_quarter_end.astype(int)
    df["is_year_start"] = d.dt.is_year_start.astype(int)
    df["is_year_end"] = d.dt.is_year_end.astype(int)

    # Cyclical encoding
    for col, period in [("dayofweek", 7), ("month", 12), ("dayofyear", 365)]:
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)

    # Fourier yearly harmonics
    for k in range(1, 6):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)

    # Linear trend
    df["trend"] = (d - pd.Timestamp("2012-07-04")).dt.days

    # Covid window flag (train-only signal, 0 for test)
    covid_mask = (d >= pd.Timestamp(COVID_START)) & (d <= pd.Timestamp(COVID_END))
    df["covid_flag"] = covid_mask.astype(int)

    # VN public holidays
    years = sorted(df["year"].unique())
    vn_holidays = _safe_vn_holidays(years)
    dates_only = d.dt.date
    df["is_vn_holiday"] = dates_only.map(lambda x: int(x in vn_holidays))

    # Tet features (moves with lunar calendar)
    tet_ts = {y: pd.Timestamp(v) for y, v in TET_DATES.items()}
    tet_for_year = df["year"].map(tet_ts)
    days_from_tet = (d - tet_for_year).dt.days
    df["days_from_tet"] = days_from_tet.fillna(0).astype(int)
    df["is_tet_week"] = ((days_from_tet >= -3) & (days_from_tet <= 7)).fillna(False).astype(int)
    df["is_tet_month"] = ((days_from_tet >= -14) & (days_from_tet <= 21)).fillna(False).astype(int)

    # Ecom sale-window heuristics (no data but well-known retail peaks)
    df["is_black_friday_week"] = ((df["month"] == 11) & (df["dayofmonth"] >= 22) &
                                   (df["dayofmonth"] <= 29)).astype(int)
    df["is_double_day"] = (((df["month"] == df["dayofmonth"]) & (df["month"].isin([9, 10, 11, 12]))) |
                           ((df["month"] == 11) & (df["dayofmonth"] == 11))).astype(int)
    df["is_xmas_week"] = ((df["month"] == 12) & (df["dayofmonth"] >= 20) &
                          (df["dayofmonth"] <= 26)).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Target lag / rolling / growth
# ─────────────────────────────────────────────────────────────────────────────

def build_lag_features(df: pd.DataFrame, col: str = "Revenue", lags=None) -> pd.DataFrame:
    if lags is None:
        lags = LAG_DAYS
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def build_rolling_features(df: pd.DataFrame, col: str = "Revenue", windows=None) -> pd.DataFrame:
    if windows is None:
        windows = ROLLING_WINDOWS
    shifted = df[col].shift(1)
    for w in windows:
        df[f"{col}_rmean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"{col}_rstd_{w}"] = shifted.rolling(w, min_periods=1).std()
        df[f"{col}_rmin_{w}"] = shifted.rolling(w, min_periods=1).min()
        df[f"{col}_rmax_{w}"] = shifted.rolling(w, min_periods=1).max()
        df[f"{col}_rmedian_{w}"] = shifted.rolling(w, min_periods=1).median()
    # Volatility regime + spike detector
    df[f"{col}_cv_7"] = (df[f"{col}_rstd_7"] / df[f"{col}_rmean_7"].replace(0, np.nan))
    df[f"{col}_spike_7"] = shifted / df[f"{col}_rmean_7"].replace(0, np.nan)
    df[f"{col}_spike_28"] = shifted / df[f"{col}_rmean_28"].replace(0, np.nan)
    return df


def build_growth_features(df: pd.DataFrame, col: str = "Revenue") -> pd.DataFrame:
    shifted = df[col].shift(1)
    df[f"{col}_yoy_ratio"] = shifted / df[col].shift(366).replace(0, np.nan)
    df[f"{col}_yoy_growth"] = df[f"{col}_yoy_ratio"] - 1.0
    df[f"{col}_wow_ratio"] = shifted / df[col].shift(8).replace(0, np.nan)
    df[f"{col}_mom_ratio"] = shifted / df[col].shift(31).replace(0, np.nan)

    recent = shifted.rolling(7, min_periods=1).mean()
    older = df[col].shift(8).rolling(28, min_periods=1).mean()
    df[f"{col}_momentum"] = recent / older.replace(0, np.nan)

    df[f"{col}_diff_1"] = shifted - df[col].shift(2)
    df[f"{col}_diff_7"] = shifted - df[col].shift(8)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Date-exact features (daily calendars we merge on Date directly)
# ─────────────────────────────────────────────────────────────────────────────

def build_promo_calendar(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build a per-day promo table for every date in ``date_range``.

    For dates where actual promotions are known (from promotions.csv we
    explode start_date..end_date), we use them directly. For dates past the
    last observed promo (test horizon) we emit a month-of-year historical
    average so the features don't collapse to all-zero.
    """
    promos = pd.read_csv(FILES["promotions"], parse_dates=["start_date", "end_date"])
    if promos.empty:
        cols = ["Date", "promo_active_cnt", "promo_max_discount",
                "promo_is_percentage", "promo_is_fixed"]
        return pd.DataFrame({c: [] for c in cols}).assign(Date=pd.to_datetime([]))

    # Explode each promo row into per-day entries
    rows = []
    for _, r in promos.iterrows():
        d = pd.date_range(r["start_date"], r["end_date"], freq="D")
        rows.append(pd.DataFrame({
            "Date": d,
            "discount_value": r["discount_value"],
            "is_percentage": int(r["promo_type"] == "percentage"),
            "is_fixed": int(r["promo_type"] != "percentage"),
            "channel": r["promo_channel"],
        }))
    long = pd.concat(rows, ignore_index=True)

    # One-hot channel
    ch_dummies = pd.get_dummies(long["channel"], prefix="promo_ch").astype(int)
    long = pd.concat([long.drop(columns=["channel"]), ch_dummies], axis=1)

    daily = long.groupby("Date").agg(
        promo_active_cnt=("discount_value", "count"),
        promo_max_discount=("discount_value", "max"),
        promo_sum_discount=("discount_value", "sum"),
        promo_is_percentage=("is_percentage", "max"),
        promo_is_fixed=("is_fixed", "max"),
        **{c: (c, "max") for c in ch_dummies.columns},
    ).reset_index()

    obs_last = daily["Date"].max()

    # Historical month-of-year avg — used as fallback for future dates
    daily["month"] = daily["Date"].dt.month
    numeric_cols = [c for c in daily.columns if c not in ("Date", "month")]
    month_avg = daily.groupby("month")[numeric_cols].mean().reset_index()

    target = pd.DataFrame({"Date": pd.DatetimeIndex(date_range)})
    merged = target.merge(daily.drop(columns=["month"]), on="Date", how="left")

    # Fill post-observation rows with month-of-year average
    future_mask = merged["Date"] > obs_last
    if future_mask.any():
        m = merged.loc[future_mask, "Date"].dt.month
        fill = m.map(
            lambda x: month_avg.loc[month_avg["month"] == x, numeric_cols].iloc[0]
            if (month_avg["month"] == x).any() else pd.Series({c: 0 for c in numeric_cols})
        ).apply(pd.Series)
        for c in numeric_cols:
            merged.loc[future_mask, c] = fill[c].values

    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    merged["promo_is_active"] = (merged["promo_active_cnt"] > 0).astype(int)
    return merged


def build_inventory_daily(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill monthly inventory snapshots to daily grain."""
    inv = pd.read_csv(FILES["inventory"], parse_dates=["snapshot_date"])
    monthly = inv.groupby("snapshot_date").agg(
        inv_stock_on_hand=("stock_on_hand", "sum"),
        inv_stockout_rate=("stockout_flag", "mean"),
        inv_fill_rate=("fill_rate", "mean"),
        inv_sell_through=("sell_through_rate", "mean"),
        inv_days_of_supply=("days_of_supply", "mean"),
    ).reset_index().rename(columns={"snapshot_date": "Date"})

    target = pd.DataFrame({"Date": pd.DatetimeIndex(date_range)})
    # Use merge_asof for forward-fill by nearest previous snapshot
    monthly = monthly.sort_values("Date").reset_index(drop=True)
    out = pd.merge_asof(
        target.sort_values("Date"),
        monthly,
        on="Date",
        direction="backward",
    )
    numeric_cols = [c for c in out.columns if c != "Date"]
    # Anything before first snapshot → 0
    out[numeric_cols] = out[numeric_cols].fillna(0)
    return out.sort_values("Date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calendar-keyed profiles
# ─────────────────────────────────────────────────────────────────────────────

def compute_historical_profiles(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Revenue/COGS seasonality profiles, plus detrended + recent variants."""
    df = train_df.copy()
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dayofmonth"] = df["Date"].dt.day
    df["quarter"] = df["Date"].dt.quarter

    profiles: Dict[str, pd.DataFrame] = {}

    profiles["dow"] = df.groupby("dayofweek").agg(
        rev_dow_mean=("Revenue", "mean"),
        rev_dow_std=("Revenue", "std"),
        rev_dow_median=("Revenue", "median"),
        cogs_dow_mean=("COGS", "mean"),
    ).reset_index()

    profiles["month"] = df.groupby("month").agg(
        rev_month_mean=("Revenue", "mean"),
        rev_month_std=("Revenue", "std"),
        rev_month_median=("Revenue", "median"),
        cogs_month_mean=("COGS", "mean"),
    ).reset_index()

    profiles["woy"] = df.groupby("weekofyear").agg(
        rev_woy_mean=("Revenue", "mean"),
        rev_woy_std=("Revenue", "std"),
        cogs_woy_mean=("COGS", "mean"),
    ).reset_index()

    profiles["dom"] = df.groupby("dayofmonth").agg(
        rev_dom_mean=("Revenue", "mean"),
        cogs_dom_mean=("COGS", "mean"),
    ).reset_index()

    profiles["quarter"] = df.groupby("quarter").agg(
        rev_qtr_mean=("Revenue", "mean"),
        cogs_qtr_mean=("COGS", "mean"),
    ).reset_index()

    profiles["month_dow"] = df.groupby(["month", "dayofweek"]).agg(
        rev_month_dow_mean=("Revenue", "mean"),
    ).reset_index()

    # Detrended: divide Revenue by trailing 365d mean, then seasonality
    df_sorted = df.sort_values("Date").copy()
    long_mean = df_sorted["Revenue"].rolling(365, min_periods=30).mean()
    df_sorted["rev_detr"] = df_sorted["Revenue"] / long_mean.replace(0, np.nan)
    profiles["dow_detr"] = df_sorted.groupby("dayofweek").agg(
        rev_dow_detr=("rev_detr", "mean"),
    ).reset_index()
    profiles["month_detr"] = df_sorted.groupby("month").agg(
        rev_month_detr=("rev_detr", "mean"),
    ).reset_index()
    profiles["month_dow_detr"] = df_sorted.groupby(["month", "dayofweek"]).agg(
        rev_month_dow_detr=("rev_detr", "mean"),
    ).reset_index()

    # Recent-window (last 3 years) — captures current business level
    cutoff = df["Date"].max() - pd.Timedelta(days=3 * 365)
    recent = df[df["Date"] >= cutoff]
    if len(recent) > 30:
        profiles["dow_recent"] = recent.groupby("dayofweek").agg(
            rev_dow_recent=("Revenue", "mean"),
            cogs_dow_recent=("COGS", "mean"),
        ).reset_index()
        profiles["month_recent"] = recent.groupby("month").agg(
            rev_month_recent=("Revenue", "mean"),
            cogs_month_recent=("COGS", "mean"),
        ).reset_index()

    return profiles


def compute_aux_profiles() -> Dict[str, pd.DataFrame]:
    """Calendar-keyed averages from auxiliary tables (order mix, returns, etc.)."""
    profiles: Dict[str, pd.DataFrame] = {}

    # ── Orders ──
    try:
        orders = pd.read_csv(
            FILES["orders"], parse_dates=["order_date"],
            usecols=["order_id", "order_date", "order_status",
                     "payment_method", "device_type", "order_source"],
        )
        orders["dayofweek"] = orders["order_date"].dt.dayofweek
        orders["month"] = orders["order_date"].dt.month

        # Daily aggregate first (one row per date)
        daily = orders.groupby("order_date").agg(
            order_count=("order_id", "count"),
            cancel_rate=("order_status", lambda x: (x == "cancelled").mean()),
            pct_paid_search=("order_source",
                             lambda x: (x == "paid_search").mean()),
            pct_mobile=("device_type", lambda x: (x == "mobile").mean()),
            pct_credit_card=("payment_method",
                             lambda x: (x == "credit_card").mean()),
        )
        daily["dayofweek"] = daily.index.dayofweek
        daily["month"] = daily.index.month

        profiles["orders_dow"] = daily.groupby("dayofweek").agg(
            avg_orders_dow=("order_count", "mean"),
            avg_cancel_rate_dow=("cancel_rate", "mean"),
            avg_paid_search_dow=("pct_paid_search", "mean"),
            avg_mobile_dow=("pct_mobile", "mean"),
            avg_credit_card_dow=("pct_credit_card", "mean"),
        ).reset_index()

        profiles["orders_month"] = daily.groupby("month").agg(
            avg_orders_month=("order_count", "mean"),
            avg_cancel_rate_month=("cancel_rate", "mean"),
            avg_paid_search_month=("pct_paid_search", "mean"),
            avg_mobile_month=("pct_mobile", "mean"),
            avg_credit_card_month=("pct_credit_card", "mean"),
        ).reset_index()
    except Exception as e:
        print(f"  Warning: orders profile failed — {e}")

    # ── Order items → discount / promo penetration ──
    try:
        items = pd.read_csv(
            FILES["order_items"],
            usecols=["order_id", "quantity", "discount_amount", "promo_id"],
        )
        orders_min = pd.read_csv(
            FILES["orders"], parse_dates=["order_date"],
            usecols=["order_id", "order_date"],
        )
        merged = items.merge(orders_min, on="order_id", how="left")
        merged["is_promo"] = merged["promo_id"].notna().astype(int)
        merged["date"] = merged["order_date"]
        daily = merged.groupby("date").agg(
            daily_units=("quantity", "sum"),
            daily_discount_amt=("discount_amount", "sum"),
            promo_penetration=("is_promo", "mean"),
        )
        daily["dayofweek"] = daily.index.dayofweek
        daily["month"] = daily.index.month
        profiles["items_dow"] = daily.groupby("dayofweek").agg(
            avg_units_dow=("daily_units", "mean"),
            avg_discount_dow=("daily_discount_amt", "mean"),
            avg_promo_pen_dow=("promo_penetration", "mean"),
        ).reset_index()
        profiles["items_month"] = daily.groupby("month").agg(
            avg_units_month=("daily_units", "mean"),
            avg_discount_month=("daily_discount_amt", "mean"),
            avg_promo_pen_month=("promo_penetration", "mean"),
        ).reset_index()
    except Exception as e:
        print(f"  Warning: order_items profile failed — {e}")

    # ── Category mix profile (month-of-year) ──
    try:
        items = pd.read_csv(
            FILES["order_items"],
            usecols=["order_id", "product_id", "quantity", "unit_price", "discount_amount"],
        )
        prods = pd.read_csv(FILES["products"], usecols=["product_id", "category"])
        orders_min = pd.read_csv(
            FILES["orders"], parse_dates=["order_date"],
            usecols=["order_id", "order_date"],
        )
        items = items.merge(prods, on="product_id", how="left")
        items = items.merge(orders_min, on="order_id", how="left")
        items["line_rev"] = items["quantity"] * items["unit_price"] - items["discount_amount"].fillna(0)
        items["month"] = items["order_date"].dt.month
        pivot = items.pivot_table(
            index="month", columns="category",
            values="line_rev", aggfunc="sum", fill_value=0,
        )
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
        pivot.columns = [f"cat_share_{c}" for c in pivot.columns]
        profiles["cat_month"] = pivot.reset_index()
    except Exception as e:
        print(f"  Warning: category mix profile failed — {e}")

    # ── Returns ──
    try:
        returns = pd.read_csv(
            FILES["returns"], parse_dates=["return_date"],
            usecols=["return_id", "return_date", "refund_amount"],
        )
        daily = returns.groupby("return_date").agg(
            return_count=("return_id", "count"),
            avg_refund=("refund_amount", "mean"),
        )
        daily["dayofweek"] = daily.index.dayofweek
        daily["month"] = daily.index.month
        profiles["returns_dow"] = daily.groupby("dayofweek").agg(
            avg_returns_dow=("return_count", "mean"),
        ).reset_index()
        profiles["returns_month"] = daily.groupby("month").agg(
            avg_returns_month=("return_count", "mean"),
            avg_refund_month=("avg_refund", "mean"),
        ).reset_index()
    except Exception as e:
        print(f"  Warning: returns profile failed — {e}")

    # ── Web traffic ──
    try:
        wt = pd.read_csv(FILES["web_traffic"], parse_dates=["date"])
        daily = wt.groupby("date").agg(
            total_sessions=("sessions", "sum"),
            total_visitors=("unique_visitors", "sum"),
            avg_bounce=("bounce_rate", "mean"),
        )
        daily["dayofweek"] = daily.index.dayofweek
        daily["month"] = daily.index.month
        profiles["traffic_dow"] = daily.groupby("dayofweek").agg(
            avg_sessions_dow=("total_sessions", "mean"),
            avg_visitors_dow=("total_visitors", "mean"),
        ).reset_index()
        profiles["traffic_month"] = daily.groupby("month").agg(
            avg_sessions_month=("total_sessions", "mean"),
            avg_bounce_month=("avg_bounce", "mean"),
        ).reset_index()
    except Exception as e:
        print(f"  Warning: web_traffic profile failed — {e}")

    # ── Inventory ──
    try:
        inv = pd.read_csv(
            FILES["inventory"],
            usecols=["month", "stockout_flag", "sell_through_rate", "fill_rate"],
        )
        profiles["inv_month"] = inv.groupby("month").agg(
            avg_stockout_rate_month=("stockout_flag", "mean"),
            avg_sell_through_month=("sell_through_rate", "mean"),
            avg_fill_rate_month=("fill_rate", "mean"),
        ).reset_index()
    except Exception as e:
        print(f"  Warning: inventory profile failed — {e}")

    return profiles


def _profile_keys(prof: pd.DataFrame) -> list:
    """Identify merge keys of a profile: non-metric columns only."""
    metric_prefixes = ("rev_", "cogs_", "avg_", "active_", "cat_share_", "promo_")
    return [c for c in prof.columns if not any(c.startswith(p) for p in metric_prefixes)]


def merge_profiles(df: pd.DataFrame, profile: pd.DataFrame, keys) -> pd.DataFrame:
    if isinstance(keys, str):
        keys = [keys]
    for k in keys:
        if k not in df.columns:
            return df
    return df.merge(profile, on=keys, how="left")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_table(
    train_df: pd.DataFrame,
    *,
    test_df: pd.DataFrame | None = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Produce a feature-rich dataframe for train_df (optionally extended with
    test_df dates). Returns (features_df, bundle) where bundle contains the
    profiles and date-exact tables needed to rebuild features at predict time.
    """
    df = train_df.copy().sort_values("Date").reset_index(drop=True)

    if test_df is not None:
        full_dates = pd.concat([df["Date"], test_df["Date"]], ignore_index=True)
    else:
        full_dates = df["Date"]
    full_range = pd.date_range(full_dates.min(), full_dates.max(), freq="D")

    if verbose:
        print("  Calendar + holidays + Fourier...")
    df = build_calendar_features(df)

    if verbose:
        print("  Target lag / rolling / growth features...")
    df = build_lag_features(df, "Revenue")
    df = build_lag_features(df, "COGS")
    df = build_rolling_features(df, "Revenue")
    df = build_rolling_features(df, "COGS", windows=[7, 14, 28, 90])
    df = build_growth_features(df, "Revenue")

    if verbose:
        print("  Promotion + inventory calendars...")
    promo_daily = build_promo_calendar(full_range)
    inv_daily = build_inventory_daily(full_range)
    df = df.merge(promo_daily, on="Date", how="left")
    df = df.merge(inv_daily, on="Date", how="left")

    if verbose:
        print("  Revenue/COGS seasonality profiles...")
    rev_profiles = compute_historical_profiles(train_df)
    df = merge_profiles(df, rev_profiles["dow"], "dayofweek")
    df = merge_profiles(df, rev_profiles["month"], "month")
    df = merge_profiles(df, rev_profiles["woy"], "weekofyear")
    df = merge_profiles(df, rev_profiles["dom"], "dayofmonth")
    df = merge_profiles(df, rev_profiles["quarter"], "quarter")
    df = merge_profiles(df, rev_profiles["month_dow"], ["month", "dayofweek"])
    df = merge_profiles(df, rev_profiles["dow_detr"], "dayofweek")
    df = merge_profiles(df, rev_profiles["month_detr"], "month")
    df = merge_profiles(df, rev_profiles["month_dow_detr"], ["month", "dayofweek"])
    if "dow_recent" in rev_profiles:
        df = merge_profiles(df, rev_profiles["dow_recent"], "dayofweek")
        df = merge_profiles(df, rev_profiles["month_recent"], "month")

    if verbose:
        print("  Auxiliary profiles (orders/returns/traffic/items/category)...")
    aux_profiles = compute_aux_profiles()
    for name, prof in aux_profiles.items():
        keys = _profile_keys(prof)
        if keys:
            df = merge_profiles(df, prof, keys)

    bundle = {
        "rev_profiles": rev_profiles,
        "aux_profiles": aux_profiles,
        "promo_daily": promo_daily,
        "inv_daily": inv_daily,
    }

    if verbose:
        print(f"  Feature table: {df.shape[0]} rows × {df.shape[1]} cols")

    return df, bundle


def apply_profiles_to_dates(dates_df: pd.DataFrame, bundle_or_profiles) -> pd.DataFrame:
    """
    Apply precomputed profiles + date-exact tables to a dataframe that already
    has calendar columns. Accepts either a bundle dict (new API) or a flat
    profiles dict (legacy API used by ex_03).
    """
    df = dates_df.copy()

    # New bundle API
    if isinstance(bundle_or_profiles, dict) and "rev_profiles" in bundle_or_profiles:
        bundle = bundle_or_profiles
        rev_profiles = bundle["rev_profiles"]
        aux_profiles = bundle["aux_profiles"]
        promo_daily = bundle["promo_daily"]
        inv_daily = bundle["inv_daily"]

        if "Date" in df.columns:
            df = df.merge(promo_daily, on="Date", how="left")
            df = df.merge(inv_daily, on="Date", how="left")
            for col in promo_daily.columns:
                if col != "Date" and col in df.columns:
                    df[col] = df[col].fillna(0)
            for col in inv_daily.columns:
                if col != "Date" and col in df.columns:
                    df[col] = df[col].fillna(0)

        combined = {**rev_profiles, **aux_profiles}
        for prof in combined.values():
            keys = _profile_keys(prof)
            keys = [k for k in keys if k in df.columns]
            if keys:
                df = df.merge(prof, on=keys, how="left")
        return df

    # Legacy flat dict API (used by ex_03_lgbm recursive_predict)
    profiles = bundle_or_profiles
    for prof in profiles.values():
        keys = _profile_keys(prof)
        keys = [k for k in keys if k in df.columns]
        if keys:
            df = df.merge(prof, on=keys, how="left")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature column names (everything except target + date)."""
    exclude = {"Date", "Revenue", "COGS"}
    return [c for c in df.columns if c not in exclude]
