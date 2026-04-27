from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "cleaning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CleanConfig:
    roll_window: int = 365
    min_periods: int = 60
    z_thresh: float = 6.5
    cap_k: float = 6.5  # cap = med ± cap_k*sigma, only when |z|>z_thresh
    keep_month_end: bool = False
    keep_payday_window: bool = False


def _read_sales() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "sales.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.reset_index(drop=True)
    return df


def _read_promos() -> pd.DataFrame:
    p = ROOT / "data" / "promotions.csv"
    if not p.exists():
        return pd.DataFrame(columns=["start_date", "end_date"])
    promos = pd.read_csv(p, parse_dates=["start_date", "end_date"])
    promos = promos.dropna(subset=["start_date", "end_date"]).copy()
    return promos


def _promo_active(dates: pd.Series, promos: pd.DataFrame) -> np.ndarray:
    if promos.empty:
        return np.zeros(len(dates), dtype=int)
    d = pd.to_datetime(dates).values.astype("datetime64[ns]")
    active = np.zeros(len(d), dtype=bool)
    for _, r in promos.iterrows():
        s = np.datetime64(r["start_date"])
        e = np.datetime64(r["end_date"])
        active |= (d >= s) & (d <= e)
    return active.astype(int)


def _double_dates(year_min: int, year_max: int) -> set[pd.Timestamp]:
    out = []
    for y in range(year_min, year_max + 1):
        for m in (9, 10, 11, 12):
            out.append(pd.Timestamp(f"{y}-{m:02d}-{m:02d}"))
    return set(out)


def _tet_dates() -> pd.DatetimeIndex:
    # Same list as feature_engineering calendar features
    return pd.to_datetime(
        [
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
    )


def keep_mask(dates: pd.Series, promo_active: np.ndarray, cfg: CleanConfig) -> pd.Series:
    dt = pd.to_datetime(dates)
    dd = _double_dates(int(dt.dt.year.min()), int(dt.dt.year.max()))
    is_dd = dt.isin(dd)
    is_month_end = dt.dt.is_month_end if cfg.keep_month_end else pd.Series(False, index=dt.index)
    is_payday_window = ((dt.dt.day >= 25) | (dt.dt.day <= 5)) if cfg.keep_payday_window else pd.Series(False, index=dt.index)

    tet = _tet_dates()
    # keep ±7 days around Tet
    is_tet_window = np.zeros(len(dt), dtype=bool)
    for t in tet:
        is_tet_window |= (dt >= (t - pd.Timedelta(days=7))) & (dt <= (t + pd.Timedelta(days=7)))

    is_promo = promo_active.astype(bool)
    return pd.Series(is_dd | is_month_end | is_payday_window | is_tet_window | is_promo, index=dt.index)


def _robust_sigma(s: pd.Series, cfg: CleanConfig) -> tuple[pd.Series, pd.Series]:
    med = s.rolling(cfg.roll_window, min_periods=cfg.min_periods).median()
    resid = (s - med).abs()
    mad = resid.rolling(cfg.roll_window, min_periods=cfg.min_periods).median()
    sigma = 1.4826 * mad.replace(0, np.nan)
    sigma = sigma.fillna(s.rolling(cfg.roll_window, min_periods=cfg.min_periods).std())
    sigma = sigma.replace(0, np.nan).fillna(sigma.median())
    return med, sigma


def clean_series(s: pd.Series, keep: pd.Series, cfg: CleanConfig) -> tuple[pd.Series, dict]:
    med, sigma = _robust_sigma(s, cfg)
    z = (s - med) / sigma
    lo = med - cfg.cap_k * sigma
    hi = med + cfg.cap_k * sigma

    # clip only on non-keep and only if |z| huge
    do_clip = (~keep) & (z.abs() > cfg.z_thresh)
    clipped = s.where(~do_clip, s.clip(lo, hi))
    clipped = clipped.clip(lower=0)

    meta = {
        "n_clipped": int(do_clip.sum()),
        "max_abs_delta": float((s - clipped).abs().max()),
        "max_abs_z": float(z.abs().max()),
    }
    return clipped, meta


def main() -> None:
    sales = _read_sales()
    promos = _read_promos()
    promo_active = _promo_active(sales["Date"], promos)
    base_report = {
        "integrity": {
            "n_rows": int(len(sales)),
            "duplicate_dates": int(sales["Date"].duplicated().sum()),
        },
        "promotions": {"pct_days_promo_active": float(promo_active.mean()), "n_promos": int(len(promos))},
    }

    configs = [
        CleanConfig(z_thresh=6.0, cap_k=6.0, keep_month_end=False, keep_payday_window=False),
        CleanConfig(z_thresh=6.5, cap_k=6.5, keep_month_end=False, keep_payday_window=False),
        CleanConfig(z_thresh=7.0, cap_k=7.0, keep_month_end=False, keep_payday_window=False),
        CleanConfig(z_thresh=6.5, cap_k=6.5, keep_month_end=True, keep_payday_window=False),
        CleanConfig(z_thresh=6.5, cap_k=6.5, keep_month_end=True, keep_payday_window=True),
    ]

    summary_rows = []
    for cfg in configs:
        keep = keep_mask(sales["Date"], promo_active, cfg)
        out = sales.copy()

        report = {
            "config": cfg.__dict__,
            **base_report,
            "keep_mask": {"pct_keep": float(keep.mean())},
            "targets": {},
        }

        changed_rows = pd.DataFrame({"Date": sales["Date"]})
        for col in ["Revenue", "COGS"]:
            cleaned, meta = clean_series(sales[col].astype(float), keep, cfg)
            out[col] = cleaned
            changed = (sales[col].astype(float) != cleaned).astype(int)
            changed_rows[f"{col}_changed"] = changed
            report["targets"][col] = {**meta, "n_changed": int(changed.sum())}

        changed_rows["any_changed"] = (changed_rows["Revenue_changed"] | changed_rows["COGS_changed"]).astype(int)
        report["targets"]["any_changed_rows"] = int(changed_rows["any_changed"].sum())

        tag = f"z{cfg.z_thresh}_k{cfg.cap_k}_me{int(cfg.keep_month_end)}_pd{int(cfg.keep_payday_window)}"
        out_path = OUT_DIR / f"sales_clean_{tag}.csv"
        rep_path = OUT_DIR / f"sales_clean_report_{tag}.json"
        ch_path = OUT_DIR / f"sales_clean_changes_{tag}.csv"

        out.to_csv(out_path, index=False)
        (rep_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

        ch = out.loc[changed_rows["any_changed"] == 1, ["Date", "Revenue", "COGS"]].copy()
        if not ch.empty:
            ch = ch.merge(sales[["Date", "Revenue", "COGS"]], on="Date", suffixes=("_clean", "_orig"))
            ch["rev_delta"] = ch["Revenue_clean"] - ch["Revenue_orig"]
            ch["cogs_delta"] = ch["COGS_clean"] - ch["COGS_orig"]
            ch["promo_active"] = promo_active[changed_rows["any_changed"].values.astype(bool)]
            ch["kept_expected_spike"] = keep[changed_rows["any_changed"].values.astype(bool)].astype(int).values
        ch.to_csv(ch_path, index=False)

        summary_rows.append(
            {
                "tag": tag,
                "pct_keep": float(keep.mean()),
                "rev_changed": report["targets"]["Revenue"]["n_changed"],
                "cogs_changed": report["targets"]["COGS"]["n_changed"],
                "any_changed": report["targets"]["any_changed_rows"],
                "rev_max_abs_delta": report["targets"]["Revenue"]["max_abs_delta"],
                "cogs_max_abs_delta": report["targets"]["COGS"]["max_abs_delta"],
                "out_csv": str(out_path),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["any_changed", "rev_changed"]).reset_index(drop=True)
    summary_path = OUT_DIR / "sales_clean_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Cleaning sweep summary:")
    with pd.option_context("display.width", 180, "display.max_columns", 50):
        print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

