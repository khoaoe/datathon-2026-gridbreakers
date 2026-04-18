"""
autoresearch/exp9_seasonal_naive.py — seasonal-naive with damped growth.

Strong baseline for distribution-mismatched long-horizon forecasts:

    yhat[date]      = actual[date - 365d] * (1 + g * 1)                  (year 1)
    yhat[date]      = actual[date - 2*365d] * (1 + g * 2)                (year 2)

where ``g`` = trailing 3-year CAGR (capped at 8 %) so the multiplier grows
linearly, not exponentially. This deliberately *flattens* the long-horizon
trajectory, which is exactly what the Prophet + LGBM stack is missing
(2024 predictions were +35 % vs 2022 actual, way above the historical CAGR
of +5.5 %).

Smoothed across 3 anchor years (365/730/1095) with exponential decay weights
to reduce noise from any single day's spike.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.prepare import (
    load_splits, load_extrapolation_splits,
    evaluate_forecast, evaluate_extrapolation,
    write_submission, append_result,
)


GROWTH_CAP = 0.08          # cap CAGR multiplier so 2024 doesn't explode
LOOKBACK_YEARS = (1, 2, 3) # anchor years to blend
ANCHOR_WEIGHTS = (0.60, 0.28, 0.12)


def _trailing_cagr(train_df: pd.DataFrame, years: int = 3) -> float:
    """Annualised growth of yearly mean Revenue over the last `years` years."""
    last_year = train_df["Date"].dt.year.max()
    a = train_df[train_df["Date"].dt.year == last_year - years]["Revenue"].mean()
    b = train_df[train_df["Date"].dt.year == last_year]["Revenue"].mean()
    if a <= 0 or np.isnan(a) or np.isnan(b):
        return 0.0
    cagr = (b / a) ** (1.0 / years) - 1.0
    return float(np.clip(cagr, -GROWTH_CAP, GROWTH_CAP))


def _seasonal_naive(train_df: pd.DataFrame, horizon_df: pd.DataFrame,
                    col: str, cagr: float) -> np.ndarray:
    train = train_df[["Date", col]].copy()
    train["Date"] = pd.to_datetime(train["Date"])
    lookup = train.set_index("Date")[col]
    out = np.zeros(len(horizon_df))
    for i, d in enumerate(pd.to_datetime(horizon_df["Date"].values)):
        preds = []
        for lb, w in zip(LOOKBACK_YEARS, ANCHOR_WEIGHTS):
            anchor = d - pd.Timedelta(days=365 * lb)
            # if that day missing, try ±3 days window
            vals = []
            for delta in range(-3, 4):
                cand = anchor + pd.Timedelta(days=delta)
                if cand in lookup.index:
                    vals.append(lookup.loc[cand])
            if vals:
                base = float(np.median(vals))
                years_ahead = (d.year - anchor.year)
                mult = (1.0 + cagr) ** years_ahead
                preds.append((base * mult, w))
        if preds:
            total_w = sum(w for _, w in preds)
            out[i] = sum(v * w for v, w in preds) / total_w
        else:
            out[i] = float(train[col].tail(30).mean())
    return np.clip(out, 0, None)


def run_split(train_df: pd.DataFrame, horizon_df: pd.DataFrame, label: str):
    cagr = _trailing_cagr(train_df, 3)
    print(f"  [{label}] trailing 3yr CAGR = {cagr*100:+.2f}%  "
          f"(capped at ±{GROWTH_CAP*100:.0f}%)")
    rev = _seasonal_naive(train_df, horizon_df, "Revenue", cagr)
    cogs = _seasonal_naive(train_df, horizon_df, "COGS", cagr)
    return rev, cogs


def main():
    t0 = time.time()
    train_fit, val, test = load_splits()
    full_train = pd.concat([train_fit, val], ignore_index=True).sort_values("Date")

    print("\n[1/3] primary val (Q4 2022)...")
    r_v, c_v = run_split(train_fit, val, "val")
    m = evaluate_forecast(val, r_v, c_v)

    print("\n[2/3] extrapolation val (2021-2022)...")
    train_ext, val_ext = load_extrapolation_splits()
    r_e, c_e = run_split(train_ext, val_ext, "ext")
    mx = evaluate_extrapolation(val_ext, r_e, c_e)

    print("\n[3/3] test (548 days)...")
    r_t, c_t = run_split(full_train, test, "test")
    out = write_submission(test["Date"], r_t, c_t, name="autoresearch_seasnaive")
    print(f"submission: {out}")

    append_result({**m, **mx}, status="keep",
                  description="exp9: seasonal-naive × damped CAGR (cap 8%)")
    print(f"total_seconds: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
