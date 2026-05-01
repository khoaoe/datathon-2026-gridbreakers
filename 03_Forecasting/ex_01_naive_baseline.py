"""
EX_01: Naive Baselines
- Seasonal naive (same day last year)
- 28-day rolling mean
- Last known value (persistence)

No ML. Just simple rules to set a floor for comparison.
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from modeling.config import SUBMISSION_DIR, VAL_START
from modeling.utils import evaluate, load_sales, make_submission
from modeling.tracker import ExperimentTracker


def main():
    start = time.time()
    train, test = load_sales()
    tracker = ExperimentTracker("ex_01_naive")

    # Split train into train/val
    val_mask = train["Date"] >= VAL_START
    val = train[val_mask].copy()
    trn = train[~val_mask].copy()

    all_data = pd.concat(
        [train, test[["Date"]].assign(Revenue=np.nan, COGS=np.nan)], ignore_index=True
    ).sort_values("Date")

    print("=" * 60)
    print("EX_01: NAIVE BASELINES")
    print("=" * 60)

    # ── Baseline 1: Seasonal Naive (same day last year) ──────────────
    print("\n--- Seasonal Naive (365-day lag) ---")
    all_data["pred_seasonal"] = all_data["Revenue"].shift(365)
    all_data["pred_cogs_seasonal"] = all_data["COGS"].shift(365)

    val_pred = all_data.loc[all_data["Date"].isin(val["Date"]), "pred_seasonal"].values
    valid_mask = ~np.isnan(val_pred) & ~np.isnan(val["Revenue"].values)
    res1 = evaluate(
        val["Revenue"].values[valid_mask], val_pred[valid_mask], "Seasonal Naive"
    )

    # ── Baseline 2: 28-day rolling mean ──────────────────────────────
    print("\n--- 28-Day Rolling Mean ---")
    all_data["pred_rolling28"] = (
        all_data["Revenue"].shift(1).rolling(28, min_periods=1).mean()
    )
    all_data["pred_cogs_rolling28"] = (
        all_data["COGS"].shift(1).rolling(28, min_periods=1).mean()
    )

    val_pred2 = all_data.loc[
        all_data["Date"].isin(val["Date"]), "pred_rolling28"
    ].values
    valid_mask2 = ~np.isnan(val_pred2) & ~np.isnan(val["Revenue"].values)
    res2 = evaluate(
        val["Revenue"].values[valid_mask2], val_pred2[valid_mask2], "Rolling 28d"
    )

    # ── Baseline 3: 7-day rolling mean ───────────────────────────────
    print("\n--- 7-Day Rolling Mean ---")
    all_data["pred_rolling7"] = (
        all_data["Revenue"].shift(1).rolling(7, min_periods=1).mean()
    )
    all_data["pred_cogs_rolling7"] = (
        all_data["COGS"].shift(1).rolling(7, min_periods=1).mean()
    )

    val_pred3 = all_data.loc[all_data["Date"].isin(val["Date"]), "pred_rolling7"].values
    valid_mask3 = ~np.isnan(val_pred3) & ~np.isnan(val["Revenue"].values)
    res3 = evaluate(
        val["Revenue"].values[valid_mask3], val_pred3[valid_mask3], "Rolling 7d"
    )

    # ── Pick best baseline and generate submission ───────────────────
    best_name = "Seasonal Naive"
    best = res1
    if res2["mae"] < best["mae"]:
        best_name = "Rolling 28d"
        best = res2
    if res3["mae"] < best["mae"]:
        best_name = "Rolling 7d"
        best = res3

    print(f"\nBest baseline: {best_name}")

    # Generate test predictions using recursive seasonal naive:
    # Year 1 (Jan-Dec 2023): use actual 2022 values (365-day lag)
    # Year 2 (Jan-Jul 2024): use predicted 2023 values (copy year 1 predictions)
    test_dates = test["Date"].values
    n_test = len(test_dates)
    rev_pred = np.full(n_test, np.nan)
    cogs_pred = np.full(n_test, np.nan)

    # Build a lookup from training data
    train_lookup_rev = dict(zip(train["Date"], train["Revenue"]))
    train_lookup_cogs = dict(zip(train["Date"], train["COGS"]))

    for i, date in enumerate(test_dates):
        dt = pd.Timestamp(date)
        # Try 365-day lag from training or previous predictions
        lag_date = dt - pd.Timedelta(days=365)
        if lag_date in train_lookup_rev:
            rev_pred[i] = train_lookup_rev[lag_date]
            cogs_pred[i] = train_lookup_cogs[lag_date]
        elif i >= 365:
            # Use our own prediction from ~1 year ago
            rev_pred[i] = rev_pred[i - 365]
            cogs_pred[i] = cogs_pred[i - 365]

    # Final fallback: fill any remaining NaN with training mean
    train_mean_rev = train["Revenue"].mean()
    train_mean_cogs = train["COGS"].mean()
    rev_pred = np.where(np.isnan(rev_pred), train_mean_rev, rev_pred)
    cogs_pred = np.where(np.isnan(cogs_pred), train_mean_cogs, cogs_pred)

    print(f"  Nulls remaining: Revenue={np.isnan(rev_pred).sum()}, COGS={np.isnan(cogs_pred).sum()}")

    make_submission(
        test["Date"], rev_pred, cogs_pred, SUBMISSION_DIR / "ex_01_naive.csv"
    )

    tracker.log_final(best)
    tracker.add_note(f"Best baseline: {best_name}")
    tracker.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return best


if __name__ == "__main__":
    main()
