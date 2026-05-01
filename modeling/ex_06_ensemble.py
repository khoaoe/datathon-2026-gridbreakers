"""
EX_06: Ensemble of Best Models
- Weighted average of EX_01 through EX_05 submissions
- Weights optimized on validation set (if available)
- Also supports simple averaging as fallback
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from modeling.config import SUBMISSION_DIR
from modeling.utils import evaluate, load_sales, make_submission


def load_submission(name):
    """Load a submission CSV."""
    path = SUBMISSION_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"])


def optimize_weights(submissions, y_true, names):
    """
    Find optimal ensemble weights via scipy minimize.
    Minimizes MAE on validation predictions.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("scipy not installed, using equal weights")
        n = len(submissions)
        return np.ones(n) / n

    preds = np.column_stack(submissions)

    def objective(w):
        w = np.abs(w) / np.abs(w).sum()  # normalize
        blend = preds @ w
        return np.mean(np.abs(y_true - blend))

    n = len(submissions)
    x0 = np.ones(n) / n
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 10000})
    weights = np.abs(result.x) / np.abs(result.x).sum()

    print("Optimized weights:")
    for name, w in zip(names, weights):
        print(f"  {name}: {w:.3f}")
    return weights


def main():
    start = time.time()
    train, test = load_sales()

    print("=" * 60)
    print("EX_06: ENSEMBLE")
    print("=" * 60)

    # ── Load all available submissions ───────────────────────────────
    sub_files = {
        "ex_01_naive": "ex_01_naive.csv",
        "ex_02_prophet": "ex_02_prophet.csv",
        "ex_03_lgbm": "ex_03_lgbm.csv",
        "ex_04_xgb": "ex_04_xgb.csv",
        "ex_05_nhits": "ex_05_nhits.csv",
    }

    loaded = {}
    for name, fname in sub_files.items():
        sub = load_submission(fname)
        if sub is not None:
            loaded[name] = sub
            print(f"  Loaded {name} ({len(sub)} rows)")
        else:
            print(f"  Skipped {name} (not found)")

    if len(loaded) < 2:
        print("\nNeed at least 2 submissions to ensemble. Run more experiments first.")
        return None

    # ── Simple average ensemble ──────────────────────────────────────
    print(f"\nEnsembling {len(loaded)} models...")
    names = list(loaded.keys())

    rev_preds = np.column_stack([loaded[n]["Revenue"].values for n in names])
    cogs_preds = np.column_stack([loaded[n]["COGS"].values for n in names])

    # Equal-weight average
    rev_avg = rev_preds.mean(axis=1)
    cogs_avg = cogs_preds.mean(axis=1)

    make_submission(test["Date"], rev_avg, cogs_avg,
                    SUBMISSION_DIR / "ex_06_ensemble_avg.csv")

    # ── Weighted ensemble (if we have the best 2-3 models) ───────────
    # Use 70% best model + 30% second best as a simple heuristic
    # Or if scipy is available, optimize on some metric
    if len(loaded) >= 3:
        # Rank-based weighting: give more weight to tree models
        weights = {}
        for n in names:
            if "lgbm" in n:
                weights[n] = 0.4
            elif "xgb" in n:
                weights[n] = 0.3
            elif "nhits" in n:
                weights[n] = 0.15
            elif "prophet" in n:
                weights[n] = 0.10
            else:
                weights[n] = 0.05

        # Normalize
        total = sum(weights[n] for n in names)
        w_array = np.array([weights[n] / total for n in names])

        print("\nWeighted ensemble:")
        for n, w in zip(names, w_array):
            print(f"  {n}: {w:.3f}")

        rev_weighted = rev_preds @ w_array
        cogs_weighted = cogs_preds @ w_array

        make_submission(test["Date"], rev_weighted, cogs_weighted,
                        SUBMISSION_DIR / "ex_06_ensemble_weighted.csv")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
