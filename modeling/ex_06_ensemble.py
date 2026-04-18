"""
EX_06: Ensemble of best submissions.

Uses validation predictions saved by each experiment to fit optimal
non-negative weights via scipy.optimize (minimizes val MAE on Revenue).
Falls back to equal average if val predictions are unavailable.

Runs three ensembles:
  • simple_avg — equal-weighted mean
  • rank_avg   — fixed weights by model type
  • optimized  — weights fit on validation Revenue MAE
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.config import SUBMISSION_DIR, VAL_START, VAL_END
from modeling.utils import evaluate, load_sales, make_submission

VAL_PRED_DIR = SUBMISSION_DIR / "val"


def load_submission(name: str):
    path = SUBMISSION_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"])


def load_val_preds(name: str):
    p = VAL_PRED_DIR / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["Date"])


def fit_weights(preds_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Non-negative weights that sum to 1, minimizing MAE. Nelder-Mead."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return np.ones(preds_matrix.shape[1]) / preds_matrix.shape[1]

    n = preds_matrix.shape[1]

    def objective(w):
        w_ = np.abs(w)
        if w_.sum() == 0:
            w_ = np.ones_like(w_)
        w_ = w_ / w_.sum()
        blend = preds_matrix @ w_
        return np.mean(np.abs(y_true - blend))

    x0 = np.ones(n) / n
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-3})
    w = np.abs(res.x)
    return w / w.sum()


def main():
    train, test = load_sales()

    print("=" * 70)
    print("EX_06: ENSEMBLE")
    print("=" * 70)

    sub_files = {
        "ex_01_naive":            "ex_01_naive.csv",
        "ex_02_prophet":          "ex_02_prophet.csv",
        "ex_03_lgbm":             "ex_03_lgbm.csv",
        "ex_04_xgb":              "ex_04_xgb.csv",
        "ex_05_nhits":            "ex_05_nhits.csv",
        "ex_07_lgbm_v3":          "ex_07_lgbm_v3.csv",
        "ex_08_prophet_residual": "ex_08_prophet_residual.csv",
        "ex_09_lgbm_direct":      "ex_09_lgbm_direct.csv",
    }

    loaded = {}
    for name, fname in sub_files.items():
        sub = load_submission(fname)
        if sub is not None:
            loaded[name] = sub
            print(f"  loaded {name} ({len(sub)})")
        else:
            print(f"  skip   {name} (not found)")

    if len(loaded) < 2:
        print("\nNeed ≥2 submissions to ensemble.")
        return None

    names = list(loaded.keys())
    rev_matrix_test = np.column_stack([loaded[n]["Revenue"].values for n in names])
    cogs_matrix_test = np.column_stack([loaded[n]["COGS"].values for n in names])

    # ── 1. Simple average ──
    make_submission(
        test["Date"], rev_matrix_test.mean(axis=1), cogs_matrix_test.mean(axis=1),
        SUBMISSION_DIR / "ex_06_ensemble_avg.csv",
    )

    # ── 2. Rank-based weighting ──
    weights_rank = {}
    for n in names:
        if "lgbm_v3" in n:        weights_rank[n] = 0.30
        elif "prophet_residual" in n: weights_rank[n] = 0.20
        elif "lgbm_direct" in n:      weights_rank[n] = 0.15
        elif "lgbm" in n:             weights_rank[n] = 0.15
        elif "xgb" in n:              weights_rank[n] = 0.10
        elif "nhits" in n:            weights_rank[n] = 0.05
        elif "prophet" in n:          weights_rank[n] = 0.03
        else:                         weights_rank[n] = 0.02
    tot = sum(weights_rank.values())
    w_rank = np.array([weights_rank[n] / tot for n in names])
    make_submission(
        test["Date"], rev_matrix_test @ w_rank, cogs_matrix_test @ w_rank,
        SUBMISSION_DIR / "ex_06_ensemble_weighted.csv",
    )

    # ── 3. Optimized weights using validation predictions ──
    val_preds_rev, val_preds_cogs, active_names = [], [], []
    val_dates = None
    for n in names:
        vp = load_val_preds(n)
        if vp is None:
            continue
        if val_dates is None:
            val_dates = vp["Date"]
        vp = vp.set_index("Date").reindex(val_dates).reset_index()
        val_preds_rev.append(vp["Revenue"].values)
        val_preds_cogs.append(vp["COGS"].values)
        active_names.append(n)

    if len(active_names) >= 2 and val_dates is not None:
        print(f"\nOptimizing weights on validation ({len(val_dates)} rows, "
              f"{len(active_names)} models: {active_names})")
        val_rev_true = train.set_index("Date").reindex(val_dates)["Revenue"].values
        val_cogs_true = train.set_index("Date").reindex(val_dates)["COGS"].values

        rev_val_mat = np.column_stack(val_preds_rev)
        cogs_val_mat = np.column_stack(val_preds_cogs)

        w_rev = fit_weights(rev_val_mat, val_rev_true)
        w_cogs = fit_weights(cogs_val_mat, val_cogs_true)

        print("  Optimized Revenue weights:")
        for nm, w in zip(active_names, w_rev):
            print(f"    {nm:30s} {w:.3f}")
        print("  Optimized COGS weights:")
        for nm, w in zip(active_names, w_cogs):
            print(f"    {nm:30s} {w:.3f}")

        # Validation score with optimized weights
        blend_rev = rev_val_mat @ w_rev
        blend_cogs = cogs_val_mat @ w_cogs
        evaluate(val_rev_true, blend_rev, "Ensemble val Revenue")
        evaluate(val_cogs_true, blend_cogs, "Ensemble val COGS")

        # Apply weights to test submissions (restricted to active_names)
        idx = [names.index(n) for n in active_names]
        rev_test_active = rev_matrix_test[:, idx]
        cogs_test_active = cogs_matrix_test[:, idx]
        make_submission(
            test["Date"], rev_test_active @ w_rev, cogs_test_active @ w_cogs,
            SUBMISSION_DIR / "ex_06_ensemble_optimized.csv",
        )
    else:
        print("\nSkipped optimized ensemble (need ≥2 experiments with val preds).")


if __name__ == "__main__":
    main()
