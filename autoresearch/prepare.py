"""
autoresearch/prepare.py — FROZEN harness.

Inspired by karpathy/autoresearch. Agents must NOT modify this file. It pins:

  • the data load path,
  • the validation split (2022-10-01 .. 2022-12-31, per datathon spec),
  • the evaluation metric (MAE / RMSE / MAPE / R² on Revenue + COGS),
  • the results.tsv schema.

Agents edit only `autoresearch/train.py`. The shape of the API below is
intentionally simple so the training script only imports three things:
``load_splits``, ``evaluate_forecast``, ``write_submission``.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "autoresearch" / "runs"
SUBMISSION_DIR = ROOT / "output" / "submissions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ── Frozen splits ────────────────────────────────────────────────────────────
# Primary val (datathon-spec compliant): Q4 2022.
TRAIN_START = pd.Timestamp("2012-07-04")
TRAIN_END   = pd.Timestamp("2022-12-31")
VAL_START   = pd.Timestamp("2022-10-01")
VAL_END     = pd.Timestamp("2022-12-31")
TEST_START  = pd.Timestamp("2023-01-01")

# Extrapolation val (honest 548-day forecast proxy): train up to 2020-12-31,
# predict all of 2021 + 2022 (730 days, close to test's 548-day horizon).
# This catches trend-extrapolation failure modes that Q4-2022 val hides.
EXT_TRAIN_END = pd.Timestamp("2020-12-31")
EXT_VAL_START = pd.Timestamp("2021-01-01")
EXT_VAL_END   = pd.Timestamp("2022-12-31")

# ── Results TSV (matches karpathy/autoresearch convention) ──────────────────
RESULTS_TSV = ROOT / "autoresearch" / "results.tsv"
TSV_HEADER = ("commit\tval_mae_rev\tval_mae_cogs\tval_rmse_rev"
              "\text_mae_rev\text_mae_cogs\tstatus\tdescription\n")


# ─── data ────────────────────────────────────────────────────────────────────

def _load_sales() -> pd.DataFrame:
    sales = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["Date"])
    return sales.sort_values("Date").reset_index(drop=True)


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (train_fit, val, test_template) where:
      • train_fit = rows strictly before VAL_START (Q4 2022)
      • val       = rows in [VAL_START, VAL_END]
      • test      = sample_submission dates (548 rows)
    """
    sales = _load_sales()
    train_fit = sales[sales["Date"] < VAL_START].copy()
    val = sales[(sales["Date"] >= VAL_START) & (sales["Date"] <= VAL_END)].copy()
    test = pd.read_csv(DATA_DIR / "sample_submission.csv", parse_dates=["Date"])
    test = test.sort_values("Date").reset_index(drop=True)
    return train_fit, val, test


def load_extrapolation_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (train_ext, val_ext) for a stricter forecast test:
      • train_ext = rows up to 2020-12-31 (8.5 yr)
      • val_ext   = rows in 2021-01-01 .. 2022-12-31 (2-yr forecast horizon)

    This mimics the real test regime (~1.5 yr ahead) much better than Q4-2022
    in-sample val. Models that ace val but flop here are extrapolation-blind
    and will flop on Kaggle (ex_07/ex_09 pattern: 365k val → 1.2M Kaggle).
    """
    sales = _load_sales()
    train_ext = sales[sales["Date"] <= EXT_TRAIN_END].copy()
    val_ext = sales[(sales["Date"] >= EXT_VAL_START) &
                    (sales["Date"] <= EXT_VAL_END)].copy()
    return train_ext, val_ext


def load_full_train() -> pd.DataFrame:
    """All sales rows (2012-07-04 .. 2022-12-31) for final retrain."""
    return _load_sales()


def data_dir() -> Path:
    """Agents may read auxiliary CSVs from here."""
    return DATA_DIR


# ─── metrics ─────────────────────────────────────────────────────────────────

def evaluate_forecast(val: pd.DataFrame,
                      rev_pred: np.ndarray,
                      cogs_pred: np.ndarray) -> dict:
    """
    Ground-truth metric. Returns dict with Revenue + COGS MAE/RMSE/MAPE/R².
    Prints a concise summary block.
    """
    y_rev = val["Revenue"].values
    y_cogs = val["COGS"].values
    rev_pred = np.asarray(rev_pred, dtype=float)
    cogs_pred = np.asarray(cogs_pred, dtype=float)

    assert rev_pred.shape == y_rev.shape, (
        f"Revenue pred shape {rev_pred.shape} != truth shape {y_rev.shape}"
    )
    assert cogs_pred.shape == y_cogs.shape, (
        f"COGS pred shape {cogs_pred.shape} != truth shape {y_cogs.shape}"
    )

    def _mape(y, yh):
        denom = np.where(np.abs(y) < 1e-9, 1.0, y)
        return float(np.mean(np.abs((y - yh) / denom)) * 100)

    out = {
        "mae_rev":  float(mean_absolute_error(y_rev, rev_pred)),
        "rmse_rev": float(np.sqrt(mean_squared_error(y_rev, rev_pred))),
        "r2_rev":   float(r2_score(y_rev, rev_pred)),
        "mape_rev": _mape(y_rev, rev_pred),
        "mae_cogs":  float(mean_absolute_error(y_cogs, cogs_pred)),
        "rmse_cogs": float(np.sqrt(mean_squared_error(y_cogs, cogs_pred))),
        "r2_cogs":   float(r2_score(y_cogs, cogs_pred)),
        "mape_cogs": _mape(y_cogs, cogs_pred),
    }
    print("---")
    print(f"val_mae_rev:   {out['mae_rev']:.2f}")
    print(f"val_mae_cogs:  {out['mae_cogs']:.2f}")
    print(f"val_rmse_rev:  {out['rmse_rev']:.2f}")
    print(f"val_rmse_cogs: {out['rmse_cogs']:.2f}")
    print(f"val_mape_rev:  {out['mape_rev']:.2f}")
    print(f"val_mape_cogs: {out['mape_cogs']:.2f}")
    print(f"val_r2_rev:    {out['r2_rev']:.4f}")
    print(f"val_r2_cogs:   {out['r2_cogs']:.4f}")
    return out


def evaluate_extrapolation(val_ext: pd.DataFrame,
                           rev_pred: np.ndarray,
                           cogs_pred: np.ndarray) -> dict:
    """
    Secondary metric on the 2021-2022 extrapolation holdout. Returns dict with
    ``ext_`` prefix so agent sees both numbers and can spot models that ace
    Q4-2022 val but fail on long-horizon extrapolation.
    """
    y_rev = val_ext["Revenue"].values
    y_cogs = val_ext["COGS"].values
    rev_pred = np.asarray(rev_pred, dtype=float)
    cogs_pred = np.asarray(cogs_pred, dtype=float)
    assert rev_pred.shape == y_rev.shape
    assert cogs_pred.shape == y_cogs.shape
    out = {
        "ext_mae_rev":  float(mean_absolute_error(y_rev, rev_pred)),
        "ext_rmse_rev": float(np.sqrt(mean_squared_error(y_rev, rev_pred))),
        "ext_r2_rev":   float(r2_score(y_rev, rev_pred)),
        "ext_mae_cogs":  float(mean_absolute_error(y_cogs, cogs_pred)),
        "ext_rmse_cogs": float(np.sqrt(mean_squared_error(y_cogs, cogs_pred))),
    }
    print("---")
    print(f"ext_mae_rev:   {out['ext_mae_rev']:.2f}")
    print(f"ext_mae_cogs:  {out['ext_mae_cogs']:.2f}")
    print(f"ext_rmse_rev:  {out['ext_rmse_rev']:.2f}")
    print(f"ext_r2_rev:    {out['ext_r2_rev']:.4f}")
    return out


def write_submission(test_dates: pd.Series,
                     rev_pred: np.ndarray,
                     cogs_pred: np.ndarray,
                     name: str = "autoresearch") -> Path:
    """
    Write a Kaggle-compatible submission CSV with exactly 548 rows.
    Ensures non-negative predictions, correct column order.
    """
    rev_pred = np.clip(np.asarray(rev_pred, dtype=float), 0, None)
    cogs_pred = np.clip(np.asarray(cogs_pred, dtype=float), 0, None)
    assert len(test_dates) == 548, f"expected 548 rows, got {len(test_dates)}"
    sub = pd.DataFrame({
        "Date": pd.to_datetime(test_dates).dt.strftime("%Y-%m-%d"),
        "Revenue": rev_pred,
        "COGS": cogs_pred,
    })
    path = SUBMISSION_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path


# ─── results tsv ─────────────────────────────────────────────────────────────

def _git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "nogit000"


def append_result(metrics: dict, status: str, description: str) -> None:
    """
    Append a row to results.tsv. Creates the header if missing.
    ``status`` is one of: keep, discard, crash, baseline, reference.
    ``metrics`` may contain both Q4-2022 val keys (mae_rev/mae_cogs/rmse_rev)
    and extrapolation keys (ext_mae_rev/ext_mae_cogs).
    """
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER)
    commit = _git_short_hash()
    mae_rev = metrics.get("mae_rev", 0.0)
    mae_cogs = metrics.get("mae_cogs", 0.0)
    rmse_rev = metrics.get("rmse_rev", 0.0)
    ext_mae_rev = metrics.get("ext_mae_rev", 0.0)
    ext_mae_cogs = metrics.get("ext_mae_cogs", 0.0)
    desc = description.replace("\t", " ").replace("\n", " ")[:140]
    with RESULTS_TSV.open("a") as f:
        f.write(f"{commit}\t{mae_rev:.2f}\t{mae_cogs:.2f}\t{rmse_rev:.2f}"
                f"\t{ext_mae_rev:.2f}\t{ext_mae_cogs:.2f}"
                f"\t{status}\t{desc}\n")


# ─── CLI self-check (use: python -m autoresearch.prepare) ────────────────────

if __name__ == "__main__":
    train, val, test = load_splits()
    print(f"train:  {len(train):>5} rows  {train['Date'].min().date()} → "
          f"{train['Date'].max().date()}")
    print(f"val:    {len(val):>5} rows  {val['Date'].min().date()} → "
          f"{val['Date'].max().date()}")
    print(f"test:   {len(test):>5} rows  {test['Date'].min().date()} → "
          f"{test['Date'].max().date()}")
    print(f"data dir: {DATA_DIR}")
