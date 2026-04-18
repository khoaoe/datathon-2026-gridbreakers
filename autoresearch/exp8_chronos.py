"""
autoresearch/exp8_chronos.py — Chronos zero-shot forecast.

Amazon Chronos foundation model, zero-shot predicts long-horizon Revenue
directly from history. No Prophet, no LGBM. Then COGS = Revenue × trailing
90-day ratio from training data.

Reason: our Prophet + LGBM stack over-extrapolates (2024 mean +35 % vs 2022
actual). Chronos was pretrained on billions of time series and should
produce a more calibrated long-horizon distribution.

Uses chronos-bolt-small (t5-efficient, ~50M params) — fast on CPU, and its
quantile-regression head supports horizons up to 64 steps natively. For the
548-day horizon we predict in 64-step chunks and roll context forward.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

_CACHE = Path(__file__).resolve().parent.parent / ".hf_cache"
_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE))

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.prepare import (
    load_splits, load_extrapolation_splits, load_full_train,
    evaluate_forecast, evaluate_extrapolation,
    write_submission, append_result,
)


MODEL_ID = "amazon/chronos-bolt-base"   # ~800MB, better quality
CHUNK = 128                              # longer chunks = less recursive drift
CONTEXT_LEN = 2048                       # give model full history


def _chronos_predict_horizon(pipeline, history: np.ndarray, horizon: int) -> np.ndarray:
    """
    Recursively predict `horizon` steps by chaining CHUNK-sized median forecasts.
    Each chunk sees the last CONTEXT_LEN days (history + previously predicted).
    """
    preds = []
    ctx = history.copy().astype(np.float32)
    remaining = horizon
    while remaining > 0:
        take = min(CHUNK, remaining)
        tensor = torch.tensor(ctx[-CONTEXT_LEN:], dtype=torch.float32)
        q, _ = pipeline.predict_quantiles(
            inputs=tensor,
            prediction_length=take,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        median = q[0, :, 1].cpu().numpy()
        preds.append(median)
        ctx = np.concatenate([ctx, median], axis=0)
        remaining -= take
    return np.concatenate(preds, axis=0)[:horizon]


def _cogs_from_rev(rev_pred: np.ndarray, train_df: pd.DataFrame, window: int = 90) -> np.ndarray:
    """COGS = Revenue × trailing-window median(COGS/Revenue) from training."""
    last = train_df.tail(window)
    ratio = (last["COGS"] / last["Revenue"].clip(lower=1)).median()
    return rev_pred * float(ratio)


def run_split(pipeline, train_df: pd.DataFrame, horizon_df: pd.DataFrame, label: str):
    print(f"  [{label}] chronos zero-shot ({len(horizon_df)} days)...")
    rev_hist = train_df["Revenue"].values.astype(np.float32)
    cogs_hist = train_df["COGS"].values.astype(np.float32)
    rev_pred = _chronos_predict_horizon(pipeline, rev_hist, len(horizon_df))
    cogs_pred = _chronos_predict_horizon(pipeline, cogs_hist, len(horizon_df))
    rev_pred = np.clip(rev_pred, 0, None)
    cogs_pred = np.clip(cogs_pred, 0, None)
    return rev_pred, cogs_pred


def main():
    t0 = time.time()
    print(f"[exp8] loading {MODEL_ID}...")
    from chronos import BaseChronosPipeline
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float32,
    )
    print(f"[exp8] loaded in {time.time() - t0:.1f}s")

    train_fit, val, test = load_splits()
    full_train = pd.concat([train_fit, val], ignore_index=True).sort_values("Date")

    print("\n[1/3] primary val (Q4 2022)...")
    rev_v, cogs_v = run_split(pipeline, train_fit, val, "val")
    m = evaluate_forecast(val, rev_v, cogs_v)

    print("\n[2/3] extrapolation val (2021-2022)...")
    train_ext, val_ext = load_extrapolation_splits()
    rev_e, cogs_e = run_split(pipeline, train_ext, val_ext, "ext")
    mx = evaluate_extrapolation(val_ext, rev_e, cogs_e)

    print("\n[3/3] test (548 days)...")
    rev_t, cogs_t = run_split(pipeline, full_train, test, "test")
    out = write_submission(test["Date"], rev_t, cogs_t, name="autoresearch_chronos")
    print(f"submission: {out}")

    combined = {**m, **mx}
    append_result(combined, status="keep",
                  description="exp8: Chronos-Bolt-Small zero-shot, chunked 64-step rollout")
    print(f"total_seconds: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
