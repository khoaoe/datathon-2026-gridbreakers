"""
EX_05: N-HiTS Deep Learning Model
- Neural Hierarchical Interpolation for Time Series (Challu et al., AAAI 2023)
- Uses neuralforecast library
- Handles multi-scale temporal patterns
- GPU accelerated (RTX 3050)
"""
import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from modeling.config import SUBMISSION_DIR, VAL_START, SEED
from modeling.utils import evaluate, load_sales, make_submission
from modeling.tracker import ExperimentTracker


def main():
    start = time.time()
    tracker = ExperimentTracker("ex_05_nhits")

    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, NBEATS
    except ImportError:
        print("neuralforecast not installed. Run: pip install neuralforecast")
        return None

    train, test = load_sales()

    print("=" * 60)
    print("EX_05: N-HiTS DEEP LEARNING")
    print("=" * 60)

    # ── Prepare data in neuralforecast format ────────────────────────
    # NeuralForecast expects: unique_id, ds, y
    df_nf = train.rename(columns={"Date": "ds", "Revenue": "y"}).copy()
    df_nf["unique_id"] = "revenue"
    df_nf = df_nf[["unique_id", "ds", "y"]].sort_values("ds")

    df_cogs = train.rename(columns={"Date": "ds", "COGS": "y"}).copy()
    df_cogs["unique_id"] = "cogs"
    df_cogs = df_cogs[["unique_id", "ds", "y"]].sort_values("ds")

    horizon = len(test)  # 548 days
    input_size = 365 * 2  # 2 years of lookback

    print(f"\n  Horizon: {horizon} days")
    print(f"  Input size: {input_size} days")

    # ── Revenue model ────────────────────────────────────────────────
    print("\n[1/2] Training Revenue N-HiTS...")
    models_rev = [
        NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=1000,
            learning_rate=1e-3,
            batch_size=32,
            n_pool_kernel_size=[16, 8, 1],
            n_freq_downsample=[168, 24, 1],
            scaler_type="robust",
            random_seed=SEED,
            accelerator="gpu",
            devices=1,
            early_stop_patience_steps=10,
            val_check_steps=50,
        ),
    ]

    nf_rev = NeuralForecast(models=models_rev, freq="D")
    nf_rev.fit(df=df_nf)
    forecast_rev = nf_rev.predict()

    # ── COGS model ───────────────────────────────────────────────────
    print("\n[2/2] Training COGS N-HiTS...")
    models_cogs = [
        NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=1000,
            learning_rate=1e-3,
            batch_size=32,
            n_pool_kernel_size=[16, 8, 1],
            n_freq_downsample=[168, 24, 1],
            scaler_type="robust",
            random_seed=SEED,
            accelerator="gpu",
            devices=1,
            early_stop_patience_steps=10,
            val_check_steps=50,
        ),
    ]

    nf_cogs = NeuralForecast(models=models_cogs, freq="D")
    nf_cogs.fit(df=df_cogs)
    forecast_cogs = nf_cogs.predict()

    # ── Validation (cross-validation on last year) ───────────────────
    print("\nCross-validation results:")
    cv_rev = nf_rev.cross_validation(df=df_nf, n_windows=1, step_size=horizon)
    val_true = cv_rev["y"].values
    val_pred = cv_rev["NHITS"].values
    valid = ~np.isnan(val_true) & ~np.isnan(val_pred)
    res_rev = evaluate(val_true[valid], val_pred[valid], "Revenue CV")

    # ── Generate submission ──────────────────────────────────────────
    rev_preds = forecast_rev["NHITS"].values[:horizon]
    cogs_preds = forecast_cogs["NHITS"].values[:horizon]

    make_submission(test["Date"], rev_preds, cogs_preds,
                    SUBMISSION_DIR / "ex_05_nhits.csv")

    tracker.log_final(res_rev)
    tracker.log_params({"model": "NHITS", "input_size": input_size, "horizon": horizon, "max_steps": 1000})
    tracker.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return res_rev


if __name__ == "__main__":
    main()
