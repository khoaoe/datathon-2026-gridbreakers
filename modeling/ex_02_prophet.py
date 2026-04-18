"""
EX_02: Prophet Baseline
- Facebook Prophet with yearly + weekly seasonality
- Vietnamese holiday approximation from promo dates
- Multiplicative seasonality for revenue data
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from modeling.config import FILES, SUBMISSION_DIR, VAL_START, SEED
from modeling.utils import evaluate, load_sales, make_submission
from modeling.tracker import ExperimentTracker


def build_holiday_df():
    """Build holiday dataframe from promotions (approximate major events)."""
    promos = pd.read_csv(FILES["promotions"], parse_dates=["start_date", "end_date"])
    holidays = []
    for _, row in promos.iterrows():
        holidays.append(
            {
                "holiday": row["promo_name"],
                "ds": row["start_date"],
                "lower_window": 0,
                "upper_window": (row["end_date"] - row["start_date"]).days,
            }
        )
    return pd.DataFrame(holidays)


def main():
    start = time.time()
    tracker = ExperimentTracker("ex_02_prophet")

    try:
        from prophet import Prophet
    except ImportError:
        print("Prophet not installed. Run: pip install prophet")
        return None

    train, test = load_sales()

    # Prophet format
    df_prophet = train.rename(columns={"Date": "ds", "Revenue": "y"})
    df_cogs = train.rename(columns={"Date": "ds", "COGS": "y"})

    val_mask = df_prophet["ds"] >= VAL_START
    trn_rev = df_prophet[~val_mask].copy()
    trn_cogs = df_cogs[~val_mask].copy()
    val = train[val_mask].copy()

    holidays = build_holiday_df()

    print("=" * 60)
    print("EX_02: PROPHET BASELINE")
    print("=" * 60)

    # ── Revenue model ────────────────────────────────────────────────
    print("\nTraining Revenue model...")
    m_rev = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        holidays=holidays,
        changepoint_prior_scale=0.05,
    )
    m_rev.fit(trn_rev[["ds", "y"]])

    # ── COGS model ───────────────────────────────────────────────────
    print("Training COGS model...")
    m_cogs = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        holidays=holidays,
        changepoint_prior_scale=0.05,
    )
    m_cogs.fit(trn_cogs[["ds", "y"]])

    # ── Validation ───────────────────────────────────────────────────
    print("\nValidation results:")
    val_dates = pd.DataFrame({"ds": val["Date"]})
    val_rev_pred = m_rev.predict(val_dates)["yhat"].values
    val_cogs_pred = m_cogs.predict(val_dates)["yhat"].values

    res_rev = evaluate(val["Revenue"].values, val_rev_pred, "Revenue")
    res_cogs = evaluate(val["COGS"].values, val_cogs_pred, "COGS")

    # ── Retrain on full training data for submission ─────────────────
    print("\nRetraining on full data for submission...")
    m_rev_full = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        holidays=holidays,
        changepoint_prior_scale=0.05,
    )
    m_rev_full.fit(df_prophet[["ds", "y"]])

    m_cogs_full = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        holidays=holidays,
        changepoint_prior_scale=0.05,
    )
    m_cogs_full.fit(df_cogs[["ds", "y"]])

    test_dates = pd.DataFrame({"ds": test["Date"]})
    test_rev = m_rev_full.predict(test_dates)["yhat"].values
    test_cogs = m_cogs_full.predict(test_dates)["yhat"].values

    make_submission(
        test["Date"], test_rev, test_cogs, SUBMISSION_DIR / "ex_02_prophet.csv"
    )

    tracker.log_final(res_rev)
    tracker.log_params(
        {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05}
    )
    tracker.add_note(
        f"COGS — MAE={res_cogs['mae']:,.0f} RMSE={res_cogs['rmse']:,.0f} R²={res_cogs['r2']:.4f}"
    )
    tracker.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return res_rev


if __name__ == "__main__":
    main()
