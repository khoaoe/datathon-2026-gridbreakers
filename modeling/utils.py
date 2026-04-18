"""
Shared utilities: evaluation metrics, data loading, plotting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from modeling.config import FILES, SEED, SUBMISSION_DIR


def evaluate(y_true, y_pred, label=""):
    """Compute MAE, RMSE, R² and print results."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((np.asarray(y_true) - np.asarray(y_pred)) /
                          np.where(np.asarray(y_true) == 0, 1, y_true))) * 100
    print(f"{'[' + label + '] ' if label else ''}"
          f"MAE={mae:,.2f}  RMSE={rmse:,.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def horizon_stratified_metrics(val_dates, y_true, y_pred, anchor_date=None,
                               buckets=((1, 7), (8, 30), (31, 90), (91, 365))):
    """
    Report MAE per forecast-horizon bucket. ``anchor_date`` defines the
    reference point: horizon = (val_date - anchor).days. If None, uses the day
    BEFORE the first val date, so horizons start at 1.
    """
    dates = pd.Series(pd.to_datetime(val_dates).values).reset_index(drop=True)
    if anchor_date is None:
        anchor = dates.min() - pd.Timedelta(days=1)
    else:
        anchor = pd.Timestamp(anchor_date)
    horizon = (dates - anchor).dt.days.values
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for lo, hi in buckets:
        mask = (horizon >= lo) & (horizon <= hi)
        if mask.sum() == 0:
            continue
        mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        out[f"h{lo}_{hi}"] = mae
        print(f"  horizon h={lo:>3}..{hi:>3}d  n={mask.sum():>3}  MAE={mae:,.0f}")
    return out


def save_val_predictions(dates, y_rev_pred, y_cogs_pred, name):
    """Persist validation predictions so ensemble weights can be optimized."""
    p = SUBMISSION_DIR / "val"
    p.mkdir(parents=True, exist_ok=True)
    dates_ser = pd.Series(pd.to_datetime(dates).values).dt.strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "Date": dates_ser.values,
        "Revenue": np.asarray(y_rev_pred).ravel(),
        "COGS": np.asarray(y_cogs_pred).ravel(),
    })
    df.to_csv(p / f"{name}.csv", index=False)
    return p / f"{name}.csv"


def load_sales():
    """Load train sales and test template."""
    train = pd.read_csv(FILES["sales"], parse_dates=["Date"])
    test = pd.read_csv(FILES["sample_sub"], parse_dates=["Date"])
    train = train.sort_values("Date").reset_index(drop=True)
    test = test.sort_values("Date").reset_index(drop=True)
    return train, test


def make_submission(dates, revenue, cogs, path):
    """Write submission CSV in correct format. Checks for nulls."""
    sub = pd.DataFrame({"Date": dates, "Revenue": revenue, "COGS": cogs})
    sub["Date"] = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d")

    # Guard: check for nulls before saving
    null_count = sub[["Revenue", "COGS"]].isnull().sum().sum()
    if null_count > 0:
        print(f"  WARNING: {null_count} null values found! Filling with column mean.")
        sub["Revenue"] = sub["Revenue"].fillna(sub["Revenue"].mean())
        sub["COGS"] = sub["COGS"].fillna(sub["COGS"].mean())

    sub.to_csv(path, index=False)
    print(f"Submission saved → {path}  ({len(sub)} rows, nulls={null_count})")
    return sub
