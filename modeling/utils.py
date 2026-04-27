"""
Shared utilities: evaluation metrics, data loading, plotting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from modeling.config import FILES, SEED


def evaluate(y_true, y_pred, label=""):
    """Compute MAE, RMSE, R² and print results."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{'[' + label + '] ' if label else ''}MAE={mae:,.2f}  RMSE={rmse:,.2f}  R²={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_sales(clean: bool = False):
    """Load train sales (optionally cleaned) and test template."""
    sales_key = "sales_clean" if clean and "sales_clean" in FILES else "sales"
    train = pd.read_csv(FILES[sales_key], parse_dates=["Date"])
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
