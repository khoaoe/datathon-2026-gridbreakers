from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "cleaning"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    roll_window: int = 365
    min_periods: int = 60
    top_n: int = 200


def _read_sales() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "sales.csv", parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def _read_promos() -> pd.DataFrame:
    p = ROOT / "data" / "promotions.csv"
    if not p.exists():
        return pd.DataFrame(columns=["start_date", "end_date"])
    promos = pd.read_csv(p, parse_dates=["start_date", "end_date"])
    promos = promos.dropna(subset=["start_date", "end_date"]).copy()
    return promos


def _promo_active(dates: pd.Series, promos: pd.DataFrame) -> np.ndarray:
    if promos.empty:
        return np.zeros(len(dates), dtype=int)
    d = pd.to_datetime(dates).values.astype("datetime64[ns]")
    active = np.zeros(len(d), dtype=bool)
    for _, r in promos.iterrows():
        s = np.datetime64(r["start_date"])
        e = np.datetime64(r["end_date"])
        active |= (d >= s) & (d <= e)
    return active.astype(int)


def _robust_zscore(s: pd.Series, cfg: Config) -> pd.Series:
    med = s.rolling(cfg.roll_window, min_periods=cfg.min_periods).median()
    resid = (s - med).abs()
    mad = resid.rolling(cfg.roll_window, min_periods=cfg.min_periods).median()
    sigma = 1.4826 * mad.replace(0, np.nan)
    sigma = sigma.fillna(s.rolling(cfg.roll_window, min_periods=cfg.min_periods).std())
    sigma = sigma.replace(0, np.nan).fillna(sigma.median())
    return (s - med) / sigma


def main() -> None:
    cfg = Config()
    sales = _read_sales()
    promos = _read_promos()

    # Basic integrity
    dates = sales["Date"]
    dup_dates = int(dates.duplicated().sum())
    min_date = str(dates.min().date())
    max_date = str(dates.max().date())
    full_range = pd.date_range(dates.min(), dates.max(), freq="D")
    missing_dates = full_range.difference(dates)

    promo_active = _promo_active(dates, promos)

    out_rows = []
    for col in ["Revenue", "COGS"]:
        s = sales[col].astype(float)
        z = _robust_zscore(s, cfg)
        d1 = s.diff()
        d7 = s.diff(7)

        # rank by |z| and by |diff|
        idx_z = z.abs().nlargest(cfg.top_n).index
        idx_d1 = d1.abs().nlargest(cfg.top_n).index
        idx_d7 = d7.abs().nlargest(cfg.top_n).index
        idx = pd.Index(idx_z.tolist() + idx_d1.tolist() + idx_d7.tolist()).unique()

        sub = sales.loc[idx, ["Date", col]].copy()
        sub["target"] = col
        sub["robust_z"] = z.loc[idx].values
        sub["abs_robust_z"] = np.abs(sub["robust_z"].values)
        sub["diff_1"] = d1.loc[idx].values
        sub["abs_diff_1"] = np.abs(sub["diff_1"].values)
        sub["diff_7"] = d7.loc[idx].values
        sub["abs_diff_7"] = np.abs(sub["diff_7"].values)
        sub["promo_active"] = promo_active[idx]
        sub["dow"] = pd.to_datetime(sub["Date"]).dt.dayofweek.astype(int)
        sub["month"] = pd.to_datetime(sub["Date"]).dt.month.astype(int)
        out_rows.append(sub)

    anomalies = pd.concat(out_rows, ignore_index=True)
    anomalies = anomalies.sort_values(["abs_robust_z", "abs_diff_1"], ascending=False).reset_index(drop=True)

    # Save
    anomalies_path = OUT_DIR / "sales_anomalies.csv"
    anomalies.to_csv(anomalies_path, index=False)

    summary = {
        "date_range": {"min": min_date, "max": max_date, "n_rows": int(len(sales))},
        "duplicates": {"duplicate_dates": dup_dates},
        "missing_dates": {"n_missing": int(len(missing_dates)), "first_10": [str(d.date()) for d in missing_dates[:10]]},
        "promotions": {"n_promos": int(len(promos)), "pct_days_promo_active": float(promo_active.mean())},
        "targets": {
            "Revenue": {"min": float(sales["Revenue"].min()), "max": float(sales["Revenue"].max())},
            "COGS": {"min": float(sales["COGS"].min()), "max": float(sales["COGS"].max())},
        },
        "outputs": {"anomalies_csv": str(anomalies_path)},
    }
    (OUT_DIR / "sales_anomaly_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print top few non-promo anomalies
    top_nonpromo = anomalies[anomalies["promo_active"] == 0].head(15)
    print("Top non-promo anomalies:")
    with pd.option_context("display.width", 160, "display.max_columns", 50):
        print(top_nonpromo[["Date", "target", "abs_robust_z", "abs_diff_1", "abs_diff_7", "month", "dow"]].to_string(index=False))
    print(f"\nWrote {anomalies_path} and {OUT_DIR/'sales_anomaly_summary.json'}")


if __name__ == "__main__":
    main()

