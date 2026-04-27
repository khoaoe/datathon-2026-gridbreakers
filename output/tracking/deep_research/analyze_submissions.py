from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SUB_DIR = ROOT / "output" / "submissions"
SALES_PATH = ROOT / "data" / "sales.csv"
OUT_JSON = (
    ROOT / "output" / "tracking" / "deep_research" / "submission_diagnostics.json"
)
OUT_CSV = ROOT / "output" / "tracking" / "deep_research" / "submission_diagnostics.csv"

TARGET_FILES = [
    "ex_24_bridge_w01.csv",
    "ex_25_bridge_w01.csv",
    "ex_49_yoy_stateless.csv",
    "ex_49_bridge_w10.csv",
    "ex_49_bridge_w15.csv",
    "ex_49_bridge_w20.csv",
    "ex_50_lag365_direct.csv",
    "ex_50_bridge_w10.csv",
    "ex_50_bridge_w20.csv",
    "ex_51_lag365_recursive.csv",
    "ex_51_bridge_w10.csv",
    "ex_51_bridge_w12.csv",
    "ex_51_bridge_w15.csv",
    "ex_51_bridge_w18.csv",
    "ex_51_bridge_w20.csv",
    "ex_51_bridge_w22.csv",
    "ex_52_blend.csv",
    "ex_52_recalib_blend.csv",
    "ex_52_growth_blend.csv",
    "ex_52_blend_bridge_w15.csv",
    "ex_52_recalib_blend_bridge_w15.csv",
]


def month_stats(df: pd.DataFrame, value_col: str) -> pd.Series:
    out = df.groupby(df["Date"].dt.month)[value_col].mean()
    out.index = out.index.astype(int)
    return out


def main() -> None:
    sales = pd.read_csv(SALES_PATH, parse_dates=["Date"])
    sales_2022 = sales[sales["Date"].dt.year == 2022].copy()

    rev_2022_month = month_stats(sales_2022, "Revenue")
    cogs_2022_month = month_stats(sales_2022, "COGS")

    rows: list[dict] = []

    for fn in TARGET_FILES:
        p = SUB_DIR / fn
        if not p.exists():
            continue
        pred = pd.read_csv(p, parse_dates=["Date"])

        rev_month = month_stats(pred, "Revenue")
        cogs_month = month_stats(pred, "COGS")

        rev_growth = rev_month / rev_2022_month
        cogs_growth = cogs_month / cogs_2022_month

        rev_d1 = pred["Revenue"].diff().dropna()
        cogs_d1 = pred["COGS"].diff().dropna()

        pred["dow"] = pred["Date"].dt.dayofweek
        rev_dow = pred.groupby("dow")["Revenue"].mean()

        rows.append(
            {
                "file": fn,
                "rev_mean": float(pred["Revenue"].mean()),
                "cogs_mean": float(pred["COGS"].mean()),
                "rev_std": float(pred["Revenue"].std()),
                "cogs_std": float(pred["COGS"].std()),
                "rev_d1_std": float(rev_d1.std()),
                "cogs_d1_std": float(cogs_d1.std()),
                "rev_month_growth_min": float(rev_growth.min()),
                "rev_month_growth_max": float(rev_growth.max()),
                "rev_month_growth_span": float(rev_growth.max() - rev_growth.min()),
                "rev_month_growth_median": float(rev_growth.median()),
                "cogs_month_growth_min": float(cogs_growth.min()),
                "cogs_month_growth_max": float(cogs_growth.max()),
                "cogs_month_growth_span": float(cogs_growth.max() - cogs_growth.min()),
                "cogs_month_growth_median": float(cogs_growth.median()),
                "rev_dow_span": float(rev_dow.max() - rev_dow.min()),
                "rev_jan_growth": float(rev_growth.loc[1]),
                "rev_nov_growth": float(rev_growth.loc[11]),
                "rev_dec_growth": float(rev_growth.loc[12]),
            }
        )

    out = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    # Pairwise correlation on top candidates
    top_files = [
        "ex_24_bridge_w01.csv",
        "ex_25_bridge_w01.csv",
        "ex_49_bridge_w10.csv",
        "ex_51_bridge_w10.csv",
        "ex_51_bridge_w15.csv",
        "ex_51_bridge_w20.csv",
        "ex_52_blend_bridge_w15.csv",
    ]
    top = []
    for fn in top_files:
        p = SUB_DIR / fn
        if p.exists():
            d = pd.read_csv(p, parse_dates=["Date"])
            top.append((fn, d["Revenue"].values))

    corr = {}
    for i, (fi, ai) in enumerate(top):
        corr[fi] = {}
        for j, (fj, aj) in enumerate(top):
            if len(ai) == len(aj):
                corr[fi][fj] = float(np.corrcoef(ai, aj)[0, 1])

    payload = {
        "n_files": int(len(out)),
        "summary": out.to_dict(orient="records"),
        "corr_revenue": corr,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print("\nTop by rev_month_growth_span (lower is smoother):")
    print(
        out.sort_values("rev_month_growth_span")[
            [
                "file",
                "rev_month_growth_span",
                "rev_mean",
                "rev_month_growth_median",
                "rev_jan_growth",
                "rev_nov_growth",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop by rev_mean (higher level):")
    print(
        out.sort_values("rev_mean", ascending=False)[
            [
                "file",
                "rev_mean",
                "rev_month_growth_span",
                "rev_jan_growth",
                "rev_nov_growth",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
