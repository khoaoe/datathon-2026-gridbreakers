from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SUB = ROOT / "output" / "submissions"
OUT = (
    ROOT
    / "output"
    / "tracking"
    / "ex_53_rectify_horizon_residual"
    / "guardrail_stats.csv"
)


def compute_guardrails(df: pd.DataFrame, rev_2022_month: pd.Series) -> dict:
    monthly = df.groupby(df["Date"].dt.month)["Revenue"].mean()
    growth = monthly / rev_2022_month
    return {
        "rev_mean": float(df["Revenue"].mean()),
        "span": float(growth.max() - growth.min()),
        "jan": float(growth.loc[1]),
        "nov": float(growth.loc[11]),
        "dec": float(growth.loc[12]),
    }


def main() -> None:
    anchor = pd.read_csv(SUB / "ex_51_bridge_w15.csv", parse_dates=["Date"])
    base = pd.read_csv(SUB / "ex_53_rectify_base.csv", parse_dates=["Date"])

    for w in (0.10, 0.15, 0.20):
        out = base.copy()
        out["Revenue"] = (1.0 - w) * anchor["Revenue"].values + w * base[
            "Revenue"
        ].values
        out["COGS"] = (1.0 - w) * anchor["COGS"].values + w * base["COGS"].values
        path = SUB / f"ex_53_base_bridge_w{int(w * 100):02d}.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path}")

    sales = pd.read_csv(ROOT / "data" / "sales.csv", parse_dates=["Date"])
    sales_2022 = sales[sales["Date"].dt.year == 2022]
    rev_2022_month = sales_2022.groupby(sales_2022["Date"].dt.month)["Revenue"].mean()

    files = [
        "ex_51_bridge_w15.csv",
        "ex_53_rectify_base.csv",
        "ex_53_rectify_horizon_residual.csv",
        "ex_53_rectify_bridge_w10.csv",
        "ex_53_rectify_bridge_w15.csv",
        "ex_53_rectify_bridge_w20.csv",
        "ex_53_base_bridge_w10.csv",
        "ex_53_base_bridge_w15.csv",
        "ex_53_base_bridge_w20.csv",
    ]

    rows = []
    for fn in files:
        d = pd.read_csv(SUB / fn, parse_dates=["Date"])
        stats = compute_guardrails(d, rev_2022_month)
        rows.append({"file": fn, **stats})

    out_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    out_df.to_csv(OUT, index=False)
    print("\nGuardrail stats:")
    print(out_df.to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
