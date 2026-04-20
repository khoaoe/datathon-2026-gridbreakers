"""
EX_11: Anchor Bridge Blends (861k -> FE Refresh)

Purpose:
- Build low-risk bridge submissions between strong 861k anchor and newer
  FE-refresh ensemble output (reported ~879k).
- Respect daily submit cap with focused 4-candidate ladder.

Outputs:
- output/submissions/ex_11_bridge_w05.csv
- output/submissions/ex_11_bridge_w10.csv
- output/submissions/ex_11_bridge_w15.csv
- output/submissions/ex_11_bridge_w20.csv
- output/tracking/ex_11_anchor_bridge/summary.csv
- output/tracking/ex_11_anchor_bridge/notes.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("output/submissions/ex_06_ensemble_weighted.csv")
FE = Path("output/submissions/ex_06_ensemble_weighted_fe_refresh.csv")
OUT = Path("output/submissions")
TRACK = Path("output/tracking/ex_11_anchor_bridge")


# Weight on FE refresh prediction. Anchor weight = 1 - w.
LADDER = [0.05, 0.10, 0.15, 0.20]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = ["Date", "Revenue", "COGS"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns {miss} in {path}")
    return df[req].sort_values("Date").reset_index(drop=True)


def _blend(base: pd.DataFrame, fe: pd.DataFrame, w_fe: float) -> pd.DataFrame:
    w_base = 1.0 - w_fe
    out = base[["Date"]].copy()
    out["Revenue"] = base["Revenue"] * w_base + fe["Revenue"] * w_fe
    out["COGS"] = base["COGS"] * w_base + fe["COGS"] * w_fe
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TRACK.mkdir(parents=True, exist_ok=True)

    base = _load(BASE)
    fe = _load(FE)
    if len(base) != len(fe):
        raise ValueError("Row count mismatch between base and fe_refresh")

    rows = []
    for w in LADDER:
        pred = _blend(base, fe, w)
        tag = int(round(w * 100))
        out_path = OUT / f"ex_11_bridge_w{tag:02d}.csv"
        pred.to_csv(out_path, index=False)

        mad_rev = (pred["Revenue"] - base["Revenue"]).abs().mean()
        mad_cogs = (pred["COGS"] - base["COGS"]).abs().mean()
        rows.append(
            {
                "file": out_path.name,
                "w_fe_refresh": w,
                "w_anchor": 1.0 - w,
                "mad_avg_vs_anchor": (mad_rev + mad_cogs) / 2,
                "mad_revenue_vs_anchor": mad_rev,
                "mad_cogs_vs_anchor": mad_cogs,
                "corr_revenue_vs_anchor": pred["Revenue"].corr(base["Revenue"]),
                "corr_cogs_vs_anchor": pred["COGS"].corr(base["COGS"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values("w_fe_refresh").reset_index(drop=True)
    summary.to_csv(TRACK / "summary.csv", index=False)

    notes = [
        "# EX_11 Anchor Bridge",
        "",
        "## Objective",
        "- Bridge from strong 861k anchor toward FE-refresh output with small steps.",
        "- Keep exactly 4 files for daily submit cap.",
        "",
        "## Inputs",
        f"- anchor: {BASE}",
        f"- fe_refresh: {FE}",
        "",
        "## Submit Order",
        "1. ex_11_bridge_w05.csv",
        "2. ex_11_bridge_w10.csv",
        "3. ex_11_bridge_w15.csv",
        "4. ex_11_bridge_w20.csv",
    ]
    (TRACK / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} files to {OUT}")
    print(f"Summary saved: {TRACK / 'summary.csv'}")


if __name__ == "__main__":
    main()
