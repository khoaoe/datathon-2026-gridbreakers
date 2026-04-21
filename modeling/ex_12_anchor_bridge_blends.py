"""
EX_12: Anchor Bridge Blends (861k anchor -> EX_12 selected deltas)

Purpose:
- Build low-drift bridge submissions from strong 861k anchor toward EX_12 output.
- Keep daily cap focused to exactly 4 candidates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ANCHOR = Path("output/submissions/ex_06_ensemble_weighted.csv")
CANDIDATE = Path("output/submissions/ex_12_lgbm_selected_deltas.csv")
OUT = Path("output/submissions")
TRACK = Path("output/tracking/ex_12_anchor_bridge")

# Weight on candidate prediction. Anchor weight = 1 - w.
LADDER = [0.02, 0.04, 0.06, 0.08]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = ["Date", "Revenue", "COGS"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns {miss} in {path}")
    return df[req].sort_values("Date").reset_index(drop=True)


def _blend(
    anchor: pd.DataFrame, candidate: pd.DataFrame, w_candidate: float
) -> pd.DataFrame:
    w_anchor = 1.0 - w_candidate
    out = anchor[["Date"]].copy()
    out["Revenue"] = anchor["Revenue"] * w_anchor + candidate["Revenue"] * w_candidate
    out["COGS"] = anchor["COGS"] * w_anchor + candidate["COGS"] * w_candidate
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TRACK.mkdir(parents=True, exist_ok=True)

    anchor = _load(ANCHOR)
    candidate = _load(CANDIDATE)
    if len(anchor) != len(candidate):
        raise ValueError("Row count mismatch between anchor and candidate")

    rows = []
    for w in LADDER:
        pred = _blend(anchor, candidate, w)
        tag = int(round(w * 100))
        out_path = OUT / f"ex_12_bridge_w{tag:02d}.csv"
        pred.to_csv(out_path, index=False)

        mad_rev = (pred["Revenue"] - anchor["Revenue"]).abs().mean()
        mad_cogs = (pred["COGS"] - anchor["COGS"]).abs().mean()
        rows.append(
            {
                "file": out_path.name,
                "w_candidate": w,
                "w_anchor": 1.0 - w,
                "mad_avg_vs_anchor": (mad_rev + mad_cogs) / 2,
                "mad_revenue_vs_anchor": mad_rev,
                "mad_cogs_vs_anchor": mad_cogs,
                "corr_revenue_vs_anchor": pred["Revenue"].corr(anchor["Revenue"]),
                "corr_cogs_vs_anchor": pred["COGS"].corr(anchor["COGS"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values("w_candidate").reset_index(drop=True)
    summary.to_csv(TRACK / "summary.csv", index=False)

    notes = [
        "# EX_12 Anchor Bridge",
        "",
        "## Objective",
        "- Build low-drift bridges from 861k anchor to EX_12 selected-delta model.",
        "- Keep exactly 4 candidates for daily submit cap.",
        "",
        "## Inputs",
        f"- anchor: {ANCHOR}",
        f"- candidate: {CANDIDATE}",
        "",
        "## Submit Order (lowest drift -> highest drift)",
        "1. ex_12_bridge_w02.csv",
        "2. ex_12_bridge_w04.csv",
        "3. ex_12_bridge_w06.csv",
        "4. ex_12_bridge_w08.csv",
    ]
    (TRACK / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"Wrote {len(summary)} files to {OUT}")
    print(f"Summary saved: {TRACK / 'summary.csv'}")


if __name__ == "__main__":
    main()
