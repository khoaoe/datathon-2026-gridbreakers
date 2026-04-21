"""
EX_16: Recursive-aware feature engineering research.

What this does:
- Compares multiple target-aligned FE variants on the 2022 holdout.
- Scores each variant by recursive performance (Revenue priority).
- Trains best variant on full data, writes submission candidate.
- Generates low-drift anchor bridges for daily submission cap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.config import LGBM_PARAMS, VAL_START
from modeling.feature_engineering import (
    apply_profiles_to_dates,
    build_calendar_features,
    build_feature_table,
    build_growth_features,
    build_lag_features,
    build_rolling_features,
    get_feature_cols,
)
from modeling.utils import evaluate, load_sales, make_submission

TRACK = Path("output/tracking/ex_16_recursive_fe_research")
SUB_DIR = Path("output/submissions")
ANCHOR_PATH = Path("output/submissions/ex_06_ensemble_weighted.csv")

AUX_PREFIXES = (
    "sessions_lag_",
    "visitors_lag_",
    "page_views_lag_",
    "bounce_rate_lag_",
    "avg_session_duration_sec_lag_",
    "order_count_lag_",
    "pay_total_lag_",
    "cancel_rate_lag_",
    "return_count_lag_",
    "refund_total_lag_",
    "return_qty_lag_",
    "ship_count_lag_",
    "ship_fee_total_lag_",
)

SELECTED_AUX = {
    "sessions_lag_7",
    "sessions_lag_14",
    "visitors_lag_7",
    "visitors_lag_14",
    "page_views_lag_7",
    "bounce_rate_lag_7",
    "order_count_lag_7",
    "pay_total_lag_7",
    "cancel_rate_lag_7",
    "return_count_lag_7",
    "refund_total_lag_7",
    "ship_count_lag_7",
}

METHODS = [
    "aligned_drop_avg",
    "aligned_keep_avg",
    "aligned_drop_avg_selected_aux",
    "aligned_keep_avg_selected_aux",
]


def _is_aux_lag(col: str) -> bool:
    return col.startswith(AUX_PREFIXES)


def _drop_opposite_target(cols: list[str], target: str) -> list[str]:
    blocked = ("COGS_",) if target == "Revenue" else ("Revenue_",)
    return [c for c in cols if not c.startswith(blocked)]


def _apply_method(cols: list[str], target: str, method: str) -> list[str]:
    out = _drop_opposite_target(cols, target)

    if "drop_avg" in method:
        out = [c for c in out if not c.startswith("avg_")]

    if "selected_aux" in method:
        filtered = []
        for c in out:
            if _is_aux_lag(c) and c not in SELECTED_AUX:
                continue
            filtered.append(c)
        out = filtered

    return out


def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.5]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def recursive_predict(
    model,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
) -> np.ndarray:
    history = history_df[["Date", target]].copy()
    preds: list[float] = []

    for date in predict_dates:
        row = pd.DataFrame({"Date": [pd.Timestamp(date)], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)

        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)

        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else val

        pred = float(model.predict(x_pred)[0])
        pred = max(0.0, pred)
        preds.append(pred)

        history = pd.concat(
            [
                history,
                pd.DataFrame({"Date": [pd.Timestamp(date)], target: [pred]}),
            ],
            ignore_index=True,
        )

    return np.array(preds)


def _fit_lgbm(x_trn, y_trn, x_val, y_val):
    import lightgbm as lgb

    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1500

    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_trn,
        y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
        ],
    )
    return model


def _evaluate_method(
    feat_df: pd.DataFrame,
    profiles,
    method: str,
) -> dict:
    val_start = pd.Timestamp(VAL_START)
    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[feat_df["Date"] >= val_start].copy()

    base_cols = get_feature_cols(feat_df)

    cols_rev = _apply_method(base_cols, "Revenue", method)
    cols_cogs = _apply_method(base_cols, "COGS", method)

    cols_rev = _finalize_cols(trn, cols_rev)
    cols_cogs = _finalize_cols(trn, cols_cogs)

    x_trn_rev = trn[cols_rev].fillna(0)
    y_trn_rev = trn["Revenue"]
    x_val_rev = val[cols_rev].fillna(0)
    y_val_rev = val["Revenue"]

    x_trn_cogs = trn[cols_cogs].fillna(0)
    y_trn_cogs = trn["COGS"]
    x_val_cogs = val[cols_cogs].fillna(0)
    y_val_cogs = val["COGS"]

    model_rev = _fit_lgbm(x_trn_rev, y_trn_rev, x_val_rev, y_val_rev)
    model_cogs = _fit_lgbm(x_trn_cogs, y_trn_cogs, x_val_cogs, y_val_cogs)

    tf_rev = np.clip(model_rev.predict(x_val_rev), 0, None)
    tf_cogs = np.clip(model_cogs.predict(x_val_cogs), 0, None)

    res_tf_rev = evaluate(y_val_rev, tf_rev, f"{method} Revenue teacher")
    res_tf_cogs = evaluate(y_val_cogs, tf_cogs, f"{method} COGS teacher")

    hist_rev = trn[["Date", "Revenue"]].copy()
    hist_cogs = trn[["Date", "COGS"]].copy()

    rec_rev = recursive_predict(
        model_rev,
        hist_rev,
        val["Date"].values,
        cols_rev,
        profiles,
        target="Revenue",
    )
    rec_cogs = recursive_predict(
        model_cogs,
        hist_cogs,
        val["Date"].values,
        cols_cogs,
        profiles,
        target="COGS",
    )

    res_rec_rev = evaluate(y_val_rev, rec_rev, f"{method} Revenue recursive")
    res_rec_cogs = evaluate(y_val_cogs, rec_cogs, f"{method} COGS recursive")

    score = float(res_rec_rev["mae"] + 0.5 * res_rec_cogs["mae"])

    return {
        "method": method,
        "n_features_revenue": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "tf_revenue_mae": float(res_tf_rev["mae"]),
        "tf_cogs_mae": float(res_tf_cogs["mae"]),
        "rec_revenue_mae": float(res_rec_rev["mae"]),
        "rec_cogs_mae": float(res_rec_cogs["mae"]),
        "score": score,
    }


def _train_full_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    method: str,
):
    val_start = pd.Timestamp(VAL_START)
    profile_source = train[train["Date"] < val_start].copy()

    feat_df, _ = build_feature_table(
        train,
        verbose=False,
        profile_source_df=profile_source,
    )

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(feat_df, _apply_method(base_cols, "Revenue", method))
    cols_cogs = _finalize_cols(feat_df, _apply_method(base_cols, "COGS", method))

    x_full_rev = feat_df[cols_rev].fillna(0)
    y_full_rev = feat_df["Revenue"]
    x_full_cogs = feat_df[cols_cogs].fillna(0)
    y_full_cogs = feat_df["COGS"]

    model_rev = _fit_lgbm(
        x_full_rev, y_full_rev, x_full_rev.tail(365), y_full_rev.tail(365)
    )
    model_cogs = _fit_lgbm(
        x_full_cogs,
        y_full_cogs,
        x_full_cogs.tail(365),
        y_full_cogs.tail(365),
    )

    _, full_profiles = build_feature_table(
        train,
        verbose=False,
        profile_source_df=train,
    )

    pred_rev = recursive_predict(
        model_rev,
        train[["Date", "Revenue"]],
        test["Date"].values,
        cols_rev,
        full_profiles,
        target="Revenue",
    )
    pred_cogs = recursive_predict(
        model_cogs,
        train[["Date", "COGS"]],
        test["Date"].values,
        cols_cogs,
        full_profiles,
        target="COGS",
    )

    return pred_rev, pred_cogs


def _make_bridges(candidate_path: Path):
    anchor = pd.read_csv(ANCHOR_PATH).sort_values("Date").reset_index(drop=True)
    cand = pd.read_csv(candidate_path).sort_values("Date").reset_index(drop=True)

    weights = [0.01, 0.02, 0.03, 0.04]
    rows = []

    for w in weights:
        w_anchor = 1.0 - w
        out = anchor[["Date"]].copy()
        out["Revenue"] = anchor["Revenue"] * w_anchor + cand["Revenue"] * w
        out["COGS"] = anchor["COGS"] * w_anchor + cand["COGS"] * w

        tag = int(round(w * 100))
        out_path = SUB_DIR / f"ex_16_bridge_w{tag:02d}.csv"
        out.to_csv(out_path, index=False)

        mad_rev = float((out["Revenue"] - anchor["Revenue"]).abs().mean())
        mad_cogs = float((out["COGS"] - anchor["COGS"]).abs().mean())

        rows.append(
            {
                "file": out_path.name,
                "w_candidate": w,
                "w_anchor": w_anchor,
                "mad_avg_vs_anchor": (mad_rev + mad_cogs) / 2,
                "mad_revenue_vs_anchor": mad_rev,
                "mad_cogs_vs_anchor": mad_cogs,
                "corr_revenue_vs_anchor": float(out["Revenue"].corr(anchor["Revenue"])),
                "corr_cogs_vs_anchor": float(out["COGS"].corr(anchor["COGS"])),
            }
        )

    bridge_df = pd.DataFrame(rows).sort_values("w_candidate").reset_index(drop=True)
    bridge_df.to_csv(TRACK / "bridge_summary.csv", index=False)


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 76)
    print("EX_16: Recursive-aware Feature Engineering Research")
    print("=" * 76)

    profile_source = train[train["Date"] < pd.Timestamp(VAL_START)].copy()
    feat_df, profiles = build_feature_table(
        train,
        verbose=True,
        profile_source_df=profile_source,
    )

    records = []
    for method in METHODS:
        print(f"\n--- Evaluating {method} ---")
        rec = _evaluate_method(feat_df, profiles, method)
        records.append(rec)

    results = pd.DataFrame(records).sort_values("score").reset_index(drop=True)
    results.to_csv(TRACK / "method_results.csv", index=False)

    best = results.iloc[0].to_dict()
    best_method = best["method"]

    print("\nBest method:", best_method)
    print(
        results[["method", "rec_revenue_mae", "rec_cogs_mae", "score"]].to_string(
            index=False
        )
    )

    pred_rev, pred_cogs = _train_full_and_predict(train, test, best_method)
    cand_path = SUB_DIR / f"ex_16_{best_method}.csv"
    make_submission(test["Date"], pred_rev, pred_cogs, cand_path)

    _make_bridges(cand_path)

    notes = [
        "# EX_16 Recursive FE Research",
        "",
        "## Goal",
        "- Rank target-aligned FE variants using recursive holdout metrics.",
        "",
        "## Methods",
    ]
    notes.extend([f"- {m}" for m in METHODS])
    notes.extend(
        [
            "",
            "## Best Method",
            f"- {best_method}",
            f"- Recursive Revenue MAE: {best['rec_revenue_mae']:,.2f}",
            f"- Recursive COGS MAE: {best['rec_cogs_mae']:,.2f}",
            "",
            "## Outputs",
            f"- Candidate: {cand_path}",
            "- Bridge summary: output/tracking/ex_16_recursive_fe_research/bridge_summary.csv",
        ]
    )
    (TRACK / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "best_method": best_method,
        "candidate_path": str(cand_path),
        "methods_tested": METHODS,
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
