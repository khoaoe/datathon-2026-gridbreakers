"""
EX_17: Recursive-aware FE research with future auxiliary imputation.

Purpose:
- Evaluate whether lagged auxiliary signals help when future aux data is unknown.
- Compare zero-imputation vs profile-imputation for future aux values.
- Use recursive holdout metrics over two yearly folds (2021, 2022).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.config import FILES, LGBM_PARAMS
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

TRACK = Path("output/tracking/ex_17_recursive_aux_impute_research")
SUB_DIR = Path("output/submissions")
ANCHOR_PATH = Path("output/submissions/ex_06_ensemble_weighted.csv")

METHODS = [
    "baseline_keep_avg",
    "keep_avg_exo_zero",
    "keep_avg_exo_profile",
]

FOLDS = [
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

AUX_RAW_COLS = [
    "order_count",
    "pay_total",
    "cancel_rate",
    "return_count",
    "refund_total",
    "ship_count",
    "sessions",
    "visitors",
    "page_views",
    "bounce_rate",
]
AUX_LAGS = [7, 14, 28]


def _fit_lgbm(x_trn, y_trn, x_val, y_val):
    import lightgbm as lgb

    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1200

    model = lgb.LGBMRegressor(**params)
    model.fit(
        x_trn,
        y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def build_aux_raw_daily(date_index: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"Date": pd.to_datetime(pd.Series(date_index).unique())})
    out = out.sort_values("Date").reset_index(drop=True)

    orders = pd.read_csv(
        FILES["orders"],
        parse_dates=["order_date"],
        usecols=["order_id", "order_date", "order_status"],
    )
    payments = pd.read_csv(
        FILES["payments"],
        usecols=["order_id", "payment_value"],
    )
    ordp = orders.merge(payments, on="order_id", how="left")
    ordp["payment_value"] = _safe_num(ordp["payment_value"])
    ord_daily = ordp.groupby("order_date", as_index=False).agg(
        order_count=("order_id", "count"),
        pay_total=("payment_value", "sum"),
        cancel_rate=("order_status", lambda s: float((s == "cancelled").mean())),
    )

    returns = pd.read_csv(
        FILES["returns"],
        parse_dates=["return_date"],
        usecols=["return_id", "return_date", "refund_amount"],
    )
    returns["refund_amount"] = _safe_num(returns["refund_amount"])
    ret_daily = returns.groupby("return_date", as_index=False).agg(
        return_count=("return_id", "count"),
        refund_total=("refund_amount", "sum"),
    )

    shipments = pd.read_csv(
        FILES["shipments"],
        parse_dates=["ship_date"],
        usecols=["order_id", "ship_date"],
    )
    ship_daily = shipments.groupby("ship_date", as_index=False).agg(
        ship_count=("order_id", "count")
    )

    web = pd.read_csv(
        FILES["web_traffic"],
        parse_dates=["date"],
        usecols=["date", "sessions", "unique_visitors", "page_views", "bounce_rate"],
    )
    web.rename(columns={"unique_visitors": "visitors"}, inplace=True)
    web["sessions"] = _safe_num(web["sessions"])
    web["visitors"] = _safe_num(web["visitors"])
    web["page_views"] = _safe_num(web["page_views"])
    web["bounce_rate"] = _safe_num(web["bounce_rate"])
    web_daily = web.groupby("date", as_index=False).agg(
        sessions=("sessions", "sum"),
        visitors=("visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
    )

    out = out.merge(
        ord_daily.rename(columns={"order_date": "Date"}), on="Date", how="left"
    )
    out = out.merge(
        ret_daily.rename(columns={"return_date": "Date"}), on="Date", how="left"
    )
    out = out.merge(
        ship_daily.rename(columns={"ship_date": "Date"}), on="Date", how="left"
    )
    out = out.merge(web_daily.rename(columns={"date": "Date"}), on="Date", how="left")

    for c in AUX_RAW_COLS:
        if c not in out.columns:
            out[c] = 0.0
    out[AUX_RAW_COLS] = out[AUX_RAW_COLS].fillna(0.0)

    return out[["Date"] + AUX_RAW_COLS].sort_values("Date").reset_index(drop=True)


def impute_future_aux(
    aux_df: pd.DataFrame, known_end: pd.Timestamp, strategy: str
) -> pd.DataFrame:
    df = aux_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    future_mask = df["Date"] > known_end

    for c in AUX_RAW_COLS:
        df.loc[future_mask, c] = np.nan

    if strategy == "zero":
        df[AUX_RAW_COLS] = df[AUX_RAW_COLS].fillna(0.0)
        return df

    if strategy != "profile":
        raise ValueError(f"Unknown strategy: {strategy}")

    hist = df[df["Date"] <= known_end].copy()
    hist["dow"] = hist["Date"].dt.dayofweek
    hist["month"] = hist["Date"].dt.month

    df["dow"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    for c in AUX_RAW_COLS:
        global_mean = float(hist[c].mean()) if not hist.empty else 0.0
        dow_map = hist.groupby("dow")[c].mean()
        month_map = hist.groupby("month")[c].mean()

        fill = 0.6 * df["dow"].map(dow_map) + 0.4 * df["month"].map(month_map)
        fill = fill.fillna(df["dow"].map(dow_map))
        fill = fill.fillna(df["month"].map(month_map))
        fill = fill.fillna(global_mean)
        fill = fill.fillna(0.0)

        df[c] = df[c].fillna(fill)

    return df.drop(columns=["dow", "month"])


def build_aux_lag_table(aux_raw: pd.DataFrame) -> pd.DataFrame:
    out = aux_raw[["Date"]].copy()
    aux = aux_raw.sort_values("Date").reset_index(drop=True)

    for c in AUX_RAW_COLS:
        for lag in AUX_LAGS:
            out[f"exo_{c}_lag_{lag}"] = aux[c].shift(lag)

    out = out.fillna(0.0)
    return out


def _drop_opposite_target(cols: list[str], target: str) -> list[str]:
    blocked = ("COGS_",) if target == "Revenue" else ("Revenue_",)
    return [c for c in cols if not c.startswith(blocked)]


def _apply_method(cols: list[str], method: str, target: str) -> list[str]:
    out = _drop_opposite_target(cols, target)
    if method == "baseline_keep_avg":
        out = [c for c in out if not c.startswith("exo_")]
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
    aux_lag_by_date: pd.DataFrame | None,
) -> np.ndarray:
    history = history_df[["Date", target]].copy()
    preds: list[float] = []

    aux_index = None
    if aux_lag_by_date is not None:
        aux_index = aux_lag_by_date.copy()
        aux_index["Date"] = pd.to_datetime(aux_index["Date"])
        aux_index = aux_index.set_index("Date")

    for date in predict_dates:
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)

        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)

        if aux_index is not None and ts in aux_index.index:
            aux_vals = aux_index.loc[[ts]].reset_index(drop=True)
            for c in aux_vals.columns:
                if c == "Date":
                    continue
                last_row[c] = aux_vals[c].values[0]

        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else val

        pred = float(np.clip(model.predict(x_pred)[0], 0, None))
        preds.append(pred)

        history = pd.concat(
            [
                history,
                pd.DataFrame({"Date": [ts], target: [pred]}),
            ],
            ignore_index=True,
        )

    return np.array(preds)


def _build_fold_feature_frame(
    sales: pd.DataFrame,
    val_start: pd.Timestamp,
    method: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(
        sales,
        verbose=False,
        profile_source_df=train_slice,
    )

    if method == "baseline_keep_avg":
        return feat_df, profiles, None

    strategy = "zero" if method == "keep_avg_exo_zero" else "profile"

    aux_raw = build_aux_raw_daily(feat_df["Date"])
    aux_imp = impute_future_aux(
        aux_raw, known_end=val_start - pd.Timedelta(days=1), strategy=strategy
    )
    aux_lag = build_aux_lag_table(aux_imp)

    feat_df = feat_df.merge(aux_lag, on="Date", how="left")
    return feat_df, profiles, aux_lag


def evaluate_fold_method(sales: pd.DataFrame, fold: dict, method: str) -> dict:
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    feat_df, profiles, aux_lag = _build_fold_feature_frame(sales, val_start, method)

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(trn, _apply_method(base_cols, method, "Revenue"))
    cols_cogs = _finalize_cols(trn, _apply_method(base_cols, method, "COGS"))

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
    res_tf_rev = evaluate(y_val_rev, tf_rev, f"{fold['name']} {method} Revenue teacher")
    res_tf_cogs = evaluate(y_val_cogs, tf_cogs, f"{fold['name']} {method} COGS teacher")

    aux_for_rec = None
    if aux_lag is not None:
        aux_for_rec = aux_lag[
            (aux_lag["Date"] >= val_start) & (aux_lag["Date"] <= val_end)
        ].copy()

    rec_rev = recursive_predict(
        model_rev,
        trn[["Date", "Revenue"]],
        val["Date"].values,
        cols_rev,
        profiles,
        target="Revenue",
        aux_lag_by_date=aux_for_rec,
    )
    rec_cogs = recursive_predict(
        model_cogs,
        trn[["Date", "COGS"]],
        val["Date"].values,
        cols_cogs,
        profiles,
        target="COGS",
        aux_lag_by_date=aux_for_rec,
    )

    res_rec_rev = evaluate(
        y_val_rev, rec_rev, f"{fold['name']} {method} Revenue recursive"
    )
    res_rec_cogs = evaluate(
        y_val_cogs, rec_cogs, f"{fold['name']} {method} COGS recursive"
    )

    return {
        "fold": fold["name"],
        "method": method,
        "n_features_revenue": len(cols_rev),
        "n_features_cogs": len(cols_cogs),
        "tf_revenue_mae": float(res_tf_rev["mae"]),
        "tf_cogs_mae": float(res_tf_cogs["mae"]),
        "rec_revenue_mae": float(res_rec_rev["mae"]),
        "rec_cogs_mae": float(res_rec_cogs["mae"]),
        "score": float(res_rec_rev["mae"] + 0.4 * res_rec_cogs["mae"]),
    }


def _train_full_and_predict(train: pd.DataFrame, test: pd.DataFrame, method: str):
    feat_df, profiles = build_feature_table(
        train, verbose=False, profile_source_df=train
    )

    aux_lag_full = None
    if method != "baseline_keep_avg":
        strategy = "zero" if method == "keep_avg_exo_zero" else "profile"
        full_dates = (
            pd.concat([train["Date"], test["Date"]], ignore_index=True)
            .sort_values()
            .reset_index(drop=True)
        )
        aux_raw = build_aux_raw_daily(full_dates)
        aux_imp = impute_future_aux(
            aux_raw, known_end=train["Date"].max(), strategy=strategy
        )
        aux_lag_full = build_aux_lag_table(aux_imp)

        aux_train = aux_lag_full[aux_lag_full["Date"].isin(train["Date"])].copy()
        feat_df = feat_df.merge(aux_train, on="Date", how="left")

    base_cols = get_feature_cols(feat_df)
    cols_rev = _finalize_cols(feat_df, _apply_method(base_cols, method, "Revenue"))
    cols_cogs = _finalize_cols(feat_df, _apply_method(base_cols, method, "COGS"))

    x_full_rev = feat_df[cols_rev].fillna(0)
    y_full_rev = feat_df["Revenue"]
    x_full_cogs = feat_df[cols_cogs].fillna(0)
    y_full_cogs = feat_df["COGS"]

    model_rev = _fit_lgbm(
        x_full_rev, y_full_rev, x_full_rev.tail(365), y_full_rev.tail(365)
    )
    model_cogs = _fit_lgbm(
        x_full_cogs, y_full_cogs, x_full_cogs.tail(365), y_full_cogs.tail(365)
    )

    aux_test = None
    if aux_lag_full is not None:
        aux_test = aux_lag_full[aux_lag_full["Date"].isin(test["Date"])].copy()

    pred_rev = recursive_predict(
        model_rev,
        train[["Date", "Revenue"]],
        test["Date"].values,
        cols_rev,
        profiles,
        target="Revenue",
        aux_lag_by_date=aux_test,
    )
    pred_cogs = recursive_predict(
        model_cogs,
        train[["Date", "COGS"]],
        test["Date"].values,
        cols_cogs,
        profiles,
        target="COGS",
        aux_lag_by_date=aux_test,
    )

    return pred_rev, pred_cogs


def _make_bridges(candidate_path: Path):
    anchor = pd.read_csv(ANCHOR_PATH).sort_values("Date").reset_index(drop=True)
    cand = pd.read_csv(candidate_path).sort_values("Date").reset_index(drop=True)

    weights = [0.01, 0.02, 0.03, 0.04]
    rows = []

    for w in weights:
        wa = 1.0 - w
        out = anchor[["Date"]].copy()
        out["Revenue"] = anchor["Revenue"] * wa + cand["Revenue"] * w
        out["COGS"] = anchor["COGS"] * wa + cand["COGS"] * w

        tag = int(round(w * 100))
        out_path = SUB_DIR / f"ex_17_bridge_w{tag:02d}.csv"
        out.to_csv(out_path, index=False)

        mad_rev = float((out["Revenue"] - anchor["Revenue"]).abs().mean())
        mad_cogs = float((out["COGS"] - anchor["COGS"]).abs().mean())
        rows.append(
            {
                "file": out_path.name,
                "w_candidate": w,
                "w_anchor": wa,
                "mad_avg_vs_anchor": (mad_rev + mad_cogs) / 2,
                "mad_revenue_vs_anchor": mad_rev,
                "mad_cogs_vs_anchor": mad_cogs,
                "corr_revenue_vs_anchor": float(out["Revenue"].corr(anchor["Revenue"])),
                "corr_cogs_vs_anchor": float(out["COGS"].corr(anchor["COGS"])),
            }
        )

    pd.DataFrame(rows).sort_values("w_candidate").reset_index(drop=True).to_csv(
        TRACK / "bridge_summary.csv", index=False
    )


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_17: Recursive FE research with future-aux imputation")
    print("=" * 78)

    records = []
    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        for method in METHODS:
            print(f"\n--- Evaluating {method} ---")
            rec = evaluate_fold_method(train, fold, method)
            records.append(rec)

    fold_df = pd.DataFrame(records)
    fold_df.to_csv(TRACK / "fold_results.csv", index=False)

    summary = (
        fold_df.groupby("method", as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_rec_revenue_mae=("rec_revenue_mae", "mean"),
            mean_rec_cogs_mae=("rec_cogs_mae", "mean"),
            mean_tf_revenue_mae=("tf_revenue_mae", "mean"),
            mean_tf_cogs_mae=("tf_cogs_mae", "mean"),
            score=("score", "mean"),
            mean_features_revenue=("n_features_revenue", "mean"),
            mean_features_cogs=("n_features_cogs", "mean"),
        )
        .sort_values("score")
        .reset_index(drop=True)
    )
    summary.to_csv(TRACK / "method_summary.csv", index=False)

    best = summary.iloc[0].to_dict()
    best_method = best["method"]

    print("\nBest method:", best_method)
    print(
        summary[
            ["method", "mean_rec_revenue_mae", "mean_rec_cogs_mae", "score"]
        ].to_string(index=False)
    )

    pred_rev, pred_cogs = _train_full_and_predict(train, test, best_method)
    candidate_path = SUB_DIR / f"ex_17_{best_method}.csv"
    make_submission(test["Date"], pred_rev, pred_cogs, candidate_path)

    _make_bridges(candidate_path)

    notes = [
        "# EX_17 Recursive Aux-Impute FE Research",
        "",
        "## Goal",
        "- Test whether lagged auxiliary features help when future aux values are unknown.",
        "",
        "## Methods",
    ]
    notes.extend([f"- {m}" for m in METHODS])
    notes.extend(
        [
            "",
            "## Validation Setup",
            "- Recursive holdout folds: 2021 and 2022.",
            "- Score = Revenue MAE + 0.4 * COGS MAE.",
            "",
            "## Best Method",
            f"- {best_method}",
            f"- Mean recursive Revenue MAE: {best['mean_rec_revenue_mae']:,.2f}",
            f"- Mean recursive COGS MAE: {best['mean_rec_cogs_mae']:,.2f}",
            "",
            "## Outputs",
            f"- Candidate: {candidate_path}",
            "- fold_results.csv",
            "- method_summary.csv",
            "- bridge_summary.csv",
        ]
    )
    (TRACK / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "best_method": best_method,
        "methods": METHODS,
        "folds": FOLDS,
        "candidate_path": str(candidate_path),
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
