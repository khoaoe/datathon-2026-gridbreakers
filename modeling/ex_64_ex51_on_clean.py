"""
EX_64: EX_51 (lag365/stateless/full) trained on cleaned targets.

Uses `load_sales(clean=True)` which loads `data/sales_clean.csv`.

Outputs:
- output/submissions/ex_64_ex51_on_clean.csv
- bridge blends with current best anchor (ex_51_bridge_w15.csv) for safety
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from modeling.config import LGBM_PARAMS, SEED
from modeling.feature_engineering import (
    build_feature_table,
    build_calendar_features,
    build_lag_features,
    build_rolling_features,
    build_growth_features,
    apply_profiles_to_dates,
    get_feature_cols,
)
from modeling.utils import load_sales, make_submission


warnings.filterwarnings("ignore")

TRACK = Path("output/tracking/ex_64_ex51_on_clean")
SUB_DIR = Path("output/submissions")

N_SEEDS = 3


def _finalize_cols(df, cols):
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.30]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def _get_stateless_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [
        c for c in base_cols
        if not c.startswith(blocked)
        and "lag" not in c and "rmean" not in c and "rstd" not in c
        and "ratio" not in c and "growth" not in c and "momentum" not in c
        and "rmin" not in c and "rmax" not in c and "rmedian" not in c
        and "spread" not in c and "margin_ratio" not in c
        and "diff" not in c and "vol" not in c and "zscore" not in c
        and c not in ["Revenue", "COGS"]
    ]


def _get_lag365_cols(base_cols, target):
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    keep = []
    for c in base_cols:
        if c.startswith(blocked) or c in ["Revenue", "COGS"]:
            continue
        if "lag365" in c or "lag364" in c or "lag366" in c or "lag730" in c:
            keep.append(c)
            continue
        if "yoy" in c:
            keep.append(c)
            continue
        if any(k in c for k in ["lag", "rmean", "rstd", "rmin", "rmax",
                                "rmedian", "ratio", "growth", "momentum",
                                "spread", "margin_ratio", "diff", "vol",
                                "zscore"]):
            continue
        keep.append(c)
    return keep


def _fit_lgbm(x_trn, y_trn, x_val, y_val, seed=SEED):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 2000
    params["random_state"] = seed
    m = lgb.LGBMRegressor(**params)
    m.fit(
        x_trn, y_trn,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return m


def _predict_ensemble(models, x):
    return np.mean([m.predict(x) for m in models], axis=0)


def recursive_predict_lag365(models, history_df, predict_dates, feature_cols, profiles, target):
    history = history_df[["Date", target]].copy()
    preds = []
    for i, date in enumerate(predict_dates):
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined[f"{target}_lag365"] = combined[target].shift(365)
        combined[f"{target}_lag364"] = combined[target].shift(364)
        combined[f"{target}_lag366"] = combined[target].shift(366)

        shifted_365 = combined[target].shift(365)
        combined[f"{target}_lag365_smooth7"] = shifted_365.rolling(7, min_periods=1, center=True).mean()
        combined[f"{target}_lag365_rmean28"] = shifted_365.rolling(28, min_periods=7).mean()
        combined[f"{target}_lag365_rstd28"] = shifted_365.rolling(28, min_periods=7).std()
        shifted_730 = combined[target].shift(730)
        combined[f"{target}_yoy_365_730"] = shifted_365 / shifted_730.replace(0, np.nan)
        combined[f"{target}_lag365_rmean30"] = shifted_365.rolling(30, min_periods=7).mean()

        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)
        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else float(val)

        pred = float(np.mean([m.predict(x_pred)[0] for m in models]))
        pred = max(0.0, pred)
        preds.append(pred)
        history = pd.concat([history, pd.DataFrame({"Date": [ts], target: [pred]})], ignore_index=True)
        if (i + 1) % 150 == 0:
            print(f"{target} lag365 day {i+1}/{len(predict_dates)} pred={pred:,.0f}", flush=True)
    return np.array(preds)


def recursive_predict_full(models, history_df, predict_dates, feature_cols, profiles, target):
    history = history_df[["Date", target]].copy()
    preds = []
    for i, date in enumerate(predict_dates):
        ts = pd.Timestamp(date)
        row = pd.DataFrame({"Date": [ts], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")
        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        combined = build_growth_features(combined, target)
        last_row = apply_profiles_to_dates(combined.iloc[-1:].copy(), profiles)

        x_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last_row.columns:
                v = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(v) else float(v)

        pred = float(np.mean([m.predict(x_pred)[0] for m in models]))
        pred = max(0.0, pred)
        preds.append(pred)
        history = pd.concat([history, pd.DataFrame({"Date": [ts], target: [pred]})], ignore_index=True)
        if (i + 1) % 150 == 0:
            print(f"{target} full day {i+1}/{len(predict_dates)} pred={pred:,.0f}", flush=True)
    return np.array(preds)


def predict_stateless(models, history_df, predict_dates, feature_cols, profiles, target):
    history = history_df[["Date", target]].copy()
    test_df = pd.DataFrame({"Date": pd.to_datetime(predict_dates), target: np.nan})
    combined = pd.concat([history, test_df], ignore_index=True).sort_values("Date")
    combined = build_calendar_features(combined)
    test_rows = combined[combined["Date"].isin(pd.to_datetime(predict_dates))].copy()
    test_rows = apply_profiles_to_dates(test_rows, profiles)

    x = pd.DataFrame(0.0, index=range(len(test_rows)), columns=feature_cols)
    for c in feature_cols:
        if c in test_rows.columns:
            vals = test_rows[c].values
            x[c] = np.where(pd.isna(vals), 0.0, vals)
    return np.maximum(0.0, _predict_ensemble(models, x))


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales(clean=True)
    print(f"Loaded cleaned train rows={len(train)}", flush=True)

    feat_df, profiles = build_feature_table(train, verbose=True, profile_source_df=train)

    # add lag365-specific derived features (same as EX_51)
    for target in ["Revenue", "COGS"]:
        shifted_365 = feat_df[target].shift(365)
        feat_df[f"{target}_lag365_smooth7"] = shifted_365.rolling(7, min_periods=1, center=True).mean()
        feat_df[f"{target}_lag365_rmean28"] = shifted_365.rolling(28, min_periods=7).mean()
        feat_df[f"{target}_lag365_rstd28"] = shifted_365.rolling(28, min_periods=7).std()
        shifted_730 = feat_df[target].shift(730)
        feat_df[f"{target}_yoy_365_730"] = shifted_365 / shifted_730.replace(0, np.nan)
        feat_df[f"{target}_lag365_rmean30"] = shifted_365.rolling(30, min_periods=7).mean()

    base_cols = get_feature_cols(feat_df)
    avg_w = (0.0, 0.7, 0.3)  # from EX_51 meta

    final_preds = {}
    for target in ["Revenue", "COGS"]:
        cols_l365 = _finalize_cols(feat_df, _get_lag365_cols(base_cols, target))
        cols_sl = _finalize_cols(feat_df, _get_stateless_cols(base_cols, target))
        cols_full = _finalize_cols(feat_df, _get_target_cols(base_cols, target))

        eval_y = feat_df[target].tail(365)
        models_l365 = [_fit_lgbm(feat_df[cols_l365].fillna(0), feat_df[target], feat_df[cols_l365].fillna(0).tail(365), eval_y, SEED + i*17) for i in range(N_SEEDS)]
        models_sl = [_fit_lgbm(feat_df[cols_sl].fillna(0), feat_df[target], feat_df[cols_sl].fillna(0).tail(365), eval_y, SEED + i*17) for i in range(N_SEEDS)]
        models_full = [_fit_lgbm(feat_df[cols_full].fillna(0), feat_df[target], feat_df[cols_full].fillna(0).tail(365), eval_y, SEED + i*17) for i in range(N_SEEDS)]

        p_l365 = recursive_predict_lag365(models_l365, train, test["Date"].values, cols_l365, profiles, target)
        p_sl = predict_stateless(models_sl, train, test["Date"].values, cols_sl, profiles, target)
        p_full = recursive_predict_full(models_full, train, test["Date"].values, cols_full, profiles, target)

        w1, w2, w3 = avg_w
        final = np.maximum(0.0, w1 * p_l365 + w2 * p_sl + w3 * p_full)
        final_preds[target] = final
        print(f"{target}: mean={final.mean():,.0f}", flush=True)

    out_path = SUB_DIR / "ex_64_ex51_on_clean.csv"
    make_submission(test["Date"], final_preds["Revenue"], final_preds["COGS"], out_path)

    anchor_path = SUB_DIR / "ex_51_bridge_w15.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path, parse_dates=["Date"])
        for w in (0.05, 0.10, 0.15, 0.20):
            br = (1 - w) * anchor["Revenue"].values + w * final_preds["Revenue"]
            bc = (1 - w) * anchor["COGS"].values + w * final_preds["COGS"]
            p = SUB_DIR / f"ex_64_bridge_w{int(w*100):02d}.csv"
            pd.DataFrame({"Date": anchor["Date"], "Revenue": br, "COGS": bc}).to_csv(p, index=False)
            print(f"wrote {p} rev_mean={br.mean():,.0f}", flush=True)

    meta = {"elapsed_sec": round(time.time() - t0, 1), "avg_w": list(avg_w), "out_path": str(out_path)}
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done in {meta['elapsed_sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

