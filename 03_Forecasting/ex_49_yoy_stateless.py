"""
EX_49: YoY-Growth Stateless Hybrid — Target 620k LB

Root cause analysis:
- EX-48 stateless hybrid had stellar CV (794k) but predicted ~2.9M test mean
- Linear/Ridge trend models trained on ANY window predict 2.8-3.0M (too low)
- The true 2022→2023 YoY growth is ~+12%, implying test mean ~3.6M
- Recursive models predict ~3.7M but snowball errors add ~175k MAE

Architecture (3 parallel paths ensembled):

Path A: YoY-Scaled Trend + LGBM Residuals
  - Trend = last_year_actual × (1 + YoY_growth_rate)
  - LGBM learns residuals (seasonality corrections) with NO lags
  - Completely stateless → zero snowball

Path B: Quadratic Trend (since 2019) + LGBM Residuals
  - Captures the V-shaped recovery curve
  - LGBM residuals on calendar/profile features

Path C: AOV Trend × Orders Trend + LGBM Residuals
  - AOV has a clean monotonic uptrend (easier to extrapolate)
  - Orders are roughly flat → simple persistence
  - Revenue_trend = AOV_trend × Orders_trend

Final = weighted ensemble of paths, weights from CV.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from modeling.config import LGBM_PARAMS, SEED, FILES
from modeling.feature_engineering import (
    build_feature_table,
    get_feature_cols,
)
from modeling.utils import evaluate, load_sales, make_submission

warnings.filterwarnings("ignore")

TRACK = Path("output/tracking/ex_49_yoy_stateless")
SUB_DIR = Path("output/submissions")

FOLDS = [
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]


def _get_stateless_cols(base_cols, target):
    """Remove ALL recursive features."""
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


def _finalize_cols(df, cols):
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


# ─── Path A: YoY-Scaled Trend ────────────────────────────────────────────────

def build_yoy_trend(train_df, predict_dates, target="Revenue"):
    """
    For each prediction date, trend = same-DOY from last available year × growth.
    Growth rate = mean(last_year) / mean(year_before_last).
    """
    train = train_df[["Date", target]].copy()
    train["doy"] = train["Date"].dt.dayofyear
    train["year"] = train["Date"].dt.year

    last_year = train["year"].max()
    prev_year = last_year - 1

    ly_mean = train[train["year"] == last_year][target].mean()
    py_mean = train[train["year"] == prev_year][target].mean()
    yoy_growth = ly_mean / py_mean if py_mean > 0 else 1.0

    # DOY profile from last year
    doy_profile = (
        train[train["year"] == last_year]
        .groupby("doy")[target]
        .mean()
        .to_dict()
    )
    # Fallback: 2-year average DOY profile
    doy_fallback = (
        train[train["year"] >= prev_year]
        .groupby("doy")[target]
        .mean()
        .to_dict()
    )
    global_mean = ly_mean

    preds = []
    for d in predict_dates:
        ts = pd.Timestamp(d)
        doy = ts.dayofyear
        years_ahead = (ts.year - last_year) + (ts.month - 6) / 12.0  # fractional
        growth_factor = yoy_growth ** max(years_ahead, 0.5)

        base = doy_profile.get(doy, doy_fallback.get(doy, global_mean))
        preds.append(base * growth_factor)

    return np.array(preds), yoy_growth


# ─── Path B: Quadratic Trend ─────────────────────────────────────────────────

def build_quadratic_trend(train_df, predict_dates, target="Revenue",
                          start="2019-01-01"):
    """Degree-2 polynomial trend fit on recent data."""
    subset = train_df[train_df["Date"] >= start].copy()
    epoch = pd.Timestamp("2013-01-01")

    ti = ((subset["Date"] - epoch).dt.days.values).reshape(-1, 1)
    test_ti = np.array([(pd.Timestamp(d) - epoch).days for d in predict_dates]).reshape(-1, 1)

    poly = PolynomialFeatures(2, include_bias=False)
    ti_poly = poly.fit_transform(ti)
    test_poly = poly.transform(test_ti)

    model = Ridge(alpha=100.0)
    model.fit(ti_poly, subset[target].values)

    train_preds = model.predict(poly.transform(
        ((train_df["Date"] - epoch).dt.days.values).reshape(-1, 1)
    ))
    test_preds = model.predict(test_poly)

    return train_preds, test_preds, model


# ─── Path C: AOV × Orders Decomposition ──────────────────────────────────────

def load_daily_orders():
    orders = pd.read_csv(
        FILES["orders"], parse_dates=["order_date"], usecols=["order_id", "order_date"]
    )
    daily = orders.groupby("order_date").agg(n_orders=("order_id", "count")).reset_index()
    daily.columns = ["Date", "n_orders"]
    return daily


def build_aov_orders_trend(train_df, predict_dates, target="Revenue"):
    """Decompose: Revenue = AOV × Orders, trend each separately."""
    daily_orders = load_daily_orders()
    merged = train_df.merge(daily_orders, on="Date", how="left")
    merged["aov"] = merged[target] / merged["n_orders"].replace(0, np.nan)

    epoch = pd.Timestamp("2013-01-01")
    test_ti = np.array([(pd.Timestamp(d) - epoch).days for d in predict_dates]).reshape(-1, 1)

    # AOV: use full history (clean monotonic uptrend)
    aov_data = merged.dropna(subset=["aov"])
    ti_aov = ((aov_data["Date"] - epoch).dt.days.values).reshape(-1, 1)
    model_aov = Ridge(alpha=10.0).fit(ti_aov, aov_data["aov"].values)
    test_aov = model_aov.predict(test_ti)

    # Orders: use since 2020 (flat/slight recovery)
    ord_data = merged[merged["Date"] >= "2020-01-01"].dropna(subset=["n_orders"])
    ti_ord = ((ord_data["Date"] - epoch).dt.days.values).reshape(-1, 1)
    model_ord = Ridge(alpha=10.0).fit(ti_ord, ord_data["n_orders"].values)
    test_orders = model_ord.predict(test_ti)

    # Revenue trend = AOV × Orders
    test_rev_trend = test_aov * test_orders

    # Train predictions for residuals
    all_ti = ((train_df["Date"] - epoch).dt.days.values).reshape(-1, 1)
    train_aov = model_aov.predict(all_ti)
    train_orders = model_ord.predict(all_ti)
    train_rev_trend = train_aov * train_orders

    return train_rev_trend, test_rev_trend


# ─── LGBM Residual Model ─────────────────────────────────────────────────────

def fit_lgbm_residuals(trn_x, trn_resid, val_x, val_resid, n_seeds=3):
    """Multi-seed LGBM ensemble on residuals."""
    models = []
    for i in range(n_seeds):
        params = LGBM_PARAMS.copy()
        params["n_estimators"] = 1500
        params["random_state"] = SEED + i * 17
        m = lgb.LGBMRegressor(**params)
        m.fit(
            trn_x, trn_resid,
            eval_set=[(val_x, val_resid)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        models.append(m)
    return models


def predict_ensemble(models, x):
    preds = np.array([m.predict(x) for m in models])
    return preds.mean(axis=0)


# ─── Fold Evaluation ─────────────────────────────────────────────────────────

def evaluate_fold(sales, fold):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, _ = build_feature_table(sales, verbose=False, profile_source_df=train_slice)
    feat_df["time_index"] = (feat_df["Date"] - pd.Timestamp("2013-01-01")).dt.days

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    base_cols = get_feature_cols(feat_df)
    results = {}

    for target in ["Revenue", "COGS"]:
        cols = _finalize_cols(trn, _get_stateless_cols(base_cols, target))
        trn_x = trn[cols].fillna(0)
        val_x = val[cols].fillna(0)
        actuals = val[target].values

        path_preds = {}

        # ── Path A: YoY trend ──
        yoy_test_preds, yoy_growth = build_yoy_trend(
            train_slice, val["Date"].values, target
        )
        # For residuals, compute trend on training dates too
        yoy_trn_preds, _ = build_yoy_trend(
            sales[sales["Date"] < (val_start - pd.DateOffset(years=1))],
            train_slice["Date"].values, target
        )
        # Use last 2 years of training for residual model (enough history)
        trn_recent = trn[trn["Date"] >= (val_start - pd.DateOffset(years=3))].copy()
        trn_recent_x = trn_recent[cols].fillna(0)
        # YoY trend for training residuals
        yoy_trn_trend, _ = build_yoy_trend(
            sales[sales["Date"] < (val_start - pd.DateOffset(years=1))],
            trn_recent["Date"].values, target
        )
        trn_resid_a = trn_recent[target].values - yoy_trn_trend
        val_resid_a = actuals - yoy_test_preds

        models_a = fit_lgbm_residuals(trn_recent_x, trn_resid_a, val_x, val_resid_a, n_seeds=3)
        resid_pred_a = predict_ensemble(models_a, val_x)
        path_preds["A_yoy"] = np.maximum(0, yoy_test_preds + resid_pred_a)

        # ── Path B: Quadratic trend ──
        trn_trend_b_full, val_trend_b, _ = build_quadratic_trend(
            train_slice, val["Date"].values, target
        )
        # Align to training rows
        trn_trend_b = trn_trend_b_full[trn["Date"].isin(train_slice["Date"]).values]
        if len(trn_trend_b) != len(trn):
            trn_trend_b = trn_trend_b_full[-len(trn):]
        trn_resid_b = trn[target].values - trn_trend_b
        val_resid_b = actuals - val_trend_b

        models_b = fit_lgbm_residuals(trn_x, trn_resid_b, val_x, val_resid_b, n_seeds=3)
        resid_pred_b = predict_ensemble(models_b, val_x)
        path_preds["B_quad"] = np.maximum(0, val_trend_b + resid_pred_b)

        # ── Path C: AOV × Orders ──
        trn_trend_c, val_trend_c = build_aov_orders_trend(
            train_slice, val["Date"].values, target
        )
        trn_trend_c_aligned = trn_trend_c[-len(trn):]
        trn_resid_c = trn[target].values - trn_trend_c_aligned
        val_resid_c = actuals - val_trend_c

        models_c = fit_lgbm_residuals(trn_x, trn_resid_c, val_x, val_resid_c, n_seeds=3)
        resid_pred_c = predict_ensemble(models_c, val_x)
        path_preds["C_aov"] = np.maximum(0, val_trend_c + resid_pred_c)

        # ── Path D: Pure stateless LGBM (no trend decomposition) ──
        models_d = fit_lgbm_residuals(trn_x, trn[target].values, val_x, actuals, n_seeds=3)
        path_preds["D_pure"] = np.maximum(0, predict_ensemble(models_d, val_x))

        # Evaluate each path
        path_scores = {}
        for pname, ppred in path_preds.items():
            mae = float(np.mean(np.abs(actuals - ppred)))
            path_scores[pname] = mae
            print(f"    {target} {pname}: MAE={mae:,.0f}, mean={ppred.mean():,.0f}")

        results[target] = {
            "path_preds": path_preds,
            "path_scores": path_scores,
            "actuals": actuals,
            "cols": cols,
        }

    # Compute combined scores for each path
    print(f"\n  Combined scores ({fold['name']}):")
    path_names = list(results["Revenue"]["path_preds"].keys())
    best_score = float("inf")
    best_path = None
    for pname in path_names:
        rev_mae = results["Revenue"]["path_scores"][pname]
        cogs_mae = results["COGS"]["path_scores"][pname]
        score = rev_mae + 0.4 * cogs_mae
        print(f"    {pname}: {score:,.0f} (rev={rev_mae:,.0f}, cogs={cogs_mae:,.0f})")
        if score < best_score:
            best_score = score
            best_path = pname

    # Ensemble: optimize weights via grid search
    print(f"\n  Searching ensemble weights...")
    best_ens_score = float("inf")
    best_weights = None

    for wa in np.arange(0, 1.05, 0.1):
        for wb in np.arange(0, 1.05 - wa, 0.1):
            for wc in np.arange(0, 1.05 - wa - wb, 0.1):
                wd = 1.0 - wa - wb - wc
                if wd < -0.01:
                    continue
                wd = max(0, wd)

                total_score = 0
                for target in ["Revenue", "COGS"]:
                    pp = results[target]["path_preds"]
                    act = results[target]["actuals"]
                    ens = (wa * pp["A_yoy"] + wb * pp["B_quad"] +
                           wc * pp["C_aov"] + wd * pp["D_pure"])
                    ens = np.maximum(0, ens)
                    mae = float(np.mean(np.abs(act - ens)))
                    weight = 1.0 if target == "Revenue" else 0.4
                    total_score += mae * weight

                if total_score < best_ens_score:
                    best_ens_score = total_score
                    best_weights = (wa, wb, wc, wd)

    print(f"  Best ensemble: score={best_ens_score:,.0f}")
    print(f"  Weights: A={best_weights[0]:.1f}, B={best_weights[1]:.1f}, "
          f"C={best_weights[2]:.1f}, D={best_weights[3]:.1f}")

    return {
        "fold": fold["name"],
        "best_path": best_path,
        "best_single_score": best_score,
        "best_ens_score": best_ens_score,
        "best_weights": best_weights,
        "path_scores": {
            t: results[t]["path_scores"] for t in ["Revenue", "COGS"]
        },
    }


# ─── Full Retrain & Submission ────────────────────────────────────────────────

def generate_submission(sales, test, fold_results):
    print("\n" + "=" * 78)
    print("Full retrain on all data...")
    print("=" * 78)

    feat_df, _ = build_feature_table(sales, verbose=True, profile_source_df=sales)
    feat_df["time_index"] = (feat_df["Date"] - pd.Timestamp("2013-01-01")).dt.days

    base_cols = get_feature_cols(feat_df)

    # Average weights from CV folds
    avg_weights = np.mean([r["best_weights"] for r in fold_results], axis=0)
    print(f"Averaged weights: A={avg_weights[0]:.2f}, B={avg_weights[1]:.2f}, "
          f"C={avg_weights[2]:.2f}, D={avg_weights[3]:.2f}")

    final_preds = {}
    for target in ["Revenue", "COGS"]:
        cols = _finalize_cols(feat_df, _get_stateless_cols(base_cols, target))
        trn_x = feat_df[cols].fillna(0)
        trn_y = feat_df[target].values
        # Use last year as eval set for early stopping
        eval_x = trn_x.tail(365)
        eval_y = trn_y[-365:]

        test_dates = test["Date"].values

        # Path A: YoY
        yoy_test, yoy_g = build_yoy_trend(sales, test_dates, target)
        yoy_trn, _ = build_yoy_trend(
            sales[sales["Date"] < "2022-01-01"], sales["Date"].values, target
        )
        trn_resid_a = trn_y - yoy_trn
        models_a = fit_lgbm_residuals(trn_x, trn_resid_a, eval_x, eval_y[-365:] - yoy_trn[-365:], n_seeds=5)

        test_dummy = pd.DataFrame({"Date": test_dates})
        test_feat, _ = build_feature_table(
            pd.concat([sales, test_dummy.assign(Revenue=np.nan, COGS=np.nan)]),
            verbose=False, profile_source_df=sales,
        )
        test_feat["time_index"] = (test_feat["Date"] - pd.Timestamp("2013-01-01")).dt.days
        test_feat = test_feat[test_feat["Date"].isin(test["Date"])].copy()
        test_x = test_feat[cols].fillna(0)

        resid_a = predict_ensemble(models_a, test_x)
        pred_a = np.maximum(0, yoy_test + resid_a)

        # Path B: Quadratic
        trn_trend_b, test_trend_b, _ = build_quadratic_trend(sales, test_dates, target)
        trn_resid_b = trn_y - trn_trend_b
        models_b = fit_lgbm_residuals(trn_x, trn_resid_b, eval_x, eval_y - trn_trend_b[-365:], n_seeds=5)
        resid_b = predict_ensemble(models_b, test_x)
        pred_b = np.maximum(0, test_trend_b + resid_b)

        # Path C: AOV × Orders
        trn_trend_c, test_trend_c = build_aov_orders_trend(sales, test_dates, target)
        trn_resid_c = trn_y - trn_trend_c
        models_c = fit_lgbm_residuals(trn_x, trn_resid_c, eval_x, eval_y - trn_trend_c[-365:], n_seeds=5)
        resid_c = predict_ensemble(models_c, test_x)
        pred_c = np.maximum(0, test_trend_c + resid_c)

        # Path D: Pure stateless
        models_d = fit_lgbm_residuals(trn_x, trn_y, eval_x, eval_y, n_seeds=5)
        pred_d = np.maximum(0, predict_ensemble(models_d, test_x))

        # Ensemble
        wa, wb, wc, wd = avg_weights
        final = wa * pred_a + wb * pred_b + wc * pred_c + wd * pred_d
        final = np.maximum(0, final)

        final_preds[target] = final

        print(f"\n  {target}:")
        print(f"    Path A (YoY):   mean={pred_a.mean():,.0f}, growth={yoy_g:.3f}")
        print(f"    Path B (Quad):  mean={pred_b.mean():,.0f}")
        print(f"    Path C (AOV):   mean={pred_c.mean():,.0f}")
        print(f"    Path D (Pure):  mean={pred_d.mean():,.0f}")
        print(f"    Ensemble:       mean={final.mean():,.0f}")

    # Save submissions
    path_ens = SUB_DIR / "ex_49_yoy_stateless.csv"
    make_submission(test["Date"], final_preds["Revenue"], final_preds["COGS"], path_ens)

    # Also save individual path submissions
    for pname, (pa, pb, pc, pd_) in [
        ("path_a", (1, 0, 0, 0)), ("path_b", (0, 1, 0, 0)),
        ("path_c", (0, 0, 1, 0)), ("path_d", (0, 0, 0, 1)),
    ]:
        # We already computed these above, just need to re-weight
        pass  # Main ensemble is the primary submission

    return final_preds


def main():
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_49: YoY-Growth Stateless Hybrid — Target 620k LB")
    print("  Path A: YoY-Scaled Trend + LGBM Residuals")
    print("  Path B: Quadratic Trend (since 2019) + LGBM Residuals")
    print("  Path C: AOV × Orders Decomposition + LGBM Residuals")
    print("  Path D: Pure Stateless LGBM (baseline)")
    print("  ALL PATHS: Zero lags, zero recursion, zero snowball")
    print("=" * 78)

    fold_results = []
    for fold in FOLDS:
        print(f"\n{'='*78}")
        print(f"=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        print(f"{'='*78}")
        result = evaluate_fold(train, fold)
        fold_results.append(result)

    print("\n\n=== FOLD SUMMARY ===")
    for r in fold_results:
        print(f"  {r['fold']}: best_single={r['best_path']}({r['best_single_score']:,.0f}), "
              f"ensemble={r['best_ens_score']:,.0f}")

    mean_ens = np.mean([r["best_ens_score"] for r in fold_results])
    print(f"\n  Mean ensemble CV score: {mean_ens:,.0f}")

    # Generate submission
    final_preds = generate_submission(train, test, fold_results)

    # Bridge blends with current best (795k anchor)
    anchor_path = SUB_DIR / "ex_24_bridge_w01.csv"
    if anchor_path.exists():
        anchor = pd.read_csv(anchor_path, parse_dates=["Date"])
        for w in [0.1, 0.2, 0.3, 0.5]:
            blended_rev = (1 - w) * anchor["Revenue"].values + w * final_preds["Revenue"]
            blended_cogs = (1 - w) * anchor["COGS"].values + w * final_preds["COGS"]
            bpath = SUB_DIR / f"ex_49_bridge_w{int(w*100):02d}.csv"
            make_submission(test["Date"], blended_rev, blended_cogs, bpath)
            print(f"  Bridge w={w}: rev_mean={blended_rev.mean():,.0f}")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "mean_ens_score": float(mean_ens),
        "fold_results": [
            {
                "fold": r["fold"],
                "best_path": r["best_path"],
                "best_single_score": r["best_single_score"],
                "best_ens_score": r["best_ens_score"],
                "weights": list(r["best_weights"]),
            }
            for r in fold_results
        ],
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']:.0f}s")


if __name__ == "__main__":
    main()
