"""
autoresearch/train.py — MUTABLE agent-edited file.

Baseline = Prophet (trend + multi-seasonality) + LightGBM residual stack on v3
features. Cloned from modeling/ex_08_prophet_residual, which beat the
tree-only LGBM baseline on Kaggle (890k vs 1.21M MAE) because Prophet
extrapolates the upward trend across the 548-day horizon while tree models
cannot.

Both metrics are reported:
  • val_mae_rev  — Q4 2022 in-sample holdout  (fast, primary)
  • ext_mae_rev  — 2021-2022 forecast from train ≤ 2020  (slow, honesty check)

An experiment that improves val_mae_rev but REGRESSES ext_mae_rev is almost
certainly overfitting to short-horizon in-sample patterns and will flop on
Kaggle. Prefer changes that improve *both* metrics, or at least leave
ext_mae_rev flat while lowering val_mae_rev.

Agents: edit freely, update EXPERIMENT_DESC, commit each experiment.
"""
from __future__ import annotations

import time
import sys
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoresearch.prepare import (
    load_splits, load_extrapolation_splits,
    evaluate_forecast, evaluate_extrapolation,
    write_submission, append_result,
)
from modeling.feature_engineering import (
    build_feature_table, build_calendar_features, build_lag_features,
    build_rolling_features, build_growth_features, apply_profiles_to_dates,
    get_feature_cols,
)
from modeling.config import LGBM_PARAMS


# ═════════════════════════════════════════════════════════════════════════════
# experiment identity — agent updates every experiment
# ═════════════════════════════════════════════════════════════════════════════
EXPERIMENT_DESC = "FINAL exp3b: log1p Prophet cp=0.2 + default LGBM residual (best val)"

# Hyperparameters the agent can tweak
PROPHET_KW = dict(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.2,
)
LOG_PROPHET = True                 # fit Prophet on log1p(target)?
USE_PROPHET_REGRESSORS = False     # add_regressor: is_promo, is_tet, etc.
PROPHET_COUNTRY_HOLIDAYS = None    # add Prophet built-in holidays (None/"VN"/"US")
DROP_LAG_FEATURES = False          # residual LGBM: drop target lag/rolling?
LGBM_KW = LGBM_PARAMS.copy()
RUN_EXTRAPOLATION_CHECK = True     # toggle the 2nd val slice (~+60s)
PROPHET_TRAIN_YEARS: float | None = None  # e.g. 4.0 = last 4yr only (dampens trend)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prophet_add_regressors(model):
    """Future-known regressors Prophet can use directly (known for test too)."""
    regs = ["is_promo", "is_vn_holiday", "is_tet_week",
            "is_black_friday_week", "is_xmas_week", "covid_flag"]
    for r in regs:
        model.add_regressor(r, standardize=True)
    return regs


def _regressor_frame(dates: pd.Series, promo_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a DataFrame of Prophet regressor values keyed by ds."""
    from modeling.feature_engineering import (
        build_calendar_features as _bc, build_promo_calendar,
    )
    d = pd.DataFrame({"Date": pd.to_datetime(dates.values)})
    d = _bc(d)
    if promo_daily is None:
        promo_daily = build_promo_calendar(pd.DatetimeIndex(d["Date"]))
    d = d.merge(promo_daily[["Date", "promo_is_active"]].rename(
        columns={"promo_is_active": "is_promo"}), on="Date", how="left")
    d["is_promo"] = d["is_promo"].fillna(0).astype(int)
    cols = ["Date", "is_promo", "is_vn_holiday", "is_tet_week",
            "is_black_friday_week", "is_xmas_week", "covid_flag"]
    out = d[cols].copy()
    out = out.rename(columns={"Date": "ds"})
    return out


def fit_prophet(train_df: pd.DataFrame, target: str,
                use_regressors: bool = False,
                log_target: bool = False,
                country_holidays: str | None = None):
    from prophet import Prophet
    df = train_df.rename(columns={"Date": "ds", target: "y"})[["ds", "y"]].copy()
    if log_target:
        df["y"] = np.log1p(df["y"].clip(lower=0))
    m = Prophet(**PROPHET_KW)
    if country_holidays:
        try:
            m.add_country_holidays(country_name=country_holidays)
        except Exception as e:
            print(f"    [warn] country_holidays={country_holidays} failed: {e}")
    regs = []
    if use_regressors:
        regs = _prophet_add_regressors(m)
        reg_df = _regressor_frame(df["ds"])
        df = df.merge(reg_df, on="ds", how="left")
        for r in regs:
            df[r] = df[r].fillna(0)
    m.fit(df)
    return m, regs


def prophet_predict(model, dates,
                    regs: list | None = None,
                    log_target: bool = False) -> np.ndarray:
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    if regs:
        reg_df = _regressor_frame(future["ds"])
        future = future.merge(reg_df, on="ds", how="left")
        for r in regs:
            future[r] = future[r].fillna(0)
    yhat = model.predict(future)["yhat"].values
    if log_target:
        yhat = np.expm1(yhat)
    return yhat


def _build_residual_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    m_rev, m_cogs, regs_rev, regs_cogs,
    log_target: bool,
) -> Tuple[pd.DataFrame, dict, list]:
    """Build residual-target feature table covering train (+ val + test dates)."""
    feat_df, bundle = build_feature_table(train_df, test_df=test_df, verbose=False)
    feat_df["prophet_rev"] = prophet_predict(m_rev, feat_df["Date"], regs_rev,
                                             log_target=log_target)
    feat_df["prophet_cogs"] = prophet_predict(m_cogs, feat_df["Date"], regs_cogs,
                                              log_target=log_target)
    feat_df["resid_rev"] = feat_df["Revenue"] - feat_df["prophet_rev"]
    feat_df["resid_cogs"] = feat_df["COGS"] - feat_df["prophet_cogs"]

    feature_cols = get_feature_cols(feat_df)
    feature_cols = [c for c in feature_cols
                    if c not in ("resid_rev", "resid_cogs",
                                 "prophet_rev", "prophet_cogs")]
    if DROP_LAG_FEATURES:
        feature_cols = [c for c in feature_cols
                        if not c.startswith(("Revenue_lag_", "COGS_lag_",
                                             "Revenue_rmean_", "Revenue_rstd_",
                                             "Revenue_rmin_", "Revenue_rmax_",
                                             "Revenue_rmedian_", "COGS_rmean_",
                                             "COGS_rstd_", "COGS_rmin_",
                                             "COGS_rmax_", "COGS_rmedian_",
                                             "Revenue_yoy_", "Revenue_wow_",
                                             "Revenue_mom_", "Revenue_momentum",
                                             "Revenue_diff_", "Revenue_spike_",
                                             "Revenue_cv_", "COGS_spike_"))]
    feature_cols = [c for c in feature_cols
                    if feat_df[c].notna().mean() > 0.5
                    and feat_df[c].nunique(dropna=True) > 1]
    return feat_df, bundle, feature_cols


def _recursive_stack_predict(
    lgbm_resid_rev, lgbm_resid_cogs,
    history_df: pd.DataFrame, horizon_dates,
    prophet_rev_yhat, prophet_cogs_yhat,
    feature_cols, bundle,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recursive: lag features rebuilt from predicted Revenue trajectory."""
    hist = history_df[["Date", "Revenue", "COGS"]].copy()
    rev_preds = np.zeros(len(horizon_dates))
    cogs_preds = np.zeros(len(horizon_dates))

    for i, date in enumerate(pd.to_datetime(horizon_dates)):
        row = pd.DataFrame({"Date": [date], "Revenue": [np.nan], "COGS": [np.nan]})
        combined = pd.concat([hist, row], ignore_index=True).sort_values("Date")
        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, "Revenue")
        combined = build_lag_features(combined, "COGS")
        combined = build_rolling_features(combined, "Revenue")
        combined = build_rolling_features(combined, "COGS",
                                          windows=[7, 14, 28, 90])
        combined = build_growth_features(combined, "Revenue")
        last = combined.iloc[-1:].copy()
        last = apply_profiles_to_dates(last, bundle)
        X = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for c in feature_cols:
            if c in last.columns:
                v = last[c].values[0]
                X[c] = 0.0 if pd.isna(v) else v
        resid_rev_hat = lgbm_resid_rev.predict(X)[0]
        resid_cogs_hat = lgbm_resid_cogs.predict(X)[0]
        rev_pred = max(prophet_rev_yhat[i] + resid_rev_hat, 0.0)
        cogs_pred = max(prophet_cogs_yhat[i] + resid_cogs_hat, 0.0)
        rev_preds[i] = rev_pred
        cogs_preds[i] = cogs_pred
        hist = pd.concat(
            [hist, pd.DataFrame({"Date": [date],
                                 "Revenue": [rev_pred],
                                 "COGS": [cogs_pred]})],
            ignore_index=True,
        )
        if (i + 1) % 150 == 0:
            print(f"    predicted {i + 1}/{len(horizon_dates)}")
    return rev_preds, cogs_preds


def _fit_and_forecast(
    train_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    *,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit the full Prophet + LGBM-residual stack on ``train_df``, then recursively
    forecast rows of ``horizon_df`` (Date column only needed).
    """
    import lightgbm as lgb

    print(f"  [{label}] fitting Prophet...")
    prophet_train = train_df
    if PROPHET_TRAIN_YEARS is not None:
        cutoff = train_df["Date"].max() - pd.Timedelta(days=int(PROPHET_TRAIN_YEARS * 365.25))
        prophet_train = train_df[train_df["Date"] >= cutoff].copy()
        print(f"  [{label}] Prophet train window: {prophet_train['Date'].min().date()} "
              f"→ {prophet_train['Date'].max().date()} ({len(prophet_train)} rows)")
    m_rev, regs_rev = fit_prophet(prophet_train, "Revenue",
                                  use_regressors=USE_PROPHET_REGRESSORS,
                                  log_target=LOG_PROPHET,
                                  country_holidays=PROPHET_COUNTRY_HOLIDAYS)
    m_cogs, regs_cogs = fit_prophet(prophet_train, "COGS",
                                    use_regressors=USE_PROPHET_REGRESSORS,
                                    log_target=LOG_PROPHET,
                                    country_holidays=PROPHET_COUNTRY_HOLIDAYS)

    print(f"  [{label}] building residual features...")
    feat_df, bundle, feature_cols = _build_residual_features(
        train_df, None, horizon_df, m_rev, m_cogs, regs_rev, regs_cogs,
        log_target=LOG_PROPHET,
    )
    train_feat = feat_df[feat_df["Date"].isin(train_df["Date"])].copy()
    X_trn = train_feat[feature_cols].fillna(0)

    print(f"  [{label}] fitting LGBM on residuals (n_features={len(feature_cols)})...")
    model_rev = lgb.LGBMRegressor(**LGBM_KW)
    model_rev.fit(X_trn, train_feat["resid_rev"])
    model_cogs = lgb.LGBMRegressor(**LGBM_KW)
    model_cogs.fit(X_trn, train_feat["resid_cogs"])

    horizon_dates = pd.to_datetime(horizon_df["Date"].values)
    prophet_rev = prophet_predict(m_rev, horizon_dates, regs_rev,
                                  log_target=LOG_PROPHET)
    prophet_cogs = prophet_predict(m_cogs, horizon_dates, regs_cogs,
                                   log_target=LOG_PROPHET)

    print(f"  [{label}] recursive stack predict ({len(horizon_dates)} days)...")
    rev_preds, cogs_preds = _recursive_stack_predict(
        model_rev, model_cogs, train_df, horizon_dates,
        prophet_rev, prophet_cogs, feature_cols, bundle,
    )
    return rev_preds, cogs_preds


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"[autoresearch] experiment = {EXPERIMENT_DESC}")
    print(f"[autoresearch] config: log_prophet={LOG_PROPHET}  "
          f"regressors={USE_PROPHET_REGRESSORS}  "
          f"country_holidays={PROPHET_COUNTRY_HOLIDAYS}  "
          f"drop_lag={DROP_LAG_FEATURES}  "
          f"changepoint_prior={PROPHET_KW['changepoint_prior_scale']}")

    train_fit, val, test = load_splits()
    full_train = pd.concat([train_fit, val], ignore_index=True)

    # ── Primary val: Q4 2022 ──
    print("\n[1/3] primary val (Q4 2022) ...")
    val_rev, val_cogs = _fit_and_forecast(train_fit, val, label="val")
    metrics = evaluate_forecast(val, val_rev, val_cogs)

    # ── Extrapolation val: predict 2021-2022 from ≤2020 train ──
    metrics_ext = {}
    if RUN_EXTRAPOLATION_CHECK:
        print("\n[2/3] extrapolation val (2021-2022 forecast) ...")
        train_ext, val_ext = load_extrapolation_splits()
        ext_rev, ext_cogs = _fit_and_forecast(train_ext, val_ext, label="ext")
        metrics_ext = evaluate_extrapolation(val_ext, ext_rev, ext_cogs)

    # ── Final retrain on full data + test submission ──
    print("\n[3/3] retraining on full train+val and predicting 548-day test ...")
    test_rev, test_cogs = _fit_and_forecast(full_train, test, label="test")
    out = write_submission(test["Date"], test_rev, test_cogs, name="autoresearch")
    print(f"submission: {out}")

    combined = {**metrics, **metrics_ext}
    append_result(combined, status="keep", description=EXPERIMENT_DESC)
    print(f"total_seconds: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
