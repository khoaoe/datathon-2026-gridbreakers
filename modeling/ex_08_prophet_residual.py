"""
EX_08: Prophet + LightGBM residual stacking.

Prophet captures trend + multi-seasonality + holiday effects. LightGBM models
the residual (actual - prophet_yhat) using v3 features, so it focuses on what
Prophet misses (short-term lags, promo spikes, etc.). Final prediction =
prophet_yhat + lgbm_residual_pred.

Reasoning: Prophet extrapolates linear trend naturally across 548 test days,
which LightGBM cannot (tree models can't extrapolate outside training range).
Residual stacking combines the strengths.
"""
from __future__ import annotations

import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modeling.config import (
    LGBM_PARAMS, MODEL_DIR, SUBMISSION_DIR, VAL_START, VAL_END, SEED, TRAIN_END,
)
from modeling.utils import (
    evaluate, load_sales, make_submission,
    horizon_stratified_metrics, save_val_predictions,
)
from modeling.feature_engineering import (
    build_feature_table, build_calendar_features, build_lag_features,
    build_rolling_features, build_growth_features, apply_profiles_to_dates,
    get_feature_cols,
)
from modeling.tracker import ExperimentTracker


def fit_prophet(train_df: pd.DataFrame, target: str):
    from prophet import Prophet
    df = train_df.rename(columns={"Date": "ds", target: "y"})[["ds", "y"]]
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    m.fit(df)
    return m


def prophet_predict(model, dates) -> np.ndarray:
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})
    return model.predict(future)["yhat"].values


def main():
    start = time.time()
    import lightgbm as lgb

    train, test = load_sales()
    tracker = ExperimentTracker("ex_08_prophet_residual")

    print("=" * 70)
    print("EX_08: PROPHET + LGBM RESIDUAL STACK")
    print("=" * 70)

    # ── Fit Prophet on pre-val train only first, for honest val ──
    val_mask_raw = (train["Date"] >= VAL_START) & (train["Date"] <= VAL_END)
    pre_val = train[~val_mask_raw].copy()

    print("\n[1/5] Fitting Prophet on pre-validation train...")
    m_rev = fit_prophet(pre_val, "Revenue")
    m_cogs = fit_prophet(pre_val, "COGS")

    prophet_rev_train = prophet_predict(m_rev, train["Date"])
    prophet_cogs_train = prophet_predict(m_cogs, train["Date"])

    train_aug = train.copy()
    train_aug["prophet_rev"] = prophet_rev_train
    train_aug["prophet_cogs"] = prophet_cogs_train
    train_aug["resid_rev"] = train_aug["Revenue"] - train_aug["prophet_rev"]
    train_aug["resid_cogs"] = train_aug["COGS"] - train_aug["prophet_cogs"]

    # ── v3 features (targets set to residual for LGBM) ──
    print("\n[2/5] Building v3 features...")
    # Note: build_feature_table uses Revenue/COGS directly for lag/rolling.
    # We still pass actual Revenue for lag features (gives LGBM access to
    # recent target history), but label is the residual.
    feat_df, bundle = build_feature_table(train, test_df=test, verbose=True)
    feat_df["prophet_rev"] = prophet_predict(m_rev, feat_df["Date"])
    feat_df["prophet_cogs"] = prophet_predict(m_cogs, feat_df["Date"])
    feat_df["resid_rev"] = feat_df["Revenue"] - feat_df["prophet_rev"]
    feat_df["resid_cogs"] = feat_df["COGS"] - feat_df["prophet_cogs"]

    val_mask = (feat_df["Date"] >= VAL_START) & (feat_df["Date"] <= VAL_END)
    trn = feat_df[~val_mask].copy()
    val = feat_df[val_mask].copy()

    feature_cols = get_feature_cols(feat_df)
    feature_cols = [c for c in feature_cols if c not in
                    ("resid_rev", "resid_cogs")]
    feature_cols = [c for c in feature_cols
                    if trn[c].notna().mean() > 0.5 and trn[c].nunique(dropna=True) > 1]
    print(f"  Features: {len(feature_cols)}")

    X_trn = trn[feature_cols].fillna(0)
    X_val = val[feature_cols].fillna(0)

    # ── LGBM on residuals ──
    print("\n[3/5] Training LGBM on Revenue residuals...")
    params = LGBM_PARAMS.copy()
    tracker.log_params(params)
    tracker.log_params({"n_features": len(feature_cols)})

    model_rev = lgb.LGBMRegressor(**params)
    model_rev.fit(
        X_trn, trn["resid_rev"],
        eval_set=[(X_trn, trn["resid_rev"]), (X_val, val["resid_rev"])],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)],
    )

    model_cogs = lgb.LGBMRegressor(**params)
    model_cogs.fit(
        X_trn, trn["resid_cogs"],
        eval_set=[(X_trn, trn["resid_cogs"]), (X_val, val["resid_cogs"])],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)],
    )

    # ── Validation ──
    print("\n[4/5] Validation...")
    val_resid_pred = model_rev.predict(X_val)
    val_pred_rev = np.clip(val["prophet_rev"].values + val_resid_pred, 0, None)
    res_rev = evaluate(val["Revenue"].values, val_pred_rev, "Revenue (stack)")

    val_resid_cogs_pred = model_cogs.predict(X_val)
    val_pred_cogs = np.clip(val["prophet_cogs"].values + val_resid_cogs_pred, 0, None)
    res_cogs = evaluate(val["COGS"].values, val_pred_cogs, "COGS (stack)")

    print("\n  Horizon-stratified Revenue MAE:")
    horizon_stratified_metrics(val["Date"], val["Revenue"].values, val_pred_rev)

    save_val_predictions(val["Date"], val_pred_rev, val_pred_cogs,
                         "ex_08_prophet_residual")

    # ── Retrain on full data + recursive stack predict for test ──
    print("\n[5/5] Retraining on full data + recursive stack predict...")
    m_rev_full = fit_prophet(train, "Revenue")
    m_cogs_full = fit_prophet(train, "COGS")

    # Refit LGBM residual models on full data using the new Prophet fits
    feat_full, bundle_full = build_feature_table(train, test_df=test, verbose=False)
    feat_full["prophet_rev"] = prophet_predict(m_rev_full, feat_full["Date"])
    feat_full["prophet_cogs"] = prophet_predict(m_cogs_full, feat_full["Date"])
    feat_full["resid_rev"] = feat_full["Revenue"] - feat_full["prophet_rev"]
    feat_full["resid_cogs"] = feat_full["COGS"] - feat_full["prophet_cogs"]
    full_feature_cols = [c for c in feature_cols if c in feat_full.columns]
    X_full = feat_full[full_feature_cols].fillna(0)

    model_rev_full = lgb.LGBMRegressor(**params)
    model_rev_full.fit(X_full, feat_full["resid_rev"])
    model_cogs_full = lgb.LGBMRegressor(**params)
    model_cogs_full.fit(X_full, feat_full["resid_cogs"])

    # Batch Prophet predictions for test
    test_dates = pd.to_datetime(test["Date"])
    prophet_rev_test = prophet_predict(m_rev_full, test_dates)
    prophet_cogs_test = prophet_predict(m_cogs_full, test_dates)

    # Recursive residual predict: LGBM sees lag features built from rolling
    # Revenue trajectory = prophet + predicted residual.
    def recursive_resid(resid_model, prophet_yhat, target: str):
        hist = train[["Date", target]].copy()
        preds = []
        for i, date in enumerate(test_dates):
            row = pd.DataFrame({"Date": [date], target: [np.nan]})
            combined = pd.concat([hist, row], ignore_index=True).sort_values("Date")
            combined = build_calendar_features(combined)
            combined = build_lag_features(combined, target)
            combined = build_rolling_features(combined, target)
            if target == "Revenue":
                combined = build_growth_features(combined, target)
            last = combined.iloc[-1:].copy()
            last = apply_profiles_to_dates(last, bundle_full)
            X = pd.DataFrame(0.0, index=[0], columns=full_feature_cols)
            for c in full_feature_cols:
                if c in last.columns:
                    v = last[c].values[0]
                    X[c] = 0.0 if pd.isna(v) else v
            resid_pred = resid_model.predict(X)[0]
            pred = max(prophet_yhat[i] + resid_pred, 0.0)
            preds.append(pred)
            hist = pd.concat(
                [hist, pd.DataFrame({"Date": [date], target: [pred]})],
                ignore_index=True,
            )
            if (i + 1) % 100 == 0:
                print(f"  {target}: predicted {i + 1}/{len(test_dates)}")
        return np.array(preds)

    test_rev = recursive_resid(model_rev_full, prophet_rev_test, "Revenue")
    test_cogs = recursive_resid(model_cogs_full, prophet_cogs_test, "COGS")

    make_submission(test_dates, test_rev, test_cogs,
                    SUBMISSION_DIR / "ex_08_prophet_residual.csv")

    # ── Save ──
    with open(MODEL_DIR / "ex_08_prophet_rev.pkl", "wb") as f:
        pickle.dump(m_rev_full, f)
    with open(MODEL_DIR / "ex_08_prophet_cogs.pkl", "wb") as f:
        pickle.dump(m_cogs_full, f)
    with open(MODEL_DIR / "ex_08_lgbm_rev.pkl", "wb") as f:
        pickle.dump(model_rev_full, f)
    with open(MODEL_DIR / "ex_08_lgbm_cogs.pkl", "wb") as f:
        pickle.dump(model_cogs_full, f)

    tracker.log_final(res_rev)
    tracker.add_note(
        f"COGS — MAE={res_cogs['mae']:,.0f} RMSE={res_cogs['rmse']:,.0f} "
        f"R²={res_cogs['r2']:.4f}"
    )
    tracker.save()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    return res_rev


if __name__ == "__main__":
    main()
