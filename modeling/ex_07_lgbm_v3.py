"""
EX_07: LightGBM on v3 features.

Upgrades over EX_03:
  • Spec-compliant val split (2022-10-01 .. 2022-12-31)
  • v3 feature engineering: future-known promo calendar, VN holidays + Tet,
    detrended & recent-window profiles, order/category-mix, covid flag
  • Log-target training (log1p on Revenue)
  • Tweedie objective variant for comparison
  • Horizon-stratified validation MAE (1-7, 8-30, 31-90, 91-365 days)
  • COGS via ratio model: predict COGS/Revenue, multiply back. Falls back to a
    direct LGBM if the ratio is unstable.
  • Saves val predictions for ensemble weighting.
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
    LGBM_PARAMS, LGBM_PARAMS_TWEEDIE,
    MODEL_DIR, SUBMISSION_DIR, VAL_START, VAL_END, SEED, TRAIN_END,
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
from modeling.tracker import ExperimentTracker, LGBMCallback


# ─────────────────────────────────────────────────────────────────────────────
# COGS ratio diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_cogs_ratio(train: pd.DataFrame) -> dict:
    ratio = train["COGS"] / train["Revenue"].replace(0, np.nan)
    ratio = ratio.dropna()
    stats = {
        "mean": float(ratio.mean()),
        "std": float(ratio.std()),
        "cv": float(ratio.std() / ratio.mean()),
        "rolling30_std": float(ratio.rolling(30, min_periods=5).std().mean()),
        "min": float(ratio.min()),
        "max": float(ratio.max()),
    }
    print("COGS/Revenue ratio diagnostics:")
    for k, v in stats.items():
        print(f"  {k:18s}: {v:.4f}")
    stats["is_stable"] = stats["cv"] < 0.05
    print(f"  → is_stable (cv<0.05): {stats['is_stable']}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Recursive prediction using v3 bundle
# ─────────────────────────────────────────────────────────────────────────────

def recursive_predict(
    model,
    train_df: pd.DataFrame,
    test_dates,
    feature_cols,
    bundle,
    target: str = "Revenue",
    log_target: bool = False,
):
    history = train_df[["Date", target]].copy()
    predictions = []

    for i, date in enumerate(test_dates):
        row = pd.DataFrame({"Date": [pd.Timestamp(date)], target: [np.nan]})
        combined = pd.concat([history, row], ignore_index=True).sort_values("Date")

        combined = build_calendar_features(combined)
        combined = build_lag_features(combined, target)
        combined = build_rolling_features(combined, target)
        if target == "Revenue":
            combined = build_growth_features(combined, target)

        last_row = combined.iloc[-1:].copy()
        last_row = apply_profiles_to_dates(last_row, bundle)

        X_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        for col in feature_cols:
            if col in last_row.columns:
                val = last_row[col].values[0]
                X_pred[col] = 0.0 if pd.isna(val) else val

        pred = model.predict(X_pred)[0]
        if log_target:
            pred = np.expm1(max(pred, 0))
        pred = max(pred, 0.0)
        predictions.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [pd.Timestamp(date)], target: [pred]})],
            ignore_index=True,
        )

        if (i + 1) % 100 == 0:
            print(f"  predicted {i + 1}/{len(test_dates)} days...")

    return np.array(predictions)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    import lightgbm as lgb

    train, test = load_sales()
    tracker = ExperimentTracker("ex_07_lgbm_v3")

    print("=" * 70)
    print("EX_07: LightGBM + v3 FEATURES")
    print("=" * 70)

    # ── Feature table (include test dates so promo/inv calendars cover them) ──
    print("\n[1/6] Building v3 feature table...")
    feat_df, bundle = build_feature_table(train, test_df=test, verbose=True)

    # ── Train / Val split ──
    print("\n[2/6] Splitting train/val...")
    val_mask = (feat_df["Date"] >= VAL_START) & (feat_df["Date"] <= VAL_END)
    trn = feat_df[~val_mask].copy()
    val = feat_df[val_mask].copy()
    feature_cols = get_feature_cols(feat_df)
    feature_cols = [c for c in feature_cols
                    if trn[c].notna().mean() > 0.5 and trn[c].nunique(dropna=True) > 1]
    print(f"  Features kept: {len(feature_cols)}")
    print(f"  Train: {len(trn)}  Val: {len(val)}")

    X_trn, y_trn = trn[feature_cols].fillna(0), trn["Revenue"]
    X_val, y_val = val[feature_cols].fillna(0), val["Revenue"]

    # ── Revenue: log-target training ──
    print("\n[3/6] Training Revenue model (log target, L1)...")
    params_log = LGBM_PARAMS.copy()
    tracker.log_params(params_log)
    tracker.log_params({"n_features": len(feature_cols), "log_target": True})

    cb = LGBMCallback(tracker, log_every=50)
    model_rev = lgb.LGBMRegressor(**params_log)
    model_rev.fit(
        X_trn, np.log1p(y_trn),
        eval_set=[(X_trn, np.log1p(y_trn)), (X_val, np.log1p(y_val))],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200), cb],
    )

    val_pred_log = model_rev.predict(X_val)
    val_pred_rev = np.clip(np.expm1(val_pred_log), 0, None)
    print("\n  Revenue validation:")
    res_rev = evaluate(y_val.values, val_pred_rev, "Revenue")

    print("  Horizon-stratified MAE:")
    horizon_rev = horizon_stratified_metrics(
        val["Date"], y_val.values, val_pred_rev,
    )
    tracker.log_params({f"h_{k}": v for k, v in horizon_rev.items()})

    # ── Tweedie variant (no log) for comparison ──
    print("\n[4/6] Training Tweedie variant for comparison...")
    tw_params = LGBM_PARAMS_TWEEDIE.copy()
    model_tw = lgb.LGBMRegressor(**tw_params)
    model_tw.fit(
        X_trn, y_trn,
        eval_set=[(X_trn, y_trn), (X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(200)],
    )
    val_pred_tw = np.clip(model_tw.predict(X_val), 0, None)
    res_tw = evaluate(y_val.values, val_pred_tw, "Revenue-Tweedie")

    # Pick the better of (log, tweedie) for Revenue
    if res_tw["mae"] < res_rev["mae"]:
        print("  → Tweedie wins. Using Tweedie for final Revenue predictions.")
        best_rev_model = model_tw
        best_log = False
        res_rev = res_tw
        val_pred_rev = val_pred_tw
    else:
        print("  → Log target wins.")
        best_rev_model = model_rev
        best_log = True

    # ── COGS: ratio model if stable, else direct LGBM ──
    print("\n[5/6] COGS strategy...")
    cogs_stats = diagnose_cogs_ratio(train)

    if cogs_stats["is_stable"]:
        print("  Ratio stable → predicting COGS = Revenue × mean_ratio")
        mean_ratio = cogs_stats["mean"]
        val_pred_cogs = val_pred_rev * mean_ratio
        res_cogs = evaluate(val["COGS"].values, val_pred_cogs, "COGS")
        model_cogs = None
    else:
        print("  Ratio drifts → training LGBM on ratio (bounded target)")
        trn_ratio = (trn["COGS"] / trn["Revenue"].replace(0, np.nan)).fillna(
            cogs_stats["mean"]
        )
        val_ratio_true = (val["COGS"] / val["Revenue"].replace(0, np.nan)).fillna(
            cogs_stats["mean"]
        )
        params_ratio = LGBM_PARAMS.copy()
        model_cogs = lgb.LGBMRegressor(**params_ratio)
        model_cogs.fit(
            X_trn, trn_ratio,
            eval_set=[(X_trn, trn_ratio), (X_val, val_ratio_true)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(200)],
        )
        val_ratio_pred = model_cogs.predict(X_val)
        val_pred_cogs = np.clip(val_pred_rev * val_ratio_pred, 0, None)
        res_cogs = evaluate(val["COGS"].values, val_pred_cogs, "COGS")

    # Persist val preds
    save_val_predictions(val["Date"], val_pred_rev, val_pred_cogs, "ex_07_lgbm_v3")

    # ── Feature importance ──
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_rev_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(MODEL_DIR / "ex_07_feature_importance.csv", index=False)
    print("\nTop 15 features (gain):")
    print(importance.head(15).to_string(index=False))

    # ── Importance plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = importance.head(30)
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.barh(top["feature"][::-1], top["importance"][::-1])
        ax.set_title("EX_07 LightGBM – Top 30 feature importances (gain)")
        fig.tight_layout()
        fig.savefig(MODEL_DIR / "ex_07_feature_importance.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  (importance plot skipped: {e})")

    # ── Save models ──
    with open(MODEL_DIR / "ex_07_lgbm_rev.pkl", "wb") as f:
        pickle.dump(best_rev_model, f)
    if model_cogs is not None:
        with open(MODEL_DIR / "ex_07_lgbm_cogs.pkl", "wb") as f:
            pickle.dump(model_cogs, f)
    with open(MODEL_DIR / "ex_07_features.pkl", "wb") as f:
        pickle.dump({"feature_cols": feature_cols, "bundle": bundle,
                     "best_log": best_log, "cogs_stats": cogs_stats}, f)

    # ── Retrain on full data + recursive predict for submission ──
    print("\n[6/6] Retraining on full data + recursive test predict...")
    full_df, full_bundle = build_feature_table(train, test_df=test, verbose=False)
    full_feature_cols = [c for c in feature_cols if c in full_df.columns]
    X_full = full_df[full_feature_cols].fillna(0)
    y_full = full_df["Revenue"]

    if best_log:
        rev_full = lgb.LGBMRegressor(**LGBM_PARAMS)
        rev_full.fit(X_full, np.log1p(y_full))
    else:
        rev_full = lgb.LGBMRegressor(**LGBM_PARAMS_TWEEDIE)
        rev_full.fit(X_full, y_full)

    print("Recursive Revenue prediction...")
    rev_preds = recursive_predict(
        rev_full, train, test["Date"].values, full_feature_cols, full_bundle,
        target="Revenue", log_target=best_log,
    )

    if cogs_stats["is_stable"]:
        cogs_preds = rev_preds * cogs_stats["mean"]
    else:
        cogs_full = lgb.LGBMRegressor(**LGBM_PARAMS)
        trn_ratio_full = (full_df["COGS"] / full_df["Revenue"].replace(0, np.nan)).fillna(
            cogs_stats["mean"]
        )
        cogs_full.fit(X_full, trn_ratio_full)
        # For test: recursive prediction of Revenue already done → compute ratio per test row
        # Build test feature snapshots matching rev_preds trajectory
        history = pd.concat(
            [train.assign(Date=pd.to_datetime(train["Date"])),
             pd.DataFrame({"Date": pd.to_datetime(test["Date"].values),
                           "Revenue": rev_preds,
                           "COGS": rev_preds * cogs_stats["mean"]})],
            ignore_index=True,
        ).sort_values("Date").reset_index(drop=True)
        history = build_calendar_features(history)
        history = build_lag_features(history, "Revenue")
        history = build_rolling_features(history, "Revenue")
        history = build_growth_features(history, "Revenue")
        test_rows = history[history["Date"].isin(pd.to_datetime(test["Date"].values))]
        test_rows = apply_profiles_to_dates(test_rows, full_bundle)
        Xt = pd.DataFrame(0.0, index=range(len(test_rows)), columns=full_feature_cols)
        for col in full_feature_cols:
            if col in test_rows.columns:
                Xt[col] = test_rows[col].fillna(0).values
        ratio_preds = cogs_full.predict(Xt)
        cogs_preds = np.clip(rev_preds * ratio_preds, 0, None)

    make_submission(test["Date"], rev_preds, cogs_preds,
                    SUBMISSION_DIR / "ex_07_lgbm_v3.csv")

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
