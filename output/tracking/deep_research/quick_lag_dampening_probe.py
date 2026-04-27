from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb

from modeling.config import LGBM_PARAMS, SEED
from modeling.feature_engineering import (
    apply_profiles_to_dates,
    build_calendar_features,
    build_feature_table,
    build_growth_features,
    build_lag_features,
    build_rolling_features,
    get_feature_cols,
)
from modeling.utils import load_sales

OUT = Path("output/tracking/deep_research/lag_dampening_probe.json")


def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
    cols = [c for c in cols if df[c].nunique(dropna=True) > 1]
    return cols


def _get_target_cols(base_cols: list[str], target: str) -> list[str]:
    blocked = ("COGS_", "cogs_") if target == "Revenue" else ("Revenue_", "rev_")
    return [c for c in base_cols if not c.startswith(blocked)]


def fit_model(trn: pd.DataFrame, val: pd.DataFrame, cols: list[str], target: str):
    params = LGBM_PARAMS.copy()
    params["n_estimators"] = 1200
    params["random_state"] = SEED
    model = lgb.LGBMRegressor(**params)
    model.fit(
        trn[cols].fillna(0),
        trn[target],
        eval_set=[(val[cols].fillna(0), val[target])],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def recursive_predict(
    model,
    history_df: pd.DataFrame,
    predict_dates,
    feature_cols: list[str],
    profiles,
    target: str,
    month_profile: dict[int, float],
    alpha: float,
):
    """
    alpha=1.0 -> standard recursion (no dampening)
    alpha<1.0 -> lag state is dampened toward month profile
    """
    history = history_df[["Date", target]].copy()
    preds = []

    for d in predict_dates:
        ts = pd.Timestamp(d)
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

        pred = float(model.predict(x_pred)[0])
        pred = max(0.0, pred)
        preds.append(pred)

        m = ts.month
        ref = month_profile.get(m, float(history[target].tail(90).mean()))
        stored = alpha * pred + (1.0 - alpha) * ref

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [stored]})],
            ignore_index=True,
        )

    return np.array(preds)


def main() -> None:
    train, _ = load_sales()

    val_start = pd.Timestamp("2022-01-01")
    val_end = pd.Timestamp("2022-12-31")

    train_slice = train[train["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(
        train, verbose=False, profile_source_df=train_slice
    )

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    base_cols = get_feature_cols(feat_df)

    results = []

    for target in ["Revenue", "COGS"]:
        cols = _finalize_cols(trn, _get_target_cols(base_cols, target))
        model = fit_model(trn, val, cols, target)

        month_profile = (
            train_slice.groupby(train_slice["Date"].dt.month)[target].mean().to_dict()
        )

        y_true = val[target].values

        for alpha in [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]:
            pred = recursive_predict(
                model=model,
                history_df=train_slice,
                predict_dates=val["Date"].values,
                feature_cols=cols,
                profiles=profiles,
                target=target,
                month_profile=month_profile,
                alpha=alpha,
            )
            mae = float(np.mean(np.abs(y_true - pred)))
            bias = float(pred.mean() / y_true.mean() - 1.0)
            results.append(
                {
                    "target": target,
                    "alpha": alpha,
                    "mae": mae,
                    "bias_pct": 100.0 * bias,
                    "pred_mean": float(pred.mean()),
                    "actual_mean": float(y_true.mean()),
                }
            )
            print(
                f"{target:7s} alpha={alpha:.2f} MAE={mae:,.0f} "
                f"bias={bias*100:+.2f}% mean={pred.mean():,.0f}"
            )

    # Weighted score summary
    rev = [r for r in results if r["target"] == "Revenue"]
    cogs = [r for r in results if r["target"] == "COGS"]
    combo = []
    for rr in rev:
        cc = next(x for x in cogs if x["alpha"] == rr["alpha"])
        score = rr["mae"] + 0.4 * cc["mae"]
        combo.append(
            {
                "alpha": rr["alpha"],
                "score": score,
                "rev_mae": rr["mae"],
                "cogs_mae": cc["mae"],
                "rev_bias_pct": rr["bias_pct"],
                "cogs_bias_pct": cc["bias_pct"],
            }
        )

    combo = sorted(combo, key=lambda x: x["score"])
    print("\nBest combined scores:")
    for row in combo:
        print(
            f"alpha={row['alpha']:.2f} score={row['score']:,.0f} "
            f"(rev={row['rev_mae']:,.0f}, cogs={row['cogs_mae']:,.0f}) "
            f"bias rev={row['rev_bias_pct']:+.2f}% cogs={row['cogs_bias_pct']:+.2f}%"
        )

    payload = {
        "rows": results,
        "combined": combo,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
