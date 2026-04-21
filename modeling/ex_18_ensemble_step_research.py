"""
EX_18: Ensemble-step deep research.

Purpose:
- Reintroduce the ensemble stage that produced the 861k anchor pattern.
- Evaluate multiple recursive LGBM component families across two yearly folds.
- Optimize non-negative ensemble weights on fold holdouts.
- Train full-data component ensemble and generate low-drift anchor bridges.

Outputs:
- output/tracking/ex_18_ensemble_step_research/fold_component_scores.csv
- output/tracking/ex_18_ensemble_step_research/fold_weight_search.csv
- output/tracking/ex_18_ensemble_step_research/ensemble_summary.csv
- output/tracking/ex_18_ensemble_step_research/notes.md
- output/tracking/ex_18_ensemble_step_research/meta.json
- output/submissions/ex_18_ensemble_step.csv
- output/submissions/ex_18_bridge_w01.csv .. ex_18_bridge_w04.csv
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

TRACK = Path("output/tracking/ex_18_ensemble_step_research")
SUB_DIR = Path("output/submissions")
ANCHOR_PATH = Path("output/submissions/ex_06_ensemble_weighted.csv")

FOLDS = [
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

COMPONENTS = [
    "core_v3_like",
    "aligned_keep_avg",
    "aligned_drop_avg",
    "naive_lag365",
]


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


def _drop_opposite(cols: list[str], target: str) -> list[str]:
    blocked = ("COGS_",) if target == "Revenue" else ("Revenue_",)
    return [c for c in cols if not c.startswith(blocked)]


def _apply_component_cols(cols: list[str], component: str, target: str) -> list[str]:
    if component == "core_v3_like":
        return cols

    out = _drop_opposite(cols, target)
    if component == "aligned_drop_avg":
        out = [c for c in out if not c.startswith("avg_")]
    return out


def _finalize_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().mean() > 0.50]
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
                val = last_row[c].values[0]
                x_pred[c] = 0.0 if pd.isna(val) else val

        pred = float(np.clip(model.predict(x_pred)[0], 0, None))
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


def naive_lag365_predict(
    history_df: pd.DataFrame, predict_dates, target: str
) -> np.ndarray:
    hist = history_df[["Date", target]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])
    idx = hist.set_index("Date")[target].to_dict()

    preds: list[float] = []
    for date in predict_dates:
        ts = pd.Timestamp(date)
        key = ts - pd.Timedelta(days=365)
        if key in idx:
            pred = float(max(idx[key], 0.0))
        else:
            pred = float(max(hist[target].iloc[-1], 0.0))
        preds.append(pred)
        idx[ts] = pred

    return np.array(preds)


def _safe_weight_optimize(pred_rev: np.ndarray, pred_cogs: np.ndarray, y_rev, y_cogs):
    n = pred_rev.shape[1]

    def obj(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        if s <= 0:
            w = np.ones(n) / n
        else:
            w = w / s

        blend_rev = pred_rev @ w
        blend_cogs = pred_cogs @ w

        mae_rev = float(np.mean(np.abs(y_rev - blend_rev)))
        mae_cogs = float(np.mean(np.abs(y_cogs - blend_cogs)))
        return mae_rev + 0.4 * mae_cogs

    try:
        from scipy.optimize import minimize

        x0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(
            obj, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints
        )
        w = np.clip(res.x, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones(n) / n
        return w, float(obj(w)), "scipy_slsqp"
    except Exception:
        rng = np.random.default_rng(42)
        best_w = np.ones(n) / n
        best = float(obj(best_w))
        for _ in range(4000):
            w = rng.dirichlet(np.ones(n))
            val = float(obj(w))
            if val < best:
                best = val
                best_w = w
        return best_w, best, "dirichlet_random"


def evaluate_fold_components(sales: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, profiles = build_feature_table(
        sales,
        verbose=False,
        profile_source_df=train_slice,
    )

    trn = feat_df[feat_df["Date"] < val_start].copy()
    val = feat_df[(feat_df["Date"] >= val_start) & (feat_df["Date"] <= val_end)].copy()

    y_val_rev = val["Revenue"].values
    y_val_cogs = val["COGS"].values

    base_cols = get_feature_cols(feat_df)

    comp_pred_rev = []
    comp_pred_cogs = []
    score_rows = []

    for comp in COMPONENTS:
        if comp == "naive_lag365":
            pred_rev = naive_lag365_predict(
                trn[["Date", "Revenue"]], val["Date"].values, "Revenue"
            )
            pred_cogs = naive_lag365_predict(
                trn[["Date", "COGS"]], val["Date"].values, "COGS"
            )
            n_feat_rev = 0
            n_feat_cogs = 0
        else:
            cols_rev = _finalize_cols(
                trn, _apply_component_cols(base_cols, comp, "Revenue")
            )
            cols_cogs = _finalize_cols(
                trn, _apply_component_cols(base_cols, comp, "COGS")
            )

            model_rev = _fit_lgbm(
                trn[cols_rev].fillna(0),
                trn["Revenue"],
                val[cols_rev].fillna(0),
                val["Revenue"],
            )
            model_cogs = _fit_lgbm(
                trn[cols_cogs].fillna(0),
                trn["COGS"],
                val[cols_cogs].fillna(0),
                val["COGS"],
            )

            pred_rev = recursive_predict(
                model_rev,
                trn[["Date", "Revenue"]],
                val["Date"].values,
                cols_rev,
                profiles,
                target="Revenue",
            )
            pred_cogs = recursive_predict(
                model_cogs,
                trn[["Date", "COGS"]],
                val["Date"].values,
                cols_cogs,
                profiles,
                target="COGS",
            )
            n_feat_rev = len(cols_rev)
            n_feat_cogs = len(cols_cogs)

        res_rev = evaluate(y_val_rev, pred_rev, f"{fold['name']} {comp} Revenue")
        res_cogs = evaluate(y_val_cogs, pred_cogs, f"{fold['name']} {comp} COGS")
        score = float(res_rev["mae"] + 0.4 * res_cogs["mae"])

        score_rows.append(
            {
                "fold": fold["name"],
                "component": comp,
                "n_features_revenue": n_feat_rev,
                "n_features_cogs": n_feat_cogs,
                "rec_revenue_mae": float(res_rev["mae"]),
                "rec_cogs_mae": float(res_cogs["mae"]),
                "score": score,
            }
        )

        comp_pred_rev.append(pred_rev)
        comp_pred_cogs.append(pred_cogs)

    pred_rev_mat = np.column_stack(comp_pred_rev)
    pred_cogs_mat = np.column_stack(comp_pred_cogs)

    w, best_score, solver = _safe_weight_optimize(
        pred_rev_mat, pred_cogs_mat, y_val_rev, y_val_cogs
    )

    blend_rev = pred_rev_mat @ w
    blend_cogs = pred_cogs_mat @ w
    blend_rev_res = evaluate(y_val_rev, blend_rev, f"{fold['name']} ensemble Revenue")
    blend_cogs_res = evaluate(y_val_cogs, blend_cogs, f"{fold['name']} ensemble COGS")

    weight_row = {
        "fold": fold["name"],
        "solver": solver,
        "opt_score": best_score,
        "ensemble_revenue_mae": float(blend_rev_res["mae"]),
        "ensemble_cogs_mae": float(blend_cogs_res["mae"]),
        "ensemble_score": float(blend_rev_res["mae"] + 0.4 * blend_cogs_res["mae"]),
    }
    for i, comp in enumerate(COMPONENTS):
        weight_row[f"w_{comp}"] = float(w[i])

    return pd.DataFrame(score_rows), pd.DataFrame([weight_row])


def train_full_component_predictions(
    train: pd.DataFrame, test: pd.DataFrame, component: str
):
    if component == "naive_lag365":
        pred_rev = naive_lag365_predict(
            train[["Date", "Revenue"]], test["Date"].values, "Revenue"
        )
        pred_cogs = naive_lag365_predict(
            train[["Date", "COGS"]], test["Date"].values, "COGS"
        )
        return pred_rev, pred_cogs

    feat_df, profiles = build_feature_table(
        train, verbose=False, profile_source_df=train
    )
    base_cols = get_feature_cols(feat_df)

    cols_rev = _finalize_cols(
        feat_df, _apply_component_cols(base_cols, component, "Revenue")
    )
    cols_cogs = _finalize_cols(
        feat_df, _apply_component_cols(base_cols, component, "COGS")
    )

    model_rev = _fit_lgbm(
        feat_df[cols_rev].fillna(0),
        feat_df["Revenue"],
        feat_df[cols_rev].fillna(0).tail(365),
        feat_df["Revenue"].tail(365),
    )
    model_cogs = _fit_lgbm(
        feat_df[cols_cogs].fillna(0),
        feat_df["COGS"],
        feat_df[cols_cogs].fillna(0).tail(365),
        feat_df["COGS"].tail(365),
    )

    pred_rev = recursive_predict(
        model_rev,
        train[["Date", "Revenue"]],
        test["Date"].values,
        cols_rev,
        profiles,
        target="Revenue",
    )
    pred_cogs = recursive_predict(
        model_cogs,
        train[["Date", "COGS"]],
        test["Date"].values,
        cols_cogs,
        profiles,
        target="COGS",
    )

    return pred_rev, pred_cogs


def _make_bridges(candidate_path: Path):
    anchor = pd.read_csv(ANCHOR_PATH).sort_values("Date").reset_index(drop=True)
    cand = pd.read_csv(candidate_path).sort_values("Date").reset_index(drop=True)

    rows = []
    for w in [0.01, 0.02, 0.03, 0.04]:
        wa = 1.0 - w
        out = anchor[["Date"]].copy()
        out["Revenue"] = anchor["Revenue"] * wa + cand["Revenue"] * w
        out["COGS"] = anchor["COGS"] * wa + cand["COGS"] * w

        tag = int(round(w * 100))
        out_path = SUB_DIR / f"ex_18_bridge_w{tag:02d}.csv"
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

    pd.DataFrame(rows).sort_values("w_candidate").to_csv(
        TRACK / "bridge_summary.csv", index=False
    )


def main() -> None:
    t0 = time.time()
    TRACK.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_sales()

    print("=" * 78)
    print("EX_18: Ensemble-step deep research")
    print("=" * 78)

    fold_score_parts = []
    fold_weight_parts = []

    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df, w_df = evaluate_fold_components(train, fold)
        fold_score_parts.append(s_df)
        fold_weight_parts.append(w_df)

    fold_scores = pd.concat(fold_score_parts, ignore_index=True)
    fold_weights = pd.concat(fold_weight_parts, ignore_index=True)

    fold_scores.to_csv(TRACK / "fold_component_scores.csv", index=False)
    fold_weights.to_csv(TRACK / "fold_weight_search.csv", index=False)

    comp_summary = (
        fold_scores.groupby("component", as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_rec_revenue_mae=("rec_revenue_mae", "mean"),
            mean_rec_cogs_mae=("rec_cogs_mae", "mean"),
            mean_score=("score", "mean"),
            mean_features_revenue=("n_features_revenue", "mean"),
            mean_features_cogs=("n_features_cogs", "mean"),
        )
        .sort_values("mean_score")
        .reset_index(drop=True)
    )

    mean_w = np.array([fold_weights[f"w_{c}"].mean() for c in COMPONENTS], dtype=float)
    mean_w = np.clip(mean_w, 0, None)
    mean_w = (
        mean_w / mean_w.sum()
        if mean_w.sum() > 0
        else np.ones(len(COMPONENTS)) / len(COMPONENTS)
    )

    ensemble_summary = pd.DataFrame(
        [
            {
                "mean_ensemble_revenue_mae": float(
                    fold_weights["ensemble_revenue_mae"].mean()
                ),
                "mean_ensemble_cogs_mae": float(
                    fold_weights["ensemble_cogs_mae"].mean()
                ),
                "mean_ensemble_score": float(fold_weights["ensemble_score"].mean()),
                **{f"w_{c}": float(mean_w[i]) for i, c in enumerate(COMPONENTS)},
            }
        ]
    )
    ensemble_summary.to_csv(TRACK / "ensemble_summary.csv", index=False)

    print("\nFinal component weights:")
    for i, comp in enumerate(COMPONENTS):
        print(f"  {comp}: {mean_w[i]:.4f}")

    pred_rev_parts = []
    pred_cogs_parts = []
    for comp in COMPONENTS:
        print(f"\nTraining full-data component: {comp}")
        p_rev, p_cogs = train_full_component_predictions(train, test, comp)
        pred_rev_parts.append(p_rev)
        pred_cogs_parts.append(p_cogs)

    pred_rev_mat = np.column_stack(pred_rev_parts)
    pred_cogs_mat = np.column_stack(pred_cogs_parts)

    final_rev = pred_rev_mat @ mean_w
    final_cogs = pred_cogs_mat @ mean_w

    candidate_path = SUB_DIR / "ex_18_ensemble_step.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    _make_bridges(candidate_path)

    notes = [
        "# EX_18 Ensemble-Step Deep Research",
        "",
        "## Goal",
        "- Reintroduce ensemble stage before anchor bridging.",
        "- Use fold-level recursive optimization, not single-candidate drift.",
        "",
        "## Components",
    ]
    notes.extend([f"- {c}" for c in COMPONENTS])
    notes.extend(
        [
            "",
            "## Validation Setup",
            "- Folds: 2021 and 2022 yearly recursive holdouts.",
            "- Objective: Revenue MAE + 0.4 * COGS MAE.",
            "",
            "## Final Weights",
        ]
    )
    for i, comp in enumerate(COMPONENTS):
        notes.append(f"- {comp}: {mean_w[i]:.4f}")

    notes.extend(
        [
            "",
            "## Outputs",
            f"- Candidate: {candidate_path}",
            "- fold_component_scores.csv",
            "- fold_weight_search.csv",
            "- ensemble_summary.csv",
            "- bridge_summary.csv",
        ]
    )
    (TRACK / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "components": COMPONENTS,
        "folds": FOLDS,
        "candidate_path": str(candidate_path),
        "weights": {c: float(mean_w[i]) for i, c in enumerate(COMPONENTS)},
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
