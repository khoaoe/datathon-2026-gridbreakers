"""
EX_22: Deep FE recency-profile dual ensemble.

Purpose:
- Extend EX_20 winner with one controlled feature-family delta:
    fold-safe recency-weighted seasonal profiles.
- Keep recursive inference leakage-safe and reproducible.
- Optimize separate global weights for Revenue and COGS on 2020-2022 folds
    with explicit recency weighting.

Outputs:
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/fold_component_scores.csv
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/fold_weight_search.csv
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/fold_global_metrics.csv
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/global_weight_search.csv
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/ensemble_summary.csv
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/bridge_summary.csv
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/notes.md
- output/tracking/ex_22_deep_fe_holidays_dual_ensemble/meta.json
- output/submissions/ex_22_deep_fe_holidays_dual_ensemble.csv
- output/submissions/ex_22_bridge_w01.csv .. ex_22_bridge_w04.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.config import LGBM_PARAMS
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

TRACK = Path("output/tracking/ex_22_deep_fe_holidays_dual_ensemble")
SUB_DIR = Path("output/submissions")
ANCHOR_PATH = Path("output/submissions/ex_21_deep_fe_recency_dual_ensemble.csv")

FOLDS = [
    {"name": "fold_2020", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"name": "fold_2021", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"name": "fold_2022", "val_start": "2022-01-01", "val_end": "2022-12-31"},
]

FOLD_RECENCY_WEIGHTS = {
    "fold_2020": 0.20,
    "fold_2021": 0.30,
    "fold_2022": 0.50,
}

COMPONENTS = [
    "core_v3_like",
    "aligned_keep_avg",
    "aligned_no_profiles",
    "aligned_recency_profiles",
    "naive_lag365",
]

MODEL_COMPONENTS = {
    "core_v3_like",
    "aligned_keep_avg",
    "aligned_no_profiles",
    "aligned_recency_profiles",
}

ROBUST_STD_PENALTY = 0.08
WEIGHT_SHRINK_L2 = 0.01
RAND_SEARCH_TRIALS = 6000
RECENCY_DECAY = 0.003


def make_recency_profiles(
    train_slice: pd.DataFrame, decay: float = RECENCY_DECAY
) -> dict[str, pd.DataFrame]:
    """Build fold-safe recency-weighted seasonal profiles from train slice."""
    df = train_slice[["Date", "Revenue", "COGS"]].copy()
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)

    ref = df["Date"].max()
    age_days = (ref - df["Date"]).dt.days
    df["w"] = np.exp(-decay * age_days)
    df["w_rev"] = df["Revenue"] * df["w"]
    df["w_cogs"] = df["COGS"] * df["w"]

    def _weighted(group_cols: list[str], suffix: str) -> pd.DataFrame:
        agg = df.groupby(group_cols, as_index=False).agg(
            w_sum=("w", "sum"),
            w_rev_sum=("w_rev", "sum"),
            w_cogs_sum=("w_cogs", "sum"),
        )
        out = agg[group_cols].copy()
        out[f"rec_rev_{suffix}_mean"] = agg["w_rev_sum"] / agg["w_sum"].replace(
            0, np.nan
        )
        out[f"rec_cogs_{suffix}_mean"] = agg["w_cogs_sum"] / agg["w_sum"].replace(
            0, np.nan
        )
        return out

    return {
        "rec_dow": _weighted(["dayofweek"], "dow"),
        "rec_month": _weighted(["month"], "month"),
        "rec_woy": _weighted(["weekofyear"], "woy"),
        "rec_month_dow": _weighted(["month", "dayofweek"], "month_dow"),
    }


def apply_recency_profiles(
    frame: pd.DataFrame,
    recency_profiles: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    out = frame.copy()
    out = out.merge(recency_profiles["rec_dow"], on=["dayofweek"], how="left")
    out = out.merge(recency_profiles["rec_month"], on=["month"], how="left")
    out = out.merge(recency_profiles["rec_woy"], on=["weekofyear"], how="left")
    out = out.merge(
        recency_profiles["rec_month_dow"],
        on=["month", "dayofweek"],
        how="left",
    )
    return out


def build_feature_frame_with_recency(
    sales: pd.DataFrame,
    profile_source_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    feat_df, profiles = build_feature_table(
        sales,
        verbose=False,
        profile_source_df=profile_source_df,
    )
    rec_profiles = make_recency_profiles(profile_source_df)
    feat_df = apply_recency_profiles(feat_df, rec_profiles)
    all_profiles = {**profiles, **rec_profiles}
    return feat_df, all_profiles


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


def _drop_profile_cols(cols: list[str]) -> list[str]:
    profile_like_prefixes = (
        "rev_",
        "cogs_",
        "avg_",
        "rec_",
    )
    return [c for c in cols if not c.startswith(profile_like_prefixes)]


def _drop_standard_profile_cols(cols: list[str]) -> list[str]:
    standard_prefixes = (
        "rev_",
        "cogs_",
        "avg_",
    )
    return [c for c in cols if not c.startswith(standard_prefixes)]


def _apply_component_cols(cols: list[str], component: str, target: str) -> list[str]:
    if component == "core_v3_like":
        return cols

    out = _drop_opposite(cols, target)

    if component == "aligned_drop_avg":
        out = [c for c in out if not c.startswith("avg_")]
    elif component == "aligned_no_profiles":
        out = _drop_profile_cols(out)
    elif component == "aligned_recency_profiles":
        out = _drop_standard_profile_cols(out)

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


def naive_predict(history_df: pd.DataFrame, predict_dates, target: str, mode: str):
    hist = history_df[["Date", target]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"])

    idx = hist.set_index("Date")[target].to_dict()
    preds: list[float] = []

    for date in predict_dates:
        ts = pd.Timestamp(date)
        fallback = float(max(hist[target].iloc[-1], 0.0))

        if mode == "naive_lag365":
            pred = float(max(idx.get(ts - pd.Timedelta(days=365), fallback), 0.0))
        elif mode == "naive_lag365_lag7_blend":
            p365 = float(max(idx.get(ts - pd.Timedelta(days=365), fallback), 0.0))
            p7 = float(max(idx.get(ts - pd.Timedelta(days=7), fallback), 0.0))
            pred = 0.70 * p365 + 0.30 * p7
        else:
            raise ValueError(f"Unknown naive mode: {mode}")

        preds.append(pred)
        idx[ts] = pred
        hist = pd.concat(
            [hist, pd.DataFrame({"Date": [ts], target: [pred]})],
            ignore_index=True,
        )

    return np.array(preds)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    s = float(w.sum())
    if s <= 0:
        return np.ones(len(w), dtype=float) / len(w)
    return w / s


def _dirichlet_search(obj_fn, n: int, trials: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    best_w = np.ones(n) / n
    best = float(obj_fn(best_w))

    for _ in range(trials):
        w = rng.dirichlet(np.ones(n))
        val = float(obj_fn(w))
        if val < best:
            best = val
            best_w = w

    return best_w, best


def _safe_weight_optimize_single(pred_mat: np.ndarray, y_true: np.ndarray):
    n = pred_mat.shape[1]

    def obj(w):
        ww = _normalize_weights(w)
        pred = pred_mat @ ww
        return float(np.mean(np.abs(y_true - pred)))

    best_w, best = _dirichlet_search(obj, n=n, trials=RAND_SEARCH_TRIALS)
    solver = "dirichlet_random"

    try:
        from scipy.optimize import minimize

        x0 = best_w.copy()
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(
            obj,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if bool(getattr(res, "success", False)):
            cand_w = _normalize_weights(res.x)
            cand = float(obj(cand_w))
            if np.isfinite(cand) and cand < best:
                best_w = cand_w
                best = cand
                solver = "dirichlet_random+scipy_slsqp"
    except Exception:
        pass

    return best_w, float(best), solver


def _safe_weight_optimize_global(
    pred_mat: np.ndarray,
    y_true: np.ndarray,
    fold_ids: np.ndarray,
    std_penalty: float,
    l2_penalty: float,
    fold_weight_map: dict[str, float] | None = None,
):
    n = pred_mat.shape[1]
    uniform = np.ones(n) / n
    uniq_folds = np.unique(fold_ids)

    if fold_weight_map is None:
        fold_weights = np.ones(len(uniq_folds), dtype=float)
    else:
        fold_weights = np.array(
            [float(fold_weight_map.get(str(f), 1.0)) for f in uniq_folds],
            dtype=float,
        )
    fold_weights = (
        fold_weights / fold_weights.sum()
        if float(fold_weights.sum()) > 0
        else np.ones(len(uniq_folds), dtype=float) / len(uniq_folds)
    )

    def obj(w):
        ww = _normalize_weights(w)
        pred = pred_mat @ ww
        abs_err = np.abs(y_true - pred)

        fold_maes = np.array(
            [float(abs_err[fold_ids == f].mean()) for f in uniq_folds], dtype=float
        )
        mean_mae = float(np.average(fold_maes, weights=fold_weights))
        std_mae = float(
            np.sqrt(np.average((fold_maes - mean_mae) ** 2, weights=fold_weights))
        )
        shrink = float(np.sum((ww - uniform) ** 2))

        return mean_mae + std_penalty * std_mae + l2_penalty * shrink

    best_w, best = _dirichlet_search(obj, n=n, trials=RAND_SEARCH_TRIALS)
    solver = "dirichlet_random"

    try:
        from scipy.optimize import minimize

        x0 = best_w.copy()
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        res = minimize(
            obj,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if bool(getattr(res, "success", False)):
            cand_w = _normalize_weights(res.x)
            cand = float(obj(cand_w))
            if np.isfinite(cand) and cand < best:
                best_w = cand_w
                best = cand
                solver = "dirichlet_random+scipy_slsqp"
    except Exception:
        pass

    return best_w, float(best), solver


def evaluate_fold_components(sales: pd.DataFrame, fold: dict):
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_slice = sales[sales["Date"] < val_start].copy()
    feat_df, profiles = build_feature_frame_with_recency(
        sales,
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
        if comp in MODEL_COMPONENTS:
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
        else:
            pred_rev = naive_predict(
                trn[["Date", "Revenue"]],
                val["Date"].values,
                "Revenue",
                mode=comp,
            )
            pred_cogs = naive_predict(
                trn[["Date", "COGS"]],
                val["Date"].values,
                "COGS",
                mode=comp,
            )
            n_feat_rev = 0
            n_feat_cogs = 0

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

    w_rev, rev_mae, solver_rev = _safe_weight_optimize_single(pred_rev_mat, y_val_rev)
    w_cogs, cogs_mae, solver_cogs = _safe_weight_optimize_single(
        pred_cogs_mat, y_val_cogs
    )

    blend_rev = pred_rev_mat @ w_rev
    blend_cogs = pred_cogs_mat @ w_cogs
    blend_rev_res = evaluate(y_val_rev, blend_rev, f"{fold['name']} ensemble Revenue")
    blend_cogs_res = evaluate(y_val_cogs, blend_cogs, f"{fold['name']} ensemble COGS")

    weight_row = {
        "fold": fold["name"],
        "solver_revenue": solver_rev,
        "solver_cogs": solver_cogs,
        "fold_opt_revenue_mae": float(rev_mae),
        "fold_opt_cogs_mae": float(cogs_mae),
        "fold_opt_score": float(blend_rev_res["mae"] + 0.4 * blend_cogs_res["mae"]),
    }
    for i, comp in enumerate(COMPONENTS):
        weight_row[f"w_rev_{comp}"] = float(w_rev[i])
        weight_row[f"w_cogs_{comp}"] = float(w_cogs[i])

    fold_oof = {
        "fold": fold["name"],
        "pred_rev_mat": pred_rev_mat,
        "pred_cogs_mat": pred_cogs_mat,
        "y_rev": y_val_rev,
        "y_cogs": y_val_cogs,
    }

    return pd.DataFrame(score_rows), pd.DataFrame([weight_row]), fold_oof


def train_full_component_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    component: str,
    feat_df: pd.DataFrame,
    profiles,
):
    if component not in MODEL_COMPONENTS:
        pred_rev = naive_predict(
            train[["Date", "Revenue"]], test["Date"].values, "Revenue", mode=component
        )
        pred_cogs = naive_predict(
            train[["Date", "COGS"]], test["Date"].values, "COGS", mode=component
        )
        return pred_rev, pred_cogs

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
        out_path = SUB_DIR / f"ex_22_bridge_w{tag:02d}.csv"
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
    print("EX_22: Deep FE recency-profile dual ensemble")
    print("=" * 78)

    fold_score_parts = []
    fold_weight_parts = []
    fold_oof_parts = []

    for fold in FOLDS:
        print(f"\n=== {fold['name']} ({fold['val_start']}..{fold['val_end']}) ===")
        s_df, w_df, oof = evaluate_fold_components(train, fold)
        fold_score_parts.append(s_df)
        fold_weight_parts.append(w_df)
        fold_oof_parts.append(oof)

    fold_scores = pd.concat(fold_score_parts, ignore_index=True)
    fold_weights = pd.concat(fold_weight_parts, ignore_index=True)

    fold_scores.to_csv(TRACK / "fold_component_scores.csv", index=False)
    fold_weights.to_csv(TRACK / "fold_weight_search.csv", index=False)

    pred_rev_oof = np.concatenate([o["pred_rev_mat"] for o in fold_oof_parts], axis=0)
    pred_cogs_oof = np.concatenate([o["pred_cogs_mat"] for o in fold_oof_parts], axis=0)
    y_rev_oof = np.concatenate([o["y_rev"] for o in fold_oof_parts], axis=0)
    y_cogs_oof = np.concatenate([o["y_cogs"] for o in fold_oof_parts], axis=0)
    fold_id_oof = np.concatenate(
        [np.array([o["fold"]] * len(o["y_rev"]), dtype=object) for o in fold_oof_parts],
        axis=0,
    )

    w_rev, rev_obj, rev_solver = _safe_weight_optimize_global(
        pred_rev_oof,
        y_rev_oof,
        fold_id_oof,
        std_penalty=ROBUST_STD_PENALTY,
        l2_penalty=WEIGHT_SHRINK_L2,
        fold_weight_map=FOLD_RECENCY_WEIGHTS,
    )
    w_cogs, cogs_obj, cogs_solver = _safe_weight_optimize_global(
        pred_cogs_oof,
        y_cogs_oof,
        fold_id_oof,
        std_penalty=ROBUST_STD_PENALTY,
        l2_penalty=WEIGHT_SHRINK_L2,
        fold_weight_map=FOLD_RECENCY_WEIGHTS,
    )

    fold_global_rows = []
    for oof in fold_oof_parts:
        p_rev = oof["pred_rev_mat"] @ w_rev
        p_cogs = oof["pred_cogs_mat"] @ w_cogs
        mae_rev = float(np.mean(np.abs(oof["y_rev"] - p_rev)))
        mae_cogs = float(np.mean(np.abs(oof["y_cogs"] - p_cogs)))
        fold_global_rows.append(
            {
                "fold": oof["fold"],
                "global_revenue_mae": mae_rev,
                "global_cogs_mae": mae_cogs,
                "global_score": float(mae_rev + 0.4 * mae_cogs),
            }
        )

    fold_global_df = pd.DataFrame(fold_global_rows).sort_values("fold")
    fold_global_df.to_csv(TRACK / "fold_global_metrics.csv", index=False)

    global_weight_row = {
        "revenue_solver": rev_solver,
        "revenue_objective": float(rev_obj),
        "cogs_solver": cogs_solver,
        "cogs_objective": float(cogs_obj),
        "std_penalty": ROBUST_STD_PENALTY,
        "weight_shrink_l2": WEIGHT_SHRINK_L2,
    }
    for i, comp in enumerate(COMPONENTS):
        global_weight_row[f"w_rev_{comp}"] = float(w_rev[i])
        global_weight_row[f"w_cogs_{comp}"] = float(w_cogs[i])
    pd.DataFrame([global_weight_row]).to_csv(
        TRACK / "global_weight_search.csv", index=False
    )

    ensemble_summary = pd.DataFrame(
        [
            {
                "mean_global_revenue_mae": float(
                    fold_global_df["global_revenue_mae"].mean()
                ),
                "mean_global_cogs_mae": float(fold_global_df["global_cogs_mae"].mean()),
                "mean_global_score": float(fold_global_df["global_score"].mean()),
                **{f"w_rev_{c}": float(w_rev[i]) for i, c in enumerate(COMPONENTS)},
                **{f"w_cogs_{c}": float(w_cogs[i]) for i, c in enumerate(COMPONENTS)},
            }
        ]
    )
    ensemble_summary.to_csv(TRACK / "ensemble_summary.csv", index=False)

    print("\nGlobal Revenue weights:")
    for i, comp in enumerate(COMPONENTS):
        print(f"  {comp}: {w_rev[i]:.4f}")

    print("\nGlobal COGS weights:")
    for i, comp in enumerate(COMPONENTS):
        print(f"  {comp}: {w_cogs[i]:.4f}")

    feat_full, profiles_full = build_feature_frame_with_recency(
        train,
        profile_source_df=train,
    )

    pred_rev_parts = []
    pred_cogs_parts = []
    for comp in COMPONENTS:
        print(f"\nTraining full-data component: {comp}")
        p_rev, p_cogs = train_full_component_predictions(
            train,
            test,
            comp,
            feat_full,
            profiles_full,
        )
        pred_rev_parts.append(p_rev)
        pred_cogs_parts.append(p_cogs)

    pred_rev_mat = np.column_stack(pred_rev_parts)
    pred_cogs_mat = np.column_stack(pred_cogs_parts)

    final_rev = pred_rev_mat @ w_rev
    final_cogs = pred_cogs_mat @ w_cogs

    candidate_path = SUB_DIR / "ex_22_deep_fe_holidays_dual_ensemble.csv"
    make_submission(test["Date"], final_rev, final_cogs, candidate_path)

    _make_bridges(candidate_path)

    notes = [
        "# EX_22 Deep FE Recency-Profile Dual Ensemble",
        "",
        "## Goal",
        "- Start from EX_21 production winner with controlled FE delta.",
        "- Add new Tet holiday and regime features to deep FE pipeline.",
        "- Use separate Revenue/COGS global weights from recursive OOF folds.",
        "",
        "## Components",
    ]
    notes.extend([f"- {c}" for c in COMPONENTS])
    notes.extend(
        [
            "",
            "## Validation Setup",
            "- Folds: 2020, 2021, 2022 yearly recursive holdouts.",
            "- Global robust objective: weighted-mean MAE + std-penalty + L2 shrink.",
            f"- Fold recency weights: {FOLD_RECENCY_WEIGHTS}",
            f"- Robust std penalty: {ROBUST_STD_PENALTY}",
            f"- Weight shrink L2: {WEIGHT_SHRINK_L2}",
            f"- Recency profile decay: {RECENCY_DECAY}",
            "",
            "## Global Revenue Weights",
        ]
    )
    for i, comp in enumerate(COMPONENTS):
        notes.append(f"- {comp}: {w_rev[i]:.4f}")

    notes.append("")
    notes.append("## Global COGS Weights")
    for i, comp in enumerate(COMPONENTS):
        notes.append(f"- {comp}: {w_cogs[i]:.4f}")

    notes.extend(
        [
            "",
            "## Outputs",
            f"- Candidate: {candidate_path}",
            "- fold_component_scores.csv",
            "- fold_weight_search.csv",
            "- fold_global_metrics.csv",
            "- global_weight_search.csv",
            "- ensemble_summary.csv",
            "- bridge_summary.csv",
        ]
    )
    (TRACK / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    meta = {
        "elapsed_sec": round(time.time() - t0, 1),
        "components": COMPONENTS,
        "model_components": sorted(MODEL_COMPONENTS),
        "folds": FOLDS,
        "fold_recency_weights": FOLD_RECENCY_WEIGHTS,
        "candidate_path": str(candidate_path),
        "anchor_path": str(ANCHOR_PATH),
        "robust_std_penalty": ROBUST_STD_PENALTY,
        "weight_shrink_l2": WEIGHT_SHRINK_L2,
        "weights_revenue": {c: float(w_rev[i]) for i, c in enumerate(COMPONENTS)},
        "weights_cogs": {c: float(w_cogs[i]) for i, c in enumerate(COMPONENTS)},
    }
    (TRACK / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone in {meta['elapsed_sec']}s")
    print(f"Tracking dir: {TRACK}")


if __name__ == "__main__":
    main()
