"""
Central configuration for paths, seeds, and hyperparameters.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# Create dirs
for d in [OUTPUT_DIR, MODEL_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data files ─────────────────────────────────────────────────────────────
FILES = {
    "sales": DATA_DIR / "sales.csv",
    "sample_sub": DATA_DIR / "sample_submission.csv",
    "orders": DATA_DIR / "orders.csv",
    "order_items": DATA_DIR / "order_items.csv",
    "payments": DATA_DIR / "payments.csv",
    "shipments": DATA_DIR / "shipments.csv",
    "returns": DATA_DIR / "returns.csv",
    "reviews": DATA_DIR / "reviews.csv",
    "products": DATA_DIR / "products.csv",
    "customers": DATA_DIR / "customers.csv",
    "promotions": DATA_DIR / "promotions.csv",
    "geography": DATA_DIR / "geography.csv",
    "inventory": DATA_DIR / "inventory.csv",
    "web_traffic": DATA_DIR / "web_traffic.csv",
}

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Time splits ───────────────────────────────────────────────────────────
TRAIN_START = "2012-07-04"
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-07-01"

# Validation: hold out last year of training for local eval
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"

# ── Feature config ────────────────────────────────────────────────────────
LAG_DAYS = [1, 2, 3, 7, 14, 21, 28, 60, 90, 180, 365]
ROLLING_WINDOWS = [7, 14, 28, 60, 90, 180, 365]

# ── LightGBM defaults ────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": -1,
}

# ── XGBoost defaults ─────────────────────────────────────────────────────
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "booster": "gbtree",
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "tree_method": "hist",
    "device": "cuda",
    "verbosity": 0,
    "n_jobs": -1,
}
