"""
Generate all explainability plots for the report:
  1. SHAP bar (already done, regenerated here for consistency)
  2. LightGBM gain-based feature importance
  3. SHAP beeswarm (dot) plot
  4. Partial dependence plots for top 4 features
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 10
warnings.filterwarnings("ignore")

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from modeling.config import MODEL_DIR, VAL_START
from modeling.utils import load_sales
from modeling.feature_engineering import build_feature_table, get_feature_cols

# ── Label map ───────────────────────────────────────────────────────────
LABEL_MAP = {
    "Revenue_lag_1": "Doanh thu ngày trước (lag 1)",
    "COGS_lag_1": "Giá vốn ngày trước (lag 1)",
    "Revenue_lag_7": "Doanh thu 7 ngày trước (lag 7)",
    "dayofmonth": "Ngày trong tháng",
    "trend": "Xu hướng tuyến tính (trend)",
    "COGS_lag_365": "Giá vốn cùng kỳ năm trước (lag 365)",
    "Revenue_lag_365": "Doanh thu cùng kỳ năm trước (lag 365)",
    "rev_dom_mean": "Doanh thu TB theo ngày-trong-tháng",
    "rev_month_dow_mean": "Doanh thu TB theo tháng × thứ",
    "Revenue_rmin_7": "Doanh thu tối thiểu 7 ngày",
    "Revenue_lag_28": "Doanh thu 28 ngày trước (lag 28)",
    "COGS_lag_7": "Giá vốn 7 ngày trước (lag 7)",
    "Revenue_rmin_14": "Doanh thu tối thiểu 14 ngày",
    "Revenue_lag_14": "Doanh thu 14 ngày trước (lag 14)",
    "dayofyear_sin": "Mùa vụ năm (sin dayofyear)",
    "COGS_lag_28": "Giá vốn 28 ngày trước (lag 28)",
    "active_discount_sum": "Tổng chiết khấu đang hoạt động",
    "cogs_dom_mean": "Giá vốn TB theo ngày-trong-tháng",
    "COGS_lag_14": "Giá vốn 14 ngày trước (lag 14)",
    "Revenue_yoy_ratio": "Tỷ lệ tăng trưởng YoY doanh thu",
    "Revenue_rmean_7": "Doanh thu TB trượt 7 ngày",
    "Revenue_rmean_28": "Doanh thu TB trượt 28 ngày",
    "Revenue_rstd_7": "Độ lệch chuẩn doanh thu 7 ngày",
    "rev_dow_mean": "Doanh thu TB theo thứ trong tuần",
    "rev_month_mean": "Doanh thu TB theo tháng",
    "rev_woy_mean": "Doanh thu TB theo tuần-trong-năm",
    "dayofweek": "Thứ trong tuần",
    "month": "Tháng",
    "year": "Năm",
    "dayofyear_cos": "Mùa vụ năm (cos dayofyear)",
    "fourier_sin_1": "Fourier sin bậc 1",
    "rev_dow_std": "Độ lệch chuẩn doanh thu theo thứ",
    "rev_woy_std": "Độ lệch chuẩn doanh thu theo tuần",
    "Revenue_momentum": "Momentum doanh thu",
    "dayofweek_cos": "Thứ trong tuần (cos)",
    "Revenue_lag_90": "Doanh thu 90 ngày trước (lag 90)",
    "days_to_tet": "Số ngày đến Tết",
    "days_to_next_mega_double": "Số ngày đến sale ngày đôi",
    "Revenue_lag_180": "Doanh thu 180 ngày trước (lag 180)",
    "COGS_rmean_365": "Giá vốn TB trượt 365 ngày",
}

def get_label(feat):
    return LABEL_MAP.get(feat, feat)

# ── Load model and data ─────────────────────────────────────────────────
print("Loading model and data...")
with open(MODEL_DIR / "ex_03_lgbm_rev.pkl", "rb") as f:
    model_rev = pickle.load(f)
with open(MODEL_DIR / "ex_03_features.pkl", "rb") as f:
    meta = pickle.load(f)

feature_cols = meta["feature_cols"]

train, _ = load_sales()
profile_source = train[train["Date"] < pd.Timestamp(VAL_START)].copy()
feat_df, profiles = build_feature_table(train, verbose=False, profile_source_df=profile_source)

val_mask = feat_df["Date"] >= pd.Timestamp(VAL_START)
val = feat_df[val_mask].copy()
X_val = val[feature_cols].fillna(0)

# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: LightGBM gain-based feature importance (top 20)
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/3] LightGBM gain importance...")

gain_imp = pd.DataFrame({
    "feature": feature_cols,
    "gain": model_rev.booster_.feature_importance(importance_type="gain"),
}).sort_values("gain", ascending=False)

top_gain = gain_imp.head(20).copy()
top_gain["label"] = top_gain["feature"].apply(get_label)
top_gain["gain_norm"] = top_gain["gain"] / top_gain["gain"].max()

fig, ax = plt.subplots(figsize=(8, 6.5))
plot_data = top_gain.iloc[::-1]
colors = ["#1a5276" if g > 0.3 else "#2980b9" if g > 0.1 else "#85c1e9"
          for g in plot_data["gain_norm"]]
ax.barh(range(len(plot_data)), plot_data["gain"].values, color=colors,
        edgecolor="white", linewidth=0.5, height=0.72)
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data["label"].values, fontsize=8.5)
ax.set_xlabel("Tổng gain (information gain tích lũy)", fontsize=10)
ax.set_title("Top 20 đặc trưng theo LightGBM gain — mô hình doanh thu",
             fontsize=11, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")
plt.tight_layout()
plt.savefig("lgbm_gain_importance.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("lgbm_gain_importance.pdf", bbox_inches="tight", facecolor="white")
print("  Saved: lgbm_gain_importance.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: SHAP beeswarm plot (top 15)
# ═══════════════════════════════════════════════════════════════════════
print("\n[2/3] SHAP beeswarm plot...")
import shap

explainer = shap.TreeExplainer(model_rev)
sample = X_val.head(500)
shap_values = explainer.shap_values(sample)

# Create Explanation object for newer SHAP API
shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=sample.values,
    feature_names=[get_label(f) for f in feature_cols],
)

fig, ax = plt.subplots(figsize=(9, 7))
plt.sca(ax)
shap.summary_plot(
    shap_values, sample,
    feature_names=[get_label(f) for f in feature_cols],
    max_display=15,
    show=False,
    plot_size=None,
)
ax.set_xlabel("Giá trị SHAP (tác động lên dự báo doanh thu, VND)", fontsize=9.5)
ax.set_title(
    "Phân phối SHAP theo đặc trưng — mô hình LightGBM (doanh thu)",
    fontsize=11, fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("shap_beeswarm.pdf", bbox_inches="tight", facecolor="white")
print("  Saved: shap_beeswarm.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: Partial dependence (SHAP dependence) for top 4 features
# ═══════════════════════════════════════════════════════════════════════
print("\n[3/3] SHAP dependence plots (top 4 features)...")

top4_features = ["Revenue_lag_1", "dayofmonth", "trend", "Revenue_lag_365"]
top4_idx = [feature_cols.index(f) for f in top4_features]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
axes = axes.flatten()

for i, (feat, feat_idx) in enumerate(zip(top4_features, top4_idx)):
    ax = axes[i]
    x_vals = sample.iloc[:, feat_idx].values
    y_vals = shap_values[:, feat_idx]

    # Color by feature value
    sc = ax.scatter(x_vals, y_vals, c=x_vals, cmap="RdBu_r", alpha=0.5,
                    s=12, edgecolors="none")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(get_label(feat), fontsize=9)
    ax.set_ylabel("Giá trị SHAP (VND)", fontsize=9)
    ax.set_title(get_label(feat), fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

fig.suptitle(
    "Biểu đồ phụ thuộc SHAP cho bốn đặc trưng hàng đầu",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("shap_dependence_top4.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig("shap_dependence_top4.pdf", bbox_inches="tight", facecolor="white")
print("  Saved: shap_dependence_top4.png")
plt.close()

print("\nAll plots generated successfully!")
