"""
Generate SHAP feature importance plot for the report.
Reads pre-computed SHAP values and creates a publication-quality bar chart.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 10

# ── Load SHAP data ──────────────────────────────────────────────────────
shap_df = pd.read_csv(
    "../output/models/ex_03_shap_importance.csv"
)

# Take top 20 features
top = shap_df.head(20).copy()

# Map raw feature names to readable Vietnamese labels
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
}

# Category color mapping
def get_category(feat):
    feat_lower = feat.lower()
    if "lag" in feat_lower or "rmin" in feat_lower or "rmax" in feat_lower or "rmean" in feat_lower or "momentum" in feat_lower or "yoy" in feat_lower:
        return "Biến trễ & thống kê trượt"
    elif any(k in feat_lower for k in ["day", "month", "week", "trend", "year", "sin", "cos", "fourier", "quarter"]):
        return "Lịch & mùa vụ"
    elif any(k in feat_lower for k in ["rev_dom", "rev_month", "rev_dow", "rev_woy", "cogs_dom", "cogs_month", "margin"]):
        return "Hồ sơ lịch sử (profiles)"
    elif any(k in feat_lower for k in ["promo", "discount", "active"]):
        return "Khuyến mãi"
    else:
        return "Khác"

CATEGORY_COLORS = {
    "Biến trễ & thống kê trượt": "#1a5276",
    "Lịch & mùa vụ": "#c0392b",
    "Hồ sơ lịch sử (profiles)": "#27ae60",
    "Khuyến mãi": "#e67e22",
    "Khác": "#7f8c8d",
}

top["label"] = top["feature"].map(LABEL_MAP).fillna(top["feature"])
top["category"] = top["feature"].apply(get_category)
top["color"] = top["category"].map(CATEGORY_COLORS)

# Scale to millions for readability
top["shap_millions"] = top["shap_mean_abs"] / 1e6

# ── Plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

# Reverse order so highest is on top
plot_data = top.iloc[::-1]

bars = ax.barh(
    range(len(plot_data)),
    plot_data["shap_millions"],
    color=plot_data["color"].values,
    edgecolor="white",
    linewidth=0.5,
    height=0.72,
)

ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data["label"].values, fontsize=8.5)
ax.set_xlabel("Giá trị SHAP trung bình tuyệt đối (triệu VND)", fontsize=10)
ax.set_title(
    "Top 20 đặc trưng theo mức đóng góp SHAP — mô hình LightGBM (doanh thu)",
    fontsize=11,
    fontweight="bold",
    pad=12,
)

# Add value labels
for i, (val, label) in enumerate(zip(plot_data["shap_millions"], plot_data["label"])):
    ax.text(val + 0.01, i, f"{val:.2f}M", va="center", fontsize=7.5, color="#333")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=c, label=k) for k, c in CATEGORY_COLORS.items()
    if k in top["category"].values
]
ax.legend(
    handles=legend_elements,
    loc="lower right",
    fontsize=8,
    framealpha=0.9,
    title="Nhóm đặc trưng",
    title_fontsize=8.5,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.2, linestyle="--")

plt.tight_layout()
plt.savefig(
    "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.savefig(
    "shap_feature_importance.pdf",
    bbox_inches="tight",
    facecolor="white",
)
print("Saved: shap_feature_importance.png & .pdf")
plt.close()
