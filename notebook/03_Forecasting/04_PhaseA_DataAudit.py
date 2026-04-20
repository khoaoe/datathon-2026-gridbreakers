# =============================================================================
# PHASE A — DATA FOUNDATION & AUDIT
# VinTelligence DATATHON 2026 | Phần 3: Sales Forecasting
#
# Mục tiêu: Trả lời 5 câu hỏi bắt buộc trước khi viết bất kỳ feature nào:
#   Q1. web_traffic.csv có cover ngày 2023-01-01 → 2024-07-01 không?
#   Q2. Có ngày nào bị thiếu trong sales.csv không?
#   Q3. Cấu trúc sample_submission có vấn đề gì không?
#   Q4. Outlier ngày nào nằm ngoài mean ± 3σ?
#   Q5. COGS/Revenue ratio có ổn định qua các năm không?
#
# Output: phase_a_audit_report.txt  (tóm tắt mọi kết quả)
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Runtime flag & đường dẫn ─────────────────────────────────────────────────
# local: đọc data từ repo/data, ghi artifacts cạnh file script
# kaggle: đọc data từ /kaggle/input, ghi artifacts vào /kaggle/working
RUN_ENV = 'local'  # đổi thành 'kaggle' khi chạy trên Kaggle

SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if RUN_ENV == 'local':
    PROJECT_DIR = SCRIPT_DIR.parents[2] if '__file__' in globals() else SCRIPT_DIR
    DATA_DIR = PROJECT_DIR / 'data'
    OUT_DIR = SCRIPT_DIR / 'artifacts'
elif RUN_ENV == 'kaggle':
    DATA_DIR = Path('/kaggle/input/competitions/datathon-2026-round-1/')
    OUT_DIR = Path('/kaggle/working/artifacts')
else:
    raise ValueError("RUN_ENV phải là 'local' hoặc 'kaggle'.")

if not DATA_DIR.exists():
    raise FileNotFoundError(f'Không tìm thấy thư mục data cho RUN_ENV={RUN_ENV}: {DATA_DIR}')

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Test period constants (NEVER CHANGE) ─────────────────────────────────────
TEST_START = pd.Timestamp('2023-01-01')
TEST_END   = pd.Timestamp('2024-07-01')
LEAKAGE_WALL = TEST_START           # mọi feature phải < ngày này

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 130, 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
})
C_BLUE  = '#2E86AB'
C_RED   = '#E84855'
C_AMBER = '#F9A72B'
C_GREEN = '#3BB273'

print('=' * 65)
print('  PHASE A — DATA FOUNDATION & AUDIT')
print('  VinTelligence DATATHON 2026')
print('=' * 65)
print(f'  RUN_ENV  : {RUN_ENV}')
print(f'  DATA_DIR : {DATA_DIR}')
print(f'  OUT_DIR  : {OUT_DIR}')

# =============================================================================
# SECTION 1 — LOAD ALL RELEVANT TABLES
# =============================================================================
print('\n📂 SECTION 1: Loading tables...\n')

sales       = pd.read_csv(DATA_DIR / 'sales.csv',        parse_dates=['Date'])
sample_sub  = pd.read_csv(DATA_DIR / 'sample_submission.csv', parse_dates=['Date'])
web_traffic = pd.read_csv(DATA_DIR / 'web_traffic.csv',  parse_dates=['date'])

# ── Chuẩn hóa tên cột web_traffic (có thể là 'date' hoặc 'Date') ─────────────
if 'date' in web_traffic.columns:
    web_traffic = web_traffic.rename(columns={'date': 'Date'})

# ── Các bảng phụ trợ (dùng cho Phase B/C) — chỉ đọc shape ở đây ──────────────
aux_tables = {
    'orders':       'orders.csv',
    'order_items':  'order_items.csv',
    'products':     'products.csv',
    'customers':    'customers.csv',
    'returns':      'returns.csv',
    'payments':     'payments.csv',
    'geography':    'geography.csv',
    'promotions':   'promotions.csv',
    'inventory':    'inventory.csv',
    'inventory_enhanced': 'inventory_enhanced.csv',
    'shipments':    'shipments.csv',
    'reviews':      'reviews.csv',
}
shapes = {}
for name, fname in aux_tables.items():
    try:
        _df = pd.read_csv(DATA_DIR / fname, nrows=0)   # chỉ đọc header
        shapes[name] = pd.read_csv(DATA_DIR / fname).shape
        print(f'  ✅  {name:<25} shape: {shapes[name]}')
    except FileNotFoundError:
        print(f'  ⚠️   {name:<25} NOT FOUND')
        shapes[name] = None

print(f'\n  ✅  sales              shape: {sales.shape}')
print(f'  ✅  sample_submission  shape: {sample_sub.shape}')
print(f'  ✅  web_traffic        shape: {web_traffic.shape}')

# =============================================================================
# SECTION 2 — SALES.CSV: DATE COMPLETENESS AUDIT
# =============================================================================
print('\n' + '─' * 65)
print('📅 SECTION 2: Sales.csv Date Completeness Audit')
print('─' * 65)

train_start = sales['Date'].min()
train_end   = sales['Date'].max()
n_rows      = len(sales)

# Tạo date range đầy đủ từ train_start đến train_end
full_date_range = pd.date_range(start=train_start, end=train_end, freq='D')
missing_dates   = full_date_range.difference(sales['Date'])
dup_dates       = sales[sales.duplicated('Date', keep=False)]['Date'].unique()

print(f'\n  Train period  : {train_start.date()} → {train_end.date()}')
print(f'  Total rows    : {n_rows:,}')
print(f'  Expected days : {len(full_date_range):,}')
print(f'  Missing dates : {len(missing_dates):,}')
print(f'  Duplicate dates: {len(dup_dates):,}')

if len(missing_dates) > 0:
    print(f'\n  ⚠️  Missing date list (first 20):')
    for d in missing_dates[:20]:
        print(f'       {d.date()}')
    if len(missing_dates) > 20:
        print(f'       ... và {len(missing_dates)-20} ngày nữa')
    # Ghi chú: cần imputation strategy trong Phase B
    MISSING_DATE_FLAG = True
else:
    print('\n  ✅ Không có ngày nào bị thiếu trong training data.')
    MISSING_DATE_FLAG = False

# Kiểm tra null trong Revenue và COGS
null_revenue = sales['Revenue'].isna().sum()
null_cogs    = sales['COGS'].isna().sum()
neg_revenue  = (sales['Revenue'] < 0).sum()
neg_cogs     = (sales['COGS'] < 0).sum()

print(f'\n  Null Revenue  : {null_revenue}')
print(f'  Null COGS     : {null_cogs}')
print(f'  Negative Rev  : {neg_revenue}')
print(f'  Negative COGS : {neg_cogs}')

if null_revenue == 0 and null_cogs == 0:
    print('  ✅ Không có null values trong Revenue/COGS.')

# =============================================================================
# SECTION 3 — SAMPLE SUBMISSION AUDIT
# =============================================================================
print('\n' + '─' * 65)
print('📋 SECTION 3: Sample Submission Audit')
print('─' * 65)

sub_start = sample_sub['Date'].min()
sub_end   = sample_sub['Date'].max()
sub_rows  = len(sample_sub)

# Kiểm tra test period
sub_missing_dates = pd.date_range(
    start=TEST_START, end=TEST_END, freq='D'
).difference(sample_sub['Date'])

print(f'\n  Test period   : {sub_start.date()} → {sub_end.date()}')
print(f'  Total rows    : {sub_rows:,}')
print(f'  Columns       : {list(sample_sub.columns)}')
print(f'  Missing dates : {len(sub_missing_dates):,}')

# Kiểm tra giá trị mẫu — là 0 hay NaN?
rev_zeros    = (sample_sub['Revenue'] == 0).sum()
rev_nulls    = sample_sub['Revenue'].isna().sum()
cogs_zeros   = (sample_sub['COGS'] == 0).sum()

print(f'\n  Revenue zeros : {rev_zeros}  (thường = placeholder)')
print(f'  Revenue nulls : {rev_nulls}')
print(f'  COGS zeros    : {cogs_zeros}')

if rev_zeros == sub_rows:
    print('  ✅ Tất cả Revenue/COGS = 0: đây là placeholder cần overwrite.')
elif rev_nulls == sub_rows:
    print('  ✅ Tất cả Revenue/COGS = NaN: đây là placeholder cần overwrite.')
else:
    print('  ℹ️  Sample có giá trị non-zero — kiểm tra kỹ trước khi overwrite.')

# =============================================================================
# SECTION 4 — WEB TRAFFIC: COVERAGE AUDIT (câu hỏi then chốt)
# =============================================================================
print('\n' + '─' * 65)
print('🌐 SECTION 4: Web Traffic Coverage Audit')
print('─' * 65)

wt_start = web_traffic['Date'].min()
wt_end   = web_traffic['Date'].max()
wt_rows  = len(web_traffic)
wt_cols  = list(web_traffic.columns)

print(f'\n  web_traffic period : {wt_start.date()} → {wt_end.date()}')
print(f'  Total rows         : {wt_rows:,}')
print(f'  Columns            : {wt_cols}')

# Kiểm tra coverage cho test period
wt_in_test = web_traffic[
    (web_traffic['Date'] >= TEST_START) & (web_traffic['Date'] <= TEST_END)
]
wt_test_dates_covered = len(wt_in_test['Date'].unique())
test_total_days = (TEST_END - TEST_START).days + 1

print(f'\n  Test period = {TEST_START.date()} → {TEST_END.date()} ({test_total_days} ngày)')
print(f'  web_traffic dates trong test period: {wt_test_dates_covered:,}')

if wt_end >= TEST_START:
    coverage_pct = wt_test_dates_covered / test_total_days * 100
    print(f'  Coverage rate: {coverage_pct:.1f}%')
    if coverage_pct > 80:
        WEB_TRAFFIC_STRATEGY = 'DIRECT'
        print('\n  🟢 QUYẾT ĐỊNH: Dùng web_traffic TRỰC TIẾP làm feature (coverage đủ cao)')
    elif coverage_pct > 0:
        WEB_TRAFFIC_STRATEGY = 'PARTIAL'
        print(f'\n  🟡 QUYẾT ĐỊNH: Dùng HYBRID — trực tiếp cho {coverage_pct:.0f}% có data, '
              f'seasonal proxy cho phần còn lại')
    else:
        WEB_TRAFFIC_STRATEGY = 'SEASONAL_PROXY'
        print('\n  🔴 QUYẾT ĐỊNH: web_traffic KHÔNG cover test period')
        print('       → Dùng SEASONAL PROXY: monthly avg từ training period')
else:
    WEB_TRAFFIC_STRATEGY = 'SEASONAL_PROXY'
    print(f'\n  🔴 web_traffic kết thúc {wt_end.date()}, trước TEST_START {TEST_START.date()}')
    print('  QUYẾT ĐỊNH: Dùng SEASONAL PROXY (historical monthly averages)')

print(f'\n  ▶ WEB_TRAFFIC_STRATEGY = "{WEB_TRAFFIC_STRATEGY}"')

# Columns breakdown của web_traffic
print(f'\n  Null counts trong web_traffic:')
print(web_traffic.isnull().sum().to_string())

# Traffic source distribution
if 'traffic_source' in web_traffic.columns:
    print(f'\n  Traffic sources:')
    print(web_traffic['traffic_source'].value_counts().to_string())

# =============================================================================
# SECTION 5 — OUTLIER DETECTION trên sales.csv
# =============================================================================
print('\n' + '─' * 65)
print('🔎 SECTION 5: Outlier Detection (mean ± 3σ)')
print('─' * 65)

rev_mean  = sales['Revenue'].mean()
rev_std   = sales['Revenue'].std()
cogs_mean = sales['COGS'].mean()
cogs_std  = sales['COGS'].std()

rev_upper  = rev_mean  + 3 * rev_std
rev_lower  = rev_mean  - 3 * rev_std
cogs_upper = cogs_mean + 3 * cogs_std
cogs_lower = cogs_mean - 3 * cogs_std

outliers_rev  = sales[(sales['Revenue'] < rev_lower) | (sales['Revenue'] > rev_upper)].copy()
outliers_cogs = sales[(sales['COGS'] < cogs_lower)   | (sales['COGS'] > cogs_upper)].copy()

# Hợp nhất outlier từ cả hai cột
outlier_dates = pd.concat([outliers_rev[['Date', 'Revenue', 'COGS']],
                            outliers_cogs[['Date', 'Revenue', 'COGS']]]).drop_duplicates('Date')

print(f'\n  Revenue  → mean: {rev_mean:>12,.0f}  std: {rev_std:>12,.0f}')
print(f'             3σ band: [{rev_lower:>12,.0f}, {rev_upper:>12,.0f}]')
print(f'  Outliers (Revenue): {len(outliers_rev):,} ngày')

print(f'\n  COGS     → mean: {cogs_mean:>12,.0f}  std: {cogs_std:>12,.0f}')
print(f'             3σ band: [{cogs_lower:>12,.0f}, {cogs_upper:>12,.0f}]')
print(f'  Outliers (COGS)  : {len(outliers_cogs):,} ngày')

print(f'\n  Tổng ngày cần flag: {len(outlier_dates):,}')
if len(outlier_dates) > 0:
    print('\n  Top outliers theo Revenue (giảm dần):')
    print(outlier_dates.sort_values('Revenue', ascending=False)
          .head(15)[['Date', 'Revenue', 'COGS']]
          .to_string(index=False))

# Lưu outlier list để dùng trong Phase B
outlier_dates.to_csv(OUT_DIR / 'phase_a_outlier_dates.csv', index=False)
print('\n  ✅ Đã lưu: phase_a_outlier_dates.csv')

# =============================================================================
# SECTION 6 — COGS/REVENUE RATIO STABILITY ANALYSIS (câu hỏi chiến lược)
# =============================================================================
print('\n' + '─' * 65)
print('💰 SECTION 6: COGS/Revenue Ratio Stability Analysis')
print('─' * 65)

sales_s6 = sales.copy()
sales_s6['year']          = sales_s6['Date'].dt.year
sales_s6['month']         = sales_s6['Date'].dt.month
sales_s6['cogs_rev_ratio'] = sales_s6['COGS'] / sales_s6['Revenue']

# Theo năm
annual_ratio = sales_s6.groupby('year')['cogs_rev_ratio'].agg(['mean', 'std', 'min', 'max'])
annual_ratio.columns = ['mean_ratio', 'std_ratio', 'min_ratio', 'max_ratio']

print('\n  COGS/Revenue ratio theo năm:')
print(annual_ratio.round(4).to_string())

ratio_overall_mean  = sales_s6['cogs_rev_ratio'].mean()
ratio_overall_std   = sales_s6['cogs_rev_ratio'].std()
ratio_cv            = ratio_overall_std / ratio_overall_mean  # coefficient of variation

print(f'\n  Overall mean ratio : {ratio_overall_mean:.4f}  ({ratio_overall_mean*100:.2f}% of Revenue)')
print(f'  Overall std ratio  : {ratio_overall_std:.4f}')
print(f'  Coefficient of Var : {ratio_cv:.4f}  ({ratio_cv*100:.2f}%)')

if ratio_cv < 0.05:
    COGS_STRATEGY = 'RATIO_LOCK'
    print('\n  🟢 QUYẾT ĐỊNH: CV < 5% → COGS = Revenue × seasonal_margin_ratio')
    print('       (Tiết kiệm 1 model, đảm bảo ràng buộc kinh tế)')
elif ratio_cv < 0.10:
    COGS_STRATEGY = 'RATIO_SEASONAL'
    print('\n  🟡 QUYẾT ĐỊNH: CV 5-10% → Model riêng cho margin theo mùa')
else:
    COGS_STRATEGY = 'INDEPENDENT_MODEL'
    print('\n  🔴 QUYẾT ĐỊNH: CV > 10% → Forecast COGS độc lập như Revenue')

print(f'\n  ▶ COGS_STRATEGY = "{COGS_STRATEGY}"')

# Monthly ratio seasonality
monthly_ratio = sales_s6.groupby('month')['cogs_rev_ratio'].mean()
print('\n  Ratio theo tháng (seasonal pattern):')
print(monthly_ratio.round(4).to_string())

# =============================================================================
# SECTION 7 — TREND ANALYSIS: Xác định breakpoints
# =============================================================================
print('\n' + '─' * 65)
print('📈 SECTION 7: Trend & Breakpoint Analysis')
print('─' * 65)

# Annual totals (chỉ năm đủ 365 ngày: 2013–2022)
annual = sales_s6.groupby('year')[['Revenue', 'COGS']].sum()
annual_full = annual.loc[2013:2022]

# YoY growth rates
annual['rev_yoy_growth'] = annual['Revenue'].pct_change() * 100
annual['cogs_yoy_growth'] = annual['COGS'].pct_change() * 100

print('\n  Annual Revenue & YoY Growth:')
print(annual[['Revenue', 'rev_yoy_growth']].round(2).to_string())

# Xác định breakpoints tự động: năm có YoY growth < -10%
breakpoints_down = annual[annual['rev_yoy_growth'] < -10].index.tolist()
breakpoints_up   = annual[(annual['rev_yoy_growth'] > 10) &
                          (annual.index > 2015)].index.tolist()

print(f'\n  ⬇️  Breakpoints suy giảm (YoY < -10%): {breakpoints_down}')
print(f'  ⬆️  Breakpoints phục hồi (YoY > +10%): {breakpoints_up}')

# Thống kê 3 giai đoạn
phases = {
    'Tăng trưởng (2013–2018)':  (2013, 2018),
    'Suy giảm   (2019–2020)':  (2019, 2020),
    'Phục hồi   (2021–2022)':  (2021, 2022),
}
print('\n  Tốc độ tăng trưởng CAGR theo giai đoạn:')
for phase_name, (y_start, y_end) in phases.items():
    if y_start in annual.index and y_end in annual.index:
        rev_start = annual.loc[y_start, 'Revenue']
        rev_end   = annual.loc[y_end,   'Revenue']
        n_years   = y_end - y_start
        if n_years > 0:
            cagr = (rev_end / rev_start) ** (1 / n_years) - 1
        else:
            cagr = (rev_end / rev_start) - 1
        print(f'  {phase_name}: CAGR = {cagr*100:+.1f}%')

# =============================================================================
# SECTION 8 — VISUALIZATIONS (4 charts)
# =============================================================================
print('\n' + '─' * 65)
print('📊 SECTION 8: Generating Audit Charts...')
print('─' * 65)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Phase A — Data Foundation Audit Report\nVinTelligence DATATHON 2026',
             fontsize=15, fontweight='bold', y=1.01)

# ── Chart 1: Revenue & COGS lịch sử + đánh dấu outliers & test period ────────
ax1 = axes[0, 0]
ax1.plot(sales['Date'], sales['Revenue'] / 1e6, lw=0.7, color=C_BLUE,
         label='Revenue (train)', alpha=0.9)
ax1.plot(sales['Date'], sales['COGS']    / 1e6, lw=0.7, color=C_AMBER,
         label='COGS (train)', alpha=0.7)

# Vẽ outlier points
if len(outlier_dates) > 0:
    ax1.scatter(outlier_dates['Date'], outlier_dates['Revenue'] / 1e6,
                color=C_RED, s=20, zorder=5, label=f'Outliers ({len(outlier_dates)})')

# Vùng test period (tô xám)
ax1.axvspan(TEST_START, TEST_END, alpha=0.08, color='green', label='Test period')
ax1.axvline(LEAKAGE_WALL, color='red', lw=1.5, ls='--', label='Leakage wall')

ax1.set_title('Historical Revenue & COGS with Outlier Flags')
ax1.set_ylabel('Value (triệu VND)')
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Chart 2: Annual Revenue + YoY growth ─────────────────────────────────────
ax2 = axes[0, 1]
bars = ax2.bar(annual_full.index, annual_full['Revenue'] / 1e9,
               color=[C_GREEN if g >= 0 else C_RED
                      for g in annual['rev_yoy_growth'].loc[2013:2022].fillna(0)],
               alpha=0.85, edgecolor='white')
ax2.set_title('Annual Revenue & YoY Growth Rate (2013–2022)')
ax2.set_ylabel('Revenue (tỷ VND)')
ax2.set_xlabel('Year')

# Thêm YoY growth labels
ax2_twin = ax2.twinx()
yoy_vals = annual['rev_yoy_growth'].loc[2013:2022]
ax2_twin.plot(annual_full.index, yoy_vals, 'o--',
              color='navy', lw=1.5, ms=5, label='YoY%')
ax2_twin.axhline(0, color='navy', lw=0.8, ls=':')
ax2_twin.set_ylabel('YoY Growth (%)', color='navy')
ax2_twin.yaxis.label.set_color('navy')
ax2_twin.tick_params(axis='y', colors='navy')
ax2_twin.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

# Annotate breakpoints
for bp in breakpoints_down:
    if bp in annual_full.index:
        ax2.annotate(f'▼{bp}', xy=(bp, annual_full.loc[bp, 'Revenue']/1e9),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9, color=C_RED, fontweight='bold')

# ── Chart 3: COGS/Revenue ratio stability ────────────────────────────────────
ax3 = axes[1, 0]
ax3.plot(sales['Date'], sales_s6['cogs_rev_ratio'], lw=0.5, color=C_BLUE, alpha=0.4)

# Rolling 30-day mean
rolling_ratio = sales_s6.set_index('Date')['cogs_rev_ratio'].rolling(30).mean()
ax3.plot(rolling_ratio.index, rolling_ratio.values, lw=1.5, color=C_BLUE,
         label='30-day rolling mean')
ax3.axhline(ratio_overall_mean, color=C_RED,   lw=1.5, ls='--',
            label=f'Overall mean = {ratio_overall_mean:.3f}')
ax3.axhline(ratio_overall_mean + 2*ratio_overall_std, color=C_RED, lw=0.8,
            ls=':', alpha=0.6, label='mean ± 2σ')
ax3.axhline(ratio_overall_mean - 2*ratio_overall_std, color=C_RED, lw=0.8,
            ls=':', alpha=0.6)
ax3.axvline(LEAKAGE_WALL, color='green', lw=1.5, ls='--', label='Leakage wall')
ax3.set_title(f'COGS/Revenue Ratio Stability (CV = {ratio_cv*100:.2f}%)')
ax3.set_ylabel('COGS / Revenue')
ax3.legend(fontsize=8)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.set_ylim(0.6, 1.1)

# ── Chart 4: Web Traffic Timeline vs Sales ───────────────────────────────────
ax4 = axes[1, 1]

# Vẽ daily sessions nếu có
traffic_col = next((c for c in web_traffic.columns
                    if 'session' in c.lower() or 'visit' in c.lower()), None)
if traffic_col:
    wt_daily = web_traffic.groupby('Date')[traffic_col].sum().reset_index()
    ax4.plot(wt_daily['Date'], wt_daily[traffic_col], lw=0.7, color=C_BLUE,
             label='Web sessions', alpha=0.7)
    ax4.set_ylabel('Sessions', color=C_BLUE)
else:
    ax4.text(0.5, 0.5, 'No sessions column\nfound in web_traffic.csv',
             ha='center', va='center', transform=ax4.transAxes, fontsize=11)

ax4.axvline(LEAKAGE_WALL, color='red', lw=1.5, ls='--', label='Leakage wall')
ax4.axvspan(TEST_START, TEST_END, alpha=0.1, color='green',
            label=f'Test period\n(Strategy: {WEB_TRAFFIC_STRATEGY})')
ax4.set_title(f'Web Traffic Timeline\n▶ Coverage Strategy: {WEB_TRAFFIC_STRATEGY}')
ax4.legend(fontsize=8)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
chart_path = OUT_DIR / 'phase_a_audit_charts.png'
plt.savefig(chart_path, bbox_inches='tight', dpi=130)
plt.show()
print(f'  📊 Chart saved: {chart_path}')

# =============================================================================
# SECTION 9 — FEATURE STRATEGY DECISION TABLE
# =============================================================================
print('\n' + '─' * 65)
print('⚙️  SECTION 9: Feature Engineering Strategy Decisions')
print('─' * 65)

decisions = {
    'Missing dates in sales.csv':    ('YES' if MISSING_DATE_FLAG else 'NO',
                                      'Impute với forward-fill trong Phase B'
                                      if MISSING_DATE_FLAG else 'Không cần imputation'),
    'Web traffic strategy':          (WEB_TRAFFIC_STRATEGY,
                                      'Dùng trực tiếp làm exogenous feature'
                                      if WEB_TRAFFIC_STRATEGY == 'DIRECT'
                                      else 'Dùng seasonal monthly average proxy'),
    'COGS forecast strategy':        (COGS_STRATEGY,
                                      'COGS = Revenue_pred × seasonal_margin_ratio'
                                      if COGS_STRATEGY != 'INDEPENDENT_MODEL'
                                      else 'Forecast COGS như target riêng biệt'),
    'Trend model':                   ('PIECEWISE LINEAR',
                                      f'Breakpoints tại: {breakpoints_down} (down), '
                                      f'{breakpoints_up} (up)'),
    'Outlier handling':              (f'{len(outlier_dates)} ngày',
                                      'Flag và clip trong Phase B/C'),
    'Lag safety for 548-day horizon':('LAG ≥ 365 ngày',
                                      'lag_365, lag_730, rolling_365d được phép'),
    'Seasonal profile source':       ('TRAIN ONLY',
                                      'Fit trên 2012-2022, lookup cho 2023-2024'),
}

print()
for key, (decision, note) in decisions.items():
    print(f'  [{key}]')
    print(f'    → Quyết định : {decision}')
    print(f'    → Ghi chú    : {note}')
    print()

# =============================================================================
# SECTION 10 — WRITE AUDIT REPORT
# =============================================================================
report_lines = [
    '=' * 65,
    'PHASE A AUDIT REPORT — VinTelligence DATATHON 2026',
    '=' * 65,
    '',
    f'Train period  : {train_start.date()} → {train_end.date()}',
    f'Test period   : {TEST_START.date()} → {TEST_END.date()}',
    f'Train rows    : {n_rows:,}',
    f'Test rows     : {sub_rows:,}',
    '',
    '--- Q1: Missing dates in sales.csv ---',
    f'Missing dates  : {len(missing_dates)}',
    f'Flag           : {"IMPUTATION NEEDED" if MISSING_DATE_FLAG else "CLEAN"}',
    '',
    '--- Q2: Sample submission structure ---',
    f'Columns        : {list(sample_sub.columns)}',
    f'Revenue zeros  : {rev_zeros}',
    '',
    '--- Q3: Web Traffic Coverage ---',
    f'WT period      : {wt_start.date()} → {wt_end.date()}',
    f'WT rows        : {wt_rows:,}',
    f'Test coverage  : {wt_test_dates_covered} days',
    f'Strategy       : {WEB_TRAFFIC_STRATEGY}',
    '',
    '--- Q4: Outliers (mean ± 3σ) ---',
    f'Revenue upper bound : {rev_upper:,.0f}',
    f'Outlier count       : {len(outlier_dates)} ngày',
    '',
    '--- Q5: COGS/Revenue Ratio ---',
    f'Overall mean   : {ratio_overall_mean:.4f}',
    f'CV             : {ratio_cv*100:.2f}%',
    f'Strategy       : {COGS_STRATEGY}',
    '',
    '--- Trend Breakpoints ---',
    f'Downward : {breakpoints_down}',
    f'Recovery : {breakpoints_up}',
    '',
    '=' * 65,
    'PHASE A COMPLETE — Ready for Phase B: Feature Store Builder',
    '=' * 65,
]

report_path = OUT_DIR / 'phase_a_audit_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print('=' * 65)
print('✅ PHASE A COMPLETE')
print('=' * 65)
print(f'  📄 Report : {report_path}')
print(f'  📊 Charts : {OUT_DIR}/phase_a_audit_charts.png')
print(f'  📋 Outliers: {OUT_DIR}/phase_a_outlier_dates.csv')
print()
print('  Các quyết định chốt cho Phase B:')
print(f'  • WEB_TRAFFIC_STRATEGY = "{WEB_TRAFFIC_STRATEGY}"')
print(f'  • COGS_STRATEGY        = "{COGS_STRATEGY}"')
print(f'  • MISSING_DATE_FLAG    = {MISSING_DATE_FLAG}')
print(f'  • Outlier dates saved  = {len(outlier_dates)} rows')