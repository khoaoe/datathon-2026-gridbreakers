# =============================================================================
# DIAGNOSIS — Xác nhận root cause trước khi fix pipeline
# Chạy script này sau Phase C. Không cần data mới, chỉ cần artifacts sẵn có.
#
# 5 câu hỏi cần trả lời:
#   D1. Test Revenue mean có ≈ Train Revenue mean không?      (root cause #1)
#   D2. rev_mean_doy có dominate feature importance không?    (root cause #1)
#   D3. t_idx có near-zero importance không?                  (root cause #1)
#   D4. Predictions năm 2024 có tệ hơn 2023 rõ rệt không?   (root cause #4 NaN)
#   D5. CV folds có overlap gây inflated CV score không?      (root cause #2)
# =============================================================================

import pickle
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
RUN_ENV = 'local'   # đổi thành 'kaggle' nếu cần

SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if RUN_ENV == 'local':
    PROJECT_DIR = SCRIPT_DIR.parents[1] if '__file__' in globals() else SCRIPT_DIR
    DATA_DIR = PROJECT_DIR / 'data'
    ART_DIR  = SCRIPT_DIR / 'artifacts'
else:
    DATA_DIR = Path('/kaggle/input/competitions/datathon-2026-round-1/')
    ART_DIR  = Path('/kaggle/working/artifacts')

plt.rcParams.update({
    'figure.dpi': 130, 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
    'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
})

print('=' * 65)
print('  DIAGNOSIS — Root Cause Confirmation')
print('  VinTelligence DATATHON 2026')
print('=' * 65)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
train = pd.read_csv(DATA_DIR / 'sales.csv', parse_dates=['Date']).sort_values('Date')
train = train[train['Date'] <= '2022-12-31']

raw = pd.read_csv(ART_DIR / 'phase_c_submission_raw.csv', parse_dates=['Date'])
oof = pd.read_csv(ART_DIR / 'oof_predictions.csv', parse_dates=['Date'])
cv  = pd.read_csv(ART_DIR / 'phase_c_cv_report.csv')

with open(ART_DIR / 'model_lgbm_revenue.pkl', 'rb') as f:
    model_rev = pickle.load(f)

# Feature names từ model
feat_names = model_rev.feature_name_
importances = model_rev.feature_importances_
fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)

# =============================================================================
# D1 — Mức Revenue: Test mean vs Train mean
# =============================================================================
print('\n' + '─' * 65)
print('D1 — Revenue Level Comparison (Test vs Train)')
print('─' * 65)

train_mean = train['Revenue'].mean()
train_std  = train['Revenue'].std()
test_mean  = raw['Revenue'].mean()
test_std   = raw['Revenue'].std()
test_max   = raw['Revenue'].max()
train_max  = train['Revenue'].max()

# So sánh với năm 2022 (năm cuối training — gần nhất với test)
train_2022 = train[train['Date'].dt.year == 2022]['Revenue'].mean()
train_2021 = train[train['Date'].dt.year == 2021]['Revenue'].mean()
yoy_2022   = (train_2022 - train_2021) / train_2021

print(f'\n  Train (2012-2022) mean : {train_mean:>15,.0f}')
print(f'  Train 2022 daily mean  : {train_2022:>15,.0f}')
print(f'  Test  (2023-2024) mean : {test_mean:>15,.0f}')
print(f'  Ratio test/train_mean  : {test_mean/train_mean:>15.4f}')
print(f'  Ratio test/2022_mean   : {test_mean/train_2022:>15.4f}')
print(f'  Train max              : {train_max:>15,.0f}')
print(f'  Test  max              : {test_max:>15,.0f}')
print(f'  Ratio test_max/train_max: {test_max/train_max:>14.4f}')
print(f'\n  YoY growth 2021→2022   : {yoy_2022*100:+.2f}%')
expected_test_mean = train_2022 * (1 + yoy_2022)
print(f'  Expected test mean     : {expected_test_mean:>15,.0f}  '
      f'(nếu growth tiếp tục)')
print(f'  Actual test mean       : {test_mean:>15,.0f}')
gap = (test_mean - expected_test_mean) / expected_test_mean
print(f'  Gap (actual vs expect) : {gap*100:>+14.2f}%')

if abs(test_mean - train_mean) / train_mean < 0.05:
    verdict_d1 = '🔴 CONFIRMED: Model bị kẹt ở historical level — test ≈ train mean'
elif test_mean < train_2022:
    verdict_d1 = '🔴 CONFIRMED: Model under-predict — test mean thấp hơn cả 2022'
elif test_mean < expected_test_mean * 0.90:
    verdict_d1 = '🟡 PARTIAL: Model under-predict so với expected growth'
else:
    verdict_d1 = '🟢 OK: Revenue level hợp lý'
print(f'\n  ▶ Verdict: {verdict_d1}')

# =============================================================================
# D2 + D3 — Feature Importance Analysis
# =============================================================================
print('\n' + '─' * 65)
print('D2+D3 — Feature Importance (Revenue model)')
print('─' * 65)

top20 = fi.head(20)
print(f'\n  Top 20 features (gain-based):')
print(f'  {"Rank":<5} {"Feature":<35} {"Importance":>12} {"Cumul%":>8}')
print('  ' + '-' * 63)

total_imp  = fi.sum()
cumul      = 0
for rank, (feat, imp) in enumerate(top20.items(), 1):
    cumul += imp
    bar = '█' * int(imp / total_imp * 200)
    print(f'  {rank:<5} {feat:<35} {imp:>12,.0f} {cumul/total_imp*100:>7.1f}%  {bar}')

# Kiểm tra các suspects
suspects = ['rev_mean_doy', 'cogs_mean_doy', 'rev_mean_month_dow',
            'rev_seasonal_index', 'rev_std_doy']
print(f'\n  Absolute level features (suspects):')
for s in suspects:
    if s in fi:
        rank_s = list(fi.index).index(s) + 1
        pct    = fi[s] / total_imp * 100
        print(f'    {s:<35} rank={rank_s:>3}  share={pct:.2f}%')

t_idx_rank = list(fi.index).index('t_idx') + 1 if 't_idx' in fi else 'N/A'
t_idx_pct  = fi.get('t_idx', 0) / total_imp * 100
print(f'\n  Trend features:')
print(f'    t_idx      rank={t_idx_rank}   share={t_idx_pct:.2f}%')
if 't_break_down' in fi:
    r = list(fi.index).index('t_break_down') + 1
    print(f'    t_break_down rank={r}  share={fi["t_break_down"]/total_imp*100:.2f}%')
if 't_break_up' in fi:
    r = list(fi.index).index('t_break_up') + 1
    print(f'    t_break_up   rank={r}  share={fi["t_break_up"]/total_imp*100:.2f}%')

# Top seasonal features share
seasonal_feats = [f for f in fi.index if any(s in f for s in
    ['mean_doy','std_doy','seasonal_index','mean_month_dow','margin'])]
seasonal_share = fi[seasonal_feats].sum() / total_imp * 100

trend_feats = ['t_idx', 't_break_down', 't_break_up']
trend_share = fi[[f for f in trend_feats if f in fi]].sum() / total_imp * 100

lag_feats = [f for f in fi.index if 'lag_' in f or 'yoy_' in f]
lag_share = fi[lag_feats].sum() / total_imp * 100

print(f'\n  Feature group shares:')
print(f'    Seasonal (absolute level) : {seasonal_share:>6.2f}%')
print(f'    Trend (t_idx + breaks)    : {trend_share:>6.2f}%')
print(f'    Lag features              : {lag_share:>6.2f}%')

if seasonal_share > 40 and trend_share < 10:
    verdict_d2 = ('🔴 CONFIRMED: Seasonal absolute-level features dominate '
                  f'({seasonal_share:.0f}%), trend signal weak ({trend_share:.0f}%)')
elif seasonal_share > 25:
    verdict_d2 = f'🟡 LIKELY: Seasonal features prominent ({seasonal_share:.0f}%)'
else:
    verdict_d2 = '🟢 OK: Feature importance well-distributed'
print(f'\n  ▶ Verdict: {verdict_d2}')

# =============================================================================
# D4 — 2024 Predictions vs 2023 (lag NaN impact)
# =============================================================================
print('\n' + '─' * 65)
print('D4 — 2024 vs 2023 Prediction Quality (lag NaN impact)')
print('─' * 65)

raw_2023 = raw[raw['Date'].dt.year == 2023]
raw_2024 = raw[raw['Date'].dt.year == 2024]

# So sánh với train cùng kỳ
train_2021 = train[train['Date'].dt.year == 2021]
train_2020 = train[train['Date'].dt.year == 2020]

print(f'\n  Test 2023: mean={raw_2023["Revenue"].mean():>12,.0f}  '
      f'std={raw_2023["Revenue"].std():>10,.0f}  n={len(raw_2023)}')
print(f'  Test 2024: mean={raw_2024["Revenue"].mean():>12,.0f}  '
      f'std={raw_2024["Revenue"].std():>10,.0f}  n={len(raw_2024)}')
print(f'  Train 2022: mean={train[train["Date"].dt.year==2022]["Revenue"].mean():>11,.0f}')

# Seasonality-adjusted: compare same months
oof_2023 = oof[oof['Date'].dt.year.isin([2019, 2020, 2021, 2022])]

def mape(a, p): return np.mean(np.abs((a - p) / (np.abs(a) + 1))) * 100

# OOF MAPE by year
oof['year'] = oof['Date'].dt.year
oof_mape_by_year = {}
for yr, grp in oof.groupby('year'):
    m = mape(grp['Revenue_actual'].values, grp['Revenue_pred'].values)
    oof_mape_by_year[yr] = m
    print(f'  OOF MAPE {yr}: {m:.2f}%')

# Variance drop in 2024 predictions?
cv_2023 = raw_2023['Revenue'].std() / raw_2023['Revenue'].mean()
cv_2024 = raw_2024['Revenue'].std() / raw_2024['Revenue'].mean()
print(f'\n  CV (std/mean) 2023 predictions: {cv_2023:.4f}')
print(f'  CV (std/mean) 2024 predictions: {cv_2024:.4f}')
print(f'  (Nếu 2024 CV << 2023 CV → model flat/mean-reverting do NaN lags)')

if cv_2024 < cv_2023 * 0.7:
    verdict_d4 = '🔴 CONFIRMED: 2024 predictions kém variance — lag NaN gây mean reversion'
else:
    verdict_d4 = '🟢 OK: 2023 vs 2024 variance tương đương'
print(f'\n  ▶ Verdict: {verdict_d4}')

# =============================================================================
# D5 — CV Fold Overlap Analysis
# =============================================================================
print('\n' + '─' * 65)
print('D5 — CV Fold Overlap & Inflation Check')
print('─' * 65)

folds_spec = [
    ('Fold1', '2018-01-01', '2019-06-30'),
    ('Fold2', '2019-01-01', '2020-06-30'),
    ('Fold3', '2020-07-01', '2022-01-01'),
    ('Fold4', '2021-07-01', '2022-12-31'),
]

print(f'\n  {"Fold":<8} {"Val Start":<14} {"Val End":<14} {"Overlap w/ prev"}')
print('  ' + '-' * 60)
prev_end = None
for name, vs, ve in folds_spec:
    vs_ts, ve_ts = pd.Timestamp(vs), pd.Timestamp(ve)
    if prev_end and vs_ts < prev_end:
        overlap_days = (prev_end - vs_ts).days
        flag = f'⚠️  {overlap_days} days overlap!'
    else:
        flag = '✅ clean'
    print(f'  {name:<8} {vs:<14} {ve:<14} {flag}')
    prev_end = ve_ts

# OOF MAPE vs reported CV MAPE
print(f'\n  Phase C CV report vs OOF recalculation:')
oof_overall_mape = mape(oof['Revenue_actual'].values, oof['Revenue_pred'].values)
cv_mean_mape     = cv['rev_MAPE'].mean()
print(f'    CV report mean MAPE  : {cv_mean_mape:.2f}%')
print(f'    OOF recalc MAPE      : {oof_overall_mape:.2f}%')
inflation = cv_mean_mape - oof_overall_mape
print(f'    Gap (CV - OOF)       : {inflation:+.2f}pp  '
      f'{"(CV inflated)" if inflation < -1 else "(consistent)"}')

# Kaggle MAE vs OOF MAE
oof_mae = np.mean(np.abs(oof['Revenue_actual'] - oof['Revenue_pred']))
kaggle_mae = 1244048.31
print(f'\n  OOF MAE (in-sample CV)  : {oof_mae:>15,.0f}')
print(f'  Kaggle MAE (test)       : {kaggle_mae:>15,.0f}')
print(f'  Generalization gap      : {(kaggle_mae - oof_mae):>+15,.0f}  '
      f'({(kaggle_mae/oof_mae - 1)*100:+.1f}%)')

if kaggle_mae > oof_mae * 1.5:
    verdict_d5 = '🔴 CONFIRMED: CV score quá lạc quan so với Kaggle — CV folds không representative'
elif kaggle_mae > oof_mae * 1.2:
    verdict_d5 = '🟡 LIKELY: CV có bias nhỏ'
else:
    verdict_d5 = '🟢 OK: Generalization gap chấp nhận được'
print(f'\n  ▶ Verdict: {verdict_d5}')

# =============================================================================
# TỔNG HỢP VÀ VERDICT CUỐI
# =============================================================================
print('\n' + '=' * 65)
print('  DIAGNOSIS SUMMARY')
print('=' * 65)

verdicts = {
    'D1 (Level bias)':       verdict_d1,
    'D2 (Feature dom.)':     verdict_d2,
    'D4 (2024 NaN lag)':     verdict_d4,
    'D5 (CV overlap)':       verdict_d5,
}

red_count    = sum(1 for v in verdicts.values() if '🔴' in v)
yellow_count = sum(1 for v in verdicts.values() if '🟡' in v)

for check, verdict in verdicts.items():
    print(f'\n  [{check}]')
    print(f'    {verdict}')

print(f'\n  ── Severity: {red_count} 🔴 critical  /  {yellow_count} 🟡 moderate ──')
if red_count >= 2:
    print('\n  ❌ Pipeline cần FULL FIX — kiến trúc cốt lõi có vấn đề nghiêm trọng.')
elif red_count == 1:
    print('\n  ⚠️  Pipeline cần targeted fix — 1 vấn đề nghiêm trọng.')
else:
    print('\n  ✅ Vấn đề nhỏ — chỉ cần tune.')

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Root Cause Diagnosis — VinTelligence DATATHON 2026',
             fontsize=14, fontweight='bold')

C_BLUE, C_RED, C_AMBER, C_GREEN = '#2E86AB', '#E84855', '#F9A72B', '#3BB273'

# Chart 1: Revenue level comparison (monthly)
ax = axes[0, 0]
train_m = (train.assign(ym=lambda x: x['Date'].dt.to_period('M'))
           .groupby('ym')['Revenue'].mean().reset_index())
train_m['Date'] = train_m['ym'].dt.to_timestamp()
test_m = (raw.assign(ym=lambda x: x['Date'].dt.to_period('M'))
          .groupby('ym')['Revenue'].mean().reset_index())
test_m['Date'] = test_m['ym'].dt.to_timestamp()
ax.plot(train_m['Date'], train_m['Revenue']/1e6, color=C_BLUE, lw=1.2,
        label='Train actual (monthly mean)', alpha=0.9)
ax.plot(test_m['Date'], test_m['Revenue']/1e6, color=C_RED, lw=2,
        label='Test predicted', alpha=0.9)
ax.axhline(train_mean/1e6, color='navy', ls='--', lw=1,
           label=f'Train overall mean = {train_mean/1e6:.1f}M')
ax.axvline(pd.Timestamp('2023-01-01'), color='black', ls='--', lw=1.5)
ax.set_title('D1: Revenue Level — Train vs Test')
ax.set_ylabel('Revenue (triệu VND)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Chart 2: Feature importance bar (top 20)
ax = axes[0, 1]
top20_plot = top20.sort_values()
colors_fi = [C_RED if any(s in f for s in
             ['mean_doy','std_doy','mean_month_dow','seasonal_index','margin'])
             else C_GREEN if any(s in f for s in ['t_idx','t_break','lag_'])
             else C_BLUE for f in top20_plot.index]
ax.barh(range(len(top20_plot)), top20_plot.values / total_imp * 100,
        color=colors_fi, alpha=0.85)
ax.set_yticks(range(len(top20_plot)))
ax.set_yticklabels(top20_plot.index, fontsize=8)
ax.set_title('D2+D3: Feature Importance\n🔴=Seasonal-abs  🟢=Trend/Lag  🔵=Other')
ax.set_xlabel('Share of total importance (%)')

# Chart 3: Prediction distribution — train vs test
ax = axes[0, 2]
ax.hist(train['Revenue']/1e6, bins=60, alpha=0.5, color=C_BLUE,
        label=f'Train (mean={train_mean/1e6:.1f}M)', density=True)
ax.hist(raw['Revenue']/1e6,   bins=60, alpha=0.5, color=C_RED,
        label=f'Test  (mean={test_mean/1e6:.1f}M)',  density=True)
ax.axvline(train_mean/1e6, color=C_BLUE, ls='--', lw=1.5)
ax.axvline(test_mean/1e6,  color=C_RED,  ls='--', lw=1.5)
ax.set_title('D1: Revenue Distribution Comparison')
ax.set_xlabel('Revenue (triệu VND)')
ax.legend(fontsize=9)

# Chart 4: OOF MAPE by year
ax = axes[1, 0]
years_s = sorted(oof_mape_by_year.keys())
mapes_s = [oof_mape_by_year[y] for y in years_s]
colors_yr = [C_GREEN if m < 15 else C_AMBER if m < 25 else C_RED for m in mapes_s]
ax.bar(years_s, mapes_s, color=colors_yr, alpha=0.85, edgecolor='white')
for y, m in zip(years_s, mapes_s):
    ax.text(y, m + 0.3, f'{m:.1f}%', ha='center', fontsize=9)
ax.axhline(25.5, color='navy', ls='--', lw=1.2, label='Baseline MAPE=25.5%')
ax.set_title('D4: OOF MAPE by Year\n(2024 high? → NaN lag issue)')
ax.set_ylabel('MAPE (%)'); ax.legend(fontsize=9)
ax.set_xticks(years_s)

# Chart 5: 2023 vs 2024 prediction variance
ax = axes[1, 1]
ax.plot(raw_2023['Date'], raw_2023['Revenue']/1e6,
        color=C_BLUE, lw=0.9, label=f'2023 (CV={cv_2023:.3f})', alpha=0.9)
ax.plot(raw_2024['Date'], raw_2024['Revenue']/1e6,
        color=C_RED, lw=0.9, label=f'2024 (CV={cv_2024:.3f})', alpha=0.9)
ax.set_title('D4: Prediction Variance 2023 vs 2024\n(flat 2024 → NaN lag dominance)')
ax.set_ylabel('Revenue (triệu VND)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

# Chart 6: Verdict summary panel
ax = axes[1, 2]
ax.axis('off')
lines = ['DIAGNOSIS VERDICT\n']
for check, v in verdicts.items():
    lines.append(f'{v[:2]}  {check}')
    lines.append(f'    {v[2:60]}')
    lines.append('')
lines.append(f'Kaggle MAE : {kaggle_mae:,.0f}')
lines.append(f'OOF MAE   : {oof_mae:,.0f}')
lines.append(f'Baseline  : 1,225,931')
lines.append(f'Gap       : {kaggle_mae - 1225931:+,.0f}')
ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e1', alpha=0.9))

plt.tight_layout()
out_path = Path(ART_DIR) / 'diagnosis_root_cause.png'
plt.savefig(out_path, bbox_inches='tight', dpi=130)
plt.show()
print(f'\n  📊 Chart saved: {out_path}')
print('  ▶ Gửi kết quả diagnosis này để lập kế hoạch fix chính xác.')
