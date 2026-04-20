# =============================================================================
# PHASE D — PREDICTION REFINEMENT & SUBMISSION BUILDER
# VinTelligence DATATHON 2026 | Phần 3: Sales Forecasting
#
# Input (từ Phase C artifacts):
#   phase_c_submission_raw.csv      — dự báo thô Revenue + COGS
#   model_lgbm_revenue.pkl          — final Revenue model
#   model_lgbm_cogs.pkl             — final COGS model
#   oof_predictions.csv             — OOF predictions (cho plot)
#   phase_c_cv_report.csv           — walk-forward CV results
#   phase_b_seasonal_profiles.pkl   — seasonal profiles (từ Phase B)
#   feature_store_test.csv/.parquet — test features (nếu cần re-predict)
#   sales.csv                       — training actuals (cho stats & sanity)
#   sample_submission.csv           — format template
#
# Pipeline trong Phase D (5 bước):
#   Step 1  Load & verify raw predictions
#   Step 2  Refined post-processing (seasonal margin + plausibility gates)
#   Step 3  Submission format validation (7 hard gates)
#   Step 4  Export submission.csv
#   Step 5  Diagnostic charts (8 plots)
#
# Output:
#   submission.csv                  — file nộp chính thức (Kaggle)
#   phase_d_postprocess_log.csv     — audit trail mọi chỉnh sửa
#   phase_d_validation_report.txt   — báo cáo pass/fail các gate
#   phase_d_submission_diagnostics.png
#   phase_d_monthly_comparison.png
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Runtime flag & paths ─────────────────────────────────────────────────────
# local: đọc artifacts từ thư mục artifacts cạnh script, data từ data/
# kaggle: đọc/ghi tại /kaggle/working
RUN_ENV = 'local'   # đổi thành 'kaggle' khi chạy trên Kaggle

SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
if RUN_ENV == 'local':
    PROJECT_DIR = SCRIPT_DIR.parents[2] if '__file__' in globals() else SCRIPT_DIR
    DATA_DIR    = PROJECT_DIR / 'data'
    ART_DIR     = SCRIPT_DIR / 'artifacts'      # đọc artifacts Phase B/C
    OUT_DIR     = SCRIPT_DIR / 'artifacts'      # ghi submission vào đây
elif RUN_ENV == 'kaggle':
    DATA_DIR = Path('/kaggle/input/competitions/datathon-2026-round-1/')
    ART_DIR  = Path('/kaggle/working/artifacts')
    OUT_DIR  = Path('/kaggle/working/artifacts')
else:
    raise ValueError("RUN_ENV phải là 'local' hoặc 'kaggle'.")

if not DATA_DIR.exists():
    raise FileNotFoundError(f'DATA_DIR not found: {DATA_DIR}')

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants (KHÔNG THAY ĐỔI) ───────────────────────────────────────────────
TRAIN_START  = pd.Timestamp('2012-07-04')
TRAIN_END    = pd.Timestamp('2022-12-31')
TEST_START   = pd.Timestamp('2023-01-01')
TEST_END     = pd.Timestamp('2024-07-01')

# Plausibility guards — điều chỉnh sau khi xem Phase C CV results
MAX_YOY_GROWTH  = 0.60      # Revenue test vs train tail không quá +60% YoY
MIN_YOY_GROWTH  = -0.50     # không giảm quá -50% YoY
COGS_RATIO_CAP  = 0.98      # COGS tối đa = 98% Revenue (hard floor cho gross margin)
COGS_RATIO_FLOOR= 0.30      # COGS tối thiểu = 30% Revenue (tránh margin phi thực tế)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 130, 'figure.facecolor': 'white',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
    'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
})
C_BLUE   = '#2E86AB'
C_RED    = '#E84855'
C_AMBER  = '#F9A72B'
C_GREEN  = '#3BB273'
C_PURPLE = '#7B2D8B'

print('=' * 65)
print('  PHASE D — PREDICTION REFINEMENT & SUBMISSION BUILDER')
print('  VinTelligence DATATHON 2026')
print('=' * 65)
print(f'  RUN_ENV : {RUN_ENV}')
print(f'  ART_DIR : {ART_DIR}')
print(f'  OUT_DIR : {OUT_DIR}')

# =============================================================================
# STEP 1 — LOAD & VERIFY RAW PREDICTIONS
# =============================================================================
print('\n' + '─' * 65)
print('📂 STEP 1: Load & verify all artifacts...')
print('─' * 65)

# ── 1a. Training actuals (cần cho statistics & sanity checks) ────────────────
sales = pd.read_csv(DATA_DIR / 'sales.csv', parse_dates=['Date']).sort_values('Date')
train = sales[sales['Date'] <= TRAIN_END].copy()
assert len(train) > 0

# Training statistics cho clipping
rev_train_mean = train['Revenue'].mean()
rev_train_std  = train['Revenue'].std()
rev_train_min  = train['Revenue'].min()
cogs_train_mean= train['COGS'].mean()
cogs_train_std = train['COGS'].std()

print(f'\n  Training Revenue: mean={rev_train_mean:,.0f}  std={rev_train_std:,.0f}')
print(f'  Training COGS   : mean={cogs_train_mean:,.0f}  std={cogs_train_std:,.0f}')

# ── 1b. Raw predictions từ Phase C ───────────────────────────────────────────
raw_path = ART_DIR / 'phase_c_submission_raw.csv'
if not raw_path.exists():
    raise FileNotFoundError(
        f'Không tìm thấy {raw_path}. Hãy chạy Phase C trước.'
    )
raw = pd.read_csv(raw_path, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
print(f'\n  Raw predictions : {raw.shape}  ({raw["Date"].min().date()} → {raw["Date"].max().date()})')
print(f'  Revenue raw stats: mean={raw["Revenue"].mean():,.0f}  '
      f'min={raw["Revenue"].min():,.0f}  max={raw["Revenue"].max():,.0f}')
print(f'  COGS    raw stats: mean={raw["COGS"].mean():,.0f}  '
      f'min={raw["COGS"].min():,.0f}  max={raw["COGS"].max():,.0f}')

# ── 1c. Sample submission (template) ─────────────────────────────────────────
sample_sub = pd.read_csv(DATA_DIR / 'sample_submission.csv', parse_dates=['Date']).sort_values('Date')
print(f'\n  Sample submission: {sample_sub.shape}')
print(f'  Columns          : {list(sample_sub.columns)}')

# ── 1d. Seasonal profiles từ Phase B (nếu tồn tại) ───────────────────────────
profiles_path = ART_DIR / 'phase_b_seasonal_profiles.pkl'
seasonal_profiles = None
if profiles_path.exists():
    with open(profiles_path, 'rb') as f:
        seasonal_profiles = pickle.load(f)
    print(f'\n  ✅ Seasonal profiles loaded: {list(seasonal_profiles.keys())}')
else:
    print('\n  ⚠️  phase_b_seasonal_profiles.pkl not found — sẽ tính lại từ train.')

# ── 1e. OOF & CV report (cho plots) ──────────────────────────────────────────
oof_path = ART_DIR / 'oof_predictions.csv'
cv_path  = ART_DIR / 'phase_c_cv_report.csv'
oof_df   = pd.read_csv(oof_path, parse_dates=['Date']) if oof_path.exists() else None
cv_df    = pd.read_csv(cv_path) if cv_path.exists() else None

if oof_df is not None:
    print(f'  ✅ OOF predictions: {oof_df.shape}')
if cv_df is not None:
    mr = cv_df['rev_MAPE'].mean()
    rr = cv_df['rev_R2'].mean()
    print(f'  ✅ CV report: Revenue MAPE={mr:.2f}%  R²={rr:.4f}')

print('\n  ✅ STEP 1 complete.')

# =============================================================================
# STEP 2 — REFINED POST-PROCESSING
# =============================================================================
print('\n' + '─' * 65)
print('⚙️  STEP 2: Refined post-processing pipeline...')
print('─' * 65)

df = raw.copy()
df['Revenue_raw']  = df['Revenue'].copy()
df['COGS_raw']     = df['COGS'].copy()

# Audit log — ghi lại mọi thay đổi theo từng hàng
audit_log = []

def log_change(date, field, old_val, new_val, reason):
    audit_log.append({
        'Date': date, 'field': field,
        'old_value': old_val, 'new_value': new_val,
        'change': new_val - old_val, 'reason': reason
    })

# ── 2a. Hard floor: không có giá trị âm ─────────────────────────────────────
print('\n  [2a] Hard floor: clip negatives to 0...')
neg_rev  = (df['Revenue'] < 0).sum()
neg_cogs = (df['COGS'] < 0).sum()
for idx in df[df['Revenue'] < 0].index:
    log_change(df.loc[idx,'Date'], 'Revenue', df.loc[idx,'Revenue'], 0, 'negative_clip')
for idx in df[df['COGS'] < 0].index:
    log_change(df.loc[idx,'Date'], 'COGS', df.loc[idx,'COGS'], 0, 'negative_clip')
df['Revenue'] = df['Revenue'].clip(lower=0)
df['COGS']    = df['COGS'].clip(lower=0)
print(f'    Revenue negatives clipped : {neg_rev}')
print(f'    COGS negatives clipped    : {neg_cogs}')

# ── [2b] Upper cap: đã được xử lý trong Phase C sau level calibration.
# Phase D KHÔNG double-clip để tránh kéo seasonal peaks về mean.
print('  [2b] Upper cap: handled in Phase C — skipped to preserve seasonal peaks ✅')

# ── 2c. Seasonal COGS margin constraint ──────────────────────────────────────
#   Dùng seasonal_profiles từ Phase B nếu có;
#   Nếu không, tính margin_mean_month từ training data
print('\n  [2c] Seasonal COGS margin constraint...')

if seasonal_profiles and 'margin_by_month' in seasonal_profiles:
    margin_by_month = seasonal_profiles['margin_by_month']
    print('    Using Phase B seasonal margin profiles.')
else:
    # Tính lại từ training data (fallback)
    train['margin'] = train['COGS'] / train['Revenue']
    train['month']  = train['Date'].dt.month
    margin_by_month = train.groupby('month')['margin'].agg(['mean', 'std']).reset_index()
    margin_by_month.columns = ['month', 'margin_mean_month', 'margin_std_month']
    print('    Fallback: computed margin_by_month from training data.')

print('    Monthly margin profile (COGS/Revenue):')
if isinstance(margin_by_month, pd.DataFrame):
    _m = margin_by_month.copy()
    _m.columns = [c.replace('margin_mean_month', 'mean').replace('margin_std_month','std')
                  for c in _m.columns]
    for _, row in _m.iterrows():
        mn = row.get('mean', row.get('margin_mean_month', np.nan))
        sd = row.get('std', row.get('margin_std_month', np.nan))
        m  = int(row.get('month', row.get('month', 0)))
        print(f'      Month {m:02d}: mean_ratio={mn:.4f}  std={sd:.4f}')

# Áp dụng: với mỗi ngày, dùng expected_ratio ± 2σ để tạo soft band
df['month'] = df['Date'].dt.month
df = df.merge(margin_by_month, on='month', how='left')

# Đặt tên cột nhất quán
if 'margin_mean_month' in df.columns:
    _mean_col = 'margin_mean_month'
    _std_col  = 'margin_std_month'
elif 'mean' in df.columns:
    _mean_col = 'mean'
    _std_col  = 'std'
else:
    # fallback nếu column không khớp
    _mean_col = margin_by_month.columns[1]
    _std_col  = margin_by_month.columns[2] if len(margin_by_month.columns) > 2 else None

# Tính implied COGS ratio và áp dụng band
df['implied_ratio'] = df['COGS'] / df['Revenue'].replace(0, np.nan)
expected_upper = (df[_mean_col] + 2 * df[_std_col]).clip(upper=COGS_RATIO_CAP)
expected_lower = (df[_mean_col] - 2 * df[_std_col]).clip(lower=COGS_RATIO_FLOOR)

# Trường hợp COGS/Rev vượt band trên → kéo COGS xuống
too_high = df['implied_ratio'] > expected_upper
if too_high.sum() > 0:
    new_cogs = df.loc[too_high, 'Revenue'] * expected_upper[too_high]
    for idx in df[too_high].index:
        log_change(df.loc[idx,'Date'], 'COGS', df.loc[idx,'COGS'],
                   new_cogs.loc[idx], 'margin_band_upper')
    df.loc[too_high, 'COGS'] = new_cogs
    print(f'    COGS ratio too high (> upper band): {too_high.sum()} rows adjusted')

# Trường hợp COGS/Rev dưới band dưới → nâng COGS lên
too_low = df['implied_ratio'] < expected_lower
if too_low.sum() > 0:
    new_cogs = df.loc[too_low, 'Revenue'] * expected_lower[too_low]
    for idx in df[too_low].index:
        log_change(df.loc[idx,'Date'], 'COGS', df.loc[idx,'COGS'],
                   new_cogs.loc[idx], 'margin_band_lower')
    df.loc[too_low, 'COGS'] = new_cogs
    print(f'    COGS ratio too low (< floor band)  : {too_low.sum()} rows adjusted')

if not too_high.any() and not too_low.any():
    print('    ✅ Toàn bộ COGS/Revenue ratios nằm trong seasonal band.')

# ── 2d. Hard economic constraint: COGS < Revenue (absolute) ─────────────────
print('\n  [2d] Hard economic constraint: COGS < Revenue...')
exceed = df['COGS'] >= df['Revenue']
if exceed.sum() > 0:
    capped = df.loc[exceed, 'Revenue'] * 0.97
    for idx in df[exceed].index:
        log_change(df.loc[idx,'Date'], 'COGS', df.loc[idx,'COGS'],
                   capped.loc[idx], 'cogs_exceeds_revenue_hard_cap')
    df.loc[exceed, 'COGS'] = capped
    print(f'    ⚠️  {exceed.sum()} rows COGS ≥ Revenue → capped at Revenue × 0.97')
else:
    print('    ✅ COGS < Revenue trên toàn bộ test period.')

# ── 2e. YoY plausibility check (monthly aggregate) ───────────────────────────
print('\n  [2e] YoY plausibility check (monthly aggregates)...')

# Lấy train tail cùng kỳ
df['year_month'] = df['Date'].dt.to_period('M')
test_monthly = df.groupby('year_month')['Revenue'].sum().reset_index()
test_monthly['year']  = test_monthly['year_month'].dt.year
test_monthly['month'] = test_monthly['year_month'].dt.month

train['year_month'] = train['Date'].dt.to_period('M')
train_monthly = train.groupby('year_month')['Revenue'].sum().reset_index()
train_monthly['year']  = train_monthly['year_month'].dt.year
train_monthly['month'] = train_monthly['year_month'].dt.month

yoy_issues = []
for _, row in test_monthly.iterrows():
    yr, mo, test_val = row['year'], row['month'], row['Revenue']
    prev_rows = train_monthly[(train_monthly['year'] == yr - 1) &
                              (train_monthly['month'] == mo)]
    if len(prev_rows) == 0:
        continue
    prev_val = prev_rows.iloc[0]['Revenue']
    if prev_val == 0:
        continue
    yoy = (test_val - prev_val) / prev_val
    if yoy > MAX_YOY_GROWTH or yoy < MIN_YOY_GROWTH:
        yoy_issues.append({'period': str(row['year_month']), 'yoy_growth': yoy,
                           'test_monthly_rev': test_val, 'prev_year_rev': prev_val})

if yoy_issues:
    print(f'    ⚠️  {len(yoy_issues)} tháng có YoY growth ngoài [{MIN_YOY_GROWTH*100:.0f}%, '
          f'+{MAX_YOY_GROWTH*100:.0f}%]:')
    for iss in yoy_issues[:10]:
        print(f'      {iss["period"]}: YoY={iss["yoy_growth"]*100:+.1f}%  '
              f'(test={iss["test_monthly_rev"]:,.0f} vs prev={iss["prev_year_rev"]:,.0f})')
    # LƯU Ý: Đây là CẢNH BÁO, không tự động chỉnh
    # Quyết định can thiệp nằm trong tay analyst — Phase D chỉ log
    print('    ℹ️  YoY issues được log nhưng KHÔNG tự động điều chỉnh.')
    print('       Hãy xem chart 4 để quyết định có cần seasonal rescaling không.')
else:
    print(f'    ✅ Toàn bộ monthly YoY growth nằm trong [{MIN_YOY_GROWTH*100:.0f}%, '
          f'+{MAX_YOY_GROWTH*100:.0f}%].')

# ── 2f. Level sanity vs expected recovery trajectory ─────────────────────────
print('\n  [2f] Level sanity check vs recovery trajectory...')
_profiles_path = ART_DIR / 'phase_b_seasonal_profiles.pkl'
if _profiles_path.exists():
    with open(_profiles_path, 'rb') as _f:
        _prof = pickle.load(_f)
    _recent_cagr   = _prof.get('recent_cagr', 0.05)
    _anchor_rev    = _prof.get('anchor_rev_2022', None)
    _anchor_date   = _prof.get('anchor_date', pd.Timestamp('2022-12-31'))

    if _anchor_rev:
        # Expected daily Revenue mean dựa trên recent CAGR
        days_to_mid_test = (pd.Timestamp('2023-10-01') - _anchor_date).days
        expected_mid = _anchor_rev * ((1 + _recent_cagr) ** (days_to_mid_test / 365))
        actual_mid   = df[df['Date'].between('2023-07-01', '2023-12-31')]['Revenue'].mean()

        print(f'    Recent CAGR (2020-2022)       : {_recent_cagr*100:.2f}%/year')
        print(f'    Expected mid-test daily mean  : {expected_mid:,.0f}')
        print(f'    Actual pred mid-test mean     : {actual_mid:,.0f}')
        ratio_check = actual_mid / expected_mid if expected_mid > 0 else 1.0
        print(f'    Ratio (pred/expected)         : {ratio_check:.4f}')
        if ratio_check < 0.80:
            print('    ⚠️  Predictions thấp hơn expected 20%+ — kiểm tra calibration')
        elif ratio_check > 1.25:
            print('    ⚠️  Predictions cao hơn expected 25%+ — có thể overfit')
        else:
            print('    ✅ Level hợp lý so với recovery trajectory')

# Lưu audit log
audit_df = pd.DataFrame(audit_log)
if len(audit_df) > 0:
    audit_df.to_csv(OUT_DIR / 'phase_d_postprocess_log.csv', index=False)
    print(f'\n  📋 Audit log: {len(audit_df)} chỉnh sửa → phase_d_postprocess_log.csv')
else:
    print('\n  ✅ Không có chỉnh sửa nào — predictions sạch từ Phase C.')

# Summary thay đổi
rev_delta  = (df['Revenue'] - df['Revenue_raw']).abs()
cogs_delta = (df['COGS']    - df['COGS_raw']).abs()
print(f'\n  Post-processing summary:')
print(f'    Revenue: max_change={rev_delta.max():,.0f}  mean_change={rev_delta.mean():,.1f}  '
      f'rows_changed={( rev_delta > 0).sum()}')
print(f'    COGS   : max_change={cogs_delta.max():,.0f}  mean_change={cogs_delta.mean():,.1f}  '
      f'rows_changed={(cogs_delta > 0).sum()}')

print('\n  ✅ STEP 2 complete.')

# =============================================================================
# STEP 3 — SUBMISSION FORMAT VALIDATION (7 HARD GATES)
# =============================================================================
print('\n' + '─' * 65)
print('🔒 STEP 3: Submission format validation (7 hard gates)...')
print('─' * 65)

gate_results = []

def check_gate(gate_id, name, passed, detail=''):
    status = '✅ PASS' if passed else '❌ FAIL'
    gate_results.append({'gate': gate_id, 'name': name, 'passed': passed, 'detail': detail})
    print(f'  [{gate_id}] {status}  {name}')
    if detail:
        print(f'         {detail}')
    if not passed:
        raise AssertionError(f'Gate [{gate_id}] FAILED: {name} — {detail}')

# Gate 1: Row count phải khớp sample_submission
check_gate('G1', 'Row count matches sample_submission',
           len(df) == len(sample_sub),
           f'predictions={len(df)}, expected={len(sample_sub)}')

# Gate 2: Date range đúng
check_gate('G2', 'Date range correct',
           df['Date'].min() == sample_sub['Date'].min() and
           df['Date'].max() == sample_sub['Date'].max(),
           f'pred: {df["Date"].min().date()} → {df["Date"].max().date()}  '
           f'expected: {sample_sub["Date"].min().date()} → {sample_sub["Date"].max().date()}')

# Gate 3: Dates sorted ascending, no duplicates
check_gate('G3', 'Dates sorted & no duplicates',
           df['Date'].is_monotonic_increasing and not df['Date'].duplicated().any(),
           f'monotonic={df["Date"].is_monotonic_increasing}, '
           f'duplicates={df["Date"].duplicated().sum()}')

# Gate 4: No NaN in Revenue or COGS
check_gate('G4', 'No NaN in predictions',
           df['Revenue'].isna().sum() == 0 and df['COGS'].isna().sum() == 0,
           f'Revenue_NaN={df["Revenue"].isna().sum()}, COGS_NaN={df["COGS"].isna().sum()}')

# Gate 5: No negative values
check_gate('G5', 'No negative values',
           (df['Revenue'] >= 0).all() and (df['COGS'] >= 0).all(),
           f'Revenue<0: {(df["Revenue"]<0).sum()}, COGS<0: {(df["COGS"]<0).sum()}')

# Gate 6: COGS < Revenue (economic constraint)
check_gate('G6', 'COGS < Revenue (economic constraint)',
           (df['COGS'] < df['Revenue']).all(),
           f'Violations: {(df["COGS"] >= df["Revenue"]).sum()}')

# Gate 7: Date alignment với sample_submission (date-by-date match)
date_match = set(df['Date'].dt.date) == set(sample_sub['Date'].dt.date)
check_gate('G7', 'Date-by-date alignment với sample_submission',
           date_match,
           f'Symmetric diff: {len(set(df["Date"].dt.date).symmetric_difference(set(sample_sub["Date"].dt.date)))} dates')

all_passed = all(g['passed'] for g in gate_results)
print(f'\n  {"✅ ALL 7 GATES PASSED" if all_passed else "❌ SOME GATES FAILED"} — '
      f'{sum(g["passed"] for g in gate_results)}/7')

# =============================================================================
# STEP 4 — EXPORT SUBMISSION.CSV
# =============================================================================
print('\n' + '─' * 65)
print('💾 STEP 4: Export submission.csv...')
print('─' * 65)

# Format phải khớp chính xác với sample_submission
submission = pd.DataFrame({
    'Date':    df['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': df['Revenue'].round(2),
    'COGS':    df['COGS'].round(2),
})

# Đảm bảo thứ tự cột giống sample_submission
expected_cols = list(sample_sub.columns)
submission = submission[expected_cols]

sub_path = OUT_DIR / 'submission.csv'
submission.to_csv(sub_path, index=False)

# Verify file vừa ghi
verify = pd.read_csv(sub_path)
print(f'\n  File: {sub_path}')
print(f'  Rows  : {len(verify):,}')
print(f'  Cols  : {list(verify.columns)}')
print(f'  Size  : {sub_path.stat().st_size / 1024:.1f} KB')
print(f'\n  Revenue final: mean={verify["Revenue"].mean():,.0f}  '
      f'min={verify["Revenue"].min():,.0f}  max={verify["Revenue"].max():,.0f}')
print(f'  COGS    final: mean={verify["COGS"].mean():,.0f}  '
      f'min={verify["COGS"].min():,.0f}  max={verify["COGS"].max():,.0f}')
print(f'  COGS/Rev ratio: mean={(verify["COGS"]/verify["Revenue"]).mean():.4f}  '
      f'min={(verify["COGS"]/verify["Revenue"]).min():.4f}  '
      f'max={(verify["COGS"]/verify["Revenue"]).max():.4f}')

print('\n  ✅ STEP 4 complete — submission.csv exported.')

# =============================================================================
# STEP 5 — VALIDATION REPORT
# =============================================================================
print('\n' + '─' * 65)
print('📄 STEP 5: Writing validation report...')
print('─' * 65)

report_lines = [
    '=' * 65,
    'PHASE D VALIDATION REPORT — VinTelligence DATATHON 2026',
    '=' * 65,
    '',
    f'Generated : {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
    f'Test period: {TEST_START.date()} → {TEST_END.date()}',
    '',
    '─── Gate Results ───────────────────────────────────────────',
]
for g in gate_results:
    s = 'PASS' if g['passed'] else 'FAIL'
    report_lines.append(f'  [{g["gate"]}] {s}  {g["name"]}')
    if g['detail']:
        report_lines.append(f'       {g["detail"]}')

report_lines += [
    '',
    '─── Final Prediction Statistics ─────────────────────────────',
    f'  Revenue: mean={verify["Revenue"].mean():,.0f}  '
    f'std={verify["Revenue"].std():,.0f}',
    f'           min={verify["Revenue"].min():,.0f}  '
    f'max={verify["Revenue"].max():,.0f}',
    f'  COGS   : mean={verify["COGS"].mean():,.0f}  '
    f'std={verify["COGS"].std():,.0f}',
    f'           min={verify["COGS"].min():,.0f}  '
    f'max={verify["COGS"].max():,.0f}',
    f'  COGS/Rev: mean={(verify["COGS"]/verify["Revenue"]).mean():.4f}',
    '',
    '─── Post-processing Changes ─────────────────────────────────',
    f'  Total audit entries : {len(audit_log)}',
    f'  Revenue rows changed: {(rev_delta > 0).sum()}  '
    f'(max Δ = {rev_delta.max():,.0f})',
    f'  COGS rows changed   : {(cogs_delta > 0).sum()}  '
    f'(max Δ = {cogs_delta.max():,.0f})',
    '',
    '─── YoY Plausibility Issues ─────────────────────────────────',
    f'  Flagged months: {len(yoy_issues)}',
]
if yoy_issues:
    for iss in yoy_issues:
        report_lines.append(
            f'    {iss["period"]}: YoY={iss["yoy_growth"]*100:+.1f}%'
        )

if cv_df is not None:
    report_lines += [
        '',
        '─── Phase C CV Summary ──────────────────────────────────────',
        f'  Revenue MAPE (mean): {cv_df["rev_MAPE"].mean():.2f}%',
        f'  Revenue R²   (mean): {cv_df["rev_R2"].mean():.4f}',
        f'  COGS MAPE    (mean): {cv_df["cogs_MAPE"].mean():.2f}%',
        f'  COGS R²      (mean): {cv_df["cogs_R2"].mean():.4f}',
    ]

report_lines += [
    '',
    '=' * 65,
    f'STATUS: {"✅ READY TO SUBMIT" if all_passed else "❌ REVIEW REQUIRED"}',
    '=' * 65,
]

rep_path = OUT_DIR / 'phase_d_validation_report.txt'
with open(rep_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f'  📄 Saved: {rep_path}')

# =============================================================================
# STEP 6 — DIAGNOSTIC CHARTS (8 plots)
# =============================================================================
print('\n' + '─' * 65)
print('📊 STEP 6: Generating diagnostic charts...')
print('─' * 65)

# ─── Figure 1: Main diagnostics (2×4 grid) ──────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle('Phase D — Submission Diagnostics\nVinTelligence DATATHON 2026',
             fontsize=15, fontweight='bold')

# ── Chart 1: Revenue — Train tail + Test predictions ─────────────────────────
ax = axes[0, 0]
tail = train[train['Date'] >= '2020-01-01'].copy()
ax.plot(tail['Date'], tail['Revenue'] / 1e6, lw=0.9, color=C_BLUE,
        label='Actual (train tail)', alpha=0.9)
ax.plot(df['Date'], df['Revenue'] / 1e6, lw=1.2, color=C_RED,
        label='Predicted (test)', alpha=0.9)
ax.fill_between(df['Date'], df['Revenue'] / 1e6, alpha=0.08, color=C_RED)
ax.axvline(TEST_START, color='black', lw=1.5, ls='--', label='Leakage wall')
ax.set_title('Revenue: Train Tail → Test Predictions')
ax.set_ylabel('Revenue (triệu VND)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

# ── Chart 2: COGS — Train tail + Test predictions ────────────────────────────
ax = axes[0, 1]
ax.plot(tail['Date'], tail['COGS'] / 1e6, lw=0.9, color=C_AMBER,
        label='Actual (train tail)', alpha=0.9)
ax.plot(df['Date'], df['COGS'] / 1e6, lw=1.2, color=C_PURPLE,
        label='Predicted (test)', alpha=0.9)
ax.fill_between(df['Date'], df['COGS'] / 1e6, alpha=0.08, color=C_PURPLE)
ax.axvline(TEST_START, color='black', lw=1.5, ls='--', label='Leakage wall')
ax.set_title('COGS: Train Tail → Test Predictions')
ax.set_ylabel('COGS (triệu VND)')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

# ── Chart 3: COGS/Revenue ratio — train vs test ───────────────────────────────
ax = axes[0, 2]
train_ratio = (train['COGS'] / train['Revenue']).rolling(30).mean()
test_ratio  = (df['COGS'] / df['Revenue'])
ax.plot(train['Date'], train_ratio, lw=0.8, color=C_BLUE, alpha=0.6,
        label='Train (30d rolling)')
ax.plot(df['Date'], test_ratio, lw=1.0, color=C_RED, alpha=0.8,
        label='Test predicted')
ax.axhline(train['COGS'].sum() / train['Revenue'].sum(), color='navy',
           lw=1.5, ls='--', label=f'Train mean = {train["COGS"].sum()/train["Revenue"].sum():.3f}')
ax.axvline(TEST_START, color='black', lw=1.5, ls='--')
ax.set_title('COGS/Revenue Ratio Continuity')
ax.set_ylabel('COGS / Revenue')
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Chart 4: Post-processing diff — Revenue ──────────────────────────────────
ax = axes[0, 3]
diff_rev = (df['Revenue'] - df['Revenue_raw']) / 1e6
ax.bar(df['Date'], diff_rev, color=np.where(diff_rev >= 0, C_GREEN, C_RED),
       alpha=0.7, width=1.0)
ax.axhline(0, color='black', lw=0.8)
ax.set_title(f'Revenue Post-process Δ\n({(diff_rev!=0).sum()} rows changed, '
             f'max={diff_rev.abs().max():.2f}M)')
ax.set_ylabel('Δ Revenue (triệu VND)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

# ── Chart 5: Monthly Revenue bar — train historical vs test forecast ──────────
ax = axes[1, 0]
# Mean monthly Revenue by month (2019-2022) vs test monthly mean
train_monthly_avg = (train[train['Date'] >= '2019-01-01']
                     .assign(month=lambda x: x['Date'].dt.month)
                     .groupby('month')['Revenue'].mean())
test_monthly_avg = (df.assign(month=lambda x: x['Date'].dt.month)
                    .groupby('month')['Revenue'].mean())
months = range(1, 13)
w = 0.38
ax.bar([m - w/2 for m in months],
       [train_monthly_avg.get(m, 0) / 1e6 for m in months],
       w, label='Train avg (2019-2022)', color=C_BLUE, alpha=0.8)
ax.bar([m + w/2 for m in months],
       [test_monthly_avg.get(m, 0) / 1e6 for m in months],
       w, label='Test forecast', color=C_RED, alpha=0.8)
ax.set_xticks(list(months))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=8)
ax.set_title('Monthly Revenue: Historical Avg vs Test Forecast')
ax.set_ylabel('Mean Revenue (triệu VND)')
ax.legend(fontsize=9)

# ── Chart 6: YoY monthly growth in test period ────────────────────────────────
ax = axes[1, 1]
yoy_data = []
for _, row in test_monthly.iterrows():
    yr, mo, test_val = row['year'], row['month'], row['Revenue']
    prev_rows = train_monthly[(train_monthly['year'] == yr - 1) &
                              (train_monthly['month'] == mo)]
    if len(prev_rows) > 0 and prev_rows.iloc[0]['Revenue'] > 0:
        yoy = (test_val - prev_rows.iloc[0]['Revenue']) / prev_rows.iloc[0]['Revenue'] * 100
        yoy_data.append({'period': str(row['year_month']), 'yoy': yoy})

if yoy_data:
    yoy_plot = pd.DataFrame(yoy_data)
    colors = [C_GREEN if v >= 0 else C_RED for v in yoy_plot['yoy']]
    ax.bar(range(len(yoy_plot)), yoy_plot['yoy'], color=colors, alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(MAX_YOY_GROWTH * 100, color=C_RED, ls='--', lw=1,
               label=f'Upper guard (+{MAX_YOY_GROWTH*100:.0f}%)')
    ax.axhline(MIN_YOY_GROWTH * 100, color=C_RED, ls='--', lw=1,
               label=f'Lower guard ({MIN_YOY_GROWTH*100:.0f}%)')
    ax.set_xticks(range(len(yoy_plot)))
    ax.set_xticklabels(yoy_plot['period'], rotation=45, ha='right', fontsize=7)
    ax.set_title('Monthly YoY Growth Rate (Test vs Same Month Prior Year)')
    ax.set_ylabel('YoY Growth (%)')
    ax.legend(fontsize=8)

# ── Chart 7: OOF residuals (từ Phase C) ──────────────────────────────────────
ax = axes[1, 2]
if oof_df is not None:
    res = (oof_df['Revenue_actual'] - oof_df['Revenue_pred']) / 1e6
    ax.hist(res.clip(lower=res.quantile(0.005), upper=res.quantile(0.995)),
            bins=55, color=C_BLUE, alpha=0.8, edgecolor='white')
    ax.axvline(0, color=C_GREEN, lw=1.5, ls='--', label='Zero residual')
    ax.axvline(res.median(), color=C_RED, lw=1.5, ls='-',
               label=f'Median={res.median():.2f}M')
    ax.set_title(f'OOF Residuals (Revenue)\nPhase C CV ({len(oof_df):,} obs)')
    ax.set_xlabel('Residual (triệu VND)')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=9)
else:
    ax.text(0.5, 0.5, 'OOF not available\n(run Phase C first)',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('OOF Residuals (N/A)')

# ── Chart 8: Gate validation result panel ────────────────────────────────────
ax = axes[1, 3]
ax.axis('off')
gate_text = ['Submission Validation Gates\n']
for g in gate_results:
    icon = '✅' if g['passed'] else '❌'
    gate_text.append(f'{icon}  [{g["gate"]}]  {g["name"]}')
gate_text.append(f'\n{"✅ ALL PASSED" if all_passed else "❌ REVIEW NEEDED"}'
                 f'  ({sum(g["passed"] for g in gate_results)}/7)')
ax.text(0.05, 0.95, '\n'.join(gate_text),
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f8ff', alpha=0.8))

plt.tight_layout()
chart_path_1 = OUT_DIR / 'phase_d_submission_diagnostics.png'
plt.savefig(chart_path_1, bbox_inches='tight', dpi=130)
plt.show()
print(f'  📊 Chart 1 saved: {chart_path_1}')

# ─── Figure 2: Monthly comparison detail ────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Phase D — Monthly Revenue & COGS Comparison\nVinTelligence DATATHON 2026',
              fontsize=13, fontweight='bold')

# Revenue monthly
ax = axes2[0]
train_m_full = (train.assign(ym=lambda x: x['Date'].dt.to_period('M'))
                .groupby('ym')[['Revenue', 'COGS']].sum().reset_index())
train_m_full['Date'] = train_m_full['ym'].dt.to_timestamp()

test_m_full = (df.assign(ym=lambda x: x['Date'].dt.to_period('M'))
               .groupby('ym')[['Revenue', 'COGS']].sum().reset_index())
test_m_full['Date'] = test_m_full['ym'].dt.to_timestamp()

ax.plot(train_m_full['Date'], train_m_full['Revenue'] / 1e6,
        color=C_BLUE, lw=1.5, label='Actual (train)', alpha=0.9)
ax.plot(test_m_full['Date'],  test_m_full['Revenue'] / 1e6,
        color=C_RED,  lw=2.0, label='Predicted (test)', alpha=0.9)
ax.fill_between(test_m_full['Date'], test_m_full['Revenue'] / 1e6,
                alpha=0.1, color=C_RED)
ax.axvline(TEST_START, color='black', lw=1.5, ls='--', label='Leakage wall')
ax.set_title('Monthly Revenue: Full Timeline')
ax.set_ylabel('Revenue (triệu VND)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# COGS monthly
ax = axes2[1]
ax.plot(train_m_full['Date'], train_m_full['COGS'] / 1e6,
        color=C_AMBER, lw=1.5, label='Actual (train)', alpha=0.9)
ax.plot(test_m_full['Date'],  test_m_full['COGS'] / 1e6,
        color=C_PURPLE, lw=2.0, label='Predicted (test)', alpha=0.9)
ax.fill_between(test_m_full['Date'], test_m_full['COGS'] / 1e6,
                alpha=0.1, color=C_PURPLE)
ax.axvline(TEST_START, color='black', lw=1.5, ls='--', label='Leakage wall')
ax.set_title('Monthly COGS: Full Timeline')
ax.set_ylabel('COGS (triệu VND)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
chart_path_2 = OUT_DIR / 'phase_d_monthly_comparison.png'
plt.savefig(chart_path_2, bbox_inches='tight', dpi=130)
plt.show()
print(f'  📊 Chart 2 saved: {chart_path_2}')

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print('\n' + '=' * 65)
print('✅ PHASE D COMPLETE — Submission Ready')
print('=' * 65)
print(f'\n  📦 submission.csv                    : {len(submission):,} rows')
print(f'  📋 phase_d_postprocess_log.csv       : {len(audit_log)} edits')
print(f'  📄 phase_d_validation_report.txt     : '
      f'{"ALL GATES PASSED" if all_passed else "REVIEW NEEDED"}')
print(f'  📊 phase_d_submission_diagnostics.png')
print(f'  📊 phase_d_monthly_comparison.png')

print(f'\n  Final submission stats:')
print(f'    Revenue : mean={verify["Revenue"].mean():>15,.0f}  '
      f'std={verify["Revenue"].std():>12,.0f}')
print(f'    COGS    : mean={verify["COGS"].mean():>15,.0f}  '
      f'std={verify["COGS"].std():>12,.0f}')
ratio_s = verify['COGS'] / verify['Revenue']
print(f'    COGS/Rev: mean={ratio_s.mean():.4f}  '
      f'min={ratio_s.min():.4f}  max={ratio_s.max():.4f}')

if cv_df is not None:
    print(f'\n  Phase C CV recap:')
    print(f'    Revenue MAPE={cv_df["rev_MAPE"].mean():.2f}%  '
          f'R²={cv_df["rev_R2"].mean():.4f}')
    print(f'    COGS MAPE={cv_df["cogs_MAPE"].mean():.2f}%  '
          f'R²={cv_df["cogs_R2"].mean():.4f}')

print(f'\n  ▶ Sẵn sàng cho Phase E: XAI Explainability Report')
print('=' * 65)