# =============================================================================
# PHASE B — FEATURE STORE BUILDER
# VinTelligence DATATHON 2026 | Phần 3: Sales Forecasting
#
# Mục tiêu: Biến đổi raw data thành feature matrix KHÔNG rò rỉ dữ liệu
#
# Kết quả từ Phase A:
#   WEB_TRAFFIC_STRATEGY = "SEASONAL_PROXY"
#   COGS_STRATEGY        = "INDEPENDENT_MODEL"
#   MISSING_DATE_FLAG    = False
#
# Output:
#   feature_store_train.csv       — train rows có Revenue + COGS target
#   feature_store_test.csv        — test rows, target = NaN (chờ predict)
#   phase_b_seasonal_profiles.pkl — fitted profiles để dùng lại ở Phase E/XAI
#   phase_b_feature_schema.csv    — tên + mô tả + leakage risk của mọi feature
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ── Ranh giới thời gian (KHÔNG THAY ĐỔI) ────────────────────────────────────
TRAIN_START = pd.Timestamp('2012-07-04')
TRAIN_END   = pd.Timestamp('2022-12-31')
TEST_START  = pd.Timestamp('2023-01-01')
TEST_END    = pd.Timestamp('2024-07-01')

# ── Breakpoints từ Phase A (xác nhận từ annual YoY growth) ──────────────────
# 2019: ~-40% YoY, 2021: recovery +x%
BP_DOWN = pd.Timestamp('2019-01-01')   # bắt đầu suy giảm
BP_UP   = pd.Timestamp('2021-01-01')   # bắt đầu phục hồi

print('=' * 65)
print('  PHASE B — FEATURE STORE BUILDER')
print('  VinTelligence DATATHON 2026')
print('=' * 65)
print(f'  RUN_ENV      : {RUN_ENV}')
print(f'  DATA_DIR     : {DATA_DIR}')
print(f'  OUT_DIR      : {OUT_DIR}')
print(f'  Leakage wall : {TEST_START.date()}')
print(f'  Breakpoints  : {BP_DOWN.date()} (down), {BP_UP.date()} (up)')

# =============================================================================
# SECTION 1 — LOAD RAW DATA
# =============================================================================
print('\n📂 SECTION 1: Loading raw data...')

sales = pd.read_csv(
    DATA_DIR / 'sales.csv', parse_dates=['Date']
).sort_values('Date').reset_index(drop=True)

web_traffic = pd.read_csv(
    DATA_DIR / 'web_traffic.csv', parse_dates=['date']
).rename(columns={'date': 'Date'})

orders = pd.read_csv(
    DATA_DIR / 'orders.csv', parse_dates=['order_date']
)

sample_sub = pd.read_csv(
    DATA_DIR / 'sample_submission.csv', parse_dates=['Date']
).sort_values('Date').reset_index(drop=True)

# Tách train / kiểm tra khoảng
train = sales[sales['Date'] <= TRAIN_END].copy()
assert len(train) == len(sales), "sales.csv phải chỉ chứa training data"
assert train['Date'].max() == TRAIN_END, f"Train không kết thúc đúng {TRAIN_END.date()}"
assert not train['Date'].duplicated().any(), "Có duplicate dates trong training data"

print(f'  Train : {train["Date"].min().date()} → {train["Date"].max().date()} ({len(train):,} rows)')
print(f'  Test  : {TEST_START.date()} → {TEST_END.date()} ({len(sample_sub):,} rows)')
print(f'  WebTF : {web_traffic["Date"].min().date()} → {web_traffic["Date"].max().date()} ({len(web_traffic):,} rows)')
print(f'  Order : {orders["order_date"].min().date()} → {orders["order_date"].max().date()} ({len(orders):,} rows)')

# =============================================================================
# SECTION 2 — XÂY DỰNG COMBINED TIMELINE
# =============================================================================
print('\n🗓️  SECTION 2: Building combined timeline...')

# Test dates từ sample_submission
test_dates_df = sample_sub[['Date']].copy()
test_dates_df['Revenue'] = np.nan
test_dates_df['COGS']    = np.nan
test_dates_df['_split']  = 'test'

train_full = train.copy()
train_full['_split'] = 'train'

# Ghép lại và sort theo Date
combined = pd.concat(
    [train_full[['Date', 'Revenue', 'COGS', '_split']],
     test_dates_df[['Date', 'Revenue', 'COGS', '_split']]],
    ignore_index=True
).sort_values('Date').reset_index(drop=True)

# Kiểm tra không có overlap
overlap = combined[combined['Date'].duplicated(keep=False)]
assert len(overlap) == 0, f"Phát hiện {len(overlap)} dates trùng nhau giữa train và test!"

print(f'  Combined: {combined["Date"].min().date()} → {combined["Date"].max().date()} ({len(combined):,} rows)')
print(f'  Train rows: {(combined["_split"]=="train").sum():,}')
print(f'  Test rows : {(combined["_split"]=="test").sum():,}')

# Lookup nhanh từ training (dùng cho lag features)
_train_rev_lookup  = train.set_index('Date')['Revenue']
_train_cogs_lookup = train.set_index('Date')['COGS']

# =============================================================================
# SECTION 3 — GROUP A: CALENDAR FEATURES
# =============================================================================
print('\n📅 SECTION 3: Group A — Calendar Features...')

df = combined.copy()

df['year']             = df['Date'].dt.year
df['month']            = df['Date'].dt.month
df['day']              = df['Date'].dt.day
df['day_of_week']      = df['Date'].dt.dayofweek          # 0=Mon, 6=Sun
df['day_of_year']      = df['Date'].dt.dayofyear
df['week_of_year']     = df['Date'].dt.isocalendar().week.astype(int)
df['quarter']          = df['Date'].dt.quarter
df['is_weekend']       = (df['day_of_week'] >= 5).astype(int)
df['is_month_start']   = df['Date'].dt.is_month_start.astype(int)
df['is_month_end']     = df['Date'].dt.is_month_end.astype(int)
df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype(int)
df['is_quarter_end']   = df['Date'].dt.is_quarter_end.astype(int)

CALENDAR_FEATURES = [
    'year', 'month', 'day', 'day_of_week', 'day_of_year',
    'week_of_year', 'quarter',
    'is_weekend', 'is_month_start', 'is_month_end',
    'is_quarter_start', 'is_quarter_end',
]
print(f'  ✅ {len(CALENDAR_FEATURES)} calendar features added.')

# =============================================================================
# SECTION 4 — GROUP B: FOURIER FEATURES (Cyclical Encoding)
# =============================================================================
print('\n🌊 SECTION 4: Group B — Fourier Features...')

FOURIER_FEATURES = []

def add_fourier(df, period, harmonics, label):
    """Thêm sin/cos Fourier pairs cho một chu kỳ."""
    feats = []
    t = df['day_of_year']                   # dùng day_of_year làm phase reference
    for k in range(1, harmonics + 1):
        sin_name = f'fourier_{label}_sin_{k}'
        cos_name = f'fourier_{label}_cos_{k}'
        df[sin_name] = np.sin(2 * np.pi * k * t / period)
        df[cos_name] = np.cos(2 * np.pi * k * t / period)
        feats += [sin_name, cos_name]
    return df, feats

# Annual seasonality (T=365.25, 3 harmonics → 6 features)
df, feats_annual = add_fourier(df, period=365.25, harmonics=3, label='annual')
FOURIER_FEATURES += feats_annual

# Weekly seasonality (T=7, dùng day_of_week làm phase, 2 harmonics → 4 features)
t_dow = df['day_of_week']
for k in range(1, 3):
    sin_name = f'fourier_weekly_sin_{k}'
    cos_name = f'fourier_weekly_cos_{k}'
    df[sin_name] = np.sin(2 * np.pi * k * t_dow / 7)
    df[cos_name] = np.cos(2 * np.pi * k * t_dow / 7)
    FOURIER_FEATURES += [sin_name, cos_name]

print(f'  ✅ {len(FOURIER_FEATURES)} Fourier features added.')
print(f'     Annual (3 harmonics): {feats_annual}')
print(f'     Weekly (2 harmonics): {[f for f in FOURIER_FEATURES if "weekly" in f]}')

# =============================================================================
# SECTION 5 — GROUP C: TREND FEATURES (Piecewise Linear)
# =============================================================================
print('\n📈 SECTION 5: Group C — Piecewise Linear Trend...')

# t_idx: số ngày kể từ ngày đầu training (int, 0-based)
t0 = TRAIN_START
df['t_idx'] = (df['Date'] - t0).dt.days.astype(float)

# Vị trí breakpoints trên trục t_idx
t_bp_down = (BP_DOWN - t0).days
t_bp_up   = (BP_UP   - t0).days

df['t_break_down'] = np.maximum(0.0, df['t_idx'] - t_bp_down)  # 0 trước 2019
df['t_break_up']   = np.maximum(0.0, df['t_idx'] - t_bp_up)    # 0 trước 2021

TREND_FEATURES = ['t_idx', 't_break_down', 't_break_up']

print(f'  t_idx range  : [{df["t_idx"].min():.0f}, {df["t_idx"].max():.0f}]')
print(f'  BP_DOWN at t = {t_bp_down} (2019-01-01)')
print(f'  BP_UP   at t = {t_bp_up} (2021-01-01)')
print(f'  ✅ {len(TREND_FEATURES)} trend features added: {TREND_FEATURES}')

# ── Recent CAGR feature (chỉ dùng 2020-2022 recovery period) ─────────────────
# Lý do: CAGR 2013-2022 = -3.8% (bị kéo bởi COVID crash 2019-2020)
# Recovery CAGR 2020-2022 ≈ +5-12%, phản ánh đúng momentum vào 2023-2024
_train_annual = (
    df[df['_split'] == 'train']
    .assign(year=lambda x: x['Date'].dt.year)
    .groupby('year')['Revenue'].sum()
)
_recent_years = _train_annual.loc[_train_annual.index >= 2020]
if len(_recent_years) >= 2:
    _recent_cagr = (
        (_recent_years.iloc[-1] / _recent_years.iloc[0])
        ** (1.0 / (len(_recent_years) - 1))
    ) - 1.0
else:
    _recent_cagr = 0.05  # fallback 5%

# Anchor = mean daily Revenue của 2022 (năm cuối, recovery level)
_anchor_rev  = df.loc[(df['_split']=='train') & (df['year']==2022), 'Revenue'].mean()
_anchor_cogs = df.loc[(df['_split']=='train') & (df['year']==2022), 'COGS'].mean()
_anchor_date = pd.Timestamp('2022-12-31')

# Projected trend cho mỗi ngày (train + test) dựa trên recent CAGR
days_from_anchor   = (df['Date'] - _anchor_date).dt.days
df['recent_trend_rev']  = _anchor_rev  * ((1 + _recent_cagr) ** (days_from_anchor / 365.0))
df['recent_trend_cogs'] = _anchor_cogs * ((1 + _recent_cagr) ** (days_from_anchor / 365.0))

# Revenue detrended ratio = Revenue / recent_trend (encode pure seasonality)
# Chỉ compute cho train rows (test rows dùng để predict ratio rồi × trend)
_train_m = df['_split'] == 'train'
df.loc[_train_m, 'rev_detrend_ratio']  = (
    df.loc[_train_m, 'Revenue'] / df.loc[_train_m, 'recent_trend_rev'].replace(0, np.nan)
)
df.loc[_train_m, 'cogs_detrend_ratio'] = (
    df.loc[_train_m, 'COGS']    / df.loc[_train_m, 'recent_trend_rev'].replace(0, np.nan)
)
df['rev_detrend_ratio']  = df.get('rev_detrend_ratio',  np.nan)
df['cogs_detrend_ratio'] = df.get('cogs_detrend_ratio', np.nan)

# Seasonal ratio profile: mean(rev_detrend_ratio | day_of_year) từ train
_ratio_seasonal = (
    df[_train_m]
    .groupby('day_of_year')['rev_detrend_ratio']
    .agg(rev_ratio_doy='mean', rev_ratio_std_doy='std')
    .reset_index()
)
df = df.merge(_ratio_seasonal, on='day_of_year', how='left')

TREND_FEATURES += ['recent_trend_rev', 'recent_trend_cogs',
                   'rev_ratio_doy', 'rev_ratio_std_doy']

print(f'  Recent CAGR (2020-2022) : {_recent_cagr*100:.2f}%/year')
print(f'  Anchor daily rev (2022) : {_anchor_rev:,.0f}')
print(f'  recent_trend @ 2024-07  : {df[df["Date"]=="2024-07-01"]["recent_trend_rev"].values[0]:,.0f}')

# =============================================================================
# SECTION 6 — GROUP D: SEASONAL PROFILES (fit on TRAIN ONLY, apply to ALL)
# =============================================================================
print('\n🗓️  SECTION 6: Group D — Seasonal Profiles (fit on train)...')

# ── Lấy phần train để fit profiles ──────────────────────────────────────────
train_mask = df['_split'] == 'train'
_train_df  = df[train_mask].copy()

# ── D1: Mean Revenue/COGS by day_of_year ────────────────────────────────────
seasonal_doy = (
    _train_df
    .groupby('day_of_year')[['Revenue', 'COGS']]
    .agg(
        rev_mean_doy  = ('Revenue', 'mean'),
        rev_std_doy   = ('Revenue', 'std'),
        cogs_mean_doy = ('COGS', 'mean'),
        cogs_std_doy  = ('COGS', 'std'),
    )
    .reset_index()
)
# Xử lý leap year: ngày 366 có thể không có dữ liệu trong nhiều năm
# Fill ngày 366 bằng mean của ngày 365 nếu thiếu
if 366 not in seasonal_doy['day_of_year'].values:
    row_365 = seasonal_doy[seasonal_doy['day_of_year'] == 365].copy()
    row_365['day_of_year'] = 366
    seasonal_doy = pd.concat([seasonal_doy, row_365], ignore_index=True)

df = df.merge(seasonal_doy, on='day_of_year', how='left')

# ── D2: Mean Revenue/COGS by (month, day_of_week) ───────────────────────────
seasonal_mdow = (
    _train_df
    .groupby(['month', 'day_of_week'])[['Revenue', 'COGS']]
    .agg(
        rev_mean_month_dow  = ('Revenue', 'mean'),
        cogs_mean_month_dow = ('COGS', 'mean'),
    )
    .reset_index()
)
df = df.merge(seasonal_mdow, on=['month', 'day_of_week'], how='left')

# ── D3: Gross margin ratio by month (COGS/Revenue) ──────────────────────────
_train_df['margin_ratio'] = _train_df['COGS'] / _train_df['Revenue']
seasonal_margin = (
    _train_df
    .groupby('month')['margin_ratio']
    .agg(margin_mean_month='mean', margin_std_month='std')
    .reset_index()
)
df = df.merge(seasonal_margin, on='month', how='left')

# ── D4: Relative seasonal index (Rev / annual_mean per year, then avg by doy)
annual_mean = _train_df.groupby('year')['Revenue'].transform('mean')
_train_df['rev_seasonal_idx'] = _train_df['Revenue'] / annual_mean
seasonal_idx = (
    _train_df
    .groupby('day_of_year')['rev_seasonal_idx']
    .mean()
    .reset_index()
    .rename(columns={'rev_seasonal_idx': 'rev_seasonal_index'})
)
df = df.merge(seasonal_idx, on='day_of_year', how='left')

SEASONAL_FEATURES = [
    # Absolute-level features bị loại (gây systematic bias khi trend ngoại suy):
    # 'rev_mean_doy', 'rev_std_doy', 'cogs_mean_doy', 'cogs_std_doy'
    'rev_mean_month_dow',   # tương tác month × dow (relative shape, giữ lại)
    'cogs_mean_month_dow',
    'margin_mean_month',    # COGS/Revenue ratio theo tháng
    'margin_std_month',
    'rev_seasonal_index',   # Revenue / annual_mean (pure shape, không encode level)
    'rev_ratio_doy',        # mean(Revenue/recent_trend | doy) — shape từ recovery period
    'rev_ratio_std_doy',    # uncertainty của ratio
]
print(f'  ✅ {len(SEASONAL_FEATURES)} seasonal profile features added.')
print(f'     D1 (doy profiles)     : rev_mean_doy, rev_std_doy, cogs_mean_doy, cogs_std_doy')
print(f'     D2 (month×dow)        : rev_mean_month_dow, cogs_mean_month_dow')
print(f'     D3 (margin by month)  : margin_mean_month, margin_std_month')
print(f'     D4 (seasonal index)   : rev_seasonal_index')

# Kiểm tra null sau merge
_null_check = df[SEASONAL_FEATURES].isnull().sum()
if _null_check.sum() > 0:
    print(f'  ⚠️  Null trong seasonal features:\n{_null_check[_null_check>0]}')
else:
    print('  ✅ Không có null trong seasonal features.')

# =============================================================================
# SECTION 7 — GROUP E: LAG FEATURES (date-matched, leak-free)
# =============================================================================
print('\n⏪ SECTION 7: Group E — Lag Features (date-matched)...')

# Hàm map lag chính xác theo ngày (tránh lỗi integer shift)
# Chỉ tra cứu từ _train_rev_lookup và _train_cogs_lookup
# → Tự động NaN nếu ngày lag nằm trong test (không có trong lookup)

def lag_by_days(date_series, days, lookup):
    """
    Map mỗi Date → value tại (Date - days) từ lookup dict.
    NaN nếu ngày lag không có trong lookup (e.g. trước training start hoặc test period).
    """
    lag_dates = date_series - pd.Timedelta(days=days)
    return lag_dates.map(lookup)

# ── Lag 365 (~ 1 năm trước, an toàn với test 2023, NaN với test 2024) ────────
df['lag_365_rev']  = lag_by_days(df['Date'], 365, _train_rev_lookup)
df['lag_365_cogs'] = lag_by_days(df['Date'], 365, _train_cogs_lookup)

# ── Lag 730 (~ 2 năm trước, an toàn với TOÀN BỘ test period) ─────────────────
df['lag_730_rev']  = lag_by_days(df['Date'], 730, _train_rev_lookup)
df['lag_730_cogs'] = lag_by_days(df['Date'], 730, _train_cogs_lookup)

# ── Lag 1095 (~ 3 năm trước, fallback cho test 2024) ─────────────────────────
df['lag_1095_rev']  = lag_by_days(df['Date'], 1095, _train_rev_lookup)
df['lag_1095_cogs'] = lag_by_days(df['Date'], 1095, _train_cogs_lookup)

# ── YoY momentum ratio ────────────────────────────────────────────────────────
# Đo tốc độ tăng trưởng tại cùng kỳ: lag_365/lag_730 = performance this year vs last year
df['yoy_rev_ratio']  = df['lag_365_rev']  / df['lag_730_rev'].replace(0, np.nan)
df['yoy_cogs_ratio'] = df['lag_365_cogs'] / df['lag_730_cogs'].replace(0, np.nan)

LAG_FEATURES = [
    'lag_365_rev', 'lag_365_cogs',
    'lag_730_rev', 'lag_730_cogs',
    'lag_1095_rev', 'lag_1095_cogs',
    'yoy_rev_ratio', 'yoy_cogs_ratio',
]

# ── Kiểm tra leakage: lag_365 của test 2024 phải là NaN ─────────────────────
test_2024_mask = (df['_split'] == 'test') & (df['Date'] >= pd.Timestamp('2024-01-01'))
lag365_test2024_nonnan = df.loc[test_2024_mask, 'lag_365_rev'].notna().sum()
assert lag365_test2024_nonnan == 0, \
    f"⚠️ LEAKAGE: lag_365_rev có {lag365_test2024_nonnan} non-NaN trong test 2024!"

lag730_test_nonnan  = df.loc[df['_split']=='test', 'lag_730_rev'].notna().sum()
lag730_test_total   = (df['_split']=='test').sum()
print(f'  lag_365 → NaN trong test 2024 : {lag365_test2024_nonnan} (expected 0) ✅')
print(f'  lag_730 → non-NaN trong test  : {lag730_test_nonnan}/{lag730_test_total} rows ✅')
print(f'  ✅ {len(LAG_FEATURES)} lag features added: {LAG_FEATURES}')

# =============================================================================
# SECTION 8 — GROUP F: WEB TRAFFIC SEASONAL PROXY
# =============================================================================
print('\n🌐 SECTION 8: Group F — Web Traffic Seasonal Proxy...')

# Chỉ dùng web_traffic từ training period (tránh mọi rủi ro)
wt_train = web_traffic[web_traffic['Date'] <= TRAIN_END].copy()
wt_train['day_of_year']  = wt_train['Date'].dt.dayofyear
wt_train['month']        = wt_train['Date'].dt.month
wt_train['day_of_week']  = wt_train['Date'].dt.dayofweek

# Reconstruct true daily conversion rate từ orders.csv (không dùng proxy metrics).
# Steps: lọc valid orders -> đếm unique order_id theo ngày -> merge vào web traffic daily
# -> conversion_rate = total_orders / sessions (safe divide + clip [0,1]).
orders_train = orders[orders['order_date'] <= TRAIN_END].copy()
valid_orders = orders_train[
    orders_train['order_status'].astype(str).str.strip().str.lower().ne('cancelled')
].copy()

daily_total_orders = (
    valid_orders
    .groupby('order_date', as_index=False)['order_id']
    .nunique()
    .rename(columns={'order_id': 'total_orders'})
)

# Aggregate daily web traffic: cộng sessions/visitors, lấy mean bounce
wt_daily = (
    wt_train
    .groupby('Date')
    .agg(
        sessions_daily        = ('sessions',        'sum'),
        bounce_rate_daily     = ('bounce_rate',     'mean'),
        unique_visitors_daily = ('unique_visitors', 'sum'),
    )
    .reset_index()
)

# Left merge theo ngày và tính conversion_rate strict = total_orders / sessions
wt_daily = wt_daily.merge(
    daily_total_orders,
    how='left',
    left_on='Date',
    right_on='order_date',
)
wt_daily['total_orders'] = wt_daily['total_orders'].fillna(0)
wt_daily['conversion_rate_daily'] = (
    wt_daily['total_orders']
    / wt_daily['sessions_daily'].replace(0, np.nan)
).clip(lower=0.0, upper=1.0)
wt_daily = wt_daily.drop(columns=['total_orders', 'order_date'])

wt_daily['day_of_year'] = wt_daily['Date'].dt.dayofyear

# ── Seasonal proxy theo day_of_year (fit trên train web_traffic) ─────────────
wt_proxy_doy = (
    wt_daily
    .groupby('day_of_year')
    .agg(
        proxy_sessions_doy         = ('sessions_daily',        'mean'),
        proxy_conversion_rate_doy  = ('conversion_rate_daily', 'mean'),
        proxy_bounce_rate_doy      = ('bounce_rate_daily',     'mean'),
        proxy_unique_visitors_doy  = ('unique_visitors_daily', 'mean'),
    )
    .reset_index()
)

# Xử lý ngày 366 (leap year) tương tự Section 6
if 366 not in wt_proxy_doy['day_of_year'].values:
    row_365 = wt_proxy_doy[wt_proxy_doy['day_of_year'] == 365].copy()
    row_365['day_of_year'] = 366
    wt_proxy_doy = pd.concat([wt_proxy_doy, row_365], ignore_index=True)

df = df.merge(wt_proxy_doy, on='day_of_year', how='left')

# ── Normalize proxy (sessions so sánh với trung bình năm) ────────────────────
annual_avg_sessions = wt_proxy_doy['proxy_sessions_doy'].mean()
df['proxy_sessions_norm'] = df['proxy_sessions_doy'] / annual_avg_sessions

WEB_FEATURES = [
    'proxy_sessions_doy', 'proxy_conversion_rate_doy',
    'proxy_bounce_rate_doy', 'proxy_unique_visitors_doy',
    'proxy_sessions_norm',
]
print(f'  wt_train period: {wt_train["Date"].min().date()} → {wt_train["Date"].max().date()} ({len(wt_train):,} rows)')
print(f'  Seasonal proxy coverage: doy 1-366')
print(f'  Annual avg sessions    : {annual_avg_sessions:,.0f}')
print(f'  ✅ {len(WEB_FEATURES)} web traffic proxy features added.')

# =============================================================================
# SECTION 9 — LEAKAGE ASSERTION TESTS (bắt buộc trước khi lưu)
# =============================================================================
print('\n🔐 SECTION 9: Leakage Assertion Tests...')

test_df_check = df[df['_split'] == 'test'].copy()
errors = []

# Test 1: Không có Revenue/COGS thực trong test rows
if test_df_check['Revenue'].notna().any():
    errors.append("FAIL: Revenue có giá trị trong test rows!")
if test_df_check['COGS'].notna().any():
    errors.append("FAIL: COGS có giá trị trong test rows!")

# Test 2: lag_365 NaN với toàn bộ test 2024
lag365_2024_ok = test_df_check.loc[
    test_df_check['Date'] >= '2024-01-01', 'lag_365_rev'
].isna().all()
if not lag365_2024_ok:
    errors.append("FAIL: lag_365_rev không NaN trong test 2024 — leakage!")

# Test 3: lag_730 không NaN với test period (mọi lag_730 test date phải reference train)
lag730_test_dates = test_df_check['Date'] - pd.Timedelta(days=730)
# Các ngày lag phải nằm trong [TRAIN_START, TRAIN_END]
lag730_in_train = lag730_test_dates.between(TRAIN_START, TRAIN_END)
if not lag730_in_train.all():
    n_bad = (~lag730_in_train).sum()
    errors.append(f"FAIL: {n_bad} test rows có lag_730 nằm ngoài train period!")

# Test 4: Seasonal profiles không dùng test Revenue
# (proxy: kiểm tra seasonal_doy được fit trên train — đã đảm bảo trong code)
# Xác nhận bằng so sánh giá trị mẫu
doy_1_train_mean = _train_df[_train_df['day_of_year'] == 1]['Revenue'].mean()
doy_1_feature    = df[df['day_of_year'] == 1]['rev_mean_doy'].iloc[0]
if abs(doy_1_train_mean - doy_1_feature) > 1.0:
    errors.append(f"FAIL: rev_mean_doy (doy=1) mismatch: "
                  f"expected {doy_1_train_mean:.2f}, got {doy_1_feature:.2f}")

# Test 5: Web proxy features không phụ thuộc vào test dates
wt_max_date_used = wt_train['Date'].max()
if wt_max_date_used >= TEST_START:
    errors.append(f"FAIL: web_traffic dùng đến {wt_max_date_used.date()} — vượt leakage wall!")

# ── Báo cáo kết quả ──────────────────────────────────────────────────────────
if errors:
    print('  ❌ LEAKAGE DETECTED:')
    for e in errors:
        print(f'     {e}')
    raise RuntimeError("Dừng pipeline: phát hiện data leakage!")
else:
    print('  ✅ Test 1 — Revenue/COGS NaN trong test rows          : PASS')
    print('  ✅ Test 2 — lag_365 NaN trong test 2024               : PASS')
    print('  ✅ Test 3 — lag_730 luôn reference training data      : PASS')
    print('  ✅ Test 4 — Seasonal profiles fit trên train only      : PASS')
    print('  ✅ Test 5 — Web proxy không dùng test-period data     : PASS')
    print('  ✅ TẤT CẢ LEAKAGE TESTS PASSED — An toàn để tiếp tục')

# =============================================================================
# SECTION 10 — TỔNG HỢP VÀ XỬ LÝ NaN
# =============================================================================
print('\n🔧 SECTION 10: NaN audit & handling...')

_all_features_raw = (
    CALENDAR_FEATURES +
    FOURIER_FEATURES  +
    TREND_FEATURES    +
    SEASONAL_FEATURES +
    LAG_FEATURES      +
    WEB_FEATURES
)

# Loại bỏ feature trùng tên (giữ thứ tự xuất hiện đầu tiên) để tránh
# nan_in_train.get(feature) trả về Series khi index bị duplicate.
ALL_FEATURES = list(dict.fromkeys(_all_features_raw))

_dup_feats = []
_seen = set()
for _f in _all_features_raw:
    if _f in _seen and _f not in _dup_feats:
        _dup_feats.append(_f)
    _seen.add(_f)
if _dup_feats:
    print(f'  ⚠️  Removed duplicated features in ALL_FEATURES: {_dup_feats}')

TARGETS = ['Revenue', 'COGS']

# Thống kê NaN theo feature trên train set
train_df_check = df[df['_split'] == 'train'].copy()
nan_in_train   = train_df_check[ALL_FEATURES].isnull().sum()
nan_in_test    = test_df_check[ALL_FEATURES].isnull().sum()

print('\n  NaN counts trong TRAIN features:')
nan_train_nonzero = nan_in_train[nan_in_train > 0]
if len(nan_train_nonzero) > 0:
    print(nan_train_nonzero.to_string())
else:
    print('  → Không có NaN nào trong train features ✅')

print('\n  NaN counts trong TEST features:')
nan_test_nonzero = nan_in_test[nan_in_test > 0]
if len(nan_test_nonzero) > 0:
    print(nan_test_nonzero.to_string())
    print('\n  ℹ️  NaN trong lag_365_rev/cogs tại test 2024: EXPECTED (safe by design)')
    print('     LightGBM xử lý NaN natively → không cần fillna thủ công')
else:
    print('  → Không có NaN nào trong test features ✅')

# Kiểm tra các lag features có bao nhiêu NaN trong train (đầu chuỗi, trước 2013)
print(f'\n  lag_365_rev NaN trong train: {train_df_check["lag_365_rev"].isna().sum()} rows')
print(f'  (Expected: các ngày trước {(TRAIN_START + pd.Timedelta(days=365)).date()})')

# =============================================================================
# SECTION 11 — LƯU FEATURE STORE
# =============================================================================
print('\n💾 SECTION 11: Saving Feature Store...')

# Tách lại train / test
final_cols = ['Date', '_split'] + TARGETS + ALL_FEATURES
output_df  = df[final_cols].copy()

feature_store_train = output_df[output_df['_split'] == 'train'].drop(columns='_split').reset_index(drop=True)
feature_store_test  = output_df[output_df['_split'] == 'test'].drop(columns='_split').reset_index(drop=True)

# Lưu CSV (không dùng parquet)
train_saved_path = OUT_DIR / 'feature_store_train.csv'
test_saved_path  = OUT_DIR / 'feature_store_test.csv'

feature_store_train.to_csv(train_saved_path, index=False)
feature_store_test.to_csv(test_saved_path, index=False)

print(f'  ✅ {train_saved_path.name} : {feature_store_train.shape}')
print(f'  ✅ {test_saved_path.name}  : {feature_store_test.shape}')

# ── Lưu seasonal profiles (dùng lại ở Phase E — XAI) ────────────────────────
seasonal_profiles = {
    'seasonal_doy':    seasonal_doy,
    'seasonal_mdow':   seasonal_mdow,
    'seasonal_margin': seasonal_margin,
    'seasonal_idx':    seasonal_idx,
    'wt_proxy_doy':    wt_proxy_doy,
    'recent_cagr':          _recent_cagr,
    'anchor_rev_2022':      _anchor_rev,
    'anchor_cogs_2022':     _anchor_cogs,
    'anchor_date':          _anchor_date,
    'ratio_seasonal_doy':   _ratio_seasonal,
    'train_rev_mean':  _train_rev_lookup.mean(),
    'train_cogs_mean': _train_cogs_lookup.mean(),
    'train_rev_std':   _train_rev_lookup.std(),
    'breakpoints': {'down': BP_DOWN, 'up': BP_UP, 't0': t0},
}
with open(OUT_DIR / 'phase_b_seasonal_profiles.pkl', 'wb') as f:
    pickle.dump(seasonal_profiles, f)
print(f'  ✅ phase_b_seasonal_profiles.pkl saved')

# =============================================================================
# SECTION 12 — FEATURE SCHEMA REPORT
# =============================================================================
print('\n📋 SECTION 12: Feature Schema Report...')

schema_rows = []

def add_schema(feats, group, leakage_risk, description_map):
    for f in feats:
        schema_rows.append({
            'feature':       f,
            'group':         group,
            'leakage_risk':  leakage_risk,
            'description':   description_map.get(f, f),
            'nan_in_train':  int(nan_in_train.get(f, 0)),
            'nan_in_test':   int(nan_in_test.get(f, 0)),
        })

add_schema(CALENDAR_FEATURES, 'A_Calendar', 'ZERO', {
    'year': 'Năm', 'month': 'Tháng (1-12)', 'day': 'Ngày trong tháng',
    'day_of_week': 'Ngày trong tuần (0=Mon)', 'day_of_year': 'Ngày trong năm (1-366)',
    'week_of_year': 'Tuần trong năm (ISO)', 'quarter': 'Quý (1-4)',
    'is_weekend': 'Cuối tuần (0/1)', 'is_month_start': 'Ngày đầu tháng (0/1)',
    'is_month_end': 'Ngày cuối tháng (0/1)', 'is_quarter_start': 'Ngày đầu quý (0/1)',
    'is_quarter_end': 'Ngày cuối quý (0/1)',
})
add_schema(FOURIER_FEATURES, 'B_Fourier', 'ZERO', {
    f: f'Fourier encoding chu kỳ {"annual" if "annual" in f else "weekly"} '
       f'harmonic {"sin" if "sin" in f else "cos"}'
    for f in FOURIER_FEATURES
})
add_schema(TREND_FEATURES, 'C_Trend', 'ZERO', {
    't_idx': 'Index thời gian tuyến tính (ngày từ TRAIN_START)',
    't_break_down': f'Breakpoint suy giảm từ {BP_DOWN.date()} (piecewise slope change)',
    't_break_up': f'Breakpoint phục hồi từ {BP_UP.date()} (piecewise slope change)',
})
add_schema(SEASONAL_FEATURES, 'D_Seasonal', 'LOW', {
    'rev_mean_doy': 'Mean Revenue theo day_of_year (từ train)',
    'rev_std_doy': 'Std Revenue theo day_of_year (từ train)',
    'cogs_mean_doy': 'Mean COGS theo day_of_year (từ train)',
    'cogs_std_doy': 'Std COGS theo day_of_year (từ train)',
    'rev_mean_month_dow': 'Mean Revenue theo (month, day_of_week) (từ train)',
    'cogs_mean_month_dow': 'Mean COGS theo (month, day_of_week) (từ train)',
    'margin_mean_month': 'Mean COGS/Revenue ratio theo month (từ train)',
    'margin_std_month': 'Std COGS/Revenue ratio theo month (từ train)',
    'rev_seasonal_index': 'Chỉ số mùa vụ tương đối (Rev / annual_mean, theo day_of_year)',
})
add_schema(LAG_FEATURES, 'E_Lags', 'LOW', {
    'lag_365_rev': 'Revenue cùng ngày năm ngoái (NaN nếu test 2024)',
    'lag_365_cogs': 'COGS cùng ngày năm ngoái (NaN nếu test 2024)',
    'lag_730_rev': 'Revenue cùng ngày 2 năm trước (luôn từ train)',
    'lag_730_cogs': 'COGS cùng ngày 2 năm trước (luôn từ train)',
    'lag_1095_rev': 'Revenue cùng ngày 3 năm trước (fallback)',
    'lag_1095_cogs': 'COGS cùng ngày 3 năm trước (fallback)',
    'yoy_rev_ratio': 'Tỷ lệ lag_365/lag_730 cho Revenue (YoY momentum)',
    'yoy_cogs_ratio': 'Tỷ lệ lag_365/lag_730 cho COGS (YoY momentum)',
})
add_schema(WEB_FEATURES, 'F_WebProxy', 'ZERO', {
    'proxy_sessions_doy': 'Mean sessions theo day_of_year (từ train web_traffic)',
    'proxy_conversion_rate_doy': 'Mean conversion_rate theo day_of_year (reconstructed from valid unique orders / sessions)',
    'proxy_bounce_rate_doy': 'Mean bounce_rate theo day_of_year (từ train)',
    'proxy_unique_visitors_doy': 'Mean unique_visitors theo day_of_year (từ train)',
    'proxy_sessions_norm': 'sessions_doy / annual_avg (normalized)',
})

schema_df = pd.DataFrame(schema_rows)
schema_df.to_csv(OUT_DIR / 'phase_b_feature_schema.csv', index=False)
print(f'  ✅ phase_b_feature_schema.csv saved')

# Summary
print('\n  Feature Groups Summary:')
group_summary = schema_df.groupby('group').agg(
    n_features=('feature', 'count'),
    leakage_risk=('leakage_risk', 'first'),
).reset_index()
print(group_summary.to_string(index=False))
print(f'\n  TOTAL FEATURES : {len(ALL_FEATURES)}')
print(f'  TARGETS        : {TARGETS}')

# =============================================================================
# SECTION 13 — VISUALIZATION: Feature Sanity Checks
# =============================================================================
print('\n📊 SECTION 13: Generating feature sanity charts...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Phase B — Feature Store Sanity Checks\nVinTelligence DATATHON 2026',
             fontsize=15, fontweight='bold')

_tr = feature_store_train.copy()
_te = feature_store_test.copy()

# ── Chart 1: Seasonal profile (rev_mean_doy) ──────────────────────────────────
ax = axes[0, 0]
ax.plot(seasonal_doy['day_of_year'], seasonal_doy['rev_mean_doy'] / 1e6,
        color='#2E86AB', lw=2)
ax.fill_between(
    seasonal_doy['day_of_year'],
    (seasonal_doy['rev_mean_doy'] - seasonal_doy['rev_std_doy']) / 1e6,
    (seasonal_doy['rev_mean_doy'] + seasonal_doy['rev_std_doy']) / 1e6,
    alpha=0.2, color='#2E86AB'
)
ax.set_title('D1: Seasonal Profile (rev_mean_doy)\nMean ± 1σ per Day-of-Year')
ax.set_xlabel('Day of Year'); ax.set_ylabel('Mean Revenue (triệu VND)')
ax.axvline(121, color='red', ls='--', lw=1, label='Day 121 ≈ May 1')
ax.legend(fontsize=9)

# ── Chart 2: Piecewise trend trên training data ───────────────────────────────
ax = axes[0, 1]
ax.scatter(_tr['Date'], _tr['Revenue'] / 1e6,
           s=1, alpha=0.3, color='#2E86AB', label='Actual')
# Vẽ t_idx theo thời gian để visualize
ax2 = ax.twinx()
ax2.plot(_tr['Date'], _tr['t_break_down'], color='#F9A72B', lw=1, alpha=0.7, label='t_break_down')
ax2.plot(_tr['Date'], _tr['t_break_up'],   color='#3BB273', lw=1, alpha=0.7, label='t_break_up')
ax2.set_ylabel('Trend feature value', color='gray')
ax.axvline(BP_DOWN, color='red', ls='--', lw=1.5, label=f'BP_DOWN {BP_DOWN.date()}')
ax.axvline(BP_UP,   color='green', ls='--', lw=1.5, label=f'BP_UP {BP_UP.date()}')
ax.set_title('C: Piecewise Trend Features vs Revenue')
ax.set_ylabel('Revenue (triệu VND)')
ax.legend(fontsize=7, loc='upper left')
ax2.legend(fontsize=7, loc='lower right')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Chart 3: Fourier annual (sin_1 và cos_1) trên 1 năm mẫu ──────────────────
ax = axes[0, 2]
_sample_year = _tr[_tr['year'] == 2019].sort_values('Date')
ax.plot(_sample_year['day_of_year'], _sample_year['fourier_annual_sin_1'],
        label='Annual sin_1', color='#2E86AB')
ax.plot(_sample_year['day_of_year'], _sample_year['fourier_annual_cos_1'],
        label='Annual cos_1', color='#E84855')
ax.plot(_sample_year['day_of_year'], _sample_year['fourier_annual_sin_2'],
        label='Annual sin_2', color='#3BB273', ls='--', alpha=0.7)
ax.set_title('B: Fourier Features (Annual Harmonics)\n2019 sample')
ax.set_xlabel('Day of Year'); ax.set_ylabel('Value (-1 to 1)')
ax.legend(fontsize=9)

# ── Chart 4: lag_365_rev vs actual Revenue (2014 onward) ─────────────────────
ax = axes[1, 0]
_tr_lag = _tr.dropna(subset=['lag_365_rev'])
ax.scatter(_tr_lag['lag_365_rev'] / 1e6, _tr_lag['Revenue'] / 1e6,
           s=1, alpha=0.2, color='#2E86AB')
# Perfect lag line
_max_val = max(_tr_lag['Revenue'].max(), _tr_lag['lag_365_rev'].max()) / 1e6
ax.plot([0, _max_val], [0, _max_val], 'r--', lw=1, label='y=x')
corr = _tr_lag[['Revenue', 'lag_365_rev']].corr().iloc[0, 1]
ax.set_title(f'E: lag_365_rev vs Revenue\n(Pearson r = {corr:.3f})')
ax.set_xlabel('lag_365_rev (triệu VND)'); ax.set_ylabel('Revenue (triệu VND)')
ax.legend(fontsize=9)

# ── Chart 5: Web traffic proxy (proxy_sessions_doy) ──────────────────────────
ax = axes[1, 1]
ax.plot(wt_proxy_doy['day_of_year'], wt_proxy_doy['proxy_sessions_doy'],
        color='#7B2D8B', lw=2, label='proxy_sessions_doy')
ax_r = ax.twinx()
ax_r.plot(wt_proxy_doy['day_of_year'], wt_proxy_doy['proxy_conversion_rate_doy'],
          color='#F9A72B', lw=2, label='proxy_conversion_rate_doy')
ax.set_title('F: Web Traffic Seasonal Proxy\n(fit from 2013–2022 web_traffic)')
ax.set_xlabel('Day of Year'); ax.set_ylabel('Sessions', color='#7B2D8B')
ax_r.set_ylabel('Conversion Rate', color='#F9A72B')
ax.legend(fontsize=9, loc='upper left')
ax_r.legend(fontsize=9, loc='upper right')

# ── Chart 6: NaN pattern trong test features ─────────────────────────────────
ax = axes[1, 2]
nan_counts = _te[ALL_FEATURES].isnull().sum()
nan_nonzero = nan_counts[nan_counts > 0].sort_values(ascending=True)
if len(nan_nonzero) > 0:
    bars = ax.barh(range(len(nan_nonzero)), nan_nonzero.values, color='#E84855', alpha=0.8)
    ax.set_yticks(range(len(nan_nonzero)))
    ax.set_yticklabels(nan_nonzero.index, fontsize=9)
    ax.set_xlabel('NaN count in test set')
    ax.set_title('NaN Distribution in Test Features\n(Expected: only lag_365 in 2024)')
    for bar, val in zip(bars, nan_nonzero.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
else:
    ax.text(0.5, 0.5, 'No NaN in test features\n(lag_365 NaN in 2024\nhandled by LightGBM)',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('NaN Distribution in Test Features')

plt.tight_layout()
chart_path = OUT_DIR / 'phase_b_feature_checks.png'
plt.savefig(chart_path, bbox_inches='tight', dpi=130)
plt.show()
print(f'  📊 Chart saved: {chart_path}')

# =============================================================================
# SECTION 14 — FINAL SUMMARY
# =============================================================================
print('\n' + '=' * 65)
print('✅ PHASE B COMPLETE — Feature Store Built Successfully')
print('=' * 65)
print(f'\n  📦 {train_saved_path.name} : {feature_store_train.shape}')
print(f'  📦 {test_saved_path.name}  : {feature_store_test.shape}')
print(f'  🔐 Leakage tests               : ALL PASSED')
print(f'  🧮 Total features              : {len(ALL_FEATURES)}')
print()
print('  Feature groups:')
for _, row in group_summary.iterrows():
    print(f'    {row["group"]:<20} {row["n_features"]:>3} features  '
          f'[Leakage: {row["leakage_risk"]}]')
print()
print('  ⚠️  Notes for Phase C:')
print(f'    • lag_365_rev/cogs = NaN trong test 2024 ({int(nan_in_test.get("lag_365_rev", 0))} rows) — LightGBM handles')
print('    • lag_730 luôn available (references 2021-2022 train data)')
print('    • COGS_STRATEGY = INDEPENDENT_MODEL → train 2 models riêng biệt')
print('    • Seasonal profiles đã lưu → dùng cho Phase E (XAI)')
print()
print('  ▶ Sẵn sàng cho Phase C: Model Training & Walk-Forward CV')