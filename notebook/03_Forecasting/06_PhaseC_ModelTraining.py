# =============================================================================
# PHASE C — MODEL TRAINING & WALK-FORWARD CROSS-VALIDATION
# VinTelligence DATATHON 2026 | Phần 3: Sales Forecasting
#
# Chiến lược:
#   • 2 LightGBM models riêng biệt: Revenue & COGS (INDEPENDENT_MODEL strategy)
#   • Walk-forward CV: 4 folds, mỗi fold val ≈ 548 ngày (mirror test horizon)
#   • Optuna hyperparameter tuning: 60 trials/target, minimize CV MAPE
#   • Final model: retrain trên toàn bộ train data với best params
#   • Post-processing: clip âm, COGS sanity check (COGS ≤ Revenue)
#
# Input:
#   feature_store_train.parquet / .csv
#   feature_store_test.parquet  / .csv
#
# Output:
#   model_lgbm_revenue.pkl
#   model_lgbm_cogs.pkl
#   oof_predictions.csv
#   phase_c_cv_report.csv
#   phase_c_submission_raw.csv
#   phase_c_feature_importance_rev.csv
#   phase_c_feature_importance_cogs.csv
#   phase_c_best_params.pkl
#   phase_c_model_diagnostics.png
#   phase_c_optuna_history.png
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Install deps nếu cần ─────────────────────────────────────────────────────
try:
    import lightgbm as lgb
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
    import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna', '-q'])
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Runtime flag & paths ─────────────────────────────────────────────────────
# local: đọc/ghi artifacts tại notebook/03_Forecasting/artifacts
# kaggle: đọc/ghi tại /kaggle/working
RUN_ENV = 'local'  # đổi thành 'kaggle' khi chạy trên Kaggle

if RUN_ENV == 'local':
    INPUT_DIR = Path('artifacts')
    OUT_DIR = Path('artifacts')
elif RUN_ENV == 'kaggle':
    INPUT_DIR = Path('/kaggle/working/artifacts')
    OUT_DIR = Path('/kaggle/working/artifacts')
else:
    raise ValueError("RUN_ENV phải là 'local' hoặc 'kaggle'.")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_END        = pd.Timestamp('2022-12-31')
TEST_START       = pd.Timestamp('2023-01-01')
TEST_END         = pd.Timestamp('2024-07-01')
N_OPTUNA_TRIALS  = 60
N_LGBM_ROUNDS    = 1500
EARLY_STOPPING   = 80

print('=' * 65)
print('  PHASE C — MODEL TRAINING & WALK-FORWARD CV')
print('  VinTelligence DATATHON 2026')
print('=' * 65)
print(f'  RUN_ENV  : {RUN_ENV}')
print(f'  INPUT_DIR: {INPUT_DIR}')
print(f'  OUT_DIR  : {OUT_DIR}')
print(f'  LightGBM : {lgb.__version__}')
print(f'  Optuna   : {optuna.__version__}')

# =============================================================================
# SECTION 1 — LOAD FEATURE STORE
# =============================================================================
print('\n📂 SECTION 1: Loading feature store...')

def load_fs(name):
    p = INPUT_DIR / f'{name}.parquet'
    c = INPUT_DIR / f'{name}.csv'
    if p.exists():
        return pd.read_parquet(p)
    if c.exists():
        return pd.read_csv(c, parse_dates=['Date'])
    raise FileNotFoundError(f'Không tìm thấy {name}.parquet/.csv trong {INPUT_DIR}')

train_fs = load_fs('feature_store_train').sort_values('Date').reset_index(drop=True)
test_fs  = load_fs('feature_store_test').sort_values('Date').reset_index(drop=True)

# ── Log-transform targets (fix multiplicative seasonality) ───────────────────
# log1p biến multiplicative structure → additive → LightGBM xử lý đúng
# expm1 ở Phase D để inverse transform
import numpy as np  # đảm bảo import

assert 'recent_trend_rev' in train_fs.columns, \
    "Chưa chạy Phase B mới — thiếu recent_trend_rev. Chạy lại Phase B trước."

train_fs['log_Revenue'] = np.log1p(train_fs['Revenue'])
train_fs['log_COGS']    = np.log1p(train_fs['COGS'])

TARGET_REV_LOG  = 'log_Revenue'
TARGET_COGS_LOG = 'log_COGS'
TARGET_REV      = 'Revenue'    # giữ để tính metrics absolute
TARGET_COGS     = 'COGS'

print(f'  log_Revenue: mean={train_fs[TARGET_REV_LOG].mean():.4f}  '
      f'std={train_fs[TARGET_REV_LOG].std():.4f}')
print(f'  (vs raw Revenue: mean={train_fs[TARGET_REV].mean():,.0f}  '
      f'std={train_fs[TARGET_REV].std():,.0f})')

NON_FEAT = {
    'Date', 'Revenue', 'COGS', '_split',
    'log_Revenue', 'log_COGS',
    'rev_detrend_ratio', 'cogs_detrend_ratio',   # targets, không phải features
}
FEATURE_COLS = [c for c in train_fs.columns if c not in NON_FEAT]

print(f'  train_fs : {train_fs.shape}')
print(f'  test_fs  : {test_fs.shape}')
print(f'  Features : {len(FEATURE_COLS)}')
print(f'  Includes recent_trend_rev: {"recent_trend_rev" in FEATURE_COLS}')

assert train_fs[TARGET_REV].notna().all(), "NaN trong Revenue train!"
assert train_fs[TARGET_COGS].notna().all(), "NaN trong COGS train!"

# =============================================================================
# SECTION 2 — METRIC FUNCTIONS
# =============================================================================

def mae(a, p):   return np.mean(np.abs(a - p))
def rmse(a, p):  return np.sqrt(np.mean((a - p) ** 2))
def mape(a, p, eps=1.0): return np.mean(np.abs((a - p) / (np.abs(a) + eps))) * 100
def r2(a, p):
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def report(tag, a, p):
    m = {'tag': tag, 'MAE': mae(a,p), 'RMSE': rmse(a,p), 'MAPE': mape(a,p), 'R2': r2(a,p)}
    print(f'  {tag:<30} MAE:{m["MAE"]:>12,.0f}  RMSE:{m["RMSE"]:>12,.0f}'
          f'  MAPE:{m["MAPE"]:>6.2f}%  R²:{m["R2"]:.4f}')
    return m

# =============================================================================
# SECTION 3 — WALK-FORWARD FOLD DEFINITIONS
# =============================================================================
print('\n📅 SECTION 3: Defining walk-forward folds...')

# 4 folds, expanding train window, val ≈ 548 ngày (mirror test horizon)
# Fold 2 (2019-2020) là fold khó nhất — chứa suy giảm mạnh
folds_spec = [
    # Non-overlapping, expanding train window
    # Fold2 chứa COVID crash (2019-2020): fold khó nhất, critical
    # Fold4 = 2022-H2: gần test boundary nhất, quan trọng cho generalization
    ('Fold1', '2017-01-01', '2018-06-30'),
    ('Fold2', '2019-01-01', '2020-06-30'),
    ('Fold3', '2020-07-01', '2021-12-31'),
    ('Fold4', '2022-01-01', '2022-12-31'),   # ← boundary fold, gần test nhất
]

FOLDS = []
for name, vs, ve in folds_spec:
    val_start = pd.Timestamp(vs)
    val_end   = pd.Timestamp(ve)
    tr_mask   = train_fs['Date'] <  val_start          # expanding window
    vl_mask   = (train_fs['Date'] >= val_start) & (train_fs['Date'] <= val_end)
    n_tr, n_vl = tr_mask.sum(), vl_mask.sum()
    if n_vl < 100:
        print(f'  ⚠️  {name}: only {n_vl} val rows — skip'); continue
    FOLDS.append({'name': name, 'val_start': val_start, 'val_end': val_end,
                  'tr': tr_mask, 'vl': vl_mask, 'n_tr': n_tr, 'n_vl': n_vl})
    print(f'  {name}: train={n_tr:,}  val={n_vl:,}  ({vs} → {ve})')

print(f'\n  Total folds: {len(FOLDS)}')

# =============================================================================
# SECTION 4 — OPTUNA HYPERPARAMETER TUNING
# =============================================================================
print(f'\n🔍 SECTION 4: Optuna tuning ({N_OPTUNA_TRIALS} trials each)...')

FIXED = {
    'objective': 'regression', 'metric': 'mae', 'verbosity': -1,
    'boosting_type': 'gbdt', 'random_state': RANDOM_SEED,
    'n_jobs': -1, 'subsample_freq': 1,
}

def make_objective(target):
    def objective(trial):
        params = {
            **FIXED,
            'num_leaves':        trial.suggest_int('num_leaves', 15, 100),
            'max_depth':         trial.suggest_int('max_depth', 3, 12),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'n_estimators':      trial.suggest_int('n_estimators', 300, N_LGBM_ROUNDS),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':         trial.suggest_float('reg_alpha', 0.1, 20.0, log=True),
            'reg_lambda':        trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
            'min_split_gain':    trial.suggest_float('min_split_gain', 0.0, 0.5),
        }
        fold_mapes = []
        for fold in FOLDS:
            X_tr = train_fs.loc[fold['tr'], FEATURE_COLS]
            y_tr = train_fs.loc[fold['tr'], target]
            X_vl = train_fs.loc[fold['vl'], FEATURE_COLS]
            y_vl = train_fs.loc[fold['vl'], target]
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                              lgb.log_evaluation(-1)])
            pred_log = np.clip(m.predict(X_vl), 0, None)
            pred_abs = np.expm1(pred_log)
            actual_abs = np.expm1(y_vl.values)   # y_vl là log target
            fold_mapes.append(mape(actual_abs, pred_abs))
        return np.mean(fold_mapes)
    return objective

# Revenue
study_rev = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
study_rev.optimize(make_objective(TARGET_REV_LOG), n_trials=N_OPTUNA_TRIALS,
                   show_progress_bar=False)
best_rev  = {**FIXED, **study_rev.best_params}
print(f'  ✅ Revenue  best CV MAPE: {study_rev.best_value:.2f}%  '
      f'(leaves={best_rev["num_leaves"]}, lr={best_rev["learning_rate"]:.4f})')

# COGS
study_cogs = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED + 1),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
study_cogs.optimize(make_objective(TARGET_COGS_LOG), n_trials=N_OPTUNA_TRIALS,
                    show_progress_bar=False)
best_cogs = {**FIXED, **study_cogs.best_params}
print(f'  ✅ COGS     best CV MAPE: {study_cogs.best_value:.2f}%  '
      f'(leaves={best_cogs["num_leaves"]}, lr={best_cogs["learning_rate"]:.4f})')

# =============================================================================
# SECTION 5 — WALK-FORWARD CV: Full evaluation với best params
# =============================================================================
print('\n📊 SECTION 5: Walk-forward CV evaluation with best params...')

cv_records  = []
oof_records = []

for fold in FOLDS:
    X_tr = train_fs.loc[fold['tr'], FEATURE_COLS]
    X_vl = train_fs.loc[fold['vl'], FEATURE_COLS]
    y_tr_r = train_fs.loc[fold['tr'], TARGET_REV_LOG]   # log Revenue
    y_vl_r = train_fs.loc[fold['vl'], TARGET_REV_LOG]
    y_tr_c = train_fs.loc[fold['tr'], TARGET_COGS_LOG]
    y_vl_c = train_fs.loc[fold['vl'], TARGET_COGS_LOG]
    dates  = train_fs.loc[fold['vl'], 'Date']

    # Revenue model
    mr = lgb.LGBMRegressor(**best_rev)
    mr.fit(X_tr, y_tr_r, eval_set=[(X_vl, y_vl_r)],
           callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                      lgb.log_evaluation(-1)])
    pr = np.expm1(np.clip(mr.predict(X_vl), 0, None))

    # COGS model
    mc = lgb.LGBMRegressor(**best_cogs)
    mc.fit(X_tr, y_tr_c, eval_set=[(X_vl, y_vl_c)],
           callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                      lgb.log_evaluation(-1)])
    pc = np.expm1(np.clip(mc.predict(X_vl), 0, None))

    y_vl_r_abs = train_fs.loc[fold['vl'], TARGET_REV].values
    y_vl_c_abs = train_fs.loc[fold['vl'], TARGET_COGS].values

    print(f'\n  ── {fold["name"]} ({fold["val_start"].date()} → {fold["val_end"].date()}) ──')
    rr = report('Revenue', y_vl_r_abs, pr)
    rc = report('COGS',    y_vl_c_abs, pc)

    cv_records.append({
        'fold': fold['name'],
        'val_start': fold['val_start'].date(), 'val_end': fold['val_end'].date(),
        'n_val': fold['n_vl'],
        'rev_MAE': rr['MAE'], 'rev_RMSE': rr['RMSE'],
        'rev_MAPE': rr['MAPE'], 'rev_R2': rr['R2'],
        'cogs_MAE': rc['MAE'], 'cogs_RMSE': rc['RMSE'],
        'cogs_MAPE': rc['MAPE'], 'cogs_R2': rc['R2'],
    })
    for d, ra, rp, ca, cp in zip(dates, y_vl_r_abs, pr, y_vl_c_abs, pc):
        oof_records.append({'Date': d, 'fold': fold['name'],
                            'Revenue_actual': ra, 'Revenue_pred': rp,
                            'COGS_actual': ca, 'COGS_pred': cp})

cv_df  = pd.DataFrame(cv_records)
oof_df = pd.DataFrame(oof_records).sort_values('Date').reset_index(drop=True)

print('\n' + '─' * 65)
print('  CV SUMMARY (mean across folds)')
print('─' * 65)
for col, lbl in [('rev_MAPE','Revenue MAPE'), ('rev_R2','Revenue R²'),
                 ('rev_MAE','Revenue MAE'), ('cogs_MAPE','COGS MAPE'),
                 ('cogs_R2','COGS R²')]:
    v = cv_df[col].values
    print(f'  {lbl:<18}: mean={v.mean():.4f}  std={v.std():.4f}'
          f'  [{v.min():.4f}, {v.max():.4f}]')

cv_df.to_csv(OUT_DIR / 'phase_c_cv_report.csv', index=False)
oof_df.to_csv(OUT_DIR / 'oof_predictions.csv', index=False)
print('\n  ✅ phase_c_cv_report.csv & oof_predictions.csv saved')

# =============================================================================
# SECTION 6 — FINAL MODELS: Retrain on FULL training data
# =============================================================================
print('\n🏋️  SECTION 6: Retraining final models on full training data...')

X_full = train_fs[FEATURE_COLS]
y_rev  = train_fs[TARGET_REV_LOG]
y_cogs = train_fs[TARGET_COGS_LOG]

final_rev = lgb.LGBMRegressor(**best_rev)
final_rev.fit(X_full, y_rev, callbacks=[lgb.log_evaluation(-1)])

final_cogs = lgb.LGBMRegressor(**best_cogs)
final_cogs.fit(X_full, y_cogs, callbacks=[lgb.log_evaluation(-1)])

# In-sample sanity (chỉ để xác nhận model fit, không phải validation metric)
print('  In-sample fit (absolute Revenue/COGS):')
report('Revenue (in-sample)', train_fs[TARGET_REV].values,
    np.expm1(np.clip(final_rev.predict(X_full), 0, None)))
report('COGS    (in-sample)', train_fs[TARGET_COGS].values,
    np.expm1(np.clip(final_cogs.predict(X_full), 0, None)))

# Save
with open(OUT_DIR / 'model_lgbm_revenue.pkl', 'wb') as f: pickle.dump(final_rev, f)
with open(OUT_DIR / 'model_lgbm_cogs.pkl',    'wb') as f: pickle.dump(final_cogs, f)
with open(OUT_DIR / 'phase_c_best_params.pkl', 'wb') as f:
    pickle.dump({'rev': best_rev, 'cogs': best_cogs}, f)
print('  ✅ Models and params saved.')

# =============================================================================
# SECTION 7 — TEST PREDICTIONS & POST-PROCESSING
# =============================================================================
print('\n🔮 SECTION 7: Test predictions & post-processing...')

X_test = test_fs[FEATURE_COLS]

# Predict trong log-space rồi inverse transform
pred_log_rev  = np.clip(final_rev.predict(X_test),  0, None)
pred_log_cogs = np.clip(final_cogs.predict(X_test), 0, None)

pred_rev  = np.expm1(pred_log_rev)
pred_cogs = np.expm1(pred_log_cogs)

# ── Level calibration (post-hoc scaling) ─────────────────────────────────────
# Dùng Fold4 (2022) OOF để tính scale factor
# Nếu model predict 2022 thấp/cao hơn actual, scale test predictions tương ứng
fold4_mask = oof_df['Date'].dt.year == 2022
if fold4_mask.sum() > 50:
    actual_2022_mean = oof_df.loc[fold4_mask, 'Revenue_actual'].mean()
    pred_2022_mean   = oof_df.loc[fold4_mask, 'Revenue_pred'].mean()
    calibration_factor = actual_2022_mean / pred_2022_mean
    calibration_factor = np.clip(calibration_factor, 0.80, 1.20)  # max 20% correction
    pred_rev  = pred_rev  * calibration_factor
    pred_cogs = pred_cogs * calibration_factor
    print(f'  Level calibration factor: {calibration_factor:.4f}')
    print(f'    actual_2022_mean = {actual_2022_mean:,.0f}')
    print(f'    pred_2022_mean   = {pred_2022_mean:,.0f}')
else:
    calibration_factor = 1.0
    print('  ⚠️  Fold4 không đủ dữ liệu — calibration skipped')

# Safety clips (sau calibration)
pred_rev  = np.clip(pred_rev,  0, y_rev.apply(np.expm1).mean() + 4*y_rev.apply(np.expm1).std())
pred_cogs = np.clip(pred_cogs, 0, y_cogs.apply(np.expm1).mean() + 4*y_cogs.apply(np.expm1).std())

# COGS ≤ Revenue constraint
exceed = pred_cogs > pred_rev
pred_cogs[exceed] = pred_rev[exceed] * 0.97

print(f'  Revenue test: mean={pred_rev.mean():,.0f}  '
      f'min={pred_rev.min():,.0f}  max={pred_rev.max():,.0f}')
print(f'  COGS    test: mean={pred_cogs.mean():,.0f}  '
      f'min={pred_cogs.min():,.0f}  max={pred_cogs.max():,.0f}')

# YoY plausibility check
pct = (pred_rev.mean() - train_fs[TARGET_REV].mean()) / train_fs[TARGET_REV].mean() * 100
print(f'\n  Test Revenue mean vs Train mean: {pct:+.1f}%')

# Save raw (Phase D sẽ làm final export/formatting)
sub_raw = pd.DataFrame({
    'Date':    test_fs['Date'].dt.strftime('%Y-%m-%d'),
    'Revenue': pred_rev.round(2),
    'COGS':    pred_cogs.round(2),
})
sub_raw.to_csv(OUT_DIR / 'phase_c_submission_raw.csv', index=False)
print(f'\n  ✅ phase_c_submission_raw.csv saved ({len(sub_raw)} rows)')

# =============================================================================
# SECTION 8 — FEATURE IMPORTANCE
# =============================================================================
fi_rev = pd.DataFrame({'feature': FEATURE_COLS,
                        'importance': final_rev.feature_importances_}
                      ).sort_values('importance', ascending=False)
fi_cogs = pd.DataFrame({'feature': FEATURE_COLS,
                         'importance': final_cogs.feature_importances_}
                       ).sort_values('importance', ascending=False)

print('\n🔑 SECTION 8: Feature Importance (gain) — Revenue top 15:')
print(fi_rev.head(15).to_string(index=False))

fi_rev.to_csv(OUT_DIR  / 'phase_c_feature_importance_rev.csv',  index=False)
fi_cogs.to_csv(OUT_DIR / 'phase_c_feature_importance_cogs.csv', index=False)

# =============================================================================
# SECTION 9 — VISUALIZATIONS (9 charts)
# =============================================================================
print('\n📊 SECTION 9: Generating diagnostic charts...')

fig = plt.figure(figsize=(20, 18))
fig.suptitle('Phase C — Log-transform Architecture\nVinTelligence DATATHON 2026',
             fontsize=15, fontweight='bold', y=1.002)

# Chart 1: OOF Revenue
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(oof_df['Date'], oof_df['Revenue_actual'] / 1e6,
         lw=0.7, color='#2E86AB', label='Actual', alpha=0.85)
ax1.plot(oof_df['Date'], oof_df['Revenue_pred'] / 1e6,
         lw=0.7, color='#E84855', ls='--', label='OOF Pred', alpha=0.85)
ax1.set_title('OOF Revenue: Actual vs Predicted')
ax1.set_ylabel('Revenue (triệu VND)'); ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Chart 2: OOF COGS
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(oof_df['Date'], oof_df['COGS_actual'] / 1e6,
         lw=0.7, color='#F9A72B', label='Actual', alpha=0.85)
ax2.plot(oof_df['Date'], oof_df['COGS_pred'] / 1e6,
         lw=0.7, color='#7B2D8B', ls='--', label='OOF Pred', alpha=0.85)
ax2.set_title('OOF COGS: Actual vs Predicted')
ax2.set_ylabel('COGS (triệu VND)'); ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Chart 3: MAPE per fold
ax3 = fig.add_subplot(3, 3, 3)
x  = np.arange(len(cv_df)); w = 0.35
ax3.bar(x-w/2, cv_df['rev_MAPE'],  w, label='Revenue', color='#2E86AB', alpha=0.85)
ax3.bar(x+w/2, cv_df['cogs_MAPE'], w, label='COGS',    color='#F9A72B', alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels([f['name'] for f in FOLDS], fontsize=9)
ax3.axhline(cv_df['rev_MAPE'].mean(), ls='--', color='#2E86AB', lw=1.2)
for xi,(rv,cv2) in enumerate(zip(cv_df['rev_MAPE'], cv_df['cogs_MAPE'])):
    ax3.text(xi-w/2, rv+0.1, f'{rv:.1f}', ha='center', fontsize=8)
    ax3.text(xi+w/2, cv2+0.1, f'{cv2:.1f}', ha='center', fontsize=8)
ax3.set_title('MAPE (%) by Walk-Forward Fold')
ax3.set_ylabel('MAPE (%)'); ax3.legend(fontsize=9)

# Chart 4: Residuals distribution
ax4 = fig.add_subplot(3, 3, 4)
res = (oof_df['Revenue_actual'] - oof_df['Revenue_pred']) / 1e6
ax4.hist(res.clip(-6, 6), bins=55, color='#2E86AB', alpha=0.8, edgecolor='white')
ax4.axvline(res.median(), color='red', lw=1.5, ls='--',
            label=f'Median {res.median():.2f}M')
ax4.axvline(0, color='green', lw=1, ls=':')
ax4.set_title('OOF Residuals Distribution — Revenue')
ax4.set_xlabel('Residual (triệu VND)'); ax4.set_ylabel('Frequency')
ax4.legend(fontsize=9)

# Chart 5: Scatter actual vs predicted
ax5 = fig.add_subplot(3, 3, 5)
ax5.scatter(oof_df['Revenue_actual']/1e6, oof_df['Revenue_pred']/1e6,
            s=2, alpha=0.25, color='#2E86AB')
_m = max(oof_df['Revenue_actual'].max(), oof_df['Revenue_pred'].max()) / 1e6
ax5.plot([0,_m],[0,_m], 'r--', lw=1.2, label='Perfect fit')
oof_r2v = r2(oof_df['Revenue_actual'].values, oof_df['Revenue_pred'].values)
ax5.set_title(f'OOF Actual vs Predicted Revenue\nR² = {oof_r2v:.4f}')
ax5.set_xlabel('Actual (M VND)'); ax5.set_ylabel('Predicted (M VND)')
ax5.legend(fontsize=9)

# Chart 6: R² per fold
ax6 = fig.add_subplot(3, 3, 6)
ax6.bar(x-w/2, cv_df['rev_R2'],  w, label='Revenue R²', color='#3BB273', alpha=0.85)
ax6.bar(x+w/2, cv_df['cogs_R2'], w, label='COGS R²',    color='#9A9A9A', alpha=0.85)
ax6.set_xticks(x); ax6.set_xticklabels([f['name'] for f in FOLDS], fontsize=9)
ax6.axhline(0.90, color='green', ls=':', lw=1, label='R²=0.9 target')
for xi,(rv,cv2) in enumerate(zip(cv_df['rev_R2'], cv_df['cogs_R2'])):
    ax6.text(xi-w/2, rv+0.003, f'{rv:.3f}', ha='center', fontsize=8)
    ax6.text(xi+w/2, cv2+0.003, f'{cv2:.3f}', ha='center', fontsize=8)
ax6.set_title('R² by Walk-Forward Fold')
ax6.set_ylabel('R²'); ax6.legend(fontsize=9)

# Chart 7: Feature importance — Revenue
ax7 = fig.add_subplot(3, 3, 7)
t20r = fi_rev.head(20).sort_values('importance')
ax7.barh(t20r['feature'], t20r['importance'], color='#2E86AB', alpha=0.85)
ax7.set_title('Feature Importance — Revenue (top 20, gain)')
ax7.set_xlabel('Importance'); ax7.tick_params(axis='y', labelsize=8)

# Chart 8: Feature importance — COGS
ax8 = fig.add_subplot(3, 3, 8)
t20c = fi_cogs.head(20).sort_values('importance')
ax8.barh(t20c['feature'], t20c['importance'], color='#F9A72B', alpha=0.85)
ax8.set_title('Feature Importance — COGS (top 20, gain)')
ax8.set_xlabel('Importance'); ax8.tick_params(axis='y', labelsize=8)

# Chart 9: Test predictions + train tail
ax9 = fig.add_subplot(3, 3, 9)
tail = train_fs[train_fs['Date'] >= '2021-01-01']
ax9.plot(tail['Date'], tail['Revenue']/1e6, lw=0.9, color='#2E86AB',
         label='Actual (train tail)', alpha=0.9)
ax9.plot(test_fs['Date'], pred_rev/1e6, lw=1.2, color='#E84855',
         label='Predicted (test)', alpha=0.9)
ax9.fill_between(test_fs['Date'], pred_rev/1e6, alpha=0.07, color='#E84855')
ax9.axvline(TEST_START, color='black', lw=1.5, ls='--', label='Leakage wall')
ax9.set_title('Test Revenue Predictions\n(Train tail 2021+ context)')
ax9.set_ylabel('Revenue (triệu VND)'); ax9.legend(fontsize=8)
ax9.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax9.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / 'phase_c_model_diagnostics.png', bbox_inches='tight', dpi=130)
plt.show()
print('  📊 phase_c_model_diagnostics.png saved')

# Optuna history
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Optuna Optimization History', fontsize=13, fontweight='bold')
for ax_o, study, title in [(axes2[0], study_rev, 'Revenue'),
                            (axes2[1], study_cogs, 'COGS')]:
    vals = [t.value for t in study.trials if t.value is not None]
    ax_o.plot(vals, 'o', ms=3, alpha=0.35, color='#9A9A9A', label='Trial')
    ax_o.plot(pd.Series(vals).cummin(), lw=2, color='#E84855', label='Best so far')
    ax_o.set_title(f'{title} — Best MAPE: {min(vals):.2f}%')
    ax_o.set_xlabel('Trial'); ax_o.set_ylabel('MAPE (%)'); ax_o.legend(fontsize=9)
plt.tight_layout()
fig2.savefig(OUT_DIR / 'phase_c_optuna_history.png', bbox_inches='tight', dpi=130)
plt.show()
print('  📊 phase_c_optuna_history.png saved')

# =============================================================================
# SECTION 10 — FINAL SUMMARY
# =============================================================================
mr  = cv_df['rev_MAPE'].mean()
rr  = cv_df['rev_R2'].mean()
mar = cv_df['rev_MAE'].mean()
mc  = cv_df['cogs_MAPE'].mean()
rc  = cv_df['cogs_R2'].mean()

print('\n' + '=' * 65)
print('✅ PHASE C COMPLETE')
print('=' * 65)
print(f'\n  Walk-Forward CV Results (mean × {len(FOLDS)} folds):')
print(f'  Revenue MAPE : {mr:.2f}%   (baseline: 25.53%,  Δ={25.53-mr:+.2f}pp)')
print(f'  Revenue R²   : {rr:.4f}   (baseline: 0.7704,  Δ={rr-0.7704:+.4f})')
print(f'  Revenue MAE  : {mar:,.0f} VND  (baseline: 612,312)')
print(f'  COGS MAPE    : {mc:.2f}%   (baseline: 23.49%,  Δ={23.49-mc:+.2f}pp)')
print(f'  COGS R²      : {rc:.4f}')
print(f'\n  Optuna best MAPE:')
print(f'    Revenue: {study_rev.best_value:.2f}%')
print(f'    COGS   : {study_cogs.best_value:.2f}%')
print(f'\n  Output files:')
for fn in ['model_lgbm_revenue.pkl','model_lgbm_cogs.pkl','phase_c_best_params.pkl',
           'oof_predictions.csv','phase_c_cv_report.csv','phase_c_submission_raw.csv',
           'phase_c_feature_importance_rev.csv','phase_c_feature_importance_cogs.csv',
           'phase_c_model_diagnostics.png','phase_c_optuna_history.png']:
    flag = '✅' if (OUT_DIR/fn).exists() else '⚠️ '
    print(f'  {flag} {fn}')
print('\n  ▶ Sẵn sàng cho Phase D: Submission Builder')