# [최종] LSTM 예측 결과 -> RL 환경 입력 CSV(LMP_Bid/Ope, Solar_Bid) 생성 (파이프라인 1단계)
"""
gen_sac_inputs_LSTM.py

New LSTM benchmark predictions → SAC/BC input CSVs

Output files (written to SAC_DATA_DIR):
  LMP_Bid_LSTM.csv   (N_orig, 8)  — new LSTM price forecast / 4
  LMP_Ope_LSTM.csv   (N_orig, 8)  — col0: actual target/4, col1-7: LMP_Bid_LSTM[t+1][0:7]
  Solar_Bid_LSTM.csv (N_orig_solar, 1)  — new LSTM solar forecast x scale

Comparison output (written to RESULTS_DIR):
  compare_LMP_Bid_orig_vs_LSTM.csv  — row-by-row diff between original and new
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (RES_FORECAST as RESULTS_DIR,
                   LEGACY_INPUTS as ORIG_DATA_DIR,
                   RL_INPUTS as SAC_DATA_DIR)

# ── Load new LSTM predictions ─────────────────────────────────────────────────
print('Loading LSTM predictions...')
lstm_price = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions_LSTM_Price.csv'))
lstm_solar = pd.read_csv(os.path.join(RESULTS_DIR, 'predictions_LSTM_Solar.csv'))
N_lstm_price = len(lstm_price)
N_lstm_solar = len(lstm_solar)
print(f'  LSTM price predictions: {N_lstm_price} rows')
print(f'  LSTM solar predictions: {N_lstm_solar} rows')

# ── Load original SAC files ───────────────────────────────────────────────────
print('Loading original SAC input files...')
orig_bid       = pd.read_csv(os.path.join(ORIG_DATA_DIR, 'LMP_Bid.csv'),   header=None, encoding='cp949').values
orig_ope       = pd.read_csv(os.path.join(ORIG_DATA_DIR, 'LMP_Ope.csv'),   header=None, encoding='cp949').values
orig_solar_bid = pd.read_csv(os.path.join(ORIG_DATA_DIR, 'Solar_Bid.csv'), header=None, encoding='cp949').values
N_orig       = len(orig_bid)
N_orig_solar = len(orig_solar_bid)
print(f'  LMP_Bid:   {N_orig} rows  |  LSTM price: {N_lstm_price}  gap: {N_orig - N_lstm_price}')
print(f'  Solar_Bid: {N_orig_solar} rows  |  LSTM solar: {N_lstm_solar}  gap: {N_orig_solar - N_lstm_solar}')

out_cols = [f'output_{i}' for i in range(1, 9)]
tgt_cols = [f'target_{i}' for i in range(1, 9)]

# ── 1. LMP_Bid_LSTM: LSTM output_1..8 / 4 ────────────────────────────────────
new_bid_vals = lstm_price[out_cols].values.astype(np.float64) / 4.0   # (N_lstm_price, 8)

if N_lstm_price <= N_orig:
    # LSTM shorter: pad tail with original values
    new_bid_full = np.vstack([new_bid_vals, orig_bid[N_lstm_price:]])
else:
    # LSTM longer: truncate to SAC length
    new_bid_full = new_bid_vals[:N_orig]
assert new_bid_full.shape == (N_orig, 8), f"Shape mismatch: {new_bid_full.shape}"

# ── 2. LMP_Ope_LSTM ───────────────────────────────────────────────────────────
# col 0 = actual realized price (target_1 / 4)
# col 1-7 = LMP_Bid_LSTM[t+1, 0:7]
actual_col0 = lstm_price['target_1'].values.astype(np.float64) / 4.0   # (N_lstm_price,)

ope_rows = np.empty((N_lstm_price, 8), dtype=np.float64)
ope_rows[:, 0]     = actual_col0
ope_rows[:-1, 1:8] = new_bid_vals[1:, 0:7]
ope_rows[-1, 1:8]  = new_bid_vals[-1, 0:7]

if N_lstm_price <= N_orig:
    new_ope_full = np.vstack([ope_rows, orig_ope[N_lstm_price:]])
else:
    new_ope_full = ope_rows[:N_orig]
assert new_ope_full.shape == (N_orig, 8), f"Shape mismatch: {new_ope_full.shape}"

# ── 3. Solar_Bid_LSTM ─────────────────────────────────────────────────────────
# Max based on target columns only (matching original pipeline)
max_solar = float(lstm_solar[tgt_cols].values.max())
scale = 50.0 / max_solar
print(f'  Solar scale factor: 50 / {max_solar:.1f} = {scale:.6f}')
print(f'  Applied: output * scale * 0.25  (MW -> MWh/15min, same as original)')

solar_pred   = lstm_solar['output_1'].values.astype(np.float64) * scale * 0.25  # MWh/15min
solar_pred   = np.clip(solar_pred, 0.0, None)
actual_solar = lstm_solar['target_1'].values.astype(np.float64)
solar_pred[actual_solar == 0.0] = 0.0                                  # zero where actual=0 (night)

if N_lstm_solar <= N_orig_solar:
    solar_gap = orig_solar_bid[N_lstm_solar:].flatten()
    new_solar_full = np.concatenate([solar_pred, solar_gap]).reshape(-1, 1)
else:
    new_solar_full = solar_pred[:N_orig_solar].reshape(-1, 1)
assert new_solar_full.shape == (N_orig_solar, 1), f"Shape mismatch: {new_solar_full.shape}"

# ── Save new files ────────────────────────────────────────────────────────────
print('\nSaving new SAC input files...')

bid_out   = os.path.join(SAC_DATA_DIR, 'LMP_Bid_LSTM.csv')
ope_out   = os.path.join(SAC_DATA_DIR, 'LMP_Ope_LSTM.csv')
solar_out = os.path.join(SAC_DATA_DIR, 'Solar_Bid_LSTM.csv')

pd.DataFrame(new_bid_full).to_csv(bid_out,   header=False, index=False)
pd.DataFrame(new_ope_full).to_csv(ope_out,   header=False, index=False)
pd.DataFrame(new_solar_full).to_csv(solar_out, header=False, index=False)

print(f'  Saved: {bid_out}')
print(f'  Saved: {ope_out}')
print(f'  Saved: {solar_out}')

# ── Comparison: original LMP_Bid vs new LMP_Bid (LSTM) ───────────────────────
N_cmp = min(N_lstm_price, N_orig)   # compare over the overlapping rows only
print(f'\n=== Comparison: Original LMP_Bid vs New LMP_Bid_LSTM (first {N_cmp} rows) ===\n')

for col_idx, label in enumerate(['slot1 (t+0)', 'slot2 (t+1)', 'slot4 (t+3)', 'slot8 (t+7)']):
    ci = [0, 1, 3, 7][col_idx]
    orig_c = orig_bid[:N_cmp, ci]
    new_c  = new_bid_vals[:N_cmp, ci]
    diff   = new_c - orig_c
    corr   = np.corrcoef(orig_c, new_c)[0, 1]
    print(f'  {label:18s}  MAE={np.abs(diff).mean():.4f}  '
          f'MeanDiff={diff.mean():+.4f}  Corr={corr:.4f}')

orig_all = orig_bid[:N_cmp, :]
new_all  = new_bid_vals[:N_cmp, :]
diff_all = new_all - orig_all
print(f'\n  All 8 slots (avg)    MAE={np.abs(diff_all).mean():.4f}  '
      f'Corr={np.corrcoef(orig_all.flatten(), new_all.flatten())[0,1]:.4f}')

N_cmp_s  = min(N_lstm_solar, N_orig_solar)
N_cmp_all = min(N_cmp, N_cmp_s)   # use the shorter of price/solar to keep all columns equal length
compare_df = pd.DataFrame({
    'orig_LMP_Bid_col0':   orig_bid[:N_cmp_all, 0],
    'new_LMP_Bid_col0':    new_bid_vals[:N_cmp_all, 0],
    'diff_col0':           new_bid_vals[:N_cmp_all, 0] - orig_bid[:N_cmp_all, 0],
    'orig_solar_bid_col0': orig_solar_bid[:N_cmp_all, 0],
    'new_solar_bid':       solar_pred[:N_cmp_all],
    'diff_solar':          solar_pred[:N_cmp_all] - orig_solar_bid[:N_cmp_all, 0],
})
cmp_path = os.path.join(RESULTS_DIR, 'compare_LMP_Bid_orig_vs_LSTM.csv')
compare_df.to_csv(cmp_path, index=False)
print(f'\n  Saved comparison -> {cmp_path}')
print('\nDone.')
