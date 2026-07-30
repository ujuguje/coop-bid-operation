# [final] Comparison CSV: original SAC inputs vs new LSTM-based inputs (validates forecast-module swap)
"""
Generate a comparison CSV between the original SAC input files and the new
LSTM-based files. Output: compare_SAC_inputs_orig_vs_LSTM.csv
"""
import numpy as np
import pandas as pd
import os

import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import LEGACY_INPUTS as ORIG, RL_INPUTS as LSTM_DIR, RES_FORECAST as RES

# Load files
orig_bid = pd.read_csv(os.path.join(ORIG, 'LMP_Bid.csv'),       header=None, encoding='cp949').values
lstm_bid = pd.read_csv(os.path.join(LSTM_DIR, 'LMP_Bid_LSTM.csv'),  header=None).values
orig_ope = pd.read_csv(os.path.join(ORIG, 'LMP_Ope.csv'),       header=None, encoding='cp949').values
lstm_ope = pd.read_csv(os.path.join(LSTM_DIR, 'LMP_Ope_LSTM.csv'),  header=None).values
orig_sol = pd.read_csv(os.path.join(ORIG, 'Solar_Bid.csv'),      header=None, encoding='cp949').values.flatten()
lstm_sol = pd.read_csv(os.path.join(LSTM_DIR, 'Solar_Bid_LSTM.csv'), header=None).values.flatten()

N  = min(len(orig_bid), len(lstm_bid))
Ns = min(len(orig_sol), len(lstm_sol))
N  = min(N, Ns)  # align all to shortest

rows = {}

# LMP_Bid: all 8 slots
for c in range(8):
    rows[f'LMPBid_orig_s{c+1}'] = orig_bid[:N, c]
    rows[f'LMPBid_lstm_s{c+1}'] = lstm_bid[:N, c]
    rows[f'LMPBid_diff_s{c+1}'] = lstm_bid[:N, c] - orig_bid[:N, c]

# LMP_Ope: all 8 cols (col0=actual, col1-7=bid shift)
for c in range(8):
    rows[f'LMPOpe_orig_c{c+1}'] = orig_ope[:N, c]
    rows[f'LMPOpe_lstm_c{c+1}'] = lstm_ope[:N, c]
    rows[f'LMPOpe_diff_c{c+1}'] = lstm_ope[:N, c] - orig_ope[:N, c]

# Solar_Bid: single column
rows['Solar_orig'] = orig_sol[:N]
rows['Solar_lstm'] = lstm_sol[:N]
rows['Solar_diff'] = lstm_sol[:N] - orig_sol[:N]

df = pd.DataFrame(rows)
out = os.path.join(RES, 'compare_SAC_inputs_orig_vs_LSTM.csv')
df.to_csv(out, index=True, index_label='row')
print(f'Saved: {out}')
print(f'Shape: {df.shape}  ({N} rows x {len(df.columns)} cols)')

# Summary stats
print()
print('=== LMP_Bid ($/MWh) ===')
for c in range(8):
    d = df[f'LMPBid_diff_s{c+1}']
    print(f'  slot{c+1}: MAE={d.abs().mean():.3f}  mean_diff={d.mean():+.3f}  Corr={df[f"LMPBid_orig_s{c+1}"].corr(df[f"LMPBid_lstm_s{c+1}"]):.4f}')

print()
print('=== LMP_Ope ($/MWh) ===')
for c in range(8):
    d = df[f'LMPOpe_diff_c{c+1}']
    label = 'actual/4' if c == 0 else f'bid_s{c}'
    print(f'  col{c+1} ({label}): MAE={d.abs().mean():.3f}  mean_diff={d.mean():+.3f}')

print()
print('=== Solar_Bid (MWh/15min) ===')
d = df['Solar_diff']
print(f'  MAE={d.abs().mean():.5f}  mean_diff={d.mean():+.5f}  Corr={df["Solar_orig"].corr(df["Solar_lstm"]):.4f}')
