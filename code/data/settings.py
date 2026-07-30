from pathlib import Path
import pandas as pd
import numpy as np

# ── Data paths (repo-relative: <root>/data/processed/...) ────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR   = _ROOT / 'data' / 'processed' / 'rl_inputs'
_EXPERT_DIR = _ROOT / 'data' / 'processed' / 'expert_actions'

# ── Raw data loading ─────────────────────────────────────────────────────────
# LSTM forecast files (8-column: current step + 7-step ahead)
_lmp_bid_raw = pd.read_csv(_DATA_DIR / 'LMP_Bid_LSTM.csv',  header=None).values   # (N, 8)
_lmp_ope_raw = pd.read_csv(_DATA_DIR / 'LMP_Ope_LSTM.csv',  header=None).values   # (N, 8)
_sol_bid_raw = pd.read_csv(_DATA_DIR / 'Solar_Bid_LSTM.csv', header=None).values  # (N, 1)

# Actual RT solar (realized, used for RT operation reward)
_sol_ope_raw = pd.read_csv(_DATA_DIR / 'Solar_Ope.csv', encoding='cp949', header=None).values  # (N, 1)

# Actual settlement price (realized market clearing price)
_lmp_set_raw = pd.read_csv(_DATA_DIR / 'LMP_Set.csv', encoding='cp949', header=None).values  # (N, 1)

# Align all arrays to the shortest file length (solar files may have 1 extra row)
_N = min(len(_lmp_bid_raw), len(_lmp_ope_raw), len(_lmp_set_raw),
         len(_sol_bid_raw), len(_sol_ope_raw))

Data_Price_Bid = _lmp_bid_raw[:_N]   # (N, 8) — DA price features (LSTM 8-step)
Data_Price_Ope = _lmp_ope_raw[:_N]   # (N, 8) — RT price features (LSTM 8-step)
Data_Price_Set = _lmp_set_raw[:_N]   # (N, 1) — actual settlement price
Data_Solar_Bid = _sol_bid_raw[:_N]   # (N, 1) — DA solar forecast
Data_Solar_Ope = _sol_ope_raw[:_N]   # (N, 1) — RT actual solar

# ── Train / Val / Test split ─────────────────────────────────────────────────
N = _N
TRAIN_DAYS = 60
VAL_DAYS   = 30
TEST_DAYS  = 0

_INTERVALS_PER_DAY = 96

idx_train_end = N - _INTERVALS_PER_DAY * TRAIN_DAYS
idx_val_end   = N - _INTERVALS_PER_DAY * VAL_DAYS

def _slice(arr, start, end):
    return arr[start:end] if end else arr[start:]

Data_Train = {
    'Data_Price_Bid': _slice(Data_Price_Bid, 0, idx_train_end),
    'Data_Price_Ope': _slice(Data_Price_Ope, 0, idx_train_end),
    'Data_Price_Set': _slice(Data_Price_Set, 0, idx_train_end),
    'Data_Solar_Bid': _slice(Data_Solar_Bid, 0, idx_train_end),
    'Data_Solar_Ope': _slice(Data_Solar_Ope, 0, idx_train_end),
}
Data_Val = {
    'Data_Price_Bid': _slice(Data_Price_Bid, idx_train_end, idx_val_end),
    'Data_Price_Ope': _slice(Data_Price_Ope, idx_train_end, idx_val_end),
    'Data_Price_Set': _slice(Data_Price_Set, idx_train_end, idx_val_end),
    'Data_Solar_Bid': _slice(Data_Solar_Bid, idx_train_end, idx_val_end),
    'Data_Solar_Ope': _slice(Data_Solar_Ope, idx_train_end, idx_val_end),
}
Data_Test = {
    'Data_Price_Bid': _slice(Data_Price_Bid, idx_val_end, None),
    'Data_Price_Ope': _slice(Data_Price_Ope, idx_val_end, None),
    'Data_Price_Set': _slice(Data_Price_Set, idx_val_end, None),
    'Data_Solar_Bid': _slice(Data_Solar_Bid, idx_val_end, None),
    'Data_Solar_Ope': _slice(Data_Solar_Ope, idx_val_end, None),
}

# ── Expert demonstrations ────────────────────────────────────────────────────
# Default: baseline scenario (deg=5.0, tol=5.0)
# For sensitivity runs, load via run_training.py --expert_csv
_expert = pd.read_csv(_EXPERT_DIR / 'Offline_Expert_Action_joint_deg5.0_tol5.0.csv')
_bid_actions = _expert['Bid_Action'].values
_ope_actions = _expert['Ope_Action'].values

Data_Expert_Train = {
    'Expert_Bid_Action': _bid_actions[:idx_train_end],
    'Expert_Ope_Action': _ope_actions[:idx_train_end],
}
Data_Expert_Val = {
    'Expert_Bid_Action': _bid_actions[idx_train_end:idx_val_end],
    'Expert_Ope_Action': _ope_actions[idx_train_end:idx_val_end],
}
Data_Expert_Test = {
    'Expert_Bid_Action': _bid_actions[idx_val_end:],
    'Expert_Ope_Action': _ope_actions[idx_val_end:],
}

# ── Battery parameters ───────────────────────────────────────────────────────
Parameters = {
    'Cap_ESS':     100,   # MWh
    'Cap_Pcs_Bid': 25,    # MW
    'Cap_Pcs_Ope': 7,     # MW
    'Delta':       0.95,
    'SoC_Min':     0,
    'SoC_Initial': 0,
    'SoC_Max':     100,
    'Action_List': None,
}
