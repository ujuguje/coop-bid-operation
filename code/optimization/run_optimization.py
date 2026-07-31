"""
Run perfect-foresight MILP optimization and generate expert demonstration CSV.

Usage (from code/):
    # Joint optimization (default) → generates Offline_Expert_Action_*.csv
    python optimization/run_optimization.py --mode joint

    # Single-agent optimization for RT operation
    python optimization/run_optimization.py --mode single --agent ope

    # Custom degradation cost
    python optimization/run_optimization.py --mode joint --deg_cost 5.0

Output CSV (joint mode):
    Offline_Expert_Action_<tag>.csv  with columns [Bid_Action, Ope_Action]
    Results_Joint_<tag>.csv          with all optimization variables per slot

    Bid_Action = B_Cha - B_Dis   (DA battery dispatch, MW)
    Ope_Action = O_Cha - O_Dis - (B_Cha - B_Dis)  (RT deviation from DA, MW)
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.settings import (Data_Price_Bid, Data_Price_Ope, Data_Price_Set,
                            Data_Solar_Bid, Data_Solar_Ope, Parameters)
from optimization.milp_optimizer import MILPOptimizer


# ── Data helpers ─────────────────────────────────────────────────────────────

def _as_scalar(arr):
    """Flatten to 1-D and return a 1-D numpy array."""
    return np.asarray(arr).reshape(-1)


def _build_daily_data(lmp_bid, lmp_oper, solar_g, solar_bd, day):
    """Extract one day's data (96 slots) as shape (96,) arrays."""
    sl = slice(day * 96, (day + 1) * 96)
    return {
        'LMP_Bid':  lmp_bid[sl],
        'LMP_Oper': lmp_oper[sl],
        'Solar_G':  solar_g[sl],
        'Solar_bd': solar_bd[sl],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    # Map multi-column arrays to per-slot scalar arrays
    # LMP_Bid  : DA market clearing price   → price_ope column 0
    # LMP_Oper : RT imbalance settle price  → price_set column 0
    # Solar_G  : RT actual solar            → solar_ope column 0
    # Solar_bd : DA solar forecast          → solar_bid column 0
    lmp_bid  = _as_scalar(Data_Price_Ope[:, 0])
    lmp_oper = _as_scalar(Data_Price_Set[:, 0] if Data_Price_Set.ndim > 1
                          else Data_Price_Set)
    solar_g  = _as_scalar(Data_Solar_Ope[:, 0] if Data_Solar_Ope.ndim > 1
                          else Data_Solar_Ope)
    solar_bd = _as_scalar(Data_Solar_Bid[:, 0] if Data_Solar_Bid.ndim > 1
                          else Data_Solar_Bid)

    n_slots = len(lmp_bid)
    n_days  = n_slots // 96
    print(f'[run_optimization] {n_days} days  mode={args.mode}  '
          f'deg_cost=${args.deg_cost}/MWh  tol={args.tol}MW')

    opt = MILPOptimizer(Parameters, degradation_cost_per_mwh=args.deg_cost, tol=args.tol)

    combined = {}

    for day in range(n_days):
        data = _build_daily_data(lmp_bid, lmp_oper, solar_g, solar_bd, day)

        if args.mode == 'joint':
            result = opt.joint_optimize(data)
        else:
            result = opt.single_optimize(data, agent=args.agent)

        for key, val in result.items():
            arr = np.asarray(val)
            if arr.shape == ():
                arr = np.full(96, float(arr))
            if key in combined:
                combined[key] = np.concatenate([combined[key], arr])
            else:
                combined[key] = arr

        if (day + 1) % 50 == 0 or day == n_days - 1:
            print(f'  day {day + 1}/{n_days}')

    # ── Save full results CSV ─────────────────────────────────────────────────
    tag = f'{args.mode}_deg{args.deg_cost}_tol{args.tol}'
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'optimization_results')
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, f'Results_{tag}.csv')
    pd.DataFrame(combined).to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f'[run_optimization] Results saved → {results_path}')

    # ── Generate expert action CSV (joint mode only) ──────────────────────────
    if args.mode == 'joint':
        bid_action = combined['B_Cha'] - combined['B_Dis']
        ope_action = (combined['O_Cha'] - combined['O_Dis']) - (combined['B_Cha'] - combined['B_Dis'])

        expert_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'processed', 'expert_actions',
            f'Offline_Expert_Action_{tag}.csv'
        )
        pd.DataFrame({'Bid_Action': bid_action,
                      'Ope_Action': ope_action}).to_csv(
            expert_path, index=False, encoding='utf-8-sig')
        print(f'[run_optimization] Expert actions saved → {expert_path}')
        print('Update data/settings.py to point to the new expert CSV if needed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',      choices=['joint', 'single'], default='joint')
    parser.add_argument('--agent',     choices=['bid', 'ope'],      default='ope',
                        help='Agent type for single mode.')
    parser.add_argument('--deg_cost',  type=float, default=5.0,
                        help='Linear degradation cost for MILP objective ($/MWh).')
    parser.add_argument('--tol',       type=float, default=5.0,
                        help='Imbalance tolerance in MILP constraints (MW). Default=5.0.')
    args = parser.parse_args()
    run(args)
