"""
Entry point for training all four algorithm variants.

Usage (from APEN_Major_Revision/code/):
    python run_training.py --algo coop_bc  --seed 0
    python run_training.py --algo coop_sac --seed 0
    python run_training.py --algo inde_sac --seed 0
    python run_training.py --algo single_sac --seed 0

For sensitivity experiments:
    python run_training.py --algo coop_bc --tol 2        --seed 0
    python run_training.py --algo coop_bc --deg_cost 2.5 --seed 0
    python run_training.py --algo coop_bc --deg_cost 7.5 --seed 0
    python run_training.py --algo coop_bc --deg_cost 10.0 --seed 0
"""
import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from data.settings import (Data_Train, Data_Val, Data_Test,
                            Data_Expert_Train, Data_Expert_Val, Data_Expert_Test,
                            Parameters, idx_train_end)
from envs.env_multi  import DualAgentEnv
from envs.env_single import SingleAgentEnv
from models.buffer   import ReplayBuffer


def _load_expert_csv(csv_path: str) -> dict:
    """Load expert CSV and slice to train portion using settings split index."""
    import pandas as pd
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    bids = df['Bid_Action'].values
    opes = df['Ope_Action'].values
    return {
        'Expert_Bid_Action': bids[:idx_train_end],
        'Expert_Ope_Action': opes[:idx_train_end],
    }


def build_dual_env(data, tol, deg_cost):
    return DualAgentEnv(data, Parameters, tol=tol,
                        degradation_cost_per_mwh=deg_cost)


def build_single_env(data, deg_cost):
    return SingleAgentEnv(data, Parameters,
                          degradation_cost_per_mwh=deg_cost)


def run_coop_bc(args):
    from envs.run_expert_buffer import populate_expert_buffers
    from algorithms.coop_bc import train

    env_train = build_dual_env(Data_Train, args.tol, args.deg_cost)
    env_val   = build_dual_env(Data_Val,   args.tol, args.deg_cost)
    env_test  = build_dual_env(Data_Test,  args.tol, args.deg_cost)

    expert_train = (_load_expert_csv(args.expert_csv)
                    if args.expert_csv else Data_Expert_Train)
    buf_bid_train, buf_ope_train, _, _ = populate_expert_buffers(
        Data_Train, expert_train, Parameters,
        tol=args.tol, degradation_cost_per_mwh=args.deg_cost,
    )

    # Best-validation actor weights are saved here (for OOD / inference reuse)
    save_dir = (args.save_dir if args.save_dir
                else os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity'))
    actor_path = os.path.join(save_dir,
                              f'actor_coop_bc_deg{args.deg_cost}_tol{args.tol}_seed{args.seed}.pth')

    val_ret, val_daily, test_daily, test_impl, timing = train(
        env_train, env_val, env_test,
        buf_bid_train, buf_ope_train,
        seed=args.seed,
        save_actor_path=actor_path,
    )
    return val_ret, val_daily, test_daily, test_impl, timing


def run_coop_sac(args):
    from algorithms.coop_sac import train

    env_train = build_dual_env(Data_Train, args.tol, args.deg_cost)
    env_val   = build_dual_env(Data_Val,   args.tol, args.deg_cost)
    env_test  = build_dual_env(Data_Test,  args.tol, args.deg_cost)

    # Best-validation actor weights saved here (for OOD comparison)
    save_dir = (args.save_dir if args.save_dir
                else os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity'))
    actor_path = os.path.join(save_dir,
                              f'actor_coop_sac_deg{args.deg_cost}_tol{args.tol}_seed{args.seed}.pth')

    return train(env_train, env_val, env_test, seed=args.seed, save_actor_path=actor_path)


def _actor_path(args, algo):
    save_dir = (args.save_dir if args.save_dir
                else os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity'))
    return os.path.join(save_dir, f'actor_{algo}_deg{args.deg_cost}_tol{args.tol}_seed{args.seed}.pth')


def run_inde_sac(args):
    from algorithms.inde_sac import train

    env_train = build_dual_env(Data_Train, args.tol, args.deg_cost)
    env_val   = build_dual_env(Data_Val,   args.tol, args.deg_cost)
    env_test  = build_dual_env(Data_Test,  args.tol, args.deg_cost)

    return train(env_train, env_val, env_test, seed=args.seed,
                 save_actor_path=_actor_path(args, 'inde_sac'))


def run_single_sac(args):
    from algorithms.single_sac import train

    env_train = build_single_env(Data_Train, args.deg_cost)
    env_val   = build_single_env(Data_Val,   args.deg_cost)
    env_test  = build_single_env(Data_Test,  args.deg_cost)

    return train(env_train, env_val, env_test, seed=args.seed,
                 save_actor_path=_actor_path(args, 'single_sac'))


RUNNERS = {
    'coop_bc':    run_coop_bc,
    'coop_sac':   run_coop_sac,
    'inde_sac':   run_inde_sac,
    'single_sac': run_single_sac,
}

# eval_freq used by each algo's train() call — saved to JSON for plotting
ALGO_EVAL_FREQ = {
    'coop_bc':    50,
    'coop_sac':   200,
    'inde_sac':   200,
    'single_sac': 200,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',      choices=list(RUNNERS), required=True)
    parser.add_argument('--seed',      type=int,   default=0)
    parser.add_argument('--tol',       type=float, default=5.0,
                        help='Imbalance tolerance (MW). Default=5. Dual-agent only.')
    parser.add_argument('--deg_cost',  type=float, default=5.0,
                        help='Linear degradation cost ($/MWh). Default=5.0.')
    parser.add_argument('--expert_csv', type=str, default=None,
                        help='Path to expert action CSV (overrides settings.py). '
                             'Used by coop_bc to load deg_cost-matched expert demos.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override output directory for result JSON. '
                             'Defaults to results/sensitivity/.')
    args = parser.parse_args()

    # Operation action limit scales with the imbalance tolerance (tol + 2 MW headroom)
    # so the stochastic/cloned policy can actually reach the ±tol band — important for
    # large tolerances (e.g. tol=10, where the old fixed cap of 7 prevented the agent
    # from using the full band). At tol=5 this evaluates to 7, exactly preserving the
    # original baseline. Parameters is the shared dict imported from settings; we mutate
    # it in place before any env is built. Each run is a separate process, so there is
    # no cross-run interference.
    Parameters['Cap_Pcs_Ope'] = args.tol + 2.0
    print(f'[Cap_Pcs_Ope = {Parameters["Cap_Pcs_Ope"]} MW  (tol={args.tol} + 2 headroom)]')

    print(f'=== {args.algo.upper()}  seed={args.seed}  tol={args.tol}MW  deg=${args.deg_cost}/MWh ===')
    result = RUNNERS[args.algo](args)
    val_last = result[0][-1] if result[0] else float('nan')
    print(f'Done. val_returns[-1]={val_last:.0f}')

    # ── Save test results to JSON for sensitivity analysis collection ─────────
    import json
    try:
        if len(result) == 5:
            val_ret, val_daily, test_daily, test_impl, timing = result
        else:
            val_ret, val_daily, test_daily, test_impl = result
            timing = {}
        test_arr = np.array(test_daily) if test_daily else np.zeros((0, 4))

        if test_arr.ndim == 2 and test_arr.shape[1] >= 4:
            summary = {
                'algo':            args.algo,
                'deg_cost':        args.deg_cost,
                'tol':             args.tol,
                'seed':            args.seed,
                'test_mean_total': float(test_arr[:, 3].mean()),
                'test_std_total':  float(test_arr[:, 3].std()),
                'test_mean_bid':   float(test_arr[:, 0].mean()),
                'test_mean_ope':   float(test_arr[:, 1].mean()),
                'test_mean_deg':   float(test_arr[:, 2].mean()),
                'val_final':       float(val_ret[-1]) if val_ret else 0.0,
                'val_returns':     [float(v) for v in val_ret] if val_ret else [],
                'eval_freq':       ALGO_EVAL_FREQ.get(args.algo, 1),
                'total_train_s':   float(timing.get('total_train_s', 0.0)),
                'n_updates':       int(timing.get('n_updates', timing.get('n_steps', 0))),
                'avg_step_ms':     float(timing.get('avg_step_ms', 0.0)),
            }
            save_dir = (args.save_dir if args.save_dir
                        else os.path.join(os.path.dirname(__file__), '..', 'results', 'sensitivity'))
            os.makedirs(save_dir, exist_ok=True)
            tag  = f'{args.algo}_deg{args.deg_cost}_tol{args.tol}_seed{args.seed}'
            path = os.path.join(save_dir, f'result_{tag}.json')
            with open(path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f'[Saved] {path}')
    except Exception as e:
        print(f'[Result save skipped] {e}')
