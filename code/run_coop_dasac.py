# -*- coding: utf-8 -*-
# [final] Training runner for the paper's Coop-DASAC (Case C) — checkpoints: results/coop_dasac/
# Example:  python run_coop_dasac.py --tol 5 --deg 5 --seed 0
# Sweep example (PowerShell):  foreach ($s in 0..9) { python run_coop_dasac.py --tol 5 --deg 5 --seed $s }
"""Coop-DASAC training runner (SAC + monotonic QMIX mixer, w_final.abs() fix).
Mirrors run_training.py's coop_sac path but uses algorithms.coop_dasac and
sets Cap_Pcs_Ope = tol + 2. Saves checkpoint + summary json to results/coop_dasac/."""
import argparse, os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # .../APEN_Major_Revision/code
ROOT = os.path.dirname(HERE)                               # .../APEN_Major_Revision
RESULTS = os.path.join(ROOT, "results", "coop_dasac")
sys.path.insert(0, HERE)

from data.settings import Data_Train, Data_Val, Data_Test, Parameters
from envs.env_multi import DualAgentEnv
from algorithms import coop_dasac


def build(data, tol, deg):
    return DualAgentEnv(data, Parameters, tol=tol, degradation_cost_per_mwh=deg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol",  type=float, default=5.0)
    ap.add_argument("--deg",  type=float, default=5.0)
    ap.add_argument("--seed", type=int,   default=0)
    ap.add_argument("--force", action="store_true", help="re-run even if result json exists")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tag      = f"coop_dasac_deg{args.deg}_tol{args.tol}_seed{args.seed}"
    res_path = os.path.join(RESULTS, f"result_{tag}.json")
    if os.path.exists(res_path) and not args.force:
        print(f"[skip] result already exists: {res_path}")
        sys.exit(0)

    # operation action cap scales with tolerance (tol=5 -> 7, preserving baseline)
    Parameters["Cap_Pcs_Ope"] = args.tol + 2.0
    print(f"[Cap_Pcs_Ope = {Parameters['Cap_Pcs_Ope']} MW  (tol={args.tol} + 2)]")
    print(f"=== COOP_SAC_QFIX  seed={args.seed}  tol={args.tol}MW  deg=${args.deg}/MWh ===", flush=True)

    actor_path = os.path.join(RESULTS, f"actor_{tag}.pth")

    val_ret, val_daily, test_daily, test_impl, timing = coop_dasac.train(
        build(Data_Train, args.tol, args.deg),
        build(Data_Val,   args.tol, args.deg),
        build(Data_Test,  args.tol, args.deg),
        seed=args.seed,
        save_actor_path=actor_path,
    )

    test_arr = np.array(test_daily) if test_daily else np.zeros((0, 4))
    if test_arr.ndim == 2 and test_arr.shape[1] >= 4:
        summary = {
            "algo": "coop_dasac", "deg_cost": args.deg, "tol": args.tol, "seed": args.seed,
            "test_mean_total": float(test_arr[:, 3].mean()),
            "test_std_total":  float(test_arr[:, 3].std()),
            "test_mean_bid":   float(test_arr[:, 0].mean()),
            "test_mean_ope":   float(test_arr[:, 1].mean()),
            "test_mean_deg":   float(test_arr[:, 2].mean()),
            "val_final":       float(val_ret[-1]) if val_ret else 0.0,
            "val_returns":     [float(v) for v in val_ret] if val_ret else [],
            "total_train_s":   float(timing.get("total_train_s", 0.0)),
            "n_updates":       int(timing.get("n_updates", 0)),
        }
        with open(res_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nDone. test_mean_total={summary['test_mean_total']:.0f}")
        print(f"[Saved] {res_path}")
    else:
        print("[Result save skipped] no test_daily")
