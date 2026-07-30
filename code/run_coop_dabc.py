# -*- coding: utf-8 -*-
# [최종] 논문 Coop-DABC(Case D) 학습 러너 — 체크포인트: results/coop_dabc/
# 사용 예:  python run_coop_dabc.py --tol 5 --deg 5 --seed 0
# 논문 설정 = 10,000 스텝 (Table 3). 시드 0–9(본선), 민감도는 0–4에 deg/tol 변경.
"""Coop-DABC training runner (TD3+BC + monotonic QMIX mixer, w_final.abs() fix).
Loads the OJPD expert demonstrations, trains offline, saves the best-validation
actor checkpoint and a result summary json to results/coop_dabc/."""
import argparse, os, sys, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # .../APEN_Major_Revision/code
ROOT = os.path.dirname(HERE)                               # .../APEN_Major_Revision
DATA = os.path.join(ROOT, "data", "processed", "expert_actions")
RESULTS = os.path.join(ROOT, "results", "coop_dabc")
sys.path.insert(0, HERE)

from data.settings import Data_Train, Data_Val, Data_Test, Parameters, idx_train_end
from envs.env_multi import DualAgentEnv
from envs.run_expert_buffer import populate_expert_buffers
from algorithms import coop_dabc


def build(data, tol, deg):
    return DualAgentEnv(data, Parameters, tol=tol, degradation_cost_per_mwh=deg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol",  type=float, default=5.0)
    ap.add_argument("--deg",  type=float, default=5.0)
    ap.add_argument("--seed", type=int,   default=0)
    ap.add_argument("--alpha", type=float, default=2.5)   # BC coefficient (Table 3)
    ap.add_argument("--w_ope", type=float, default=1.0)   # operation BC weight
    ap.add_argument("--w_bid", type=float, default=1.0)   # bid BC weight
    ap.add_argument("--steps", type=int,   default=10000) # paper setting (Table 3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    wtag = "" if args.w_ope == 1.0 else f"_wope{args.w_ope}"
    wtag += "" if args.w_bid == 1.0 else f"_wbid{args.w_bid}"
    wtag += "" if args.alpha == 2.5 else f"_alpha{args.alpha}"
    tag = f"coop_dabc_deg{args.deg}_tol{args.tol}{wtag}_seed{args.seed}"
    res_path = os.path.join(RESULTS, f"result_{tag}.json")
    if os.path.exists(res_path) and not args.force:
        print(f"[skip] {res_path}"); sys.exit(0)

    Parameters["Cap_Pcs_Ope"] = args.tol + 2.0
    print(f"=== COOP_BC_MONO tol={args.tol} deg={args.deg} seed={args.seed} alpha={args.alpha} steps={args.steps} ===", flush=True)

    csv = os.path.join(DATA, f"Offline_Expert_Action_joint_deg{args.deg}_tol{args.tol}.csv")
    df = pd.read_csv(csv, encoding="utf-8-sig")
    EXP = {"Expert_Bid_Action": df["Bid_Action"].values[:idx_train_end],
           "Expert_Ope_Action": df["Ope_Action"].values[:idx_train_end]}
    bb, bo, _, _ = populate_expert_buffers(Data_Train, EXP, Parameters, tol=args.tol,
                                           degradation_cost_per_mwh=args.deg)

    actor_path = os.path.join(RESULTS, f"actor_{tag}.pth")
    vr, vd, test_daily, ti, timing = coop_dabc.train(
        build(Data_Train, args.tol, args.deg), build(Data_Val, args.tol, args.deg), build(Data_Test, args.tol, args.deg),
        bb, bo, seed=args.seed, max_steps=args.steps, alpha=args.alpha, bc_ope_w=args.w_ope, bc_bid_w=args.w_bid, eval_freq=50, save_actor_path=actor_path)

    a = np.array(test_daily) if test_daily else np.zeros((0, 4))
    if a.ndim == 2 and a.shape[1] >= 4:
        summary = {"algo": "coop_dabc", "deg_cost": args.deg, "tol": args.tol, "seed": args.seed, "alpha": args.alpha,
                   "w_ope": args.w_ope, "w_bid": args.w_bid, "steps": args.steps,
                   "test_mean_total": float(a[:, 3].mean()), "test_std_total": float(a[:, 3].std()),
                   "test_mean_bid": float(a[:, 0].mean()), "test_mean_ope": float(a[:, 1].mean()),
                   "test_mean_deg": float(a[:, 2].mean()),
                   "val_final": float(vr[-1]) if vr else 0.0, "val_returns": [float(v) for v in vr] if vr else [],
                   "total_train_s": float(timing.get("total_train_s", 0.0))}
        with open(res_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Done. total={summary['test_mean_total']:.0f}  bid={summary['test_mean_bid']:.0f}  "
              f"ope={summary['test_mean_ope']:.0f}  deg={summary['test_mean_deg']:.0f}  [Saved] {res_path}", flush=True)
