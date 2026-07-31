# -*- coding: utf-8 -*-
# [final] Shared evaluation helpers for reproducing the paper's numbers — used by
# notebooks B1 (Table 4), B2 (Tables B.1/B.2), and B3 (Table 5).
# Key rules:
#   * Every evaluation runs in eval mode (dropout OFF) with deterministic actions
#     - the paper's evaluation standard
#   * Parameters['Cap_Pcs_Ope'] = tol + 2 must be set before creating environments (use set_cap_ope())
#   * Checkpoints: results/baseline & sensitivity (Single/Inde), results/coop_dasac (Coop-DASAC),
#     results/coop_dabc (Coop-DABC). Filenames: actor_{tag}_deg{D}_tol{T}_seed{S}.pth
"""Shared evaluation helpers for reproducing the paper's tables.

Typical use (from a notebook that did sys.path.insert(0, CODE)):

    from algorithms.eval_utils import (METHODS, set_cap_ope, make_envs,
                                       evaluate_checkpoint, find_ckpt, expert_daily)
    set_cap_ope(5.0)
    env_dual, env_sing = make_envs(Data_Test, tol=5.0, deg=5.0)
    daily = evaluate_checkpoint('det', find_ckpt(METHODS['Coop-DABC']['dirs'],
                                'coop_dabc', 5.0, 5.0, 0), env_dual, env_sing)
    # daily: (n_days, 4) = [bid, ope, deg, total] per test day
"""
import os
import numpy as np
import torch

from data.settings import Parameters
from models.networks import Actor, SquashedGaussianMLPActor, SingleAgentActor
from envs.env_multi import DualAgentEnv
from envs.env_single import SingleAgentEnv
from algorithms.utils import fast_evaluate, fast_evaluate_single

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../code/algorithms
ROOT = os.path.dirname(os.path.dirname(_HERE))              # .../APEN_Major_Revision
RES = os.path.join(ROOT, "results")
DIR_BASE = os.path.join(RES, "baseline")
DIR_SENS = os.path.join(RES, "sensitivity")
DIR_DASAC = os.path.join(RES, "coop_dasac")
DIR_DABC = os.path.join(RES, "coop_dabc")

# (checkpoint tag, loader kind, search dirs) for the paper's four learned methods.
# kind: 'single'=single-agent SAC, 'inde'=independent dual SAC, 'sac'=Coop-DASAC, 'det'=Coop-DABC (deterministic actor)
METHODS = {
    "Single-SAC": dict(tag="single_sac",    kind="single", dirs=[DIR_BASE, DIR_SENS]),
    "Inde-DASAC": dict(tag="inde_sac",      kind="inde",   dirs=[DIR_BASE, DIR_SENS]),
    "Coop-DASAC": dict(tag="coop_dasac", kind="sac",    dirs=[DIR_DASAC]),
    "Coop-DABC":  dict(tag="coop_dabc",  kind="det",    dirs=[DIR_DABC]),
}

OBS_BID, OBS_OPE, ACT, HID = 107, 108, 1, 128


def set_cap_ope(tol):
    """Operation action-cap rule (fix #1): Cap_Pcs_Ope = tol + 2. Call before creating envs/networks."""
    Parameters["Cap_Pcs_Ope"] = float(tol) + 2.0
    return Parameters["Cap_Pcs_Ope"]


def make_envs(data, tol, deg):
    """Create the (dual, single) evaluation environment pair. Call set_cap_ope(tol) first."""
    env_dual = DualAgentEnv(data, Parameters, tol=tol, degradation_cost_per_mwh=deg)
    env_sing = SingleAgentEnv(data, Parameters, degradation_cost_per_mwh=deg)
    return env_dual, env_sing


def find_ckpt(dirs, tag, deg, tol, seed):
    """Search the directory list for actor_{tag}_deg{deg}_tol{tol}_seed{seed}.pth."""
    for d in dirs:
        p = os.path.join(d, f"actor_{tag}_deg{deg}_tol{tol}_seed{seed}.pth")
        if os.path.exists(p):
            return p
    return None


def load_policy(kind, path):
    """Load a checkpoint and return the policy (or policies) in eval mode."""
    cap_b, cap_o = Parameters["Cap_Pcs_Bid"], Parameters["Cap_Pcs_Ope"]
    if kind == "single":
        pi = SingleAgentActor(OBS_BID, ACT, HID, cap_b)
        pi.load_state_dict(torch.load(path, map_location="cpu"))
        return pi.eval()
    if kind == "inde":
        ck = torch.load(path, map_location="cpu")
        pb = SingleAgentActor(OBS_BID, ACT, HID, cap_b); pb.load_state_dict(ck["bid"])
        po = SingleAgentActor(OBS_OPE, ACT, HID, cap_o); po.load_state_dict(ck["ope"])
        return pb.eval(), po.eval()
    if kind == "sac":
        pi = SquashedGaussianMLPActor(OBS_BID, OBS_OPE, ACT, HID, cap_b, cap_o)
        pi.load_state_dict(torch.load(path, map_location="cpu"))
        return pi.eval()
    pi = Actor(OBS_BID, OBS_OPE, ACT, HID, cap_b, cap_o)   # 'det' (Coop-DABC)
    pi.load_state_dict(torch.load(path, map_location="cpu"))
    return pi.eval()


def evaluate_checkpoint(kind, path, env_dual, env_sing):
    """Evaluate one checkpoint in eval mode. Returns (n_days, 4) = [bid, ope, deg, total]."""
    model = load_policy(kind, path)
    if kind == "single":
        def f(s, _pi=model):
            with torch.inference_mode():
                a, _ = _pi(torch.from_numpy(s), deterministic=True, with_logprob=False)
            return a[:, 0].numpy()
        _, daily = fast_evaluate_single(env_sing, f)
    elif kind == "inde":
        pb, po = model
        def f(sb, so, _pb=pb, _po=po):
            with torch.inference_mode():
                ab, _ = _pb(torch.from_numpy(sb), deterministic=True, with_logprob=False)
                ao, _ = _po(torch.from_numpy(so), deterministic=True, with_logprob=False)
            return ab[:, 0].numpy(), ao[:, 0].numpy()
        _, daily = fast_evaluate(env_dual, f)
    elif kind == "sac":
        def f(sb, so, _pi=model):
            with torch.inference_mode():
                ab, ao = _pi(torch.from_numpy(sb), torch.from_numpy(so),
                             deterministic=True, with_logprob=False)[:2]
            return ab[:, 0].numpy(), ao[:, 0].numpy()
        _, daily = fast_evaluate(env_dual, f)
    else:  # 'det'
        def f(sb, so, _pi=model):
            with torch.inference_mode():
                ab, ao = _pi(torch.from_numpy(sb), torch.from_numpy(so))
            return ab[:, 0].numpy(), ao[:, 0].numpy()
        _, daily = fast_evaluate(env_dual, f)
    return np.array(daily)


def evaluate_method(name, deg, tol, data, seeds):
    """Evaluate several seeds of a named method; returns the per-seed day-mean matrix (n_seed, 4).
    Missing seeds are skipped and reported on the console."""
    spec = METHODS[name]
    set_cap_ope(tol)
    env_dual, env_sing = make_envs(data, tol, deg)
    rows, used = [], []
    for s in seeds:
        p = find_ckpt(spec["dirs"], spec["tag"], deg, tol, s)
        if p is None:
            print(f"  [missing] {name} deg{deg} tol{tol} seed{s}")
            continue
        rows.append(evaluate_checkpoint(spec["kind"], p, env_dual, env_sing).mean(0))
        used.append(s)
    return np.array(rows), used


def expert_daily(deg, tol, data, idx_start):
    """Replay the OJPD (perfect-foresight expert) demonstrations through the test
    environment; returns daily (bid, ope, deg). idx_start: test-window start index
    (idx_val_end from settings)."""
    import pandas as pd
    from envs.run_expert_buffer import populate_expert_buffers
    csv = os.path.join(ROOT, "data", "processed", "expert_actions",
                       f"Offline_Expert_Action_joint_deg{deg}_tol{tol}.csv")
    df = pd.read_csv(csv, encoding="utf-8-sig")
    exp = {"Expert_Bid_Action": df["Bid_Action"].values[idx_start:],
           "Expert_Ope_Action": df["Ope_Action"].values[idx_start:]}
    set_cap_ope(tol)
    _, _, _, daily = populate_expert_buffers(data, exp, Parameters,
                                             tol=tol, degradation_cost_per_mwh=deg)
    return np.array(daily)   # (days, 3): [r_bid, r_ope, r_deg]


# ─── OOD (Table 5) construction helpers ─────────────────────────────────────
PRICE_KEYS = ["Data_Price_Bid", "Data_Price_Ope", "Data_Price_Set"]
SOLAR_KEYS = ["Data_Solar_Bid", "Data_Solar_Ope"]


def _col0(a):
    a = np.asarray(a, dtype=float)
    return a[:, 0] if (a.ndim == 2 and a.shape[1] > 1) else a.reshape(-1)


def ood_subsets(data):
    """Part A stress subsets (dict of test-day indices). Fixed criteria from
    manuscript sections 4.3.3/B.1: top-third price std, days containing spikes above
    the 95th percentile, bottom/top-third solar."""
    price = _col0(data["Data_Price_Ope"]); solar = _col0(data["Data_Solar_Ope"])
    n_days = len(price) // 96
    pday = price[:n_days * 96].reshape(n_days, 96)
    sday = solar[:n_days * 96].reshape(n_days, 96)
    day_std, day_solar = pday.std(axis=1), sday.sum(axis=1)
    p95 = np.percentile(pday, 95)
    vh, sl, sh = (np.percentile(day_std, 67), np.percentile(day_solar, 33),
                  np.percentile(day_solar, 67))
    return {
        "All Test Days": np.arange(n_days),
        "High Volatility": np.where(day_std >= vh)[0],
        "Extreme Price": np.where((pday > p95).any(axis=1))[0],
        "Low Solar": np.where(day_solar <= sl)[0],
        "High Solar": np.where(day_solar >= sh)[0],
    }


def slice_days(data, idx):
    """Slice a data dict by a list of day indices; returns a new dict."""
    return {k: np.concatenate([np.asarray(v)[i * 96:(i + 1) * 96] for i in idx], axis=0)
            for k, v in data.items()}


def perturb_data(data, price_scale=1.0, vol_scale=1.0, price_noise=0.0,
                 solar_noise=0.0, seed=0):
    """Part B synthetic distribution shifts: price-level scaling, volatility
    amplification (around daily means), Gaussian noise."""
    rng = np.random.default_rng(seed)
    out = {k: np.asarray(v, dtype=float).copy() for k, v in data.items()}
    for k in PRICE_KEYS:
        x = out[k]; mu = x.mean(); x = mu + vol_scale * (x - mu); x = x * price_scale
        if price_noise > 0:
            x = x + rng.normal(0, price_noise * (x.std() + 1e-9), size=x.shape)
        out[k] = x
    for k in SOLAR_KEYS:
        x = out[k]
        if solar_noise > 0:
            x = np.clip(x + rng.normal(0, solar_noise * (x.std() + 1e-9), size=x.shape), 0, None)
        out[k] = x
    return out


OOD_SCENARIOS = {
    "Nominal": dict(), "Price -20%": dict(price_scale=0.8), "Price +20%": dict(price_scale=1.2),
    "Volatility x1.5": dict(vol_scale=1.5), "Price noise 10%": dict(price_noise=0.10),
    "Solar noise 10%": dict(solar_noise=0.10),
}
