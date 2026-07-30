# -*- coding: utf-8 -*-
# [최종] 논문 수치 재현용 공통 평가 헬퍼 — 노트북 B1(Table 4)·B2(Table B.1/B.2)·B3(Table 5)이 사용.
# 핵심 규칙:
#   * 모든 평가는 eval 모드(dropout OFF) + deterministic 액션 (학습 중 검증에 있던 train-mode 버그의 수정판)
#   * 환경 생성 전 Parameters['Cap_Pcs_Ope'] = tol + 2 를 반드시 설정 (set_cap_ope() 사용)
#   * 체크포인트 위치: results/baseline·sensitivity (Single/Inde), results/coop_dasac (Coop-DASAC),
#     results/coop_dabc (Coop-DABC). 파일명 actor_{tag}_deg{D}_tol{T}_seed{S}.pth
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

# 논문 4개 학습기법의 (체크포인트 태그, 로더 종류, 탐색 폴더).
# kind: 'single'=단일 에이전트 SAC, 'inde'=독립 이중 SAC, 'sac'=Coop-DASAC, 'det'=Coop-DABC(결정론적 actor)
METHODS = {
    "Single-SAC": dict(tag="single_sac",    kind="single", dirs=[DIR_BASE, DIR_SENS]),
    "Inde-DASAC": dict(tag="inde_sac",      kind="inde",   dirs=[DIR_BASE, DIR_SENS]),
    "Coop-DASAC": dict(tag="coop_dasac", kind="sac",    dirs=[DIR_DASAC]),
    "Coop-DABC":  dict(tag="coop_dabc",  kind="det",    dirs=[DIR_DABC]),
}

OBS_BID, OBS_OPE, ACT, HID = 107, 108, 1, 128


def set_cap_ope(tol):
    """운영 액션 상한 규칙 (fix #1): Cap_Pcs_Ope = tol + 2. 환경/네트워크 생성 전에 호출."""
    Parameters["Cap_Pcs_Ope"] = float(tol) + 2.0
    return Parameters["Cap_Pcs_Ope"]


def make_envs(data, tol, deg):
    """평가용 (dual, single) 환경 쌍 생성. set_cap_ope(tol)를 먼저 호출한다."""
    env_dual = DualAgentEnv(data, Parameters, tol=tol, degradation_cost_per_mwh=deg)
    env_sing = SingleAgentEnv(data, Parameters, degradation_cost_per_mwh=deg)
    return env_dual, env_sing


def find_ckpt(dirs, tag, deg, tol, seed):
    """actor_{tag}_deg{deg}_tol{tol}_seed{seed}.pth 를 폴더 목록에서 탐색."""
    for d in dirs:
        p = os.path.join(d, f"actor_{tag}_deg{deg}_tol{tol}_seed{seed}.pth")
        if os.path.exists(p):
            return p
    return None


def load_policy(kind, path):
    """체크포인트 로드 후 eval 모드 정책(들) 반환."""
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
    """체크포인트 1개를 eval 모드로 평가. 반환: (n_days, 4) = [bid, ope, deg, total]."""
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
    """method 이름으로 여러 시드를 평가해 per-seed day-mean 행렬 (n_seed, 4) 반환.
    누락 시드는 건너뛰고 콘솔에 표시."""
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
    """OJPD(완전예견 전문가) 시연을 테스트 환경으로 리플레이해 일별 (bid, ope, deg) 반환.
    idx_start: 테스트 구간 시작 인덱스 (settings의 idx_val_end)."""
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


# ─── OOD (Table 5) 구성 헬퍼 ─────────────────────────────────────────────────
PRICE_KEYS = ["Data_Price_Bid", "Data_Price_Ope", "Data_Price_Set"]
SOLAR_KEYS = ["Data_Solar_Bid", "Data_Solar_Ope"]


def _col0(a):
    a = np.asarray(a, dtype=float)
    return a[:, 0] if (a.ndim == 2 and a.shape[1] > 1) else a.reshape(-1)


def ood_subsets(data):
    """Part A 스트레스 서브셋 (테스트일 인덱스 dict). 원고 4.3.3/B.1의 고정 기준:
    가격 표준편차 상위 1/3, 95퍼센타일 초과 스파이크 보유일, 태양광 하위/상위 1/3."""
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
    """일 인덱스 목록으로 데이터 dict를 잘라 새 dict 반환."""
    return {k: np.concatenate([np.asarray(v)[i * 96:(i + 1) * 96] for i in idx], axis=0)
            for k, v in data.items()}


def perturb_data(data, price_scale=1.0, vol_scale=1.0, price_noise=0.0,
                 solar_noise=0.0, seed=0):
    """Part B 합성 분포이동: 가격 레벨 스케일, 변동성 확대(일평균 기준), 가우시안 노이즈."""
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
