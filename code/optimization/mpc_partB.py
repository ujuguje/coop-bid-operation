# -*- coding: utf-8 -*-
# [final] MPC rows of Table 5 Part B — re-runs MPC on the perturbed datasets. Cache: results/ood_analysis/mpc_partB.json
"""Run the rolling-horizon MPC on the Part B perturbed datasets (same perturbations
the RL policies saw, seed=0), so MPC can be added to the Part B comparison.
Saves per-scenario daily totals to results/ood_analysis/mpc_partB.json (incremental).
Nominal reuses the cached mpc_daily.json (no re-run)."""
import sys, os, json, time
import numpy as np
import cvxpy as cp

REV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(REV, "code")
OOD = os.path.join(REV, "results", "ood_analysis")
BASE = os.path.join(REV, "results", "baseline")
sys.path.insert(0, CODE)
from data.settings import Data_Train, Data_Test, Parameters

DEG, TOL = 5.0, 5.0
CAP_ESS = float(Parameters['Cap_ESS']); CAP_PCS = float(Parameters['Cap_Pcs_Bid'])
EFF = float(Parameters['Delta']); DEGRAD_COST = DEG * 0.25; M_BIG = 1e6

# ── time-of-day means from TRAIN data (NOT perturbed — far-horizon fill) ─────
pb_tr = np.asarray(Data_Train['Data_Price_Bid'], float); po_tr = np.asarray(Data_Train['Data_Price_Ope'], float)
sb_tr = np.asarray(Data_Train['Data_Solar_Bid'], float).flatten(); so_tr = np.asarray(Data_Train['Data_Solar_Ope'], float).flatten()
mean_lmp_bid_q   = np.array([pb_tr[q::96, 0].mean() for q in range(96)])
mean_lmp_ope_q   = np.array([po_tr[q::96, 0].mean() for q in range(96)])
mean_solar_bid_q = np.array([np.clip(sb_tr[q::96], 0, None).mean() for q in range(96)])
mean_solar_ope_q = np.array([np.clip(so_tr[q::96], 0, None).mean() for q in range(96)])

def build_forecast(t_abs, lstm_arr, mean_q, n_lstm=8):
    q = t_abs % 96; n_remain = 96 - q; arr = np.zeros(n_remain)
    n_use = min(n_lstm, n_remain); arr[:n_use] = lstm_arr[t_abs, :n_use]
    for j in range(n_use, n_remain): arr[j] = mean_q[(q + j) % 96]
    return arr

def mpc_bid_solve(soc_init, lmp_bid, sol_bid):
    n = len(lmp_bid)
    B_Cha = cp.Variable(n, nonneg=True); B_Dis = cp.Variable(n, nonneg=True)
    B_Sell = cp.Variable(n, nonneg=True); B_Purc = cp.Variable(n, nonneg=True)
    SoC_b = cp.Variable(n, nonneg=True)
    zb = cp.Variable(n, boolean=True); zb_ = cp.Variable(n, boolean=True)
    con = []
    for q in range(n):
        sp = soc_init if q == 0 else SoC_b[q-1]
        con += [B_Sell[q] == B_Purc[q] + sol_bid[q] + B_Dis[q] - B_Cha[q],
                B_Cha[q] <= CAP_PCS, B_Dis[q] <= CAP_PCS, SoC_b[q] <= CAP_ESS,
                B_Sell[q] <= M_BIG*zb[q], B_Purc[q] <= M_BIG*(1-zb[q]),
                B_Cha[q] <= M_BIG*zb_[q], B_Dis[q] <= M_BIG*(1-zb_[q]),
                SoC_b[q] == sp + B_Cha[q]*EFF - B_Dis[q]/EFF]
    con += [SoC_b[n-1] == 0]
    prob = cp.Problem(cp.Maximize((B_Sell - B_Purc) @ lmp_bid), con); prob.solve(solver=cp.HIGHS)
    if prob.status in ('infeasible', 'unbounded') or B_Cha.value is None: return None
    return {'B_Cha': float(B_Cha.value[0]), 'B_Dis': float(B_Dis.value[0]),
            'commit_vec': (B_Sell.value - B_Purc.value).copy()}

def mpc_ope_solve(soc_init, lmp_ope, sol_ope, commit_vec):
    n = len(lmp_ope)
    O_Cha = cp.Variable(n, nonneg=True); O_Dis = cp.Variable(n, nonneg=True)
    O_Sell = cp.Variable(n, nonneg=True); O_Purc = cp.Variable(n, nonneg=True)
    SoC = cp.Variable(n, nonneg=True)
    zo = cp.Variable(n, boolean=True); zo_ = cp.Variable(n, boolean=True)
    con = []
    for q in range(n):
        sp = soc_init if q == 0 else SoC[q-1]; cq = float(commit_vec[q])
        con += [O_Sell[q] == O_Purc[q] + sol_ope[q] + O_Dis[q] - O_Cha[q],
                O_Cha[q] <= CAP_PCS, O_Dis[q] <= CAP_PCS, SoC[q] <= CAP_ESS,
                O_Sell[q] <= M_BIG*zo[q], O_Purc[q] <= M_BIG*(1-zo[q]),
                O_Cha[q] <= M_BIG*zo_[q], O_Dis[q] <= M_BIG*(1-zo_[q]),
                SoC[q] == sp + O_Cha[q]*EFF - O_Dis[q]/EFF,
                O_Sell[q] - O_Purc[q] <= cq + TOL, O_Sell[q] - O_Purc[q] >= cq - TOL]
    con += [SoC[n-1] == 0]
    profit = (O_Sell - O_Purc) @ lmp_ope; deg = cp.sum(O_Dis + O_Cha) * DEGRAD_COST
    prob = cp.Problem(cp.Maximize(profit - deg), con); prob.solve(solver=cp.HIGHS)
    if prob.status in ('infeasible', 'unbounded') or O_Cha.value is None: return None
    return {'O_Cha': float(O_Cha.value[0]), 'O_Dis': float(O_Dis.value[0])}

def run_mpc(data):
    pb = np.asarray(data['Data_Price_Bid'], float); po = np.asarray(data['Data_Price_Ope'], float)
    ps = np.asarray(data['Data_Price_Set'], float).flatten()
    sb = np.asarray(data['Data_Solar_Bid'], float).flatten(); so = np.asarray(data['Data_Solar_Ope'], float).flatten()
    N = len(ps); ND = N // 96; daily = []
    for day in range(ND):
        soc = 0.0; d_bid = d_ope = d_deg = 0.0
        for q in range(96):
            t = day*96 + q; n_rem = 96 - q
            lmp_b = build_forecast(t, pb, mean_lmp_bid_q); lmp_o = build_forecast(t, po, mean_lmp_ope_q)
            sol_b = np.array([max(0., float(sb[t]))] + [mean_solar_bid_q[(q+j) % 96] for j in range(1, n_rem)])
            sol_o = np.array([max(0., float(so[t]))] + [mean_solar_ope_q[(q+j) % 96] for j in range(1, n_rem)])
            br = mpc_bid_solve(soc, lmp_b, sol_b)
            B_Cha0, B_Dis0 = (max(0., br['B_Cha']), max(0., br['B_Dis'])) if br else (0., 0.)
            orr = mpc_ope_solve(soc, lmp_o, sol_o, br['commit_vec'] if br else np.zeros(n_rem))
            O_Cha0, O_Dis0 = (max(0., orr['O_Cha']), max(0., orr['O_Dis'])) if orr else (0., 0.)
            sell_bid = max(0., float(sb[t])) + B_Dis0 - B_Cha0
            sell_ope = max(0., float(so[t])) + O_Dis0 - O_Cha0
            act = O_Cha0 - O_Dis0
            d_bid += sell_bid * float(po[t, 0]); d_ope += (sell_ope - sell_bid) * float(ps[t]); d_deg += -abs(act) * DEGRAD_COST
            soc = float(np.clip(soc + O_Cha0*EFF - O_Dis0/EFF, 0., CAP_ESS))
        daily.append(d_bid + d_ope + d_deg)
    return daily

PRICE_KEYS = ['Data_Price_Bid', 'Data_Price_Ope', 'Data_Price_Set']
SOLAR_KEYS = ['Data_Solar_Bid', 'Data_Solar_Ope']
def perturb_data(data, price_scale=1.0, vol_scale=1.0, price_noise=0.0, solar_noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: np.asarray(v, dtype=float).copy() for k, v in data.items()}
    for k in PRICE_KEYS:
        x = out[k]; mu = x.mean(); x = mu + vol_scale*(x-mu); x = x*price_scale
        if price_noise > 0: x = x + rng.normal(0, price_noise*(x.std()+1e-9), size=x.shape)
        out[k] = x
    for k in SOLAR_KEYS:
        x = out[k]
        if solar_noise > 0: x = np.clip(x + rng.normal(0, solar_noise*(x.std()+1e-9), size=x.shape), 0, None)
        out[k] = x
    return out

# Part B scenarios needing MPC (Nominal reuses cache)
scen = {'Price -20%': dict(price_scale=0.8), 'Price +20%': dict(price_scale=1.2),
        'Volatility x1.5': dict(vol_scale=1.5), 'Price noise 10%': dict(price_noise=0.10),
        'Solar noise 10%': dict(solar_noise=0.10)}

out_path = os.path.join(OOD, 'mpc_partB.json')
results = json.load(open(out_path)) if os.path.exists(out_path) else {}
# Nominal from cache
mpc_cache = os.path.join(BASE, 'mpc_daily.json')
if os.path.exists(mpc_cache):
    results['Nominal'] = [float(r[3]) for r in json.load(open(mpc_cache))]

for name, kw in scen.items():
    if name in results:
        print(f'[skip] {name} (already done)', flush=True); continue
    t0 = time.time()
    daily = run_mpc(perturb_data(Data_Test, **kw))
    results[name] = daily
    json.dump(results, open(out_path, 'w'))   # incremental save
    print(f'[done] {name}  mean={np.mean(daily):,.0f}  ({time.time()-t0:.0f}s)', flush=True)

print('\nSaved:', out_path)
for k, v in results.items():
    print(f'  {k:<16} mean daily = {np.mean(v):,.0f}')
