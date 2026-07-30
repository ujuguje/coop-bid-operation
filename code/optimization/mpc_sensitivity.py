# -*- coding: utf-8 -*-
# [최종] 민감도 (deg,tol) 설정별 MPC 총수익 — result_mpc_deg{d}_tol{t}_seed0.json 생성
"""Run MPC at the sensitivity (deg, tol) scenarios on the real test data, so MPC
can be added to the sensitivity figure. Saves result_mpc_deg{d}_tol{t}_seed0.json
(format compatible with the sensitivity figure: has 'test_mean_total')."""
import sys, os, json, time
import numpy as np
import cvxpy as cp

REV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(REV, "code"); SENS = os.path.join(REV, "results", "sensitivity"); BASE = os.path.join(REV, "results", "baseline")
sys.path.insert(0, CODE)
from data.settings import Data_Train, Data_Test, Parameters

CAP_ESS = float(Parameters['Cap_ESS']); CAP_PCS = float(Parameters['Cap_Pcs_Bid']); EFF = float(Parameters['Delta']); M_BIG = 1e6
pb_tr = np.asarray(Data_Train['Data_Price_Bid'], float); po_tr = np.asarray(Data_Train['Data_Price_Ope'], float)
sb_tr = np.asarray(Data_Train['Data_Solar_Bid'], float).flatten(); so_tr = np.asarray(Data_Train['Data_Solar_Ope'], float).flatten()
m_lb = np.array([pb_tr[q::96,0].mean() for q in range(96)]); m_lo = np.array([po_tr[q::96,0].mean() for q in range(96)])
m_sb = np.array([np.clip(sb_tr[q::96],0,None).mean() for q in range(96)]); m_so = np.array([np.clip(so_tr[q::96],0,None).mean() for q in range(96)])
pb = np.asarray(Data_Test['Data_Price_Bid'], float); po = np.asarray(Data_Test['Data_Price_Ope'], float)
ps = np.asarray(Data_Test['Data_Price_Set'], float).flatten()
sb = np.asarray(Data_Test['Data_Solar_Bid'], float).flatten(); so = np.asarray(Data_Test['Data_Solar_Ope'], float).flatten()

def fc(t, arr, mq, n_lstm=8):
    q = t % 96; nr = 96 - q; a = np.zeros(nr); nu = min(n_lstm, nr); a[:nu] = arr[t,:nu]
    for j in range(nu, nr): a[j] = mq[(q+j)%96]
    return a

def bid_solve(soc, lmp, sol):
    n = len(lmp); BC=cp.Variable(n,nonneg=True);BD=cp.Variable(n,nonneg=True);BS=cp.Variable(n,nonneg=True);BP=cp.Variable(n,nonneg=True)
    S=cp.Variable(n,nonneg=True);z=cp.Variable(n,boolean=True);z2=cp.Variable(n,boolean=True);con=[]
    for q in range(n):
        sp = soc if q==0 else S[q-1]
        con += [BS[q]==BP[q]+sol[q]+BD[q]-BC[q], BC[q]<=CAP_PCS, BD[q]<=CAP_PCS, S[q]<=CAP_ESS,
                BS[q]<=M_BIG*z[q], BP[q]<=M_BIG*(1-z[q]), BC[q]<=M_BIG*z2[q], BD[q]<=M_BIG*(1-z2[q]),
                S[q]==sp+BC[q]*EFF-BD[q]/EFF]
    con += [S[n-1]==0]; p=cp.Problem(cp.Maximize((BS-BP)@lmp),con); p.solve(solver=cp.HIGHS)
    if p.status in ('infeasible','unbounded') or BC.value is None: return None
    return float(BC.value[0]), float(BD.value[0]), (BS.value-BP.value).copy()

def ope_solve(soc, lmp, sol, commit, tol, degc):
    n=len(lmp); OC=cp.Variable(n,nonneg=True);OD=cp.Variable(n,nonneg=True);OS=cp.Variable(n,nonneg=True);OP=cp.Variable(n,nonneg=True)
    S=cp.Variable(n,nonneg=True);z=cp.Variable(n,boolean=True);z2=cp.Variable(n,boolean=True);con=[]
    for q in range(n):
        sp=soc if q==0 else S[q-1]; cq=float(commit[q])
        con += [OS[q]==OP[q]+sol[q]+OD[q]-OC[q], OC[q]<=CAP_PCS, OD[q]<=CAP_PCS, S[q]<=CAP_ESS,
                OS[q]<=M_BIG*z[q], OP[q]<=M_BIG*(1-z[q]), OC[q]<=M_BIG*z2[q], OD[q]<=M_BIG*(1-z2[q]),
                S[q]==sp+OC[q]*EFF-OD[q]/EFF, OS[q]-OP[q]<=cq+tol, OS[q]-OP[q]>=cq-tol]
    con += [S[n-1]==0]; p=cp.Problem(cp.Maximize((OS-OP)@lmp - cp.sum(OD+OC)*degc),con); p.solve(solver=cp.HIGHS)
    if p.status in ('infeasible','unbounded') or OC.value is None: return None
    return float(OC.value[0]), float(OD.value[0])

def run_mpc(tol, degc):
    N=len(ps); ND=N//96; daily=[]
    for day in range(ND):
        soc=0.0; tot=0.0
        for q in range(96):
            t=day*96+q; nr=96-q
            lb=fc(t,pb,m_lb); lo=fc(t,po,m_lo)
            s_b=np.array([max(0.,float(sb[t]))]+[m_sb[(q+j)%96] for j in range(1,nr)])
            s_o=np.array([max(0.,float(so[t]))]+[m_so[(q+j)%96] for j in range(1,nr)])
            br=bid_solve(soc,lb,s_b); BC0,BD0,cv=(max(0.,br[0]),max(0.,br[1]),br[2]) if br else (0.,0.,np.zeros(nr))
            orr=ope_solve(soc,lo,s_o,cv,tol,degc); OC0,OD0=(max(0.,orr[0]),max(0.,orr[1])) if orr else (0.,0.)
            sell_bid=max(0.,float(sb[t]))+BD0-BC0; sell_ope=max(0.,float(so[t]))+OD0-OC0; act=OC0-OD0
            tot += sell_bid*float(po[t,0]) + (sell_ope-sell_bid)*float(ps[t]) - abs(act)*degc
            soc=float(np.clip(soc+OC0*EFF-OD0/EFF,0.,CAP_ESS))
        daily.append(tot)
    return daily

# scenarios (deg, tol); (5,5) reused from baseline cache
scen = [(2.5,5.0),(7.5,5.0),(10.0,5.0),(5.0,2.0),(5.0,10.0)]

# (5,5) from cache
cache = os.path.join(BASE,'mpc_daily.json')
if os.path.exists(cache):
    dd=[float(r[3]) for r in json.load(open(cache))]
    json.dump({'algo':'mpc','deg_cost':5.0,'tol':5.0,'seed':0,'test_mean_total':float(np.mean(dd)),'test_std_total':float(np.std(dd))},
              open(os.path.join(SENS,'result_mpc_deg5.0_tol5.0_seed0.json'),'w'))
    print('[ (5,5) from cache ]', flush=True)

for deg, tol in scen:
    fp = os.path.join(SENS, f'result_mpc_deg{deg}_tol{tol}_seed0.json')
    if os.path.exists(fp): print(f'[skip] deg={deg} tol={tol}', flush=True); continue
    t0=time.time(); daily=run_mpc(tol, deg*0.25)
    json.dump({'algo':'mpc','deg_cost':deg,'tol':tol,'seed':0,
               'test_mean_total':float(np.mean(daily)),'test_std_total':float(np.std(daily))}, open(fp,'w'))
    print(f'[done] deg={deg} tol={tol}  mean={np.mean(daily):,.0f}  ({time.time()-t0:.0f}s)', flush=True)
print('MPC sensitivity done')
