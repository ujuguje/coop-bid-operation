"""
Perfect-foresight MILP optimizer for PV-ESS two-settlement bidding.

Degradation: linear proxy (deg_cost $/MWh × 0.25 h × throughput MW) — cvxpy-compatible.
"""
import cvxpy as cp
import numpy as np


# ── MILP Optimizer ───────────────────────────────────────────────────────────

class MILPOptimizer:
    """
    Perfect-foresight MILP for a 96-slot (15-min) day.

    Objective per day:
        max  profit_bid + profit_oper  −  deg_linear
    where
        profit_bid   = Σ (B_Sell − B_Purc) × LMP_Bid
        profit_oper  = Σ (O_Sell − B_Sell − O_Purc + B_Purc) × LMP_Oper
        deg_linear   = Σ (O_Dis + O_Cha) × deg_rate

    Args:
        params                  : battery parameter dict (from data/settings.py)
        degradation_cost_per_mwh: linear degradation cost ($/MWh).  Default 5.
    """

    _BIG_M = 1e6

    def __init__(self, params: dict, degradation_cost_per_mwh: float = 5.0,
                 tol: float = 5.0):
        self.cap_ess     = params['Cap_ESS']
        self.cap_pcs_bid = params['Cap_Pcs_Bid']
        self.cap_pcs_ope = params['Cap_Pcs_Ope']
        self.delta       = params['Delta']
        self.deg_rate    = degradation_cost_per_mwh * 0.25  # $/MW/slot
        self.tol         = tol                               # MW imbalance tolerance
        self._Q          = range(96)

    def joint_optimize(self, data: dict) -> dict:
        """
        Joint DA-bid + RT-operation optimization.

        data keys: 'LMP_Bid', 'LMP_Oper', 'Solar_G', 'Solar_bd'  — each shape (96,)
        """
        LMP_Bid  = data['LMP_Bid']
        LMP_Oper = data['LMP_Oper']
        Solar_G  = data['Solar_G']
        Solar_bd = data['Solar_bd']
        M = self._BIG_M

        B_Sell = cp.Variable(96, nonneg=True)
        B_Purc = cp.Variable(96, nonneg=True)
        O_Sell = cp.Variable(96, nonneg=True)
        O_Purc = cp.Variable(96, nonneg=True)
        SoC    = cp.Variable(96, nonneg=True)
        B_Cha  = cp.Variable(96, nonneg=True)
        B_Dis  = cp.Variable(96, nonneg=True)
        O_Cha  = cp.Variable(96, nonneg=True)
        O_Dis  = cp.Variable(96, nonneg=True)

        z_bid   = cp.Variable(96, boolean=True)
        z_oper  = cp.Variable(96, boolean=True)
        z_bid_  = cp.Variable(96, boolean=True)
        z_oper_ = cp.Variable(96, boolean=True)

        con = []
        for q in self._Q:
            con += [
                B_Sell[q] == B_Purc[q] + Solar_bd[q] + B_Dis[q] - B_Cha[q],
                O_Sell[q] == O_Purc[q] + Solar_G[q]  + O_Dis[q] - O_Cha[q],
                B_Cha[q] <= self.cap_pcs_bid,
                B_Dis[q] <= self.cap_pcs_bid,
                O_Cha[q] <= self.cap_pcs_bid,
                O_Dis[q] <= self.cap_pcs_bid,
                SoC[q]   <= self.cap_ess,
                # Mutual exclusivity (big-M)
                B_Sell[q] <= M * z_bid[q],
                B_Purc[q] <= M * (1 - z_bid[q]),
                O_Sell[q] <= M * z_oper[q],
                O_Purc[q] <= M * (1 - z_oper[q]),
                B_Cha[q]  <= M * z_bid_[q],
                B_Dis[q]  <= M * (1 - z_bid_[q]),
                O_Cha[q]  <= M * z_oper_[q],
                O_Dis[q]  <= M * (1 - z_oper_[q]),
                # Imbalance tolerance: |net_bid − net_oper| ≤ tol MW
                B_Sell[q] - B_Purc[q] - O_Sell[q] + O_Purc[q] <=  self.tol,
                B_Sell[q] - B_Purc[q] - O_Sell[q] + O_Purc[q] >= -self.tol,
            ]
            if q == 0:
                con += [
                    SoC[q] == self.delta * O_Cha[q] - O_Dis[q] / self.delta,
                    B_Dis[q] == 0,
                ]
            else:
                con += [
                    SoC[q] == SoC[q - 1] + self.delta * O_Cha[q] - O_Dis[q] / self.delta,
                    SoC[q - 1] + B_Cha[q] * self.delta <= self.cap_ess,
                    SoC[q - 1] - B_Dis[q] / self.delta >= 0,
                ]
        con += [SoC[95] == 0]

        profit_bid  = cp.sum((B_Sell - B_Purc) * LMP_Bid)
        profit_oper = cp.sum((O_Sell - B_Sell - O_Purc + B_Purc) * LMP_Oper)
        deg_linear  = cp.sum((O_Dis + O_Cha) * self.deg_rate)

        problem = cp.Problem(cp.Maximize(profit_bid + profit_oper - deg_linear), con)
        problem.solve()

        soc_val   = SoC.value
        b_sell    = B_Sell.value;  b_purc = B_Purc.value
        o_sell    = O_Sell.value;  o_purc = O_Purc.value
        b_cha     = B_Cha.value;   b_dis  = B_Dis.value
        o_cha     = O_Cha.value;   o_dis  = O_Dis.value

        return {
            'B_Sell': b_sell, 'B_Purc': b_purc,
            'O_Sell': o_sell, 'O_Purc': o_purc,
            'B_Cha':  b_cha,  'B_Dis':  b_dis,
            'O_Cha':  o_cha,  'O_Dis':  o_dis,
            'SoC':    soc_val,
            'LMP_Bid': LMP_Bid, 'LMP_Oper': LMP_Oper,
            'Solar_G': Solar_G, 'Solar_bd': Solar_bd,
            'Profit_bid':    (b_sell - b_purc) * LMP_Bid,
            'Profit_oper':   (o_sell - b_sell - o_purc + b_purc) * LMP_Oper,
            'Degradation':   (o_dis + o_cha) * self.deg_rate,
            'Total_Profit':  (b_sell - b_purc) * LMP_Bid
                             + (o_sell - b_sell - o_purc + b_purc) * LMP_Oper
                             - (o_dis + o_cha) * self.deg_rate,
        }

    def single_optimize(self, data: dict, agent: str = 'ope') -> dict:
        """
        Single-agent optimization.

        agent='bid' : optimizes DA bid market only (uses LMP_Bid, Solar_bd).
        agent='ope' : optimizes RT operation only  (uses LMP_Oper, Solar_G).
        """
        LMP     = data['LMP_Bid']  if agent == 'bid' else data['LMP_Oper']
        Solar_G = data['Solar_bd'] if agent == 'bid' else data['Solar_G']
        M       = self._BIG_M

        Sell = cp.Variable(96, nonneg=True)
        Purc = cp.Variable(96, nonneg=True)
        SoC  = cp.Variable(96, nonneg=True)
        Cha  = cp.Variable(96, nonneg=True)
        Dis  = cp.Variable(96, nonneg=True)
        z    = cp.Variable(96, boolean=True)
        z_   = cp.Variable(96, boolean=True)

        con = []
        for q in self._Q:
            con += [
                Sell[q] == Purc[q] + Solar_G[q] + Dis[q] - Cha[q],
                Cha[q]  <= self.cap_pcs_bid,
                Dis[q]  <= self.cap_pcs_bid,
                SoC[q]  <= self.cap_ess,
                Sell[q] <= M * z[q],
                Purc[q] <= M * (1 - z[q]),
                Cha[q]  <= M * z_[q],
                Dis[q]  <= M * (1 - z_[q]),
            ]
            if q == 0:
                con += [
                    SoC[q] == self.delta * Cha[q] - Dis[q] / self.delta,
                    Dis[q] == 0,
                ]
            else:
                con += [SoC[q] == SoC[q - 1] + self.delta * Cha[q] - Dis[q] / self.delta]
        con += [SoC[95] == 0]

        profit    = cp.sum((Sell - Purc) * LMP)
        deg_linear = cp.sum((Dis + Cha) * self.deg_rate)

        problem = cp.Problem(cp.Maximize(profit - deg_linear), con)
        problem.solve()

        sell_val = Sell.value;  purc_val = Purc.value
        cha_val  = Cha.value;   dis_val  = Dis.value
        soc_val  = SoC.value

        return {
            'Sell': sell_val, 'Purc': purc_val,
            'Cha':  cha_val,  'Dis':  dis_val,
            'SoC':  soc_val,
            'LMP':  LMP, 'Solar_G': Solar_G,
            'Profit':       (sell_val - purc_val) * LMP,
            'Degradation':  (dis_val + cha_val) * self.deg_rate,
            'Total_Profit': (sell_val - purc_val) * LMP
                            - (dis_val + cha_val) * self.deg_rate,
        }
