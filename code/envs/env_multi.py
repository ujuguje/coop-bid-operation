"""
Two-settlement electricity market environment for a dual-agent (bid + operation) PV-ESS system.

State spaces
  Bid agent   : [one-hot time (96), SoC, DA price forecast (8), solar forecast (1)] → dim 107
  Ope agent   : [one-hot time (96), SoC, DA commit (1), RT price (8), solar (1)]    → dim 108

Action spaces (continuous)
  Bid : battery power dispatch in DA market  ∈ [-Cap_Pcs_Bid, +Cap_Pcs_Bid] MW
  Ope : deviation from DA commitment         ∈ [-Cap_Pcs_Bid, +Cap_Pcs_Bid] MW

Key parameters
  tol                      : imbalance tolerance (MW) — symmetric ± band around DA commitment
  degradation_cost_per_mwh : linear battery degradation cost ($/MWh); deg_rate = cost × 0.25

Performance notes
  - All arrays cast to float32 at init (NN-compatible; 2× memory bandwidth vs float64)
  - Price and solar arrays pre-normalised at init → zero per-step arithmetic in state builders
  - State vectors built in-place into pre-allocated float32 buffers → zero heap allocation per step
  - Single-column arrays squeezed to 1D at init → plain scalar indexing, no .item() needed
"""
import numpy as np

_PRICE_STD = np.float32(10.0)
_SOLAR_STD = np.float32(50.0)


class DualAgentEnv:

    def __init__(self, data: dict, params: dict, tol: float = 5.0,
                 degradation_cost_per_mwh: float = 5.0):

        def _f32(x):
            return np.asarray(x, dtype=np.float32)

        def _sq(a):
            return a[:, 0] if (a.ndim > 1 and a.shape[1] == 1) else a

        price_bid = _f32(data['Data_Price_Bid'])   # (N, 8) DA price features
        price_ope = _f32(data['Data_Price_Ope'])   # (N, 8) RT price features
        price_set = _sq(_f32(data['Data_Price_Set']))  # (N,)  settlement price
        solar_bid = _sq(_f32(data['Data_Solar_Bid']))  # (N,)  DA solar forecast
        solar_ope = _sq(_f32(data['Data_Solar_Ope']))  # (N,)  RT actual solar

        # Raw values used in reward and action clipping
        self.solar_bid     = solar_bid
        self.solar_ope     = solar_ope
        self.price_set     = price_set
        self._p_ope_reward = price_ope[:, 0]   # (N,) — RT price for bid reward
        self._p_bid_reward = price_bid[:, 0]   # (N,) — DA price (getter only)

        # Pre-normalised arrays: state builders do a single array copy per step
        self._price_bid_norm = price_bid / _PRICE_STD  # (N, 8)
        self._price_ope_norm = price_ope / _PRICE_STD  # (N, 8)
        self._solar_bid_norm = solar_bid / _SOLAR_STD  # (N,)
        self._solar_ope_norm = solar_ope / _SOLAR_STD  # (N,)

        self.cap_ess     = params['Cap_ESS']
        self.cap_pcs_bid = params['Cap_Pcs_Bid']
        self.cap_pcs_ope = params['Cap_Pcs_Ope']
        self.delta       = params['Delta']
        self.soc_min     = params['SoC_Min']
        self.soc_initial = params['SoC_Initial']
        self.soc_max     = params['SoC_Max']
        self.action_list = params['Action_List']

        self.tol      = tol
        self.deg_rate = degradation_cost_per_mwh * 0.25

        self._n       = len(solar_bid)
        self._t       = 0
        self._soc_ope = self.soc_initial
        self._soc_bid = self.soc_initial
        self._commit  = 0.0

        # Pre-allocated state buffers — reused every step, no heap allocation
        self._state_bid = np.zeros(107, dtype=np.float32)
        self._state_ope = np.zeros(108, dtype=np.float32)
        self._snext_bid = np.zeros(107, dtype=np.float32)

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        self._soc_ope = self.soc_initial
        self._soc_bid = self.soc_initial

    # ── State builders ───────────────────────────────────────────────────────

    def get_state_bid(self):
        t = self._t
        s = self._state_bid
        s[(t - 1) % 96] = 0;  s[t % 96] = 1
        s[96]     = self._soc_bid / self.cap_ess
        s[97:105] = self._price_bid_norm[t]
        s[106]    = self._solar_bid_norm[t]
        return s

    def get_state_ope(self):
        t = self._t
        s = self._state_ope
        s[(t - 1) % 96] = 0;  s[t % 96] = 1
        s[96]     = self._soc_ope / self.cap_ess
        s[97]     = self._commit / _SOLAR_STD
        s[98:106] = self._price_ope_norm[t]
        s[107]    = self._solar_ope_norm[t]
        return s

    # ── Action processing ────────────────────────────────────────────────────

    def save_action_bid(self, raw_action):
        action, commit, charge = self._clip_bid(raw_action, self.solar_bid[self._t], self._soc_bid)
        self._action_bid = action
        self._commit     = commit
        self._charge_bid = charge
        return action, commit, None, None

    def save_action_ope(self, raw_action, da_commit, da_battery):
        action, sell_pur, charge = self._clip_ope(
            raw_action, self.solar_ope[self._t], self._soc_ope, da_commit, da_battery
        )
        self._action_ope   = action
        self._sell_pur_ope = sell_pur
        self._charge_ope   = charge
        return action, sell_pur, charge, self.solar_ope[self._t]

    def _clip_bid(self, action, solar, soc):
        if action >= 0:
            action = min(action, (self.soc_max - soc) / self.delta, self.cap_pcs_bid)
            charge = True
        else:
            action = -min(-action, (soc - self.soc_min) * self.delta, self.cap_pcs_bid)
            charge = False
        commit = float(-action + solar)   # solar is np.float32 scalar after squeeze
        return action, commit, charge

    def _clip_ope(self, action, solar, soc, da_commit, da_battery):
        solar = float(solar)
        raw   = action + da_battery

        max_ch  = min(self.cap_pcs_bid, (self.soc_max - soc) / self.delta)
        max_dis = min(self.cap_pcs_bid, (soc - self.soc_min) * self.delta)

        lower = max(-max_dis,   solar - da_commit - self.tol)
        upper = min( max_ch,    solar - da_commit + self.tol)

        final  = min(max(raw, lower), upper)
        charge = final >= 0
        return final, solar - final, charge

    # ── Reward ───────────────────────────────────────────────────────────────

    def save_reward(self):
        t     = self._t
        p_ope = self._p_ope_reward[t]   # float32 scalar
        p_set = self.price_set[t]       # float32 scalar (squeezed)

        self._reward_bid = float(self._commit * p_ope)
        imbalance        = (self._sell_pur_ope - self._commit) * p_set
        self._reward_ope = (float(imbalance), -abs(self._action_ope) * self.deg_rate)

    # ── SoC + time update ────────────────────────────────────────────────────

    def soc_update(self):
        if self._charge_ope:
            self._soc_ope += self._action_ope * self.delta
        else:
            self._soc_ope += self._action_ope / self.delta
        self._soc_bid = self._soc_ope

    def idx_tq_update(self):
        self._t  = (self._t + 1) % self._n
        self.Done = (self._t % 96 == 0)

    # ── Step outputs ─────────────────────────────────────────────────────────

    def step_bid(self):
        t = self._t
        s = self._snext_bid
        s[(t - 1) % 96] = 0;  s[t % 96] = 1
        s[96]     = self._soc_bid / self.cap_ess
        s[97:105] = self._price_bid_norm[t]
        s[106]    = self._solar_bid_norm[t]
        return s, self._reward_bid, self.Done, None

    def step_ope(self):
        return self.get_state_ope(), self._reward_ope, self.Done, None

    # ── Expert data helpers ──────────────────────────────────────────────────

    def expert_action_load(self, bid_actions, ope_actions):
        self._expert_bid = bid_actions
        self._expert_ope = ope_actions

    def expert_action_sample(self, bid: bool):
        return self._expert_bid[self._t] if bid else self._expert_ope[self._t]

    # ── Observation / action dimensions ─────────────────────────────────────

    def observation_dim_bid(self): return 107
    def observation_dim_ope(self): return 108
    def action_dim(self):          return 1 if self.action_list is None else len(self.action_list)
    def action_limit_bid(self):    return self.cap_pcs_bid
    def action_limit_ope(self):    return self.cap_pcs_ope
    def len_data(self):            return self._n

    # ── Getters used by evaluation ───────────────────────────────────────────

    def get_soc_ope(self):       return self._soc_ope
    def get_pv_bid(self):        return float(self.solar_bid[self._t])
    def get_pv_ope(self):        return float(self.solar_ope[self._t])
    def get_purchase_bid(self):  return self._commit
    def get_purchase_ope(self):  return self._sell_pur_ope
    def get_price_bid(self):     return float(self._p_bid_reward[self._t])
    def get_price_ope(self):     return float(self._p_ope_reward[self._t])

    def sample_action(self):
        bid = np.random.uniform(-self.cap_pcs_bid, self.cap_pcs_bid)
        ope = np.random.uniform(-self.cap_pcs_ope, self.cap_pcs_ope)
        return bid, ope
