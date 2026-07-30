"""
Two-settlement market environment for a single-agent PV-ESS system.
The single agent controls battery dispatch for both DA and RT markets simultaneously.

State: [one-hot time (96), SoC, DA price forecast (8), solar (1)] → dim 107 (bid)
       [one-hot time (96), SoC, DA commit (1), RT price (8), solar (1)] → dim 108 (ope)
Action: battery power ∈ [-Cap_Pcs_Bid, +Cap_Pcs_Bid] MW (same action applied to both markets)

Performance notes
  - All arrays cast to float32 at init (NN-compatible; 2× memory bandwidth vs float64)
  - Price and solar arrays pre-normalised at init → zero per-step arithmetic in state builders
  - State vectors built in-place into pre-allocated float32 buffers → zero heap allocation per step
    (replaces previous np.hstack which allocated a new 107/108-element array every step)
  - Single-column arrays squeezed to 1D at init → plain scalar indexing
"""
import numpy as np

_PRICE_STD = np.float32(10.0)
_SOLAR_STD = np.float32(50.0)


class SingleAgentEnv:

    def __init__(self, data: dict, params: dict, degradation_cost_per_mwh: float = 5.0):

        def _f32(x):
            return np.asarray(x, dtype=np.float32)

        def _sq(a):
            return a[:, 0] if (a.ndim > 1 and a.shape[1] == 1) else a

        price_bid = _f32(data['Data_Price_Bid'])   # (N, 8)
        price_ope = _f32(data['Data_Price_Ope'])   # (N, 8)
        price_set = _sq(_f32(data['Data_Price_Set']))  # (N,)
        solar_bid = _sq(_f32(data['Data_Solar_Bid']))  # (N,)
        solar_ope = _sq(_f32(data['Data_Solar_Ope']))  # (N,)

        self.solar_bid     = solar_bid
        self.solar_ope     = solar_ope
        self.price_set     = price_set
        self._p_ope_reward = price_ope[:, 0]
        self._p_bid_reward = price_bid[:, 0]

        # Pre-normalised arrays: state builders copy directly, no per-step arithmetic
        self._price_bid_norm = price_bid / _PRICE_STD
        self._price_ope_norm = price_ope / _PRICE_STD
        self._solar_bid_norm = solar_bid / _SOLAR_STD
        self._solar_ope_norm = solar_ope / _SOLAR_STD

        self.cap_ess     = params['Cap_ESS']
        self.cap_pcs     = params['Cap_Pcs_Bid']
        self.delta       = params['Delta']
        self.soc_min     = params['SoC_Min']
        self.soc_initial = params['SoC_Initial']
        self.soc_max     = params['SoC_Max']
        self.action_list = params['Action_List']

        self.deg_rate = degradation_cost_per_mwh * 0.25

        self._n       = len(solar_bid)
        self._t       = 0
        self._soc     = self.soc_initial
        self._commit  = 0.0

        # Pre-allocated state buffers (replaces np.hstack allocation per step)
        self._state_bid = np.zeros(107, dtype=np.float32)
        self._state_ope = np.zeros(108, dtype=np.float32)

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self):
        self._soc = self.soc_initial

    # ── State builders ───────────────────────────────────────────────────────

    def get_state_bid(self):
        t = self._t
        s = self._state_bid
        s[(t - 1) % 96] = 0;  s[t % 96] = 1
        s[96]     = self._soc / self.cap_ess
        s[97:105] = self._price_bid_norm[t]
        s[106]    = self._solar_bid_norm[t]
        return s

    def get_state_ope(self):
        t = self._t
        s = self._state_ope
        s[(t - 1) % 96] = 0;  s[t % 96] = 1
        s[96]     = self._soc / self.cap_ess
        s[97]     = self._commit / _SOLAR_STD
        s[98:106] = self._price_ope_norm[t]
        s[107]    = self._solar_ope_norm[t]
        return s

    # ── Action processing ────────────────────────────────────────────────────

    def save_action_bid(self, action):
        action, commit, charge = self._clip(action, self.solar_bid[self._t], self._soc)
        self._action  = action
        self._commit  = commit
        self._charge  = charge
        return action, commit, charge, self.solar_ope[self._t]

    def save_action_ope(self, action):
        action, sell_pur, charge = self._clip(action, self.solar_ope[self._t], self._soc)
        self._action_ope   = action
        self._sell_pur_ope = sell_pur
        self._charge       = charge
        return action, sell_pur, charge, self.solar_ope[self._t]

    def _clip(self, action, solar, soc):
        if action >= 0:
            action = min(action, (self.soc_max - soc) / self.delta, self.cap_pcs)
            charge = True
        else:
            action = -min(-action, (soc - self.soc_min) * self.delta, self.cap_pcs)
            charge = False
        sell_pur = float(-action + solar)
        return action, sell_pur, charge

    # ── Reward ───────────────────────────────────────────────────────────────

    def save_reward(self):
        t     = self._t
        p_ope = self._p_ope_reward[t]
        p_set = self.price_set[t]

        self._reward_bid = float(self._commit * p_ope)
        imbalance        = (self._sell_pur_ope - self._commit) * p_set
        self._reward_ope = (float(imbalance), -abs(self._action_ope) * self.deg_rate)

    # ── SoC + time ───────────────────────────────────────────────────────────

    def soc_update(self):
        if self._charge:
            self._soc += self._action_ope * self.delta
        else:
            self._soc += self._action_ope / self.delta

    def idx_tq_update(self):
        self._t   = (self._t + 1) % self._n
        self.Done = (self._t % 96 == 0)

    # ── Step outputs ─────────────────────────────────────────────────────────

    def step_bid(self):
        return self.get_state_bid(), self._reward_bid, self.Done, None

    def step_ope(self):
        return self._reward_ope

    # ── Dimensions ───────────────────────────────────────────────────────────

    def observation_dim_bid(self): return 107
    def observation_dim_ope(self): return 108
    def action_dim(self):          return 1 if self.action_list is None else len(self.action_list)
    def action_limit(self):        return self.cap_pcs
    def len_data(self):            return self._n

    # ── Getters ──────────────────────────────────────────────────────────────

    def get_soc_ope(self):       return self._soc
    def get_pv_bid(self):        return float(self.solar_bid[self._t])
    def get_pv_ope(self):        return float(self.solar_ope[self._t])
    def get_purchase_bid(self):  return self._commit
    def get_purchase_ope(self):  return self._sell_pur_ope
    def get_price_bid(self):     return float(self._p_bid_reward[self._t])
    def get_price_ope(self):     return float(self._p_ope_reward[self._t])

    def sample_action(self):
        return np.random.uniform(-self.cap_pcs, self.cap_pcs)
