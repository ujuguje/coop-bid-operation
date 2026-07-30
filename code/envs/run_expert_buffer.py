"""
Populate expert replay buffers from offline MILP-optimal demonstrations.

The expert actions are pre-computed by a perfect-foresight MILP optimizer
and stored in Offline_Expert_Action_250908.csv.  This module replays those
actions through the environment to collect (obs, action, reward, next_obs, done)
transitions for both the bid and operation agents.

Temporal structure
------------------
  t=0 : save_action_bid → save_action_ope → save_reward/soc_update/idx_update
        → step_bid stores (obs_bid_0, a_bid_0, r_bid_0, obs_bid_1)
  t=1 : step_ope delivers reward for ope action taken at t=0
        → stores (obs_ope_0, a_ope_0, r_ope_0, obs_ope_1)
"""
import numpy as np

from envs.env_multi import DualAgentEnv
from models.buffer import ReplayBuffer


def populate_expert_buffers(data: dict,
                             expert_data: dict,
                             params: dict,
                             tol: float = 5.0,
                             degradation_cost_per_mwh: float = 5.0):
    """
    Replay expert demonstrations to fill bid and operation replay buffers.

    Args:
        data                     : market data dict
        expert_data              : dict with 'Expert_Bid_Action' and 'Expert_Ope_Action'
        params                   : battery parameter dict (from settings.Parameters)
        tol                      : imbalance tolerance (MW)
        degradation_cost_per_mwh : linear degradation cost ($/MWh)

    Returns:
        buf_bid       : ReplayBuffer — bid-agent transitions
        buf_ope       : ReplayBuffer — ope-agent transitions
        total_return  : float
        daily_returns : list of [r_bid, r_ope, r_deg] per episode
    """
    env = DualAgentEnv(data, params, tol=tol,
                       degradation_cost_per_mwh=degradation_cost_per_mwh)
    env.expert_action_load(expert_data['Expert_Bid_Action'],
                           expert_data['Expert_Ope_Action'])

    buf_bid = ReplayBuffer(env.observation_dim_bid(), env.action_dim(), env.len_data())
    buf_ope = ReplayBuffer(env.observation_dim_ope(), env.action_dim(), env.len_data())

    episodes     = env.len_data() // 96
    total_return = 0.0
    daily_returns = []

    print(f'[Expert buffer] {episodes} episodes  tol={tol}MW  deg=${degradation_cost_per_mwh}/MWh')

    for _ in range(episodes):
        env.reset()
        step         = 0
        done_ope     = False
        r_bid        = 0.0
        prev_obs_ope    = None
        prev_action_ope = None
        daily = [0.0, 0.0, 0.0]   # [r_bid, r_ope, r_deg]

        while not done_ope:
            obs_bid    = env.get_state_bid()
            action_bid = env.expert_action_sample(bid=True)
            act_bid, act_commit, _, _ = env.save_action_bid(action_bid)

            # step_ope() delivers the reward earned by the *previous* ope action
            if step > 0:
                next_obs_ope, r_ope, done_ope, _ = env.step_ope()
                r_ope_total = r_ope[0] + r_ope[1]
                buf_ope.store(prev_obs_ope, prev_action_ope,
                              r_ope_total / 1000.0, next_obs_ope, done_ope)
                total_return += r_bid + r_ope_total
                daily[0] += r_bid
                daily[1] += r_ope[0]
                daily[2] += r_ope[1]
                if done_ope:
                    break

            obs_ope    = env.get_state_ope()
            action_ope = env.expert_action_sample(bid=False)
            env.save_action_ope(action_ope, act_commit, act_bid)

            env.save_reward()
            env.soc_update()
            env.idx_tq_update()

            next_obs_bid, r_bid, done_bid, _ = env.step_bid()
            buf_bid.store(obs_bid, action_bid, r_bid / 1000.0, next_obs_bid, done_bid)

            prev_obs_ope    = obs_ope
            prev_action_ope = action_ope
            step += 1

        daily_returns.append(daily)

    print(f'[Expert buffer] total_return={total_return:.0f}  '
          f'buf_bid.size={buf_bid.size}  buf_ope.size={buf_ope.size}')
    return buf_bid, buf_ope, total_return, daily_returns
