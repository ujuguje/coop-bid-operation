"""Evaluation utilities shared by all algorithms."""
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def fast_evaluate(env, batch_action_fn):
    """
    Vectorized multi-episode evaluation.

    Runs all N episodes in lockstep: 2 batch forward passes per timestep
    (batch_size=N) instead of 2*N individual passes (batch_size=1).
    Typically 10-20x faster than evaluation_multi_agent for large N.

    Args:
        env              : DualAgentEnv instance
        batch_action_fn  : callable(s_bid, s_ope) where both are np.ndarray
                           of shape (n, obs_dim), returns (a_bid, a_ope) each (n,)

    Returns:
        ep_ret       : float — total return over all episodes
        daily_returns: list[list[float]] — [r_bid, r_ope, r_deg, total] per episode
    """
    T = 96
    n = env.len_data() // T

    pb_n = env._price_bid_norm.reshape(n, T, 8)   # (n, 96, 8)
    po_n = env._price_ope_norm.reshape(n, T, 8)
    po_r = env._p_ope_reward.reshape(n, T)         # RT price for bid reward
    ps   = env.price_set.reshape(n, T)             # settlement price
    sb_n = env._solar_bid_norm.reshape(n, T)
    so_n = env._solar_ope_norm.reshape(n, T)
    sb   = env.solar_bid.reshape(n, T)             # raw solar bid
    so   = env.solar_ope.reshape(n, T)             # raw solar ope

    cap_bid  = env.cap_pcs_bid
    delta    = env.delta
    soc_min  = env.soc_min
    soc_max  = env.soc_max
    cap_ess  = env.cap_ess
    tol      = env.tol
    deg_rate = env.deg_rate

    onehot   = np.eye(T, dtype=np.float32)  # (96, 96)

    # Contiguous float32 state buffers for zero-copy torch.from_numpy
    s_bid      = np.zeros((n, 107), dtype=np.float32)
    s_ope_curr = np.zeros((n, 108), dtype=np.float32)
    # Initial ope obs at t=0: SoC=soc_init, commit=0
    s_ope_prev = np.zeros((n, 108), dtype=np.float32)
    s_ope_prev[:, 0]      = 1.0                        # one-hot position 0
    s_ope_prev[:, 96]     = env.soc_initial / cap_ess
    s_ope_prev[:, 98:106] = po_n[:, 0]                 # RT price at t=0
    s_ope_prev[:, 107]    = so_n[:, 0]                 # solar at t=0

    soc   = np.full(n, env.soc_initial, dtype=np.float32)
    daily = np.zeros((n, 4), dtype=np.float32)  # [r_bid, r_ope, r_deg, total]

    for t in range(T):
        # Bid obs: [one-hot(96), SoC, price(8), gap, solar]
        s_bid[:, :96]    = onehot[t]
        s_bid[:, 96]     = soc / cap_ess
        s_bid[:, 97:105] = pb_n[:, t]
        # index 105 stays 0 (structural gap matching env state layout)
        s_bid[:, 106]    = sb_n[:, t]

        # Pass 1: bid action using PREVIOUS ope obs
        a_bid, _ = batch_action_fn(s_bid, s_ope_prev)

        # Vectorized clip_bid → commit
        max_ch  = np.minimum((soc_max - soc) / delta, cap_bid)
        max_dis = np.minimum((soc - soc_min) * delta, cap_bid)
        a_bid_c = np.where(a_bid >= 0,
                           np.minimum(a_bid, max_ch),
                           -np.minimum(-a_bid, max_dis)).astype(np.float32)
        commit  = (-a_bid_c + sb[:, t]).astype(np.float32)

        # Fresh ope obs: [one-hot(96), SoC, commit_norm, RT_price(8), gap, solar]
        s_ope_curr[:, :96]    = onehot[t]
        s_ope_curr[:, 96]     = soc / cap_ess
        s_ope_curr[:, 97]     = commit / 50.0
        s_ope_curr[:, 98:106] = po_n[:, t]
        # index 106 stays 0 (structural gap)
        s_ope_curr[:, 107]    = so_n[:, t]

        # Pass 2: ope action using CURRENT ope obs
        _, a_ope = batch_action_fn(s_bid, s_ope_curr)

        # Vectorized clip_ope → sell_pur
        raw_ope   = a_ope + a_bid_c
        max_ch_o  = np.minimum(cap_bid, (soc_max - soc) / delta)
        max_dis_o = np.minimum(cap_bid, (soc - soc_min) * delta)
        lower     = np.maximum(-max_dis_o, so[:, t] - commit - tol)
        upper     = np.minimum( max_ch_o,  so[:, t] - commit + tol)
        a_ope_c   = np.clip(raw_ope, lower, upper).astype(np.float32)
        sell_pur  = (so[:, t] - a_ope_c).astype(np.float32)

        # Rewards
        r_bid_v = commit * po_r[:, t]
        r_ope_v = (sell_pur - commit) * ps[:, t]
        r_deg_v = -np.abs(a_ope_c) * deg_rate

        daily[:, 0] += r_bid_v
        daily[:, 1] += r_ope_v
        daily[:, 2] += r_deg_v
        daily[:, 3] += r_bid_v + r_ope_v + r_deg_v

        # SoC update
        soc = np.where(a_ope_c >= 0,
                       soc + a_ope_c * delta,
                       soc + a_ope_c / delta).astype(np.float32)

        np.copyto(s_ope_prev, s_ope_curr)

    return float(daily[:, 3].sum()), daily.tolist()


def fast_evaluate_single(env, batch_action_fn):
    """
    Vectorized single-agent evaluation.

    Runs all N episodes in lockstep: 1 batch forward pass per timestep
    (batch_size=N) instead of N individual passes (batch_size=1).

    The same clipped action is applied to both DA bid and RT operation.

    Args:
        env              : SingleAgentEnv instance
        batch_action_fn  : callable(s_bid: np.ndarray (n, 107)) -> np.ndarray (n,)

    Returns:
        ep_ret       : float — total return over all episodes
        daily_returns: list[list[float]] — [r_bid, r_ope, r_deg, total] per episode
    """
    T = 96
    n = env.len_data() // T

    pb_n = env._price_bid_norm.reshape(n, T, 8)
    po_r = env._p_ope_reward.reshape(n, T)
    ps   = env.price_set.reshape(n, T)
    sb_n = env._solar_bid_norm.reshape(n, T)
    sb   = env.solar_bid.reshape(n, T)
    so   = env.solar_ope.reshape(n, T)

    cap_pcs  = env.cap_pcs
    delta    = env.delta
    soc_min  = env.soc_min
    soc_max  = env.soc_max
    cap_ess  = env.cap_ess
    deg_rate = env.deg_rate

    onehot = np.eye(T, dtype=np.float32)
    s_bid  = np.zeros((n, 107), dtype=np.float32)

    soc   = np.full(n, env.soc_initial, dtype=np.float32)
    daily = np.zeros((n, 4), dtype=np.float32)

    for t in range(T):
        s_bid[:, :96]    = onehot[t]
        s_bid[:, 96]     = soc / cap_ess
        s_bid[:, 97:105] = pb_n[:, t]
        # index 105: structural gap (stays 0)
        s_bid[:, 106]    = sb_n[:, t]

        action = batch_action_fn(s_bid)   # (n,)

        # Vectorized clip — solar doesn't affect the clipped action value
        max_ch = np.minimum((soc_max - soc) / delta, cap_pcs)
        max_dis = np.minimum((soc - soc_min) * delta, cap_pcs)
        a_c = np.where(action >= 0,
                       np.minimum(action, max_ch),
                       -np.minimum(-action, max_dis)).astype(np.float32)

        commit   = (-a_c + sb[:, t]).astype(np.float32)   # bid solar
        sell_pur = (-a_c + so[:, t]).astype(np.float32)   # ope solar

        r_bid_v = commit * po_r[:, t]
        r_ope_v = (sell_pur - commit) * ps[:, t]
        r_deg_v = -np.abs(a_c) * deg_rate

        daily[:, 0] += r_bid_v
        daily[:, 1] += r_ope_v
        daily[:, 2] += r_deg_v
        daily[:, 3] += r_bid_v + r_ope_v + r_deg_v

        soc = np.where(a_c >= 0,
                       soc + a_c * delta,
                       soc + a_c / delta).astype(np.float32)

    return float(daily[:, 3].sum()), daily.tolist()


def evaluation_multi_agent(env, get_action_fn, collect_impl=True, n_episodes=None):
    """
    Evaluate a dual-agent policy over the entire dataset in env.

    Args:
        env            : DualAgentEnv instance
        get_action_fn  : callable(obs_bid, obs_ope, determine, turn_on) → (a_bid, a_ope)
        collect_impl   : whether to collect per-step implementation details
        n_episodes     : if set, evaluate only the first N episodes (faster checkpoint selection)

    Returns:
        ep_ret         : float — total return over all episodes
        daily_returns  : list of [bid, ope, degradation, total] per day
        impl           : dict of per-step records (empty if collect_impl=False)
    """
    episodes = env.len_data() // 96
    if n_episodes is not None:
        episodes = min(episodes, n_episodes)
    ep_ret   = 0.0
    daily_returns = []
    impl = {k: [] for k in ['soc', 'bid_price', 'ope_price', 'pv_forecast',
                             'pv_real', 'sell_bid', 'sell_ope']}

    for _ in range(episodes):
        env.reset()
        step = 0
        obs_ope = env.get_state_ope()
        d_bid = d_ope = False
        r_bid = 0.0
        daily = [0.0, 0.0, 0.0, 0.0]

        while not d_ope:
            obs_bid = env.get_state_bid()
            a_bid, a_ope = get_action_fn(obs_bid, obs_ope, determine=True, turn_on=True)
            act_bid, act_commit, _, _ = env.save_action_bid(a_bid)

            if collect_impl:
                impl['soc'].append(env.get_soc_ope())
                impl['pv_forecast'].append(env.get_pv_bid())
                impl['bid_price'].append(env.get_price_bid())
                impl['sell_bid'].append(env.get_purchase_bid())

            if step > 0:
                next_obs_ope, r_ope, d_ope, _ = env.step_ope()
                ep_ret        += r_bid + r_ope[0] + r_ope[1]
                daily[0]      += r_bid
                daily[1]      += r_ope[0]
                daily[2]      += r_ope[1]
                daily[3]      += r_bid + r_ope[0] + r_ope[1]

                if d_ope:
                    obs_ope = env.get_state_ope()
                    _, a_ope_final = get_action_fn(obs_bid, obs_ope, determine=True, turn_on=True)
                    env.save_action_ope(a_ope_final, act_commit, act_bid)
                    if collect_impl:
                        impl['ope_price'].append(env.get_price_ope())
                        impl['pv_real'].append(env.get_pv_ope())
                        impl['sell_ope'].append(env.get_purchase_ope())
                    break

            step += 1
            obs_ope = env.get_state_ope()
            _, a_ope = get_action_fn(obs_bid, obs_ope, determine=True, turn_on=True)
            env.save_action_ope(a_ope, act_commit, act_bid)

            if collect_impl:
                impl['ope_price'].append(env.get_price_ope())
                impl['pv_real'].append(env.get_pv_ope())
                impl['sell_ope'].append(env.get_purchase_ope())

            env.save_reward()
            env.soc_update()
            env.idx_tq_update()
            next_obs_bid, r_bid, d_bid, _ = env.step_bid()

        daily_returns.append(daily)

    return ep_ret, daily_returns, impl if collect_impl else {}


def evaluation_single_agent(env, get_action_fn, collect_impl=True):
    """
    Evaluate a single-agent policy over the entire dataset in env.

    Args:
        env            : SingleAgentEnv instance
        get_action_fn  : callable(obs_bid, determine, turn_on) → action
        collect_impl   : whether to collect per-step implementation details

    Returns: (ep_ret, daily_returns, impl)
    """
    episodes = env.len_data() // 96
    ep_ret   = 0.0
    daily_returns = []
    impl = {k: [] for k in ['soc', 'bid_price', 'ope_price', 'pv_forecast',
                             'pv_real', 'sell_bid', 'sell_ope']}

    for _ in range(episodes):
        env.reset()
        d_bid = False
        daily = [0.0, 0.0, 0.0, 0.0]

        while not d_bid:
            obs_bid = env.get_state_bid()
            action  = get_action_fn(obs_bid, determine=True, turn_on=True)
            env.save_action_bid(action)
            env.save_action_ope(action)
            env.save_reward()
            env.soc_update()
            env.idx_tq_update()

            _, r_bid, d_bid, _ = env.step_bid()
            r_ope = env.step_ope()

            ep_ret   += r_bid + r_ope[0] + r_ope[1]
            daily[0] += r_bid
            daily[1] += r_ope[0]
            daily[2] += r_ope[1]
            daily[3] += r_bid + r_ope[0] + r_ope[1]

            if collect_impl:
                impl['soc'].append(env.get_soc_ope())
                impl['pv_forecast'].append(env.get_pv_bid())
                impl['bid_price'].append(env.get_price_bid())
                impl['sell_bid'].append(env.get_purchase_bid())
                impl['ope_price'].append(env.get_price_ope())
                impl['pv_real'].append(env.get_pv_ope())
                impl['sell_ope'].append(env.get_purchase_ope())

        daily_returns.append(daily)

    return ep_ret, daily_returns, impl if collect_impl else {}


def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}
