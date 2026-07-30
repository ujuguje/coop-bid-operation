"""
Cooperative Dual-Agent SAC (Coop-DASAC).

Both bid and operation agents share a QMIX mixer (cooperative value function).
Entropy is regularised with a single learnable temperature α shared across agents.
"""
import os
import time
import numpy as np
import torch
from copy import deepcopy
from torch.optim import Adam

from envs.env_multi import DualAgentEnv
from models.buffer import ReplayBuffer
from models.networks import SquashedGaussianMLPActor, MLPQFunction, Hypernet
from algorithms.utils import evaluation_multi_agent, fast_evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def _global_state(obs_bid, obs_ope):
    return torch.cat([obs_bid, obs_ope[:, 96:]], dim=1)


def _to_tensor(arr):
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def train(env_: DualAgentEnv,
          env_val:  DualAgentEnv,
          env_test: DualAgentEnv,
          seed:           int   = 0,
          replay_size:    int   = int(1e6),
          gamma:          float = 0.98,
          polyak:         float = 0.999,
          lr:             float = 3e-4,
          batch_size:     int   = 512,
          start_steps:    int   = 30_000,
          update_after:   int   = 5_000,
          update_every:   int   = 10,
          update_limit:   int   = 50000,
          hidden_size: int   = 128,
          eval_freq:   int   = 200,
          save_actor_path: str = None):
    """
    Online SAC training with cooperative QMIX value decomposition.

    Args:
        env_       : training environment (DualAgentEnv)
        env_val    : validation environment
        env_test   : test environment
        replay_size: replay buffer capacity
        gamma      : discount factor
        polyak     : soft-update coefficient
        lr         : learning rate for all optimisers
        batch_size : mini-batch size
        start_steps: random-action warm-up steps before gradient updates
        update_after: minimum steps before first update
        update_every : environment steps per gradient update
        update_limit : total number of gradient updates before stopping
        hidden_size  : network hidden dimension

    Returns: (val_returns, val_daily, test_daily, test_impl)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = env_
    obs_bid_dim = env.observation_dim_bid()
    obs_ope_dim = env.observation_dim_ope()
    act_dim     = env.action_dim()
    lim_bid     = env.action_limit_bid()
    lim_ope     = env.action_limit_ope()
    global_dim  = obs_bid_dim + obs_ope_dim - 96
    n_agents    = 2

    log_alpha   = torch.zeros(1, requires_grad=True, device=device)
    target_ent  = torch.tensor(-float(n_agents), device=device)
    opt_alpha   = Adam([log_alpha], lr=1e-4)

    actor   = SquashedGaussianMLPActor(obs_bid_dim, obs_ope_dim, act_dim, hidden_size, lim_bid, lim_ope).to(device)
    q1      = MLPQFunction(obs_bid_dim, obs_ope_dim, act_dim, hidden_size).to(device)
    q2      = MLPQFunction(obs_bid_dim, obs_ope_dim, act_dim, hidden_size).to(device)
    mixer1  = Hypernet(global_dim, n_agents).to(device)
    mixer2  = Hypernet(global_dim, n_agents).to(device)

    q1_targ     = deepcopy(q1).to(device)
    q2_targ     = deepcopy(q2).to(device)
    mixer1_targ = deepcopy(mixer1).to(device)
    mixer2_targ = deepcopy(mixer2).to(device)
    for p in (list(q1_targ.parameters()) + list(q2_targ.parameters()) +
              list(mixer1_targ.parameters()) + list(mixer2_targ.parameters())):
        p.requires_grad = False

    q_params   = list(q1.parameters()) + list(q2.parameters()) + list(mixer1.parameters()) + list(mixer2.parameters())
    opt_q      = Adam(q_params, lr=lr)
    opt_actor  = Adam(actor.parameters(), lr=lr)

    buf_bid = ReplayBuffer(obs_bid_dim, act_dim, replay_size)
    buf_ope = ReplayBuffer(obs_ope_dim, act_dim, replay_size)

    from copy import deepcopy as _deepcopy
    actor_cpu = _deepcopy(actor).cpu()

    def _sync_cpu():
        actor_cpu.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})

    def _batch_act(s_bid, s_ope):
        with torch.inference_mode():
            a_b, a_o, _, _ = actor_cpu(
                torch.from_numpy(s_bid), torch.from_numpy(s_ope),
                deterministic=True, with_logprob=False)
        return a_b[:, 0].numpy(), a_o[:, 0].numpy()

    def get_action(o_bid, o_ope, determine, turn_on):
        """GPU actor — for online exploration during training."""
        if not turn_on:
            return env.sample_action()
        with torch.no_grad():
            a_bid, a_ope, _, _ = actor(
                _to_tensor(o_bid).unsqueeze(0),
                _to_tensor(o_ope).unsqueeze(0),
                deterministic=determine, with_logprob=False)
        return a_bid[0, 0].item(), a_ope[0, 0].item()

    def get_action_eval(o_bid, o_ope, determine=True, turn_on=True):
        """CPU actor — for evaluation (batch_size=1, CPU faster than GPU)."""
        with torch.inference_mode():
            a_bid, a_ope, _, _ = actor_cpu(
                torch.as_tensor(o_bid, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(o_ope, dtype=torch.float32).unsqueeze(0),
                deterministic=True, with_logprob=False)
        return a_bid[0, 0].item(), a_ope[0, 0].item()

    def update():
        idxs = np.random.randint(0, buf_bid.size, size=batch_size)
        bb = {k: _to_tensor(v) for k, v in buf_bid.sample_batch(idxs).items()}
        bo = {k: _to_tensor(v) for k, v in buf_ope.sample_batch(idxs).items()}

        o_bid, a_bid, r_bid, o2_bid, d = bb['obs'], bb['act'], bb['rew'], bb['obs2'], bb['done']
        o_ope, a_ope, r_ope, o2_ope    = bo['obs'], bo['act'], bo['rew'], bo['obs2']
        alpha = log_alpha.exp()

        with torch.no_grad():
            a2_bid, a2_ope, lp2_bid, lp2_ope = actor(o2_bid, o2_ope)
            gs_next = _global_state(o2_bid, o2_ope)

            q1b_t, q1o_t = q1_targ(o2_bid, a2_bid, o2_ope, a2_ope)
            q2b_t, q2o_t = q2_targ(o2_bid, a2_bid, o2_ope, a2_ope)
            qs1 = torch.stack([q1b_t, q1o_t], dim=2)
            qs2 = torch.stack([q2b_t, q2o_t], dim=2)
            q_tot_t = torch.min(mixer1_targ(qs1, gs_next).view(-1),
                                mixer2_targ(qs2, gs_next).view(-1))
            backup = (r_bid + r_ope) + gamma * (1 - d) * (q_tot_t - alpha * (lp2_bid + lp2_ope))

        q1b, q1o = q1(o_bid, a_bid, o_ope, a_ope)
        q2b, q2o = q2(o_bid, a_bid, o_ope, a_ope)
        gs = _global_state(o_bid, o_ope)
        qs1_cur = torch.stack([q1b, q1o], dim=1).view(-1, 1, n_agents)
        qs2_cur = torch.stack([q2b, q2o], dim=1).view(-1, 1, n_agents)
        loss_q = (((mixer1(qs1_cur, gs).view(-1) - backup)**2) +
                  ((mixer2(qs2_cur, gs).view(-1) - backup)**2)).mean()

        opt_q.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(q_params, 5.0)
        opt_q.step()

        for p in q_params:
            p.requires_grad = False

        a_bid_pi, a_ope_pi, lp_bid, lp_ope = actor(o_bid, o_ope)
        q1b_pi, q1o_pi = q1(o_bid, a_bid_pi, o_ope, a_ope_pi)
        q2b_pi, q2o_pi = q2(o_bid, a_bid_pi, o_ope, a_ope_pi)
        qs_pi = torch.stack([q1b_pi, q1o_pi], dim=1).view(-1, 1, n_agents)
        q_tot_pi = torch.min(mixer1(qs_pi, gs), mixer2(qs_pi, gs))
        loss_a = (alpha.detach() * (lp_bid + lp_ope) - q_tot_pi).mean()

        opt_actor.zero_grad()
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        alpha_loss = -(alpha * ((lp_bid + lp_ope).detach() + target_ent)).mean()
        opt_alpha.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([log_alpha], 1.0)
        opt_alpha.step()

        for p in q_params:
            p.requires_grad = True

        _soft_update(q1_targ, q1, 1 - polyak)
        _soft_update(q2_targ, q2, 1 - polyak)
        _soft_update(mixer1_targ, mixer1, 1 - polyak)
        _soft_update(mixer2_targ, mixer2, 1 - polyak)

    nsteps     = 0
    update_num = 0
    turn_on    = False
    val_returns, val_daily = [], []
    best_val, test_daily, test_impl = -1e9, None, None
    t_train_start = time.time()


    for _ in range(1000 * 96):
        env.reset()
        step_in_epi = 0
        obs_ope = env.get_state_ope()
        done_ope = False
        r_bid = 0.0

        while not done_ope:
            obs_bid = env.get_state_bid()

            if nsteps == start_steps:
                turn_on = True

            a_bid, _ = get_action(obs_bid, obs_ope, False, turn_on)
            act_bid, act_commit, _, _ = env.save_action_bid(a_bid)

            if step_in_epi > 0:
                next_obs_ope, r_ope, done_ope, _ = env.step_ope()
                buf_ope.store(obs_ope, a_ope, (r_ope[0] + r_ope[1]) / 1000, next_obs_ope, done_ope)
                nsteps += 1

                if nsteps >= update_after and nsteps % update_every == 0:
                    update()
                    update_num += 1

                    if update_num % eval_freq == 0 or update_num == update_limit:
                        _sync_cpu()
                        ret_val, daily_val = fast_evaluate(env_val, _batch_act)
                        if ret_val > best_val:
                            best_val = ret_val
                            _, test_daily = fast_evaluate(env_test, _batch_act)
                            test_impl = {}
                            # ── Save best-validation actor weights (for OOD comparison) ──
                            if save_actor_path is not None:
                                os.makedirs(os.path.dirname(save_actor_path), exist_ok=True)
                                torch.save(actor.state_dict(), save_actor_path)
                        val_returns.append(ret_val)
                        val_daily.append(daily_val)

                        elapsed = time.time() - t_train_start
                        pct = update_num / update_limit * 100
                        print(f'[{update_num:4d}/{update_limit}  {pct:.0f}%  {elapsed:.0f}s] '
                              f'val={ret_val:.0f}  best={best_val:.0f}', flush=True)

                    if update_num == update_limit:
                        timing = {'total_train_s': time.time() - t_train_start, 'n_updates': update_num}
                        return val_returns, val_daily, test_daily, test_impl, timing

                if done_ope:
                    break

            step_in_epi += 1
            obs_ope = env.get_state_ope()
            _, a_ope = get_action(obs_bid, obs_ope, False, turn_on)
            env.save_action_ope(a_ope, act_commit, act_bid)
            env.save_reward()
            env.soc_update()
            env.idx_tq_update()
            _, r_bid, _, _ = env.step_bid()
            buf_bid.store(obs_bid, a_bid, r_bid / 1000, env.get_state_bid(), env.Done)
