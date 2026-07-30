"""
Independent Dual-Agent SAC (Inde-DASAC).

Bid and operation agents each maintain their own Q-functions and actor.
No value-function sharing — agents are trained independently but observe the same market.
"""
import os
import time
import numpy as np
import torch
from copy import deepcopy
from torch.optim import Adam

from envs.env_multi import DualAgentEnv
from models.buffer import ReplayBuffer
from models.networks import SingleAgentActor, SingleAgentQFunction
from algorithms.utils import evaluation_multi_agent, fast_evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


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
          hidden_size:    int   = 128,
          eval_freq:      int   = 200,
          save_actor_path: str = None):
    """
    Online SAC training with independent (non-cooperative) dual agents.

    Returns: (val_returns, val_daily, test_daily, test_impl)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_
    obs_bid_dim = env.observation_dim_bid()
    obs_ope_dim = env.observation_dim_ope()
    act_dim     = env.action_dim()
    lim_bid     = env.action_limit_bid()
    lim_ope     = env.action_limit_ope()

    # Per-agent learnable temperatures
    log_alpha_bid = torch.zeros(1, requires_grad=True, device=device)
    log_alpha_ope = torch.zeros(1, requires_grad=True, device=device)
    target_ent    = torch.tensor(-1.0, device=device)
    opt_alpha_bid = Adam([log_alpha_bid], lr=3e-4)
    opt_alpha_ope = Adam([log_alpha_ope], lr=3e-4)

    # Bid agent
    pi_bid   = SingleAgentActor(obs_bid_dim, act_dim, hidden_size, lim_bid).to(device)
    q1_bid   = SingleAgentQFunction(obs_bid_dim, act_dim, hidden_size).to(device)
    q2_bid   = SingleAgentQFunction(obs_bid_dim, act_dim, hidden_size).to(device)
    q1_bid_t = deepcopy(q1_bid).to(device)
    q2_bid_t = deepcopy(q2_bid).to(device)

    # Ope agent
    pi_ope   = SingleAgentActor(obs_ope_dim, act_dim, hidden_size, lim_ope).to(device)
    q1_ope   = SingleAgentQFunction(obs_ope_dim, act_dim, hidden_size).to(device)
    q2_ope   = SingleAgentQFunction(obs_ope_dim, act_dim, hidden_size).to(device)
    q1_ope_t = deepcopy(q1_ope).to(device)
    q2_ope_t = deepcopy(q2_ope).to(device)

    for p in list(q1_bid_t.parameters()) + list(q2_bid_t.parameters()) + \
             list(q1_ope_t.parameters()) + list(q2_ope_t.parameters()):
        p.requires_grad = False

    q_bid_params = list(q1_bid.parameters()) + list(q2_bid.parameters())
    q_ope_params = list(q1_ope.parameters()) + list(q2_ope.parameters())
    opt_q_bid  = Adam(q_bid_params, lr=lr)
    opt_q_ope  = Adam(q_ope_params, lr=lr)
    opt_pi_bid = Adam(pi_bid.parameters(), lr=lr)
    opt_pi_ope = Adam(pi_ope.parameters(), lr=lr)

    buf_bid = ReplayBuffer(obs_bid_dim, act_dim, replay_size)
    buf_ope = ReplayBuffer(obs_ope_dim, act_dim, replay_size)

    pi_bid_cpu = deepcopy(pi_bid).cpu()
    pi_ope_cpu = deepcopy(pi_ope).cpu()

    def _sync_cpu():
        pi_bid_cpu.load_state_dict({k: v.cpu() for k, v in pi_bid.state_dict().items()})
        pi_ope_cpu.load_state_dict({k: v.cpu() for k, v in pi_ope.state_dict().items()})

    def _batch_act(s_bid, s_ope):
        with torch.inference_mode():
            a_b, _ = pi_bid_cpu(torch.from_numpy(s_bid), deterministic=True, with_logprob=False)
            a_o, _ = pi_ope_cpu(torch.from_numpy(s_ope), deterministic=True, with_logprob=False)
        return a_b[:, 0].numpy(), a_o[:, 0].numpy()

    def get_action(o_bid, o_ope, determine, turn_on):
        if not turn_on:
            return env.sample_action()
        with torch.no_grad():
            a_bid, _ = pi_bid(_to_tensor(o_bid).unsqueeze(0), deterministic=determine, with_logprob=False)
            a_ope, _ = pi_ope(_to_tensor(o_ope).unsqueeze(0), deterministic=determine, with_logprob=False)
        return a_bid.cpu().numpy()[0][0], a_ope.cpu().numpy()[0][0]

    def _update_agent(buf, pi, q1, q2, q1_t, q2_t, q_params, opt_q, opt_pi, log_alpha, opt_alpha, idxs):
        b = {k: _to_tensor(v) for k, v in buf.sample_batch(idxs).items()}
        o, a, r, o2, d = b['obs'], b['act'], b['rew'], b['obs2'], b['done']
        alpha = log_alpha.exp()

        with torch.no_grad():
            a2, lp2 = pi(o2)
            backup = r + gamma * (1 - d) * (torch.min(q1_t(o2, a2), q2_t(o2, a2)) - alpha * lp2)

        loss_q = ((q1(o, a) - backup)**2 + (q2(o, a) - backup)**2).mean()
        opt_q.zero_grad(); loss_q.backward()
        torch.nn.utils.clip_grad_norm_(q_params, 10.0)
        opt_q.step()

        for p in q_params:
            p.requires_grad = False
        a_pi, lp_pi = pi(o)
        loss_pi = (alpha.detach() * lp_pi - torch.min(q1(o, a_pi), q2(o, a_pi))).mean()
        opt_pi.zero_grad(); loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(pi.parameters(), 10.0)
        opt_pi.step()

        alpha_loss = -(alpha * (lp_pi.detach() + target_ent)).mean()
        opt_alpha.zero_grad(); alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([log_alpha], 1.0)
        opt_alpha.step()

        for p in q_params:
            p.requires_grad = True
        _soft_update(q1_t, q1, 1 - polyak)
        _soft_update(q2_t, q2, 1 - polyak)

    nsteps = 0; update_num = 0; turn_on = False
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
                    idxs = np.random.randint(0, buf_bid.size, size=batch_size)
                    _update_agent(buf_bid, pi_bid, q1_bid, q2_bid, q1_bid_t, q2_bid_t,
                                  q_bid_params, opt_q_bid, opt_pi_bid, log_alpha_bid, opt_alpha_bid, idxs)
                    _update_agent(buf_ope, pi_ope, q1_ope, q2_ope, q1_ope_t, q2_ope_t,
                                  q_ope_params, opt_q_ope, opt_pi_ope, log_alpha_ope, opt_alpha_ope, idxs)
                    update_num += 1

                    if update_num % eval_freq == 0 or update_num == update_limit:
                        _sync_cpu()
                        ret_val, daily_val = fast_evaluate(env_val, _batch_act)
                        if ret_val > best_val:
                            best_val = ret_val
                            _, test_daily = fast_evaluate(env_test, _batch_act)
                            test_impl = {}
                            if save_actor_path is not None:
                                os.makedirs(os.path.dirname(save_actor_path), exist_ok=True)
                                torch.save({'bid': pi_bid.state_dict(),
                                            'ope': pi_ope.state_dict()}, save_actor_path)
                        val_returns.append(ret_val); val_daily.append(daily_val)
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
            env.save_reward(); env.soc_update(); env.idx_tq_update()
            _, r_bid, _, _ = env.step_bid()
            buf_bid.store(obs_bid, a_bid, r_bid / 1000, env.get_state_bid(), env.Done)
