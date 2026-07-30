"""
Single-Agent SAC (Single-SAC).

A single RL agent controls battery dispatch for both DA and RT markets using
the same action. No dual-agent coordination; serves as the simplest baseline.
"""
import os
import time
import numpy as np
import torch
from copy import deepcopy
from torch.optim import Adam

from envs.env_single import SingleAgentEnv
from models.buffer import ReplayBuffer
from models.networks import SingleAgentActor, SingleAgentQFunction
from algorithms.utils import evaluation_single_agent, fast_evaluate_single

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def _to_tensor(arr):
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def train(env_: SingleAgentEnv,
          env_val:  SingleAgentEnv,
          env_test: SingleAgentEnv,
          seed:           int   = 0,
          replay_size:    int   = int(1e6),
          gamma:          float = 0.98,
          polyak:         float = 0.995,
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
    Online SAC training for single-agent battery bidding.

    Returns: (val_returns, val_daily, test_daily, test_impl)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_
    obs_dim = env.observation_dim_bid()
    act_dim = env.action_dim()
    lim     = env.action_limit()

    log_alpha  = torch.zeros(1, requires_grad=True, device=device)
    target_ent = torch.tensor(-1.0, device=device)
    opt_alpha  = Adam([log_alpha], lr=3e-4)

    pi     = SingleAgentActor(obs_dim, act_dim, hidden_size, lim).to(device)
    q1     = SingleAgentQFunction(obs_dim, act_dim, hidden_size).to(device)
    q2     = SingleAgentQFunction(obs_dim, act_dim, hidden_size).to(device)
    q1_t   = deepcopy(q1).to(device)
    q2_t   = deepcopy(q2).to(device)

    for p in list(q1_t.parameters()) + list(q2_t.parameters()):
        p.requires_grad = False

    q_params   = list(q1.parameters()) + list(q2.parameters())
    opt_q      = Adam(q_params, lr=lr)
    opt_pi     = Adam(pi.parameters(), lr=lr)

    buf = ReplayBuffer(obs_dim, act_dim, replay_size)

    pi_cpu = deepcopy(pi).cpu()

    def _sync_cpu():
        pi_cpu.load_state_dict({k: v.cpu() for k, v in pi.state_dict().items()})

    def _batch_act(s):
        with torch.inference_mode():
            a, _ = pi_cpu(torch.from_numpy(s), deterministic=True, with_logprob=False)
        return a[:, 0].numpy()

    def get_action(obs, determine, turn_on):
        if not turn_on:
            return env.sample_action()
        with torch.no_grad():
            a, _ = pi(_to_tensor(obs).unsqueeze(0), deterministic=determine, with_logprob=False)
        return a.cpu().numpy()[0][0]

    def update():
        idxs = np.random.randint(0, buf.size, size=batch_size)
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

    for _ in range(10_000 * 96):
        env.reset()
        done = False

        while not done:
            obs = env.get_state_bid()
            if nsteps == start_steps:
                turn_on = True

            action = get_action(obs, False, turn_on)
            env.save_action_bid(action)
            env.save_action_ope(action)
            env.save_reward()
            env.soc_update()
            env.idx_tq_update()

            next_obs, r_bid, done, _ = env.step_bid()
            r_ope  = env.step_ope()
            r_train = (r_bid + r_ope[0] + r_ope[1]) / 1000
            buf.store(obs, action, r_train, next_obs, done)
            nsteps += 1

            if nsteps >= update_after and nsteps % update_every == 0:
                update()
                update_num += 1

                if update_num % eval_freq == 0 or update_num == update_limit:
                    _sync_cpu()
                    ret_val, daily_val = fast_evaluate_single(env_val, _batch_act)
                    if ret_val > best_val:
                        best_val = ret_val
                        _, test_daily = fast_evaluate_single(env_test, _batch_act)
                        test_impl = {}
                        if save_actor_path is not None:
                            os.makedirs(os.path.dirname(save_actor_path), exist_ok=True)
                            torch.save(pi.state_dict(), save_actor_path)
                    val_returns.append(ret_val); val_daily.append(daily_val)
                    elapsed = time.time() - t_train_start
                    pct = update_num / update_limit * 100
                    print(f'[{update_num:4d}/{update_limit}  {pct:.0f}%  {elapsed:.0f}s] '
                          f'val={ret_val:.0f}  best={best_val:.0f}', flush=True)

                if update_num == update_limit:
                    timing = {'total_train_s': time.time() - t_train_start, 'n_updates': update_num}
                    return val_returns, val_daily, test_daily, test_impl, timing
