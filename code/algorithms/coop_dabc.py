# [final] Paper's Coop-DABC (Case D) — TD3+BC with a monotonic QMIX mixer (w_final.abs() fix)
"""
Coop-DABC — Q-FIXED + MONOTONIC-MIXER variant (ver2, experimental).

IDENTICAL to coop_bc_qfix.py EXCEPT the QMIX mixer uses a local FixedHypernet
that applies .abs() to BOTH w1 AND w_final (proper QMIX monotonicity
dQ_tot/dQ_local >= 0). The shipped networks.Hypernet used by coop_bc_qfix
applies .abs() only to w1 (w_final missing it), breaking monotonicity. Since
coop_bc_qfix removed the no_grad wrap (Q now flows to the actor), that broken
mixer feeds a non-monotonic Q-gradient into the actor — this file tests whether
fixing it improves bid-operation coordination (operation cost).
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam

from envs.env_multi import DualAgentEnv
from models.networks import Actor, MLPQFunction
from algorithms.utils import evaluation_multi_agent, fast_evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FixedHypernet(nn.Module):
    """QMIX hypernetwork with .abs() on BOTH w1 and w_final (proper monotonicity).
    Drop-in replacement for networks.Hypernet (same __init__ signature)."""

    def __init__(self, global_state_dim, agent_num, hidden_size=128):
        super().__init__()
        self.agent_num   = agent_num
        self.hidden_size = hidden_size
        self.w1      = nn.Linear(global_state_dim, hidden_size * agent_num)
        self.b1      = nn.Linear(global_state_dim, hidden_size)
        self.w_final = nn.Linear(global_state_dim, hidden_size)
        self.V       = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, q_n, global_state):
        w1 = self.w1(global_state).abs().view(-1, self.agent_num, self.hidden_size)
        b1 = self.b1(global_state).view(-1, 1, self.hidden_size)
        hidden   = F.elu(torch.bmm(q_n, w1) + b1)
        w_final  = self.w_final(global_state).abs().view(-1, self.hidden_size, 1)  # <-- .abs() ADDED (fix #4)
        v        = self.V(global_state).view(-1, 1, 1)
        return torch.bmm(hidden, w_final) + v


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def _global_state(obs_bid, obs_ope):
    return torch.cat([obs_bid, obs_ope[:, 96:]], dim=1)


def _to_tensor(arr):
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def train(env_train: DualAgentEnv,
          env_val:   DualAgentEnv,
          env_test:  DualAgentEnv,
          buf_bid,
          buf_ope,
          seed:          int   = 0,
          max_steps:     int   = 10000,
          discount:      float = 0.99,
          policy_noise:  float = 0.2,
          noise_clip:    float = 0.5,
          alpha:         float = 2.5,
          bc_ope_w:      float = 1.0,
          bc_bid_w:      float = 1.0,
          policy_freq:   int   = 1,
          hidden_size:   int   = 128,
          tau:           float = 5e-3,
          lr_actor:      float = 3e-4,
          lr_critic:     float = 3e-4,
          batch_size:  int   = 256,
          eval_freq:   int   = 50,
          record_curves: bool = False,
          save_actor_path: str = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    obs_bid_dim = env_train.observation_dim_bid()
    obs_ope_dim = env_train.observation_dim_ope()
    act_dim     = env_train.action_dim()
    lim_bid     = env_train.action_limit_bid()
    lim_ope     = env_train.action_limit_ope()
    global_dim  = obs_bid_dim + obs_ope_dim - 96
    n_agents    = 2

    actor   = Actor(obs_bid_dim, obs_ope_dim, act_dim, hidden_size, lim_bid, lim_ope).to(device)
    critic  = MLPQFunction(obs_bid_dim, obs_ope_dim, act_dim, hidden_size).to(device)
    mixer1  = FixedHypernet(global_dim, n_agents).to(device)
    mixer2  = FixedHypernet(global_dim, n_agents).to(device)

    critic_targ = deepcopy(critic).to(device)
    mixer1_targ = deepcopy(mixer1).to(device)
    mixer2_targ = deepcopy(mixer2).to(device)
    for p in list(critic_targ.parameters()) + list(mixer1_targ.parameters()) + list(mixer2_targ.parameters()):
        p.requires_grad = False

    critic_params = list(critic.parameters()) + list(mixer1.parameters()) + list(mixer2.parameters())
    opt_critic = Adam(critic_params, lr=lr_critic)
    opt_actor  = Adam(actor.parameters(), lr=lr_actor)

    actor_cpu = deepcopy(actor).cpu()
    _nan_action_reported = [False]

    def _sync_cpu():
        actor_cpu.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})

    def _batch_act(s_bid, s_ope):
        with torch.inference_mode():
            a_b, a_o = actor_cpu(torch.from_numpy(s_bid), torch.from_numpy(s_ope))
        return a_b[:, 0].numpy(), a_o[:, 0].numpy()

    val_returns, val_daily = [], []
    test_curve = []   # (record_curves) test total at EVERY eval, for checkpoint-selection analysis
    best_val, test_daily, test_impl = -1e9, None, None
    step_times = []
    t_train_start = time.time()

    print(f'[Coop-DABC-Mono] train={env_train.len_data()//96}d  steps={max_steps}  alpha={alpha}', flush=True)

    recent_loss_q = 0.0
    recent_loss_a = 0.0

    for step in range(max_steps):
        t0 = time.perf_counter()
        idxs = np.random.randint(0, buf_bid.max_size, size=batch_size)

        bb = {k: _to_tensor(v) for k, v in buf_bid.sample_batch(idxs).items()}
        bo = {k: _to_tensor(v) for k, v in buf_ope.sample_batch(idxs).items()}

        o_bid, a_bid_exp, r_bid, o2_bid, d = bb['obs'], bb['act'], bb['rew'], bb['obs2'], bb['done']
        o_ope, a_ope_exp, r_ope, o2_ope    = bo['obs'], bo['act'], bo['rew'], bo['obs2']

        # Critic update
        with torch.no_grad():
            noise_b = (torch.randn_like(a_bid_exp) * policy_noise).clamp(-noise_clip, noise_clip)
            noise_o = (torch.randn_like(a_ope_exp) * policy_noise).clamp(-noise_clip, noise_clip)
            a2_bid, a2_ope = actor(o2_bid, o2_ope)
            a2_bid = (a2_bid + noise_b).clamp(-lim_bid, lim_bid)
            a2_ope = (a2_ope + noise_o).clamp(-lim_ope, lim_ope)

            q1b_t, q1o_t = critic_targ(o2_bid, a2_bid, o2_ope, a2_ope)
            gs_next = _global_state(o2_bid, o2_ope)
            q_mix1  = mixer1_targ(torch.stack([q1b_t, q1o_t], dim=2), gs_next).view(-1)
            q_mix2  = mixer2_targ(torch.stack([q1b_t, q1o_t], dim=2), gs_next).view(-1)
            backup  = (r_bid + r_ope) + (1 - d) * discount * torch.min(q_mix1, q_mix2)

        q1b, q1o = critic(o_bid, a_bid_exp, o_ope, a_ope_exp)
        gs = _global_state(o_bid, o_ope)
        q_stacked = torch.stack([q1b, q1o], dim=1).view(-1, 1, n_agents)
        loss_q = (((mixer1(q_stacked, gs).view(-1) - backup) ** 2) +
                  ((mixer2(q_stacked, gs).view(-1) - backup) ** 2)).mean()

        opt_critic.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(critic_params, 10.0)
        opt_critic.step()
        recent_loss_q = loss_q.item()

        # Actor update  ── Q-FIXED: q_tot keeps gradient; only lmbda is detached ──
        if step % policy_freq == 0:
            a_bid, a_ope = actor(o_bid, o_ope)
            q1b_pi, q1o_pi = critic(o_bid, a_bid, o_ope, a_ope)
            qs = torch.stack([q1b_pi, q1o_pi], dim=1).view(-1, 1, n_agents)
            q_tot = torch.min(mixer1(qs, gs).view(-1), mixer2(qs, gs).view(-1))
            lmbda = alpha / (q_tot.abs().mean().detach() + 1e-8)

            loss_a = (-lmbda * q_tot.mean()
                      + bc_bid_w * F.mse_loss(a_bid, a_bid_exp)
                      + bc_ope_w * F.mse_loss(a_ope, a_ope_exp))
            if not torch.isnan(loss_a):
                opt_actor.zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                opt_actor.step()
                recent_loss_a = loss_a.item()

            _soft_update(critic_targ, critic, tau)
            _soft_update(mixer1_targ, mixer1, tau)
            _soft_update(mixer2_targ, mixer2, tau)

        step_times.append(time.perf_counter() - t0)

        # Validation
        if (step + 1) % eval_freq == 0:
            _sync_cpu()
            ret_val, daily_val = fast_evaluate(env_val, _batch_act)
            if record_curves:
                t_ret, t_daily = fast_evaluate(env_test, _batch_act)
                test_curve.append(float(np.array(t_daily)[:, 3].mean()))
            if ret_val > best_val:
                best_val = ret_val
                if record_curves:
                    test_daily = t_daily
                else:
                    _, test_daily = fast_evaluate(env_test, _batch_act)
                test_impl = {}
                if save_actor_path is not None:
                    os.makedirs(os.path.dirname(save_actor_path), exist_ok=True)
                    torch.save(actor.state_dict(), save_actor_path)
            val_returns.append(ret_val)
            val_daily.append(daily_val)

    timing = {'total_train_s': time.time() - t_train_start,
              'avg_step_ms': np.mean(step_times) * 1000, 'n_steps': max_steps,
              'test_curve': test_curve}
    return val_returns, val_daily, test_daily, test_impl, timing
