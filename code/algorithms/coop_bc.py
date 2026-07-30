"""
Cooperative Dual-Agent Behavior Cloning (Coop-DABC).

Algorithm: TD3+BC with QMIX-style cooperative value decomposition.
  - Policy is cloned from offline expert demonstrations (BC loss).
  - Value function uses QMIX Hypernet to aggregate per-agent Q-values into Q_total.
  - No online environment interaction during training (offline RL).

Reference: Fujimoto & Gu, "A Minimalist Approach to Offline RL" (TD3+BC, 2021).
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import Adam

from envs.env_multi import DualAgentEnv
from models.networks import Actor, MLPQFunction, Hypernet
from algorithms.utils import evaluation_multi_agent, fast_evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
          policy_freq:   int   = 1,
          hidden_size:   int   = 128,
          tau:           float = 5e-3,
          lr_actor:      float = 3e-4,
          lr_critic:     float = 3e-4,
          batch_size:  int   = 256,
          eval_freq:   int   = 50,
          save_actor_path: str = None):
    """
    Train the Coop-DABC agent on offline expert data.

    Args:
        env_train / env_val / env_test : market environments
        buf_bid / buf_ope              : expert replay buffers (ReplayBuffer)
        seed                           : random seed
        max_steps                      : number of gradient update steps
        discount                       : reward discount factor γ
        policy_noise / noise_clip      : target-policy smoothing (TD3)
        alpha                          : BC regularisation coefficient
        policy_freq                    : actor update frequency (per critic step)
        hidden_size                    : MLP / GRU hidden dimension
        tau                            : soft-update rate
        lr_actor / lr_critic           : learning rates
        batch_size                     : mini-batch size

    Returns:
        val_returns     : list of per-step validation returns
        val_daily       : list of per-step validation daily returns
        test_daily      : daily returns on best-val-selected test rollout
        test_impl       : implementation details on test rollout
        timing          : dict with training/inference timing stats
    """
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

    # ── Networks ──────────────────────────────────────────────────────────────
    actor   = Actor(obs_bid_dim, obs_ope_dim, act_dim, hidden_size, lim_bid, lim_ope).to(device)
    critic  = MLPQFunction(obs_bid_dim, obs_ope_dim, act_dim, hidden_size).to(device)
    mixer1  = Hypernet(global_dim, n_agents).to(device)
    mixer2  = Hypernet(global_dim, n_agents).to(device)

    critic_targ = deepcopy(critic).to(device)
    mixer1_targ = deepcopy(mixer1).to(device)
    mixer2_targ = deepcopy(mixer2).to(device)
    for p in list(critic_targ.parameters()) + list(mixer1_targ.parameters()) + list(mixer2_targ.parameters()):
        p.requires_grad = False

    critic_params = list(critic.parameters()) + list(mixer1.parameters()) + list(mixer2.parameters())
    opt_critic = Adam(critic_params, lr=lr_critic)
    opt_actor  = Adam(actor.parameters(), lr=lr_actor)

    # ── Inference helper (CPU actor avoids PCIe round-trip at batch_size=1) ──
    actor_cpu = deepcopy(actor).cpu()
    _nan_action_reported = [False]

    def _sync_cpu():
        actor_cpu.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})

    def _batch_act(s_bid, s_ope):
        with torch.inference_mode():
            a_b, a_o = actor_cpu(torch.from_numpy(s_bid), torch.from_numpy(s_ope))
        return a_b[:, 0].numpy(), a_o[:, 0].numpy()

    def _check_actor_nan():
        for name, p in actor_cpu.named_parameters():
            if torch.isnan(p).any():
                print(f'  [NaN-WEIGHT] actor_cpu.{name}  nan={torch.isnan(p).sum().item()}/{p.numel()}', flush=True)
                return True
        return False

    def get_action(obs_bid, obs_ope, determine=True, turn_on=True):
        with torch.inference_mode():
            a_bid, a_ope = actor_cpu(
                torch.as_tensor(obs_bid, dtype=torch.float32).unsqueeze(0),
                torch.as_tensor(obs_ope, dtype=torch.float32).unsqueeze(0))
        ab, ao = a_bid[0, 0].item(), a_ope[0, 0].item()
        if (np.isnan(ab) or np.isnan(ao)) and not _nan_action_reported[0]:
            _nan_action_reported[0] = True
            obs_nan = np.isnan(obs_bid).any() or np.isnan(obs_ope).any()
            print(f'  [NaN-ACTION] a_bid={ab}  a_ope={ao}  obs_has_nan={obs_nan}', flush=True)
        return ab, ao

    # ── Training loop ─────────────────────────────────────────────────────────
    val_returns, val_daily = [], []
    best_val, test_daily, test_impl = -1e9, None, None
    step_times = []
    t_train_start = time.time()

    TRAIN_LOG_FREQ = 50   # print training loss every N steps
    EVAL_LOG_FREQ  = 200  # print val result every N steps (must be multiple of eval_freq)

    print(f'[Coop-DABC] train={env_train.len_data()//96}d  val={env_val.len_data()//96}d  test={env_test.len_data()//96}d  steps={max_steps}', flush=True)

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

        # Actor update
        if step % policy_freq == 0:
            a_bid, a_ope = actor(o_bid, o_ope)
            with torch.no_grad():
                q1b_pi, q1o_pi = critic(o_bid, a_bid, o_ope, a_ope)
                qs = torch.stack([q1b_pi, q1o_pi], dim=1).view(-1, 1, n_agents)
                q_tot = torch.min(mixer1(qs, gs).view(-1), mixer2(qs, gs).view(-1))
                lmbda = alpha / (q_tot.abs().mean() + 1e-8)

            loss_a = (-lmbda * q_tot.mean()
                      + F.mse_loss(a_bid, a_bid_exp)
                      + F.mse_loss(a_ope, a_ope_exp))
            if torch.isnan(loss_a):
                if step < 5 or (step + 1) % TRAIN_LOG_FREQ == 0:
                    print(f'  [NaN-LOSS_A] step={step}  lmbda={lmbda:.4f}  '
                          f'q_tot_mean={q_tot.mean().item():.4f}  '
                          f'a_bid_nan={torch.isnan(a_bid).any().item()}  '
                          f'a_ope_nan={torch.isnan(a_ope).any().item()}', flush=True)
            else:
                opt_actor.zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                opt_actor.step()
                recent_loss_a = loss_a.item()

            _soft_update(critic_targ, critic, tau)
            _soft_update(mixer1_targ, mixer1, tau)
            _soft_update(mixer2_targ, mixer2, tau)

        step_times.append(time.perf_counter() - t0)

        # Training loss log (no eval needed)
        if (step + 1) % TRAIN_LOG_FREQ == 0:
            elapsed = time.time() - t_train_start
            pct = (step + 1) / max_steps * 100
            print(f'  [{step+1:4d}/{max_steps}  {pct:.0f}%  {elapsed:.1f}s] '
                  f'loss_q={recent_loss_q:.4f}  loss_a={recent_loss_a:.4f}', flush=True)

        # Validation (every eval_freq steps)
        if (step + 1) % eval_freq == 0:
            _sync_cpu()
            _nan_action_reported[0] = False  # reset per eval
            has_nan_w = _check_actor_nan()
            ret_val, daily_val = fast_evaluate(env_val, _batch_act)
            if np.isnan(ret_val):
                print(f'  [NaN-RETURN] step={step+1}  nan_weights={has_nan_w}', flush=True)
            if ret_val > best_val:
                best_val = ret_val
                _, test_daily = fast_evaluate(env_test, _batch_act)
                test_impl = {}
                # ── Save best-validation actor weights (for OOD / inference) ──
                if save_actor_path is not None:
                    os.makedirs(os.path.dirname(save_actor_path), exist_ok=True)
                    torch.save(actor.state_dict(), save_actor_path)
            val_returns.append(ret_val)
            val_daily.append(daily_val)

            if (step + 1) % EVAL_LOG_FREQ == 0:
                elapsed = time.time() - t_train_start
                pct = (step + 1) / max_steps * 100
                print(f'★ [{step+1:4d}/{max_steps}  {pct:.0f}%  {elapsed:.1f}s] '
                      f'val={ret_val:.0f}  best={best_val:.0f}', flush=True)

    # ── Timing report ────────────────────────────────────────────────────────
    timing = {
        'total_train_s': time.time() - t_train_start,
        'avg_step_ms':   np.mean(step_times) * 1000,
        'n_steps':       max_steps,
    }
    print(f'\n[Coop-DABC] total={timing["total_train_s"]:.0f}s  '
          f'step={timing["avg_step_ms"]:.1f}ms')

    return val_returns, val_daily, test_daily, test_impl, timing
