"""
Neural network architectures for the dual-agent bidding framework.

Dual-agent networks (bid + operation markets):
  Actor              — deterministic policy (used by Coop-DABC)
  MLPQFunction       — local Q-function with shared GRU critic
  Hypernet           — QMIX-style global value mixer
  SquashedGaussianMLPActor — stochastic policy (used by Coop-SAC)

Single-agent networks (single market):
  SingleAgentActor   — stochastic policy (used by Single-SAC, Inde-DASAC)
  SingleAgentQFunction — Q-function
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

_LOG_STD_MAX =  1
_LOG_STD_MIN = -20


# ── Dual-agent ───────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Deterministic dual-agent policy with GRU coupling between bid and operation."""

    def __init__(self, obs_dim_bid, obs_dim_ope, act_dim, hidden_size, act_limit_bid, act_limit_ope):
        super().__init__()
        self.mlp_bid = nn.Sequential(
            nn.Linear(obs_dim_bid, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mlp_ope = nn.Sequential(
            nn.Linear(obs_dim_ope, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.head_bid = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )
        self.head_ope = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )
        self.act_limit_bid = act_limit_bid
        self.act_limit_ope = act_limit_ope

    def forward(self, obs_bid, obs_ope):
        h0 = torch.zeros(1, obs_bid.size(0), self.gru.hidden_size, device=obs_bid.device)
        gru_bid, h_bid = self.gru(self.mlp_bid(obs_bid).unsqueeze(1), h0)
        gru_ope, _     = self.gru(self.mlp_ope(obs_ope).unsqueeze(1), h_bid)

        a_bid = torch.tanh(self.head_bid(gru_bid)).squeeze(2) * self.act_limit_bid
        a_ope = torch.tanh(self.head_ope(gru_ope)).squeeze(2) * self.act_limit_ope
        return a_bid, a_ope


class MLPQFunction(nn.Module):
    """Local Q-function for dual-agent setting, using shared GRU to couple markets."""

    def __init__(self, obs_dim_bid, obs_dim_ope, act_dim, hidden_size):
        super().__init__()
        self.net_bid = nn.Sequential(
            nn.Linear(obs_dim_bid + act_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),           nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.net_ope = nn.Sequential(
            nn.Linear(obs_dim_ope + act_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),           nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.final = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs_bid, act_bid, obs_ope, act_ope):
        h0 = torch.zeros(1, obs_bid.size(0), self.gru.hidden_size, device=obs_bid.device)
        gru_bid, h_bid = self.gru(self.net_bid(torch.cat([obs_bid, act_bid], -1)).unsqueeze(1), h0)
        gru_ope, _     = self.gru(self.net_ope(torch.cat([obs_ope, act_ope], -1)).unsqueeze(1), h_bid)

        q_bid = self.final(F.relu(gru_bid.squeeze(1)))
        q_ope = self.final(F.relu(gru_ope.squeeze(1)))
        return q_bid, q_ope


class Hypernet(nn.Module):
    """QMIX-style hypernetwork: mixes per-agent Q-values into a global Q-total."""

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
        # q_n: (B, 1, agent_num)
        w1 = self.w1(global_state).abs().view(-1, self.agent_num, self.hidden_size)
        b1 = self.b1(global_state).view(-1, 1, self.hidden_size)
        hidden   = F.elu(torch.bmm(q_n, w1) + b1)
        w_final  = self.w_final(global_state).view(-1, self.hidden_size, 1)
        v        = self.V(global_state).view(-1, 1, 1)
        return torch.bmm(hidden, w_final) + v


class SquashedGaussianMLPActor(nn.Module):
    """Stochastic dual-agent policy with GRU coupling (used by Coop-SAC)."""

    def __init__(self, obs_dim_bid, obs_dim_ope, act_dim, hidden_size, act_limit_bid, act_limit_ope):
        super().__init__()
        def _mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_size), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
            )
        def _head(out_dim):
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_size, out_dim),
            )
        self.mlp_bid = _mlp(obs_dim_bid)
        self.mlp_ope = _mlp(obs_dim_ope)
        self.gru     = nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)

        self.mu_bid      = _head(act_dim)
        self.log_std_bid = _head(act_dim)
        self.mu_ope      = _head(act_dim)
        self.log_std_ope = _head(act_dim)

        self.act_limit_bid = act_limit_bid
        self.act_limit_ope = act_limit_ope

    def _sample(self, mu_layer, log_std_layer, gru_out, act_limit, deterministic, with_logprob):
        mu      = mu_layer(gru_out)
        log_std = log_std_layer(gru_out).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mu, std)
        raw     = mu if deterministic else dist.rsample()

        logp = None
        if with_logprob:
            logp = dist.log_prob(raw).sum(-1)
            logp -= (2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(-1)

        action = torch.tanh(raw).squeeze(2) * act_limit
        return action, logp

    def forward(self, obs_bid, obs_ope, deterministic=False, with_logprob=True):
        h0 = torch.zeros(1, obs_bid.size(0), self.gru.hidden_size, device=obs_bid.device)
        gru_bid, h_bid = self.gru(self.mlp_bid(obs_bid).unsqueeze(1), h0)
        gru_ope, _     = self.gru(self.mlp_ope(obs_ope).unsqueeze(1), h_bid)

        a_bid, logp_bid = self._sample(self.mu_bid, self.log_std_bid, gru_bid, self.act_limit_bid, deterministic, with_logprob)
        a_ope, logp_ope = self._sample(self.mu_ope, self.log_std_ope, gru_ope, self.act_limit_ope, deterministic, with_logprob)
        return a_bid, a_ope, logp_bid, logp_ope


# ── Single-agent ─────────────────────────────────────────────────────────────

class SingleAgentActor(nn.Module):
    """Stochastic MLP policy for single-obs-space agent (Single-SAC, Inde-DASAC)."""

    def __init__(self, obs_dim, act_dim, hidden_size, act_limit):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
        )
        def _head():
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_size, act_dim),
            )
        self.mu_layer      = _head()
        self.log_std_layer = _head()
        self.act_limit     = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.mlp(obs)
        mu      = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std     = log_std.exp()
        dist    = Normal(mu, std)
        raw     = mu if deterministic else dist.rsample()

        logp = None
        if with_logprob:
            logp = dist.log_prob(raw).sum(-1)
            logp -= (2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(-1)

        action = torch.tanh(raw) * self.act_limit
        return action, logp


class SingleAgentQFunction(nn.Module):
    """Q-function for single-obs-space agent."""

    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))
