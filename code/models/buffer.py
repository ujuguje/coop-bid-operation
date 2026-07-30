import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size FIFO experience replay buffer."""

    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs_buf  = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf  = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.idx]  = obs
        self.obs2_buf[self.idx] = next_obs
        self.act_buf[self.idx]  = act
        self.rew_buf[self.idx]  = rew
        self.done_buf[self.idx] = done
        self.idx  = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, idxs):
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
