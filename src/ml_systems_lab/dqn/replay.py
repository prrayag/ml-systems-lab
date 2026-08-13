from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Fixed-size replay buffer for DQN transitions."""

    def __init__(self, capacity: int, state_dim: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self.size:
            raise ValueError("batch_size cannot exceed buffer size")

        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size

