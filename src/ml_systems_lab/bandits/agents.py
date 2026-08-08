from __future__ import annotations

import numpy as np


class BanditAgent:
    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: int) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class EpsilonGreedyAgent(BanditAgent):
    """Epsilon-greedy agent with sample-average value estimates."""

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int | None = None):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        self.n_arms = n_arms
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts.fill(0)
        self.values.fill(0.0)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, action: int, reward: int) -> None:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"invalid action {action}")

        self.counts[action] += 1
        step_size = 1.0 / self.counts[action]
        self.values[action] += step_size * (reward - self.values[action])

