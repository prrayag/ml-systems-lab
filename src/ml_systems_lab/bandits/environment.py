from __future__ import annotations

import numpy as np


class BernoulliBandit:
    """Stochastic K-armed bandit with Bernoulli rewards."""

    def __init__(self, probabilities: list[float] | np.ndarray, seed: int | None = None):
        self.probabilities = np.asarray(probabilities, dtype=float)
        if self.probabilities.ndim != 1 or len(self.probabilities) == 0:
            raise ValueError("probabilities must be a non-empty 1D array")
        if np.any((self.probabilities < 0.0) | (self.probabilities > 1.0)):
            raise ValueError("probabilities must be between 0 and 1")

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_arms = len(self.probabilities)
        self.optimal_action = int(np.argmax(self.probabilities))
        self.optimal_expected_reward = float(np.max(self.probabilities))

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def step(self, action: int) -> int:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"invalid action {action}")
        return int(self.rng.random() < self.probabilities[action])

