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


class RandomAgent(BanditAgent):
    """Uniform random action selection."""

    def __init__(self, n_arms: int, seed: int | None = None):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")

        self.n_arms = n_arms
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def select_action(self) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, action: int, reward: int) -> None:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"invalid action {action}")


class OptimisticInitialValuesAgent(BanditAgent):
    """Greedy agent initialized with optimistic value estimates."""

    def __init__(
        self,
        n_arms: int,
        initial_value: float = 1.0,
        seed: int | None = None,
    ):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")

        self.n_arms = n_arms
        self.initial_value = initial_value
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.full(n_arms, initial_value, dtype=float)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.counts.fill(0)
        self.values.fill(self.initial_value)

    def select_action(self) -> int:
        return int(np.argmax(self.values))

    def update(self, action: int, reward: int) -> None:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"invalid action {action}")

        self.counts[action] += 1
        step_size = 1.0 / self.counts[action]
        self.values[action] += step_size * (reward - self.values[action])


class UCBAgent(BanditAgent):
    """Upper Confidence Bound action selection."""

    def __init__(self, n_arms: int, c: float = 2.0):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        if c < 0.0:
            raise ValueError("c must be non-negative")

        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        self.t = 0

    def reset(self) -> None:
        self.counts.fill(0)
        self.values.fill(0.0)
        self.t = 0

    def select_action(self) -> int:
        # Pull every arm once before applying the UCB formula.
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            return int(untried[0])

        scores = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(scores))

    def update(self, action: int, reward: int) -> None:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"invalid action {action}")

        self.t += 1
        self.counts[action] += 1
        step_size = 1.0 / self.counts[action]
        self.values[action] += step_size * (reward - self.values[action])


class ThompsonSamplingAgent(BanditAgent):
    """Thompson Sampling for Bernoulli rewards."""

    def __init__(self, n_arms: int, seed: int | None = None):
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")

        self.n_arms = n_arms
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_arms, dtype=float)
        self.beta = np.ones(n_arms, dtype=float)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.alpha.fill(1.0)
        self.beta.fill(1.0)

    def select_action(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, action: int, reward: int) -> None:
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"invalid action {action}")
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")

        if reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
