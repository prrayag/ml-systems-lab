from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy action selection."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount: float = 0.99,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive")
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be between 0 and 1")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.q_values = np.zeros((n_states, n_actions), dtype=float)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.q_values.fill(0.0)

    def select_action(self, state: int, greedy: bool = False) -> int:
        if state < 0 or state >= self.n_states:
            raise ValueError(f"invalid state {state}")
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        target = reward
        if not done:
            target += self.discount * np.max(self.q_values[next_state])

        error = target - self.q_values[state, action]
        self.q_values[state, action] += self.learning_rate * error

