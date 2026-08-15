from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(states)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values


class PolicyGradientAgent:
    """Categorical policy over discrete actions."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 32,
        learning_rate: float = 1e-2,
        seed: int | None = None,
    ):
        if state_dim <= 0 or n_actions <= 0:
            raise ValueError("state_dim and n_actions must be positive")

        if seed is not None:
            torch.manual_seed(seed)

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        self.policy = PolicyNetwork(state_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def encode_state(self, state: int) -> np.ndarray:
        if state < 0 or state >= self.state_dim:
            raise ValueError(f"invalid state {state}")
        encoded = np.zeros(self.state_dim, dtype=np.float32)
        encoded[state] = 1.0
        return encoded

    def action_distribution(self, state: int) -> torch.distributions.Categorical:
        state_tensor = torch.as_tensor(self.encode_state(state)).unsqueeze(0)
        logits = self.policy(state_tensor)
        return torch.distributions.Categorical(logits=logits)

    def select_action(self, state: int) -> tuple[int, torch.Tensor]:
        distribution = self.action_distribution(state)
        action = distribution.sample()
        return int(action.item()), distribution.log_prob(action).squeeze(0)

    def select_greedy_action(self, state: int) -> int:
        state_tensor = torch.as_tensor(self.encode_state(state)).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(state_tensor)
        return int(torch.argmax(logits, dim=1).item())


class ActorCriticAgent:
    """Categorical policy with a learned state-value baseline."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 32,
        learning_rate: float = 3e-3,
        seed: int | None = None,
    ):
        if state_dim <= 0 or n_actions <= 0:
            raise ValueError("state_dim and n_actions must be positive")

        if seed is not None:
            torch.manual_seed(seed)

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.model = ActorCriticNetwork(state_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def encode_state(self, state: int) -> np.ndarray:
        if state < 0 or state >= self.state_dim:
            raise ValueError(f"invalid state {state}")
        encoded = np.zeros(self.state_dim, dtype=np.float32)
        encoded[state] = 1.0
        return encoded

    def encode_states(self, states: np.ndarray | list[int]) -> torch.Tensor:
        encoded = np.zeros((len(states), self.state_dim), dtype=np.float32)
        for row, state in enumerate(states):
            if state < 0 or state >= self.state_dim:
                raise ValueError(f"invalid state {state}")
            encoded[row, state] = 1.0
        return torch.as_tensor(encoded)

    def distribution_and_value(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        logits, values = self.model(states)
        return torch.distributions.Categorical(logits=logits), values

    def select_action(self, state: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        state_tensor = torch.as_tensor(self.encode_state(state)).unsqueeze(0)
        distribution, value = self.distribution_and_value(state_tensor)
        action = distribution.sample()
        return int(action.item()), distribution.log_prob(action).squeeze(0), value.squeeze(0)

    def select_greedy_action(self, state: int) -> int:
        state_tensor = torch.as_tensor(self.encode_state(state)).unsqueeze(0)
        with torch.no_grad():
            distribution, _ = self.distribution_and_value(state_tensor)
        return int(torch.argmax(distribution.logits, dim=1).item())

    def evaluate_actions(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tensor = self.encode_states(states)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        distribution, values = self.distribution_and_value(state_tensor)
        return distribution.log_prob(action_tensor), distribution.entropy(), values
