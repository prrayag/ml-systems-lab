from __future__ import annotations

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class DQNAgent:
    """Small DQN agent for discrete states and actions."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 32,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        if state_dim <= 0 or n_actions <= 0:
            raise ValueError("state_dim and n_actions must be positive")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be between 0 and 1")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        if seed is not None:
            torch.manual_seed(seed)

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_network = QNetwork(state_dim, n_actions, hidden_dim)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dim)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.sync_target()

    def encode_state(self, state: int) -> np.ndarray:
        if state < 0 or state >= self.state_dim:
            raise ValueError(f"invalid state {state}")
        encoded = np.zeros(self.state_dim, dtype=np.float32)
        encoded[state] = 1.0
        return encoded

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        state_tensor = torch.as_tensor(self.encode_state(state)).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_on_batch(self, batch: dict[str, np.ndarray]) -> float:
        states = torch.as_tensor(batch["states"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32)

        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.discount * next_values * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
