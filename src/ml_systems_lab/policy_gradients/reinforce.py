from __future__ import annotations

import numpy as np
import torch

from ml_systems_lab.policy_gradients.agent import PolicyGradientAgent


def discounted_returns(rewards: list[float], discount: float) -> np.ndarray:
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be between 0 and 1")

    returns = np.zeros(len(rewards), dtype=np.float32)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + discount * running_return
        returns[index] = running_return
    return returns


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return values
    std = np.std(values)
    if std == 0.0:
        return values - np.mean(values)
    return (values - np.mean(values)) / (std + 1e-8)


def reinforce_update(
    agent: PolicyGradientAgent,
    log_probs: list[torch.Tensor],
    rewards: list[float],
    discount: float = 0.99,
) -> float:
    returns = normalize(discounted_returns(rewards, discount))
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
    log_prob_tensor = torch.stack(log_probs)

    loss = -(log_prob_tensor * returns_tensor).sum()
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    return float(loss.item())

