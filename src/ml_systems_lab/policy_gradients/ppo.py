from __future__ import annotations

import numpy as np
import torch

from ml_systems_lab.policy_gradients.agent import ActorCriticAgent
from ml_systems_lab.policy_gradients.reinforce import normalize
from ml_systems_lab.tabular.gridworld import GridWorld


def discounted_returns_with_resets(
    rewards: np.ndarray,
    dones: np.ndarray,
    discount: float,
) -> np.ndarray:
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be between 0 and 1")

    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + discount * running_return * (1.0 - float(dones[index]))
        returns[index] = running_return
    return returns


def collect_rollout(
    env: GridWorld,
    agent: ActorCriticAgent,
    steps: int = 256,
    max_episode_steps: int = 50,
) -> dict[str, np.ndarray]:
    if steps <= 0 or max_episode_steps <= 0:
        raise ValueError("steps and max_episode_steps must be positive")

    states = np.zeros(steps, dtype=int)
    actions = np.zeros(steps, dtype=int)
    old_log_probs = np.zeros(steps, dtype=np.float32)
    values = np.zeros(steps, dtype=np.float32)
    rewards = np.zeros(steps, dtype=np.float32)
    dones = np.zeros(steps, dtype=bool)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []

    state = env.reset()
    episode_return = 0.0
    episode_length = 0

    for step in range(steps):
        with torch.no_grad():
            action, log_prob, value = agent.select_action(state)

        result = env.step(action)
        episode_return += result.reward
        episode_length += 1
        episode_done = result.done or episode_length >= max_episode_steps

        states[step] = state
        actions[step] = action
        old_log_probs[step] = float(log_prob.item())
        values[step] = float(value.item())
        rewards[step] = result.reward
        dones[step] = episode_done

        if episode_done:
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            episode_successes.append(result.done)
            state = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            state = result.state

    return {
        "states": states,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "values": values,
        "rewards": rewards,
        "dones": dones,
        "episode_returns": np.array(episode_returns, dtype=float),
        "episode_lengths": np.array(episode_lengths, dtype=int),
        "episode_successes": np.array(episode_successes, dtype=bool),
    }


def clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()


def ppo_update(
    agent: ActorCriticAgent,
    rollout: dict[str, np.ndarray],
    discount: float = 0.95,
    clip_range: float = 0.2,
    epochs: int = 4,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    returns = discounted_returns_with_resets(rollout["rewards"], rollout["dones"], discount)
    advantages = normalize(returns - rollout["values"])

    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32)
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
    advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32)

    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0

    for _ in range(epochs):
        log_probs, entropy, values = agent.evaluate_actions(rollout["states"], rollout["actions"])
        policy_loss = clipped_policy_loss(log_probs, old_log_probs, advantages_tensor, clip_range)
        value_loss = torch.mean((returns_tensor - values) ** 2)
        entropy_bonus = torch.mean(entropy)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        last_policy_loss = float(policy_loss.item())
        last_value_loss = float(value_loss.item())
        last_entropy = float(entropy_bonus.item())

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
    }
