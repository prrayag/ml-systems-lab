from __future__ import annotations

import numpy as np
import torch

from ml_systems_lab.policy_gradients.agent import PolicyGradientAgent
from ml_systems_lab.policy_gradients.reinforce import reinforce_update
from ml_systems_lab.tabular.gridworld import GridWorld, default_gridworld


def train_reinforce(
    env: GridWorld,
    agent: PolicyGradientAgent,
    episodes: int = 500,
    max_steps: int = 50,
    discount: float = 0.99,
) -> dict[str, np.ndarray]:
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")

    returns = np.zeros(episodes, dtype=float)
    lengths = np.zeros(episodes, dtype=int)
    successes = np.zeros(episodes, dtype=bool)
    losses = np.zeros(episodes, dtype=float)

    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            result = env.step(action)
            log_probs.append(log_prob)
            rewards.append(result.reward)

            returns[episode] += result.reward
            state = result.state
            if result.done:
                successes[episode] = True
                lengths[episode] = step + 1
                break
        else:
            lengths[episode] = max_steps

        losses[episode] = reinforce_update(agent, log_probs, rewards, discount)

    return {
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
        "losses": losses,
    }


def evaluate_policy(
    env: GridWorld,
    agent: PolicyGradientAgent,
    episodes: int = 20,
    max_steps: int = 50,
) -> dict[str, float]:
    successes = 0
    total_steps = 0
    total_return = 0.0

    for _ in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_greedy_action(state)
            result = env.step(action)
            total_return += result.reward
            state = result.state
            if result.done:
                successes += 1
                total_steps += step + 1
                break
        else:
            total_steps += max_steps

    return {
        "success_rate": successes / episodes,
        "average_steps": total_steps / episodes,
        "average_return": total_return / episodes,
    }


def run_reinforce_training(
    episodes: int = 500,
    max_steps: int = 50,
    seed: int = 7,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = default_gridworld()
    agent = PolicyGradientAgent(
        state_dim=env.n_states,
        n_actions=env.n_actions,
        hidden_dim=32,
        learning_rate=5e-3,
        seed=seed,
    )
    history = train_reinforce(env, agent, episodes=episodes, max_steps=max_steps, discount=0.95)
    evaluation = evaluate_policy(env, agent, episodes=20, max_steps=max_steps)
    return history, evaluation

