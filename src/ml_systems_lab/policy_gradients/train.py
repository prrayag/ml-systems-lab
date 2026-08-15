from __future__ import annotations

import numpy as np
import torch

from ml_systems_lab.policy_gradients.agent import ActorCriticAgent, PolicyGradientAgent
from ml_systems_lab.policy_gradients.ppo import collect_rollout, ppo_update
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


def policy_action_probabilities(env: GridWorld, agent: PolicyGradientAgent) -> np.ndarray:
    probabilities = np.zeros((env.n_states, env.n_actions), dtype=float)
    for state in range(env.n_states):
        with torch.no_grad():
            distribution = agent.action_distribution(state)
        probabilities[state] = distribution.probs.squeeze(0).numpy()
    return probabilities


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


def train_ppo(
    env: GridWorld,
    agent: ActorCriticAgent,
    updates: int = 80,
    rollout_steps: int = 128,
    max_steps: int = 50,
    discount: float = 0.95,
    clip_range: float = 0.2,
    epochs: int = 4,
) -> dict[str, np.ndarray]:
    if updates <= 0 or rollout_steps <= 0 or max_steps <= 0:
        raise ValueError("updates, rollout_steps, and max_steps must be positive")

    returns = np.zeros(updates, dtype=float)
    lengths = np.zeros(updates, dtype=float)
    successes = np.zeros(updates, dtype=float)
    policy_losses = np.zeros(updates, dtype=float)
    value_losses = np.zeros(updates, dtype=float)
    entropies = np.zeros(updates, dtype=float)

    for update in range(updates):
        rollout = collect_rollout(env, agent, steps=rollout_steps, max_episode_steps=max_steps)
        metrics = ppo_update(
            agent,
            rollout,
            discount=discount,
            clip_range=clip_range,
            epochs=epochs,
        )

        if len(rollout["episode_returns"]) > 0:
            returns[update] = float(np.mean(rollout["episode_returns"]))
            lengths[update] = float(np.mean(rollout["episode_lengths"]))
            successes[update] = float(np.mean(rollout["episode_successes"]))

        policy_losses[update] = metrics["policy_loss"]
        value_losses[update] = metrics["value_loss"]
        entropies[update] = metrics["entropy"]

    return {
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "entropies": entropies,
    }


def run_ppo_training(
    updates: int = 80,
    rollout_steps: int = 128,
    max_steps: int = 50,
    seed: int = 23,
) -> tuple[dict[str, np.ndarray], dict[str, float], ActorCriticAgent]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = default_gridworld()
    agent = ActorCriticAgent(
        state_dim=env.n_states,
        n_actions=env.n_actions,
        hidden_dim=32,
        learning_rate=3e-3,
        seed=seed,
    )
    history = train_ppo(
        env,
        agent,
        updates=updates,
        rollout_steps=rollout_steps,
        max_steps=max_steps,
        discount=0.95,
        clip_range=0.2,
        epochs=4,
    )
    evaluation = evaluate_policy(env, agent, episodes=20, max_steps=max_steps)
    return history, evaluation, agent
