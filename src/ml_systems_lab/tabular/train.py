from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ml_systems_lab.tabular.agents import QLearningAgent, SarsaAgent
from ml_systems_lab.tabular.gridworld import GridWorld, default_gridworld


def train_q_learning(
    env: GridWorld,
    agent: QLearningAgent,
    episodes: int,
    max_steps: int,
) -> dict[str, np.ndarray]:
    returns = np.zeros(episodes, dtype=float)
    lengths = np.zeros(episodes, dtype=int)
    successes = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        state = env.reset()

        for step in range(max_steps):
            action = agent.select_action(state)
            result = env.step(action)
            agent.update(state, action, result.reward, result.state, result.done)

            returns[episode] += result.reward
            state = result.state
            if result.done:
                successes[episode] = True
                lengths[episode] = step + 1
                break
        else:
            lengths[episode] = max_steps

    return {"returns": returns, "lengths": lengths, "successes": successes}


def train_sarsa(
    env: GridWorld,
    agent: SarsaAgent,
    episodes: int,
    max_steps: int,
) -> dict[str, np.ndarray]:
    returns = np.zeros(episodes, dtype=float)
    lengths = np.zeros(episodes, dtype=int)
    successes = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        state = env.reset()
        action = agent.select_action(state)

        for step in range(max_steps):
            result = env.step(action)
            next_action = agent.select_action(result.state)
            agent.update(
                state,
                action,
                result.reward,
                result.state,
                next_action,
                result.done,
            )

            returns[episode] += result.reward
            state = result.state
            action = next_action
            if result.done:
                successes[episode] = True
                lengths[episode] = step + 1
                break
        else:
            lengths[episode] = max_steps

    return {"returns": returns, "lengths": lengths, "successes": successes}


def evaluate_greedy_policy(
    env: GridWorld,
    agent: QLearningAgent | SarsaAgent,
    episodes: int = 20,
    max_steps: int = 50,
) -> dict[str, float]:
    successes = 0
    total_steps = 0
    total_return = 0.0

    for _ in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.select_action(state, greedy=True)
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


def run_training(
    build_agent: Callable[[int, int, int], QLearningAgent | SarsaAgent],
    train_agent: Callable[[GridWorld, QLearningAgent | SarsaAgent, int, int], dict[str, np.ndarray]],
    episodes: int = 500,
    max_steps: int = 50,
    seed: int = 7,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    env = default_gridworld()
    agent = build_agent(env.n_states, env.n_actions, seed)
    history = train_agent(env, agent, episodes, max_steps)
    evaluation = evaluate_greedy_policy(env, agent, episodes=20, max_steps=max_steps)
    return history, evaluation

