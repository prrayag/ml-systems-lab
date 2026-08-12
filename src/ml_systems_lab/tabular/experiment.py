from __future__ import annotations

import numpy as np

from ml_systems_lab.bandits.metrics import mean_confidence_interval
from ml_systems_lab.tabular.agents import QLearningAgent, SarsaAgent
from ml_systems_lab.tabular.gridworld import default_gridworld
from ml_systems_lab.tabular.train import evaluate_greedy_policy, train_q_learning, train_sarsa


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    if window == 1:
        return values.copy()

    averaged = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        averaged[index] = np.mean(values[start : index + 1])
    return averaged


def run_tabular_comparison(
    episodes: int = 500,
    runs: int = 50,
    max_steps: int = 50,
    seed: int = 7,
) -> dict[str, dict[str, np.ndarray | float]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    seed_sequence = np.random.SeedSequence(seed)
    run_seeds = [int(child.generate_state(1)[0]) for child in seed_sequence.spawn(runs)]
    experiments = {
        "q-learning": _run_q_learning,
        "sarsa": _run_sarsa,
    }

    results: dict[str, dict[str, np.ndarray | float]] = {}
    for name, run_algorithm in experiments.items():
        returns = np.empty((runs, episodes), dtype=float)
        successes = np.empty((runs, episodes), dtype=float)
        eval_success = np.empty(runs, dtype=float)
        eval_steps = np.empty(runs, dtype=float)

        for run_index, run_seed in enumerate(run_seeds):
            history, evaluation = run_algorithm(episodes, max_steps, run_seed)
            returns[run_index] = history["returns"]
            successes[run_index] = history["successes"].astype(float)
            eval_success[run_index] = evaluation["success_rate"]
            eval_steps[run_index] = evaluation["average_steps"]

        return_mean, return_lower, return_upper = mean_confidence_interval(returns)
        success_mean, success_lower, success_upper = mean_confidence_interval(successes)

        results[name] = {
            "return": return_mean,
            "return_lower": return_lower,
            "return_upper": return_upper,
            "return_smoothed": moving_average(return_mean),
            "success_rate": success_mean,
            "success_rate_lower": success_lower,
            "success_rate_upper": success_upper,
            "success_rate_smoothed": moving_average(success_mean),
            "final_return": float(np.mean(returns[:, -50:])),
            "final_success_rate": float(np.mean(successes[:, -50:])),
            "eval_success_rate": float(np.mean(eval_success)),
            "eval_average_steps": float(np.mean(eval_steps)),
        }

    return results


def _run_q_learning(
    episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    env = default_gridworld()
    agent = QLearningAgent(
        env.n_states,
        env.n_actions,
        learning_rate=0.2,
        discount=0.95,
        epsilon=0.1,
        seed=seed,
    )
    history = train_q_learning(env, agent, episodes, max_steps)
    return history, evaluate_greedy_policy(env, agent, episodes=20, max_steps=max_steps)


def _run_sarsa(
    episodes: int,
    max_steps: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    env = default_gridworld()
    agent = SarsaAgent(
        env.n_states,
        env.n_actions,
        learning_rate=0.2,
        discount=0.95,
        epsilon=0.1,
        seed=seed,
    )
    history = train_sarsa(env, agent, episodes, max_steps)
    return history, evaluate_greedy_policy(env, agent, episodes=20, max_steps=max_steps)

