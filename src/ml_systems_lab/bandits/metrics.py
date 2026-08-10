from __future__ import annotations

import numpy as np


def average_reward(rewards: np.ndarray) -> np.ndarray:
    return np.mean(rewards, axis=0)


def cumulative_reward(rewards: np.ndarray) -> np.ndarray:
    return np.cumsum(rewards, axis=-1)


def optimal_action_rate(actions: np.ndarray, optimal_action: int) -> np.ndarray:
    return np.mean(actions == optimal_action, axis=0)


def mean_confidence_interval(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] < 2:
        return mean, mean.copy(), mean.copy()

    standard_error = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    margin = 1.96 * standard_error
    return mean, mean - margin, mean + margin


def instantaneous_regret(
    actions: np.ndarray,
    probabilities: list[float] | np.ndarray,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    optimal_expected_reward = np.max(probabilities)
    selected_expected_rewards = probabilities[actions]
    return optimal_expected_reward - selected_expected_rewards


def cumulative_regret(
    actions: np.ndarray,
    probabilities: list[float] | np.ndarray,
) -> np.ndarray:
    regret = instantaneous_regret(actions, probabilities)
    return np.cumsum(regret, axis=-1)
