from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ml_systems_lab.bandits.agents import (
    BanditAgent,
    EpsilonGreedyAgent,
    OptimisticInitialValuesAgent,
    ThompsonSamplingAgent,
    UCBAgent,
)
from ml_systems_lab.bandits.environment import BernoulliBandit
from ml_systems_lab.bandits.metrics import (
    average_reward,
    cumulative_regret,
    cumulative_reward,
    mean_confidence_interval,
    optimal_action_rate,
)


DEFAULT_PROBABILITIES = np.array([0.10, 0.35, 0.60, 0.45, 0.80])

AgentFactory = Callable[[int, int], BanditAgent]


def _scalar_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean, lower, upper = mean_confidence_interval(values[:, np.newaxis])
    return float(mean[0]), float(lower[0]), float(upper[0])


def standard_agents() -> dict[str, AgentFactory]:
    return {
        "epsilon-greedy (0.1)": lambda n_arms, seed: EpsilonGreedyAgent(
            n_arms=n_arms,
            epsilon=0.1,
            seed=seed,
        ),
        "optimistic values": lambda n_arms, seed: OptimisticInitialValuesAgent(
            n_arms=n_arms,
            initial_value=2.0,
            seed=seed,
        ),
        "ucb1": lambda n_arms, seed: UCBAgent(n_arms=n_arms, c=2.0),
        "thompson sampling": lambda n_arms, seed: ThompsonSamplingAgent(
            n_arms=n_arms,
            seed=seed,
        ),
    }


def epsilon_agents() -> dict[str, AgentFactory]:
    return {
        f"epsilon-greedy ({epsilon})": (
            lambda n_arms, seed, epsilon=epsilon: EpsilonGreedyAgent(
                n_arms=n_arms,
                epsilon=epsilon,
                seed=seed,
            )
        )
        for epsilon in (0.01, 0.1, 0.2)
    }


def run_agent(
    agent: BanditAgent,
    probabilities: np.ndarray,
    steps: int,
    env_seed: int,
) -> dict[str, np.ndarray]:
    env = BernoulliBandit(probabilities, seed=env_seed)
    actions = np.empty(steps, dtype=int)
    rewards = np.empty(steps, dtype=int)

    for step in range(steps):
        action = agent.select_action()
        reward = env.step(action)
        agent.update(action, reward)

        actions[step] = action
        rewards[step] = reward

    return {"actions": actions, "rewards": rewards}


def run_comparison(
    agent_factories: dict[str, AgentFactory],
    probabilities: list[float] | np.ndarray | None = None,
    steps: int = 2000,
    runs: int = 200,
    seed: int = 7,
) -> dict[str, dict[str, np.ndarray | float]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")

    probabilities = np.asarray(
        DEFAULT_PROBABILITIES if probabilities is None else probabilities,
        dtype=float,
    )
    n_arms = len(probabilities)
    optimal_action = int(np.argmax(probabilities))

    results: dict[str, dict[str, np.ndarray | float]] = {}
    seed_sequence = np.random.SeedSequence(seed)

    for name, build_agent in agent_factories.items():
        rewards = np.empty((runs, steps), dtype=int)
        actions = np.empty((runs, steps), dtype=int)
        run_seeds = seed_sequence.spawn(runs)

        for run_index, run_seed in enumerate(run_seeds):
            env_seed, agent_seed = run_seed.generate_state(2)
            agent = build_agent(n_arms, int(agent_seed))
            run = run_agent(agent, probabilities, steps, int(env_seed))
            rewards[run_index] = run["rewards"]
            actions[run_index] = run["actions"]

        regret = cumulative_regret(actions, probabilities)
        total_reward = cumulative_reward(rewards)
        optimal_actions = actions == optimal_action
        reward_mean, reward_lower, reward_upper = mean_confidence_interval(rewards)
        optimal_mean, optimal_lower, optimal_upper = mean_confidence_interval(optimal_actions)
        regret_mean, regret_lower, regret_upper = mean_confidence_interval(regret)
        cumulative_reward_mean = np.mean(total_reward, axis=0)
        final_reward = np.mean(rewards[:, -100:], axis=1)
        final_optimal = np.mean(optimal_actions[:, -100:], axis=1)
        reward_value, reward_final_lower, reward_final_upper = _scalar_interval(final_reward)
        optimal_value, optimal_final_lower, optimal_final_upper = _scalar_interval(final_optimal)
        regret_value, regret_final_lower, regret_final_upper = _scalar_interval(regret[:, -1])

        results[name] = {
            "average_reward": average_reward(rewards),
            "average_reward_lower": reward_lower,
            "average_reward_upper": reward_upper,
            "cumulative_reward": cumulative_reward_mean,
            "optimal_action_rate": optimal_action_rate(actions, optimal_action),
            "optimal_action_rate_lower": optimal_lower,
            "optimal_action_rate_upper": optimal_upper,
            "cumulative_regret": regret_mean,
            "cumulative_regret_lower": regret_lower,
            "cumulative_regret_upper": regret_upper,
            "final_average_reward": reward_value,
            "final_average_reward_lower": reward_final_lower,
            "final_average_reward_upper": reward_final_upper,
            "final_cumulative_reward": float(np.mean(total_reward[:, -1])),
            "final_optimal_action_rate": optimal_value,
            "final_optimal_action_rate_lower": optimal_final_lower,
            "final_optimal_action_rate_upper": optimal_final_upper,
            "final_cumulative_regret": regret_value,
            "final_cumulative_regret_lower": regret_final_lower,
            "final_cumulative_regret_upper": regret_final_upper,
        }

    return results
