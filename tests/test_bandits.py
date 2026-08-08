import numpy as np
import pytest

from ml_systems_lab.bandits.agents import (
    EpsilonGreedyAgent,
    OptimisticInitialValuesAgent,
    ThompsonSamplingAgent,
    UCBAgent,
)
from ml_systems_lab.bandits.environment import BernoulliBandit
from ml_systems_lab.bandits.experiment import (
    epsilon_agents,
    run_comparison,
    standard_agents,
)
from ml_systems_lab.bandits.metrics import (
    average_reward,
    cumulative_regret,
    cumulative_reward,
    instantaneous_regret,
    optimal_action_rate,
)


def test_environment_returns_bernoulli_rewards():
    bandit = BernoulliBandit([0.0, 1.0], seed=3)

    assert bandit.step(0) == 0
    assert bandit.step(1) == 1


def test_environment_rejects_invalid_action():
    bandit = BernoulliBandit([0.2, 0.8])

    with pytest.raises(ValueError):
        bandit.step(2)


def test_environment_reproducible_with_fixed_seed():
    first = BernoulliBandit([0.25, 0.75], seed=11)
    second = BernoulliBandit([0.25, 0.75], seed=11)

    first_rewards = [first.step(1) for _ in range(20)]
    second_rewards = [second.step(1) for _ in range(20)]

    assert first_rewards == second_rewards


def test_environment_reset_restarts_random_sequence():
    bandit = BernoulliBandit([0.5], seed=19)

    before_reset = [bandit.step(0) for _ in range(10)]
    bandit.reset()
    after_reset = [bandit.step(0) for _ in range(10)]

    assert before_reset == after_reset


def test_epsilon_greedy_incremental_value_update():
    agent = EpsilonGreedyAgent(n_arms=2, epsilon=0.0, seed=7)

    agent.update(0, 1)
    agent.update(0, 0)
    agent.update(0, 1)

    assert agent.counts[0] == 3
    assert agent.values[0] == pytest.approx(2 / 3)


def test_epsilon_greedy_tracks_action_counts():
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0)

    agent.update(0, 1)
    agent.update(2, 0)
    agent.update(2, 1)

    np.testing.assert_array_equal(agent.counts, np.array([1, 0, 2]))


def test_epsilon_zero_selects_best_estimated_action():
    agent = EpsilonGreedyAgent(n_arms=3, epsilon=0.0, seed=5)
    agent.values[:] = [0.1, 0.6, 0.3]

    assert agent.select_action() == 1


def test_epsilon_one_explores_with_seeded_rng():
    first = EpsilonGreedyAgent(n_arms=4, epsilon=1.0, seed=23)
    second = EpsilonGreedyAgent(n_arms=4, epsilon=1.0, seed=23)

    first_actions = [first.select_action() for _ in range(10)]
    second_actions = [second.select_action() for _ in range(10)]

    assert first_actions == second_actions
    assert len(set(first_actions)) > 1


def test_optimistic_agent_starts_with_configured_values():
    agent = OptimisticInitialValuesAgent(n_arms=3, initial_value=2.5)

    np.testing.assert_allclose(agent.values, np.array([2.5, 2.5, 2.5]))


def test_optimistic_agent_uses_incremental_value_update():
    agent = OptimisticInitialValuesAgent(n_arms=2, initial_value=1.0)

    agent.update(0, 0)
    agent.update(0, 1)

    assert agent.counts[0] == 2
    assert agent.values[0] == pytest.approx(0.5)


def test_ucb_selects_each_arm_once_initially():
    agent = UCBAgent(n_arms=3, c=2.0)

    actions = []
    for _ in range(3):
        action = agent.select_action()
        actions.append(action)
        agent.update(action, 0)

    assert actions == [0, 1, 2]


def test_ucb_selection_after_all_arms_are_tried():
    agent = UCBAgent(n_arms=2, c=1.0)
    agent.update(0, 1)
    agent.update(1, 0)

    assert agent.select_action() == 0


def test_ucb_reset_clears_state():
    agent = UCBAgent(n_arms=2, c=1.0)
    agent.update(0, 1)
    agent.update(1, 0)

    agent.reset()

    np.testing.assert_array_equal(agent.counts, np.array([0, 0]))
    np.testing.assert_allclose(agent.values, np.array([0.0, 0.0]))
    assert agent.t == 0


def test_thompson_sampling_alpha_update_after_success():
    agent = ThompsonSamplingAgent(n_arms=2, seed=31)

    agent.update(1, 1)

    np.testing.assert_allclose(agent.alpha, np.array([1.0, 2.0]))
    np.testing.assert_allclose(agent.beta, np.array([1.0, 1.0]))


def test_thompson_sampling_beta_update_after_failure():
    agent = ThompsonSamplingAgent(n_arms=2, seed=31)

    agent.update(0, 0)

    np.testing.assert_allclose(agent.alpha, np.array([1.0, 1.0]))
    np.testing.assert_allclose(agent.beta, np.array([2.0, 1.0]))


def test_thompson_sampling_reproducible_with_fixed_seed():
    first = ThompsonSamplingAgent(n_arms=4, seed=41)
    second = ThompsonSamplingAgent(n_arms=4, seed=41)

    assert [first.select_action() for _ in range(8)] == [
        second.select_action() for _ in range(8)
    ]


def test_thompson_sampling_rejects_non_bernoulli_reward():
    agent = ThompsonSamplingAgent(n_arms=2)

    with pytest.raises(ValueError):
        agent.update(0, 2)


def test_average_and_cumulative_reward():
    rewards = np.array([[1, 0, 1], [0, 1, 1]])

    np.testing.assert_allclose(average_reward(rewards), np.array([0.5, 0.5, 1.0]))
    np.testing.assert_array_equal(
        cumulative_reward(rewards),
        np.array([[1, 1, 2], [0, 1, 2]]),
    )


def test_optimal_action_rate():
    actions = np.array([[0, 2, 2], [1, 2, 0]])

    np.testing.assert_allclose(optimal_action_rate(actions, 2), np.array([0.0, 1.0, 0.5]))


def test_regret_uses_expected_arm_rewards():
    probabilities = np.array([0.2, 0.8])
    actions = np.array([[0, 1, 0]])

    np.testing.assert_allclose(
        instantaneous_regret(actions, probabilities),
        np.array([[0.6, 0.0, 0.6]]),
    )
    np.testing.assert_allclose(
        cumulative_regret(actions, probabilities),
        np.array([[0.6, 0.6, 1.2]]),
    )


def test_run_comparison_returns_expected_shapes():
    results = run_comparison(
        {"epsilon": standard_agents()["epsilon-greedy (0.1)"]},
        probabilities=[0.2, 0.8],
        steps=12,
        runs=4,
        seed=101,
    )

    assert set(results["epsilon"]) == {
        "average_reward",
        "cumulative_reward",
        "optimal_action_rate",
        "cumulative_regret",
        "final_average_reward",
        "final_cumulative_reward",
        "final_optimal_action_rate",
        "final_cumulative_regret",
    }
    assert results["epsilon"]["average_reward"].shape == (12,)
    assert results["epsilon"]["cumulative_regret"].shape == (12,)


def test_run_comparison_reproducible_with_same_seed():
    first = run_comparison(
        {"epsilon": epsilon_agents()["epsilon-greedy (0.1)"]},
        probabilities=[0.3, 0.7],
        steps=15,
        runs=5,
        seed=202,
    )
    second = run_comparison(
        {"epsilon": epsilon_agents()["epsilon-greedy (0.1)"]},
        probabilities=[0.3, 0.7],
        steps=15,
        runs=5,
        seed=202,
    )

    np.testing.assert_allclose(
        first["epsilon"]["average_reward"],
        second["epsilon"]["average_reward"],
    )
    np.testing.assert_allclose(
        first["epsilon"]["cumulative_regret"],
        second["epsilon"]["cumulative_regret"],
    )
