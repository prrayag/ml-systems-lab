import numpy as np
import pytest

from ml_systems_lab.bandits.agents import (
    EpsilonGreedyAgent,
    OptimisticInitialValuesAgent,
    UCBAgent,
)
from ml_systems_lab.bandits.environment import BernoulliBandit


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
