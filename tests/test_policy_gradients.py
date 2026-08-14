import numpy as np
import pytest
import torch

from ml_systems_lab.policy_gradients.agent import PolicyGradientAgent, PolicyNetwork
from ml_systems_lab.policy_gradients.reinforce import (
    discounted_returns,
    normalize,
    reinforce_update,
)


def test_policy_network_output_shape():
    network = PolicyNetwork(state_dim=5, n_actions=4, hidden_dim=8)
    states = torch.zeros((3, 5))

    assert network(states).shape == (3, 4)


def test_policy_agent_one_hot_encodes_states():
    agent = PolicyGradientAgent(state_dim=4, n_actions=2, seed=1)

    np.testing.assert_allclose(agent.encode_state(2), np.array([0.0, 0.0, 1.0, 0.0]))


def test_policy_agent_samples_valid_action():
    agent = PolicyGradientAgent(state_dim=4, n_actions=3, seed=3)

    action, log_prob = agent.select_action(0)

    assert 0 <= action < 3
    assert log_prob.ndim == 0


def test_policy_agent_rejects_invalid_state():
    agent = PolicyGradientAgent(state_dim=4, n_actions=2)

    with pytest.raises(ValueError):
        agent.encode_state(4)


def test_discounted_returns_work_backward():
    returns = discounted_returns([1.0, 1.0, 1.0], discount=0.9)

    np.testing.assert_allclose(returns, np.array([2.71, 1.9, 1.0]), atol=1e-6)


def test_normalize_zero_mean_unit_scale():
    values = normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    assert values.mean() == pytest.approx(0.0)
    assert values.std() == pytest.approx(1.0)


def test_reinforce_update_changes_policy_parameters():
    agent = PolicyGradientAgent(state_dim=3, n_actions=2, hidden_dim=8, seed=5)
    log_probs = []
    for state in [0, 1, 2]:
        _, log_prob = agent.select_action(state)
        log_probs.append(log_prob)
    before = [parameter.detach().clone() for parameter in agent.policy.parameters()]

    loss = reinforce_update(agent, log_probs, [0.0, 0.0, 1.0], discount=0.9)

    assert isinstance(loss, float)
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, agent.policy.parameters(), strict=True)
    )
