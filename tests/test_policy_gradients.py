import numpy as np
import pytest
import torch

from ml_systems_lab.policy_gradients.agent import PolicyGradientAgent, PolicyNetwork


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

