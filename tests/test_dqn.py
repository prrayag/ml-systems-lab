import numpy as np
import pytest
import torch

from ml_systems_lab.dqn.agent import DQNAgent, QNetwork
from ml_systems_lab.dqn.replay import ReplayBuffer


def test_replay_buffer_adds_transitions():
    buffer = ReplayBuffer(capacity=3, state_dim=2, seed=1)

    buffer.add(np.array([1.0, 0.0]), 1, 0.5, np.array([0.0, 1.0]), False)

    assert len(buffer) == 1
    np.testing.assert_allclose(buffer.states[0], np.array([1.0, 0.0]))
    assert buffer.actions[0] == 1
    assert buffer.rewards[0] == pytest.approx(0.5)


def test_replay_buffer_overwrites_old_transitions():
    buffer = ReplayBuffer(capacity=2, state_dim=1, seed=1)

    buffer.add(np.array([1.0]), 0, 0.0, np.array([2.0]), False)
    buffer.add(np.array([2.0]), 1, 0.0, np.array([3.0]), False)
    buffer.add(np.array([3.0]), 0, 1.0, np.array([4.0]), True)

    assert len(buffer) == 2
    np.testing.assert_allclose(buffer.states, np.array([[3.0], [2.0]], dtype=np.float32))


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=5, state_dim=3, seed=3)
    for index in range(5):
        state = np.full(3, index, dtype=np.float32)
        buffer.add(state, index % 2, float(index), state + 1, False)

    batch = buffer.sample(batch_size=4)

    assert batch["states"].shape == (4, 3)
    assert batch["actions"].shape == (4,)
    assert batch["next_states"].shape == (4, 3)


def test_replay_buffer_rejects_invalid_sample_size():
    buffer = ReplayBuffer(capacity=2, state_dim=1)

    with pytest.raises(ValueError):
        buffer.sample(1)


def test_q_network_output_shape():
    network = QNetwork(state_dim=5, n_actions=3, hidden_dim=8)
    states = torch.zeros((4, 5))

    assert network(states).shape == (4, 3)


def test_dqn_agent_one_hot_encodes_states():
    agent = DQNAgent(state_dim=4, n_actions=2, seed=1)

    np.testing.assert_allclose(agent.encode_state(2), np.array([0.0, 0.0, 1.0, 0.0]))


def test_dqn_agent_random_actions_reproducible_with_seed():
    first = DQNAgent(state_dim=4, n_actions=3, epsilon=1.0, seed=5)
    second = DQNAgent(state_dim=4, n_actions=3, epsilon=1.0, seed=5)

    assert [first.select_action(0) for _ in range(8)] == [
        second.select_action(0) for _ in range(8)
    ]


def test_dqn_agent_syncs_target_network():
    agent = DQNAgent(state_dim=4, n_actions=2, seed=7)

    with torch.no_grad():
        for parameter in agent.q_network.parameters():
            parameter.add_(1.0)
    agent.sync_target()

    for q_param, target_param in zip(
        agent.q_network.parameters(),
        agent.target_network.parameters(),
        strict=True,
    ):
        assert torch.allclose(q_param, target_param)


def test_dqn_agent_rejects_invalid_state():
    agent = DQNAgent(state_dim=4, n_actions=2)

    with pytest.raises(ValueError):
        agent.encode_state(4)


def test_dqn_train_on_batch_updates_network():
    agent = DQNAgent(state_dim=3, n_actions=2, hidden_dim=8, seed=11)
    batch = {
        "states": np.eye(3, dtype=np.float32),
        "actions": np.array([0, 1, 0]),
        "rewards": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "next_states": np.eye(3, dtype=np.float32),
        "dones": np.array([False, True, False]),
    }
    before = [parameter.detach().clone() for parameter in agent.q_network.parameters()]

    loss = agent.train_on_batch(batch)

    assert loss >= 0.0
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, agent.q_network.parameters(), strict=True)
    )
