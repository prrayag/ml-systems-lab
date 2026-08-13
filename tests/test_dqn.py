import numpy as np
import pytest

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

