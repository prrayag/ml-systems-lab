import pytest

from ml_systems_lab.tabular.agents import QLearningAgent, SarsaAgent
from ml_systems_lab.tabular.gridworld import DOWN, LEFT, RIGHT, UP, GridWorld
from ml_systems_lab.tabular.train import (
    evaluate_greedy_policy,
    train_q_learning,
    train_sarsa,
)


def test_gridworld_reset_returns_start_state():
    env = GridWorld(start=(0, 1), goal=(1, 1), rows=2, cols=2)

    assert env.reset() == env.to_state((0, 1))


def test_gridworld_moves_between_cells():
    env = GridWorld(rows=2, cols=2, goal=(1, 1))

    result = env.step(RIGHT)

    assert result.state == env.to_state((0, 1))
    assert result.reward == pytest.approx(-0.01)
    assert not result.done


def test_gridworld_blocks_walls_and_edges():
    env = GridWorld(rows=3, cols=3, walls={(0, 1)}, goal=(2, 2))

    assert env.step(RIGHT).state == env.to_state((0, 0))
    assert env.step(LEFT).state == env.to_state((0, 0))


def test_gridworld_goal_ends_episode():
    env = GridWorld(rows=2, cols=2, goal=(1, 1))

    env.step(DOWN)
    result = env.step(RIGHT)

    assert result.state == env.to_state((1, 1))
    assert result.reward == pytest.approx(1.0)
    assert result.done


def test_gridworld_rejects_invalid_action():
    env = GridWorld()

    with pytest.raises(ValueError):
        env.step(4)


def test_gridworld_state_cell_roundtrip():
    env = GridWorld(rows=3, cols=4, goal=(2, 3))

    for row in range(env.rows):
        for col in range(env.cols):
            state = env.to_state((row, col))
            assert env.to_cell(state) == (row, col)


def test_gridworld_rejects_invalid_layout():
    with pytest.raises(ValueError):
        GridWorld(rows=0)
    with pytest.raises(ValueError):
        GridWorld(start=(0, 0), goal=(0, 1), walls={(0, 0)})


def test_q_learning_update_uses_max_next_state_value():
    agent = QLearningAgent(
        n_states=3,
        n_actions=2,
        learning_rate=0.5,
        discount=0.9,
        epsilon=0.0,
    )
    agent.q_values[1] = [0.2, 0.8]

    agent.update(state=0, action=1, reward=1.0, next_state=1, done=False)

    assert agent.q_values[0, 1] == pytest.approx(0.86)


def test_q_learning_terminal_update_uses_reward_only():
    agent = QLearningAgent(n_states=3, n_actions=2, learning_rate=0.5, discount=0.9)
    agent.q_values[1] = [10.0, 10.0]

    agent.update(state=0, action=1, reward=1.0, next_state=1, done=True)

    assert agent.q_values[0, 1] == pytest.approx(0.5)


def test_q_learning_greedy_action_uses_largest_q_value():
    agent = QLearningAgent(n_states=2, n_actions=3, epsilon=0.0, seed=5)
    agent.q_values[0] = [0.1, 0.7, 0.2]

    assert agent.select_action(0) == 1


def test_q_learning_random_actions_reproducible_with_seed():
    first = QLearningAgent(n_states=2, n_actions=4, epsilon=1.0, seed=19)
    second = QLearningAgent(n_states=2, n_actions=4, epsilon=1.0, seed=19)

    assert [first.select_action(0) for _ in range(10)] == [
        second.select_action(0) for _ in range(10)
    ]


def test_q_learning_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        QLearningAgent(n_states=0, n_actions=2)
    with pytest.raises(ValueError):
        QLearningAgent(n_states=2, n_actions=2, discount=1.1)


def test_sarsa_update_uses_selected_next_action():
    agent = SarsaAgent(
        n_states=3,
        n_actions=2,
        learning_rate=0.5,
        discount=0.9,
        epsilon=0.0,
    )
    agent.q_values[1] = [0.2, 0.8]

    agent.update(
        state=0,
        action=1,
        reward=1.0,
        next_state=1,
        next_action=0,
        done=False,
    )

    assert agent.q_values[0, 1] == pytest.approx(0.59)


def test_sarsa_terminal_update_uses_reward_only():
    agent = SarsaAgent(n_states=3, n_actions=2, learning_rate=0.5, discount=0.9)
    agent.q_values[1] = [10.0, 10.0]

    agent.update(
        state=0,
        action=1,
        reward=1.0,
        next_state=1,
        next_action=0,
        done=True,
    )

    assert agent.q_values[0, 1] == pytest.approx(0.5)


def test_sarsa_select_action_uses_epsilon_greedy_policy():
    agent = SarsaAgent(n_states=2, n_actions=3, epsilon=0.0)
    agent.q_values[0] = [0.2, 0.1, 0.8]

    assert agent.select_action(0) == 2


def test_train_q_learning_returns_episode_history():
    env = GridWorld(rows=2, cols=2, goal=(1, 1))
    agent = QLearningAgent(env.n_states, env.n_actions, epsilon=0.2, seed=3)

    history = train_q_learning(env, agent, episodes=5, max_steps=10)

    assert history["returns"].shape == (5,)
    assert history["lengths"].shape == (5,)
    assert history["successes"].shape == (5,)


def test_train_sarsa_returns_episode_history():
    env = GridWorld(rows=2, cols=2, goal=(1, 1))
    agent = SarsaAgent(env.n_states, env.n_actions, epsilon=0.2, seed=3)

    history = train_sarsa(env, agent, episodes=5, max_steps=10)

    assert history["returns"].shape == (5,)
    assert history["lengths"].shape == (5,)
    assert history["successes"].shape == (5,)


def test_q_learning_learns_short_gridworld_path():
    env = GridWorld(rows=2, cols=2, goal=(1, 1))
    agent = QLearningAgent(
        env.n_states,
        env.n_actions,
        learning_rate=0.4,
        discount=0.95,
        epsilon=0.2,
        seed=11,
    )

    train_q_learning(env, agent, episodes=120, max_steps=10)
    evaluation = evaluate_greedy_policy(env, agent, episodes=5, max_steps=10)

    assert evaluation["success_rate"] == pytest.approx(1.0)
    assert evaluation["average_steps"] <= 3
