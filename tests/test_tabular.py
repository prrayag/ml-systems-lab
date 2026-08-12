import pytest

from ml_systems_lab.tabular.gridworld import DOWN, LEFT, RIGHT, UP, GridWorld


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

