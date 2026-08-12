from __future__ import annotations

from dataclasses import dataclass


Action = int
State = int

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    done: bool


class GridWorld:
    """Small deterministic gridworld for tabular control."""

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (3, 3),
        walls: set[tuple[int, int]] | None = None,
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
    ):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")

        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.walls = set() if walls is None else set(walls)
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.n_states = rows * cols
        self.n_actions = 4
        self.state = self.to_state(start)

        for cell in {self.start, self.goal, *self.walls}:
            self._validate_cell(cell)
        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("start and goal cannot be walls")

    def reset(self) -> State:
        self.state = self.to_state(self.start)
        return self.state

    def step(self, action: Action) -> StepResult:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"invalid action {action}")

        row, col = self.to_cell(self.state)
        next_cell = self._move((row, col), action)

        if not self._inside(next_cell) or next_cell in self.walls:
            next_cell = (row, col)

        self.state = self.to_state(next_cell)
        done = next_cell == self.goal
        reward = self.goal_reward if done else self.step_reward
        return StepResult(self.state, reward, done)

    def to_state(self, cell: tuple[int, int]) -> State:
        self._validate_cell(cell)
        row, col = cell
        return row * self.cols + col

    def to_cell(self, state: State) -> tuple[int, int]:
        if state < 0 or state >= self.n_states:
            raise ValueError(f"invalid state {state}")
        return divmod(state, self.cols)

    def _move(self, cell: tuple[int, int], action: Action) -> tuple[int, int]:
        row, col = cell
        if action == UP:
            return row - 1, col
        if action == RIGHT:
            return row, col + 1
        if action == DOWN:
            return row + 1, col
        return row, col - 1

    def _inside(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _validate_cell(self, cell: tuple[int, int]) -> None:
        if not self._inside(cell):
            raise ValueError(f"cell outside grid: {cell}")


def default_gridworld() -> GridWorld:
    return GridWorld(walls={(1, 1), (2, 1)})

