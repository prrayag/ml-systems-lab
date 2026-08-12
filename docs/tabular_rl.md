# Tabular RL Notes

This module uses a small deterministic Gridworld to compare Q-learning and SARSA.

The state is the agent's grid cell. The actions are up, right, down, and left. Walls and grid boundaries leave the agent in the same state. Reaching the goal ends the episode.

## Temporal Difference Updates

Both algorithms update a table of action values:

```text
Q[state, action]
```

The update has the same shape in both methods:

```text
Q(s, a) <- Q(s, a) + learning_rate * (target - Q(s, a))
```

The difference is the target.

## Q-learning

Q-learning uses the best next action according to the current value table:

```text
reward + discount * max_a Q(next_state, a)
```

This is off-policy because the update assumes greedy behavior at the next state, even when the agent is still exploring with epsilon-greedy action selection.

## SARSA

SARSA uses the next action actually selected by the current policy:

```text
reward + discount * Q(next_state, next_action)
```

This is on-policy because the update includes the effect of exploration.

## Reading The Results

The current Gridworld is intentionally small. Both algorithms learn a successful greedy policy quickly, so this experiment is mainly useful for understanding the update rules and training loop.

For future experiments, a harder environment will be more useful for comparing stability, exploration, and sample efficiency.

