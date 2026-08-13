# DQN Notes

This module uses a small Deep Q-Network on the same Gridworld used by the tabular RL experiments.

The state is encoded as a one-hot vector. The network predicts one Q-value per action:

```text
state -> Q(state, up), Q(state, right), Q(state, down), Q(state, left)
```

## Why This Is DQN

The implementation includes two ideas that distinguish DQN from the earlier tabular methods.

Experience replay stores transitions:

```text
state, action, reward, next_state, done
```

Training samples mini-batches from this buffer instead of updating only from the most recent transition. This reduces the dependence between consecutive updates.

The target network is a delayed copy of the online network. The online network is optimized, while the target network is used to compute the bootstrapped target:

```text
reward + discount * max_a target_network(next_state, a)
```

The target network is synced periodically to keep targets from moving every single gradient step.

## Scope

This is intentionally a small DQN experiment. It demonstrates the mechanics of function approximation, replay, target networks, and gradient updates. It is not meant to be a benchmark result.

The environment is simple enough that tabular methods solve it more directly. DQN is included here to make the transition from tabular values to neural value approximation explicit.

