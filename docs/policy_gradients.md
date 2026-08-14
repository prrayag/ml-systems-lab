# Policy Gradients

This module trains a small REINFORCE agent on the same deterministic Gridworld used by the tabular and DQN examples.

The policy is a neural network that maps a one-hot state vector to action logits. Instead of estimating the value of each action directly, it represents a probability distribution over actions.

## REINFORCE Update

For one episode, the agent stores:

- the log probability of each sampled action
- the reward observed after each action

After the episode ends, discounted returns are computed backward:

```text
G_t = r_t + gamma * G_{t+1}
```

The update minimizes:

```text
-log pi(a_t | s_t) * G_t
```

Actions followed by higher return become more likely. Actions followed by lower return become less likely.

Returns are normalized inside each episode. That does not change the ordering of the returns, but it usually makes the gradient step less noisy.

## What To Notice

- REINFORCE learns from complete episodes, not from one-step bootstrapped targets.
- The update can have high variance because there is no value baseline yet.
- The learned policy is stochastic during training, but evaluation uses the greedy action from the final policy.
- The saved `policy.json` file shows the learned action probabilities for each Gridworld state.

## Limitations

This is a small learning example, not a benchmark. The environment is deterministic and the shortest path is simple. A stronger next step would add a value baseline or move to PPO-style clipped policy updates.
