# PPO

This module trains a small PPO agent on the same Gridworld used by the tabular, DQN, and REINFORCE examples.

PPO still learns a policy directly, but it also learns a value estimate for each state. The value estimate is used as a baseline so the policy update can focus on whether an action was better or worse than expected.

## Rollouts

The agent first collects a batch of transitions using the current policy:

```text
state, action, old log probability, reward, done, value estimate
```

Discounted returns are computed backward. Episode boundaries reset the running return so one episode does not leak into the next.

Advantages are estimated as:

```text
return - value
```

The advantages are normalized before the update.

## Clipped Objective

PPO compares the new policy to the policy that collected the rollout:

```text
ratio = exp(new_log_prob - old_log_prob)
```

The policy loss uses the smaller of the unclipped and clipped objective:

```text
min(ratio * advantage, clip(ratio, 1 - eps, 1 + eps) * advantage)
```

The clipping step discourages a single update from moving the policy too far away from the behavior that generated the rollout.

## What Is Included

- actor-critic network
- fixed-size rollouts
- clipped policy loss
- value loss
- entropy bonus
- greedy evaluation after training

## Limitations

This is intentionally small. It does not use minibatches, generalized advantage estimation, parallel environments, or continuous actions. Those would be reasonable extensions later, but the first goal here is to make the core PPO update readable.
