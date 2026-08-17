# Interview Guide

This is a short checklist of ideas worth understanding before presenting the repository.

## Bandits

- Exploration means trying actions that may be worse now to improve future estimates.
- Epsilon-greedy explores at a fixed rate, even after it has mostly learned the best arm.
- Optimistic initial values encourage early exploration because unseen arms look valuable.
- UCB adds an uncertainty bonus that shrinks as an arm is sampled more often.
- Thompson Sampling samples from a posterior belief over Bernoulli success probabilities.
- Regret is calculated from expected arm rewards, not sampled reward noise.

## Tabular Control

- Q-learning is off-policy because it updates toward the greedy next action.
- SARSA is on-policy because it updates toward the action actually selected by the behavior policy.
- The Bellman target combines immediate reward with discounted next-state value.
- Epsilon affects both exploration and the final behavior during training.

## DQN

- The neural network replaces the tabular Q-table with a function approximator.
- Experience replay breaks up the order of correlated transitions.
- The target network makes bootstrapped targets move more slowly.
- Huber loss is less sensitive to large temporal-difference errors than squared loss.

## Policy Gradients

- REINFORCE directly increases the probability of actions that led to higher returns.
- It learns from full episode returns rather than one-step bootstrapped targets.
- Normalizing returns reduces update scale issues but does not remove all variance.
- A learned value baseline is a common next step because plain REINFORCE can be noisy.

## PPO

- PPO uses an actor-critic model: the actor chooses actions, the critic estimates state value.
- Advantages are estimated as return minus value.
- The probability ratio compares the new policy to the policy that collected the rollout.
- Clipping discourages a single update from moving the policy too far.
- Entropy keeps the policy from collapsing too early during training.

## Limitations To Say Out Loud

- The Gridworld is intentionally small and deterministic.
- These results are implementation checks, not benchmark claims.
- PPO here omits minibatches, GAE, parallel environments, and continuous actions.
- The next serious extension would use a larger environment and more careful experiment comparisons.
