# Bandit Notes

This module uses a stationary Bernoulli bandit. Each arm has a fixed reward probability, but the agent only observes sampled rewards.

## Core Terms

- Exploration: choosing actions to learn more about uncertain arms.
- Exploitation: choosing the arm that currently looks best.
- Expected regret: the gap between the best arm's expected reward and the selected arm's expected reward.
- Independent run: a fresh simulation with its own random seeds.

Regret is calculated from the true arm probabilities:

```text
best_expected_reward - selected_arm_expected_reward
```

It is not calculated from sampled rewards. Sampled rewards include noise, while regret should measure whether the chosen action was good.

## Algorithms

Epsilon-greedy keeps a sample-average estimate for each arm. With probability `epsilon`, it explores randomly. Otherwise it chooses the arm with the highest current estimate.

Optimistic initial values use greedy action selection, but start every value estimate high. Early failures reduce the selected arm's estimate, which encourages trying other arms.

UCB1 chooses the arm with the largest value:

```text
Q(a) + c * sqrt(log(t) / N(a))
```

The second term is larger for arms with fewer samples. In this implementation, every arm is selected once before the formula is used.

Thompson Sampling keeps a Beta distribution for each Bernoulli arm. A reward of `1` increments `alpha`; a reward of `0` increments `beta`. Each step samples a possible success rate for every arm and chooses the largest sample.

## Reading The Results

The random baseline is a sanity check. If a learning strategy cannot beat it on this stationary bandit, something is probably wrong.

The confidence bands show variation across independent runs. Wide bands usually mean the method is sensitive to early random outcomes or explores inconsistently.

The epsilon comparison shows a tradeoff:

- very low epsilon can learn slowly if early samples are unlucky
- high epsilon keeps exploring even after the best arm is known
- moderate epsilon is often a practical compromise

