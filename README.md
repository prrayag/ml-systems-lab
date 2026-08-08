# ML Systems Lab

Small implementations and experiments for understanding machine learning and reinforcement learning systems from first principles.

The current module compares exploration strategies for stochastic multi-armed bandits.

## Multi-Armed Bandits

A Bernoulli bandit has several arms, each with an unknown probability of returning reward `1`. The agent chooses one arm per step and tries to learn which arm has the highest expected reward.

The environment used here has five arms:

```text
[0.10, 0.35, 0.60, 0.45, 0.80]
```

The agent does not observe these probabilities directly. They are used by the environment to sample rewards and by the experiment code to calculate expected regret.

## Algorithms

| Algorithm | Main idea |
| --- | --- |
| Epsilon-greedy | Mostly choose the best estimated arm, sometimes explore randomly. |
| Optimistic initial values | Start value estimates high so early greedy choices try different arms. |
| UCB1 | Add an uncertainty bonus to arms with fewer samples. |
| Thompson Sampling | Sample each arm's success probability from a Beta posterior. |

## Experiments

Default run:

- 5 arms
- 2000 steps
- 200 independent runs
- fixed master seed

The main comparison uses epsilon-greedy with `epsilon = 0.1`, optimistic initial values, UCB1, and Thompson Sampling.

A smaller comparison also runs epsilon-greedy with `epsilon = 0.01`, `0.1`, and `0.2`.

## Results

Results below come from:

```bash
.venv/bin/python scripts/run_bandits.py
```

| Algorithm | Final avg reward | Final optimal action rate | Final cumulative regret |
| --- | ---: | ---: | ---: |
| Epsilon-greedy (0.1) | 0.7705 | 0.9194 | 101.5830 |
| Optimistic values | 0.7058 | 0.7150 | 171.8568 |
| UCB1 | 0.7640 | 0.8670 | 163.4660 |
| Thompson Sampling | 0.8023 | 0.9947 | 21.7337 |

![Average reward](results/bandits/average_reward.png)

![Optimal action rate](results/bandits/optimal_action_rate.png)

![Cumulative regret](results/bandits/cumulative_regret.png)

![Epsilon comparison](results/bandits/epsilon_comparison.png)

## Running

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/run_bandits.py
```

Useful options:

```bash
.venv/bin/python scripts/run_bandits.py --steps 2000 --runs 200 --seed 7
```

## Tests

```bash
.venv/bin/python -m pytest
```

## Notes

- Thompson Sampling performs best in this setup because it quickly concentrates probability mass on the best arm while still exploring uncertain arms.
- Epsilon-greedy with `epsilon = 0.01` has lower final regret slope but often learns too slowly early on.
- Regret is calculated from expected arm rewards, not sampled rewards, so it measures action quality rather than reward noise.

## Next

Value-based reinforcement learning will be added later, starting with small tabular environments.
