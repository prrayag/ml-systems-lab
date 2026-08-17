# Reproducibility

The experiments use fixed seeds and saved summaries so results can be checked without rerunning every training job.

## Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Verification

Run the test suite:

```bash
.venv/bin/python -m pytest
```

Check the saved experiment artifacts:

```bash
.venv/bin/python scripts/check_all_results.py
```

Regenerate the aggregate summary:

```bash
.venv/bin/python scripts/summarize_results.py
```

## Seeds

| Experiment | Seed |
| --- | ---: |
| Bandits | 7 |
| Tabular RL | 7 |
| DQN | 23 |
| REINFORCE | 19 |
| PPO | 23 |

## Notes

- The bandit experiment averages many independent runs.
- Gridworld training is deterministic enough to make the saved results stable with the listed seeds.
- The saved plots and JSON summaries are part of the repository because the README reports actual generated values.
- If an experiment is rerun with different settings, update the matching summary file and README table together.
