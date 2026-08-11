from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "bandits"

EXPECTED_FILES = [
    "average_reward.png",
    "optimal_action_rate.png",
    "cumulative_regret.png",
    "cumulative_reward.png",
    "epsilon_comparison.png",
    "summary.json",
    "summary.csv",
]

EXPECTED_STANDARD_ALGORITHMS = {
    "random",
    "epsilon-greedy (0.1)",
    "optimistic values",
    "ucb1",
    "thompson sampling",
}

EXPECTED_EPSILON_ALGORITHMS = {
    "epsilon-greedy (0.01)",
    "epsilon-greedy (0.1)",
    "epsilon-greedy (0.2)",
}


def main() -> None:
    missing = [name for name in EXPECTED_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        raise SystemExit(f"missing result files: {', '.join(missing)}")

    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    standard = set(summary["standard_comparison"])
    epsilon = set(summary["epsilon_comparison"])

    if standard != EXPECTED_STANDARD_ALGORITHMS:
        raise SystemExit(f"unexpected standard algorithms: {sorted(standard)}")
    if epsilon != EXPECTED_EPSILON_ALGORITHMS:
        raise SystemExit(f"unexpected epsilon algorithms: {sorted(epsilon)}")

    print("bandit results look complete")


if __name__ == "__main__":
    try:
        main()
    except KeyError as error:
        print(f"missing summary field: {error}", file=sys.stderr)
        raise SystemExit(1)

