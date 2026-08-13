from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "dqn"

EXPECTED_FILES = [
    "gridworld.txt",
    "return.png",
    "success_rate.png",
    "loss.png",
    "summary.json",
]


def main() -> None:
    missing = [name for name in EXPECTED_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        raise SystemExit(f"missing result files: {', '.join(missing)}")

    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    if summary["eval_success_rate"] < 0.9:
        raise SystemExit("greedy DQN evaluation did not solve the gridworld")

    print("dqn results look complete")


if __name__ == "__main__":
    try:
        main()
    except KeyError as error:
        print(f"missing summary field: {error}", file=sys.stderr)
        raise SystemExit(1)

