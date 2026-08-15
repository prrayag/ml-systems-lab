from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "ppo"

EXPECTED_FILES = [
    "gridworld.txt",
    "return.png",
    "success_rate.png",
    "policy_loss.png",
    "value_loss.png",
    "entropy.png",
    "policy.json",
    "summary.json",
]


def main() -> None:
    missing = [name for name in EXPECTED_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        raise SystemExit(f"missing result files: {', '.join(missing)}")

    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    if summary["eval_success_rate"] < 0.8:
        raise SystemExit("greedy PPO evaluation did not solve the gridworld")

    policy_rows = json.loads((RESULTS_DIR / "policy.json").read_text())
    if len(policy_rows) != 16:
        raise SystemExit("expected one policy row per gridworld state")

    print("ppo results look complete")


if __name__ == "__main__":
    try:
        main()
    except KeyError as error:
        print(f"missing summary field: {error}", file=sys.stderr)
        raise SystemExit(1)
