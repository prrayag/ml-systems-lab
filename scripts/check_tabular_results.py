from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tabular"

EXPECTED_FILES = [
    "return.png",
    "success_rate.png",
    "summary.json",
    "summary.csv",
]

EXPECTED_ALGORITHMS = {"q-learning", "sarsa"}


def main() -> None:
    missing = [name for name in EXPECTED_FILES if not (RESULTS_DIR / name).exists()]
    if missing:
        raise SystemExit(f"missing result files: {', '.join(missing)}")

    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    algorithms = set(summary["algorithms"])
    if algorithms != EXPECTED_ALGORITHMS:
        raise SystemExit(f"unexpected algorithms: {sorted(algorithms)}")

    print("tabular results look complete")


if __name__ == "__main__":
    try:
        main()
    except KeyError as error:
        print(f"missing summary field: {error}", file=sys.stderr)
        raise SystemExit(1)

