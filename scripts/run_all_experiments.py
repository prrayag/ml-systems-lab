from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    "run_bandits.py",
    "run_tabular_rl.py",
    "run_dqn.py",
    "run_policy_gradients.py",
    "run_ppo.py",
]


def run(script: str) -> None:
    print(f"\n== {script} ==")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)


def main() -> None:
    for script in EXPERIMENTS:
        run(script)

    run("summarize_results.py")
    run("check_all_results.py")


if __name__ == "__main__":
    main()
