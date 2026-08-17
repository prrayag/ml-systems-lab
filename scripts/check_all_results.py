from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    "check_bandit_results.py",
    "check_tabular_results.py",
    "check_dqn_results.py",
    "check_policy_gradient_results.py",
    "check_ppo_results.py",
    "check_markdown_links.py",
    "check_readme_results.py",
]


def main() -> None:
    for script in CHECKS:
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)

    print("all saved results look complete")


if __name__ == "__main__":
    main()
