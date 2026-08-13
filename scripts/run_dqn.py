from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT / "src"))

from ml_systems_lab.dqn.train import run_dqn_training  # noqa: E402
from ml_systems_lab.tabular.experiment import moving_average  # noqa: E402
from ml_systems_lab.tabular.gridworld import default_gridworld  # noqa: E402


RESULTS_DIR = ROOT / "results" / "dqn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small DQN Gridworld experiment.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def plot_curve(values: np.ndarray, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(moving_average(values, window=20))
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gridworld.txt").write_text(default_gridworld().render() + "\n")

    history, evaluation = run_dqn_training(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    plot_curve(history["returns"], "DQN Gridworld Return", "Episode return", RESULTS_DIR / "return.png")
    plot_curve(
        history["successes"].astype(float),
        "DQN Gridworld Success Rate",
        "Smoothed success",
        RESULTS_DIR / "success_rate.png",
    )
    if len(history["losses"]) > 0:
        plot_curve(history["losses"], "DQN Training Loss", "Huber loss", RESULTS_DIR / "loss.png")

    summary = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "final_success_rate": round(float(np.mean(history["successes"][-50:])), 4),
        "final_return": round(float(np.mean(history["returns"][-50:])), 4),
        "eval_success_rate": round(float(evaluation["success_rate"]), 4),
        "eval_average_steps": round(float(evaluation["average_steps"]), 4),
        "updates": int(len(history["losses"])),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"saved results to {RESULTS_DIR.relative_to(ROOT)}")
    print(
        "dqn: "
        f"final_success={summary['final_success_rate']:.3f}, "
        f"eval_success={summary['eval_success_rate']:.3f}, "
        f"eval_steps={summary['eval_average_steps']:.1f}"
    )


if __name__ == "__main__":
    main()

