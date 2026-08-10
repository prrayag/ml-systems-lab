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

sys.path.insert(0, str(ROOT / "src"))

from ml_systems_lab.bandits.experiment import (  # noqa: E402
    DEFAULT_PROBABILITIES,
    epsilon_agents,
    run_comparison,
    standard_agents,
)


RESULTS_DIR = ROOT / "results" / "bandits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bernoulli bandit experiments.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def plot_metric(
    results: dict,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for name, values in results.items():
        line = plt.plot(values[metric], label=name)[0]
        lower = values.get(f"{metric}_lower")
        upper = values.get(f"{metric}_upper")
        if lower is not None and upper is not None:
            steps = range(len(values[metric]))
            plt.fill_between(steps, lower, upper, color=line.get_color(), alpha=0.12)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def summarize(results: dict) -> dict[str, dict[str, float]]:
    fields = [
        "final_average_reward",
        "final_average_reward_lower",
        "final_average_reward_upper",
        "final_cumulative_reward",
        "final_optimal_action_rate",
        "final_optimal_action_rate_lower",
        "final_optimal_action_rate_upper",
        "final_cumulative_regret",
        "final_cumulative_regret_lower",
        "final_cumulative_regret_upper",
    ]
    return {
        name: {field: round(float(values[field]), 4) for field in fields}
        for name, values in results.items()
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    standard = run_comparison(
        standard_agents(),
        probabilities=DEFAULT_PROBABILITIES,
        steps=args.steps,
        runs=args.runs,
        seed=args.seed,
    )
    epsilon = run_comparison(
        epsilon_agents(),
        probabilities=DEFAULT_PROBABILITIES,
        steps=args.steps,
        runs=args.runs,
        seed=args.seed + 1,
    )

    plot_metric(
        standard,
        "average_reward",
        "Average Reward Over Time",
        "Average reward",
        RESULTS_DIR / "average_reward.png",
    )
    plot_metric(
        standard,
        "optimal_action_rate",
        "Optimal Action Rate",
        "Fraction of runs choosing best arm",
        RESULTS_DIR / "optimal_action_rate.png",
    )
    plot_metric(
        standard,
        "cumulative_regret",
        "Cumulative Regret",
        "Expected regret",
        RESULTS_DIR / "cumulative_regret.png",
    )
    plot_metric(
        epsilon,
        "cumulative_regret",
        "Epsilon-Greedy Cumulative Regret",
        "Expected regret",
        RESULTS_DIR / "epsilon_comparison.png",
    )

    summary = {
        "probabilities": [float(p) for p in DEFAULT_PROBABILITIES],
        "steps": args.steps,
        "runs": args.runs,
        "seed": args.seed,
        "standard_comparison": summarize(standard),
        "epsilon_comparison": summarize(epsilon),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"saved results to {RESULTS_DIR.relative_to(ROOT)}")
    for name, values in summary["standard_comparison"].items():
        reward = values["final_average_reward"]
        regret = values["final_cumulative_regret"]
        optimal = values["final_optimal_action_rate"]
        print(f"{name}: reward={reward:.3f}, regret={regret:.1f}, optimal={optimal:.3f}")


if __name__ == "__main__":
    main()
