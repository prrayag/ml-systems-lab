from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

sys.path.insert(0, str(ROOT / "src"))

from ml_systems_lab.tabular.experiment import run_tabular_comparison  # noqa: E402
from ml_systems_lab.tabular.gridworld import default_gridworld  # noqa: E402


RESULTS_DIR = ROOT / "results" / "tabular"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tabular Gridworld experiments.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def plot_metric(results: dict, metric: str, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, values in results.items():
        y = values.get(f"{metric}_smoothed", values[metric])
        line = plt.plot(y, label=name)[0]
        lower = values.get(f"{metric}_lower")
        upper = values.get(f"{metric}_upper")
        if lower is not None and upper is not None:
            episodes = range(len(values[metric]))
            plt.fill_between(episodes, lower, upper, color=line.get_color(), alpha=0.12)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def summarize(results: dict) -> dict[str, dict[str, float]]:
    fields = [
        "final_return",
        "final_success_rate",
        "eval_success_rate",
        "eval_average_steps",
    ]
    return {
        name: {field: round(float(values[field]), 4) for field in fields}
        for name, values in results.items()
    }


def write_summary_csv(summary: dict, output_path: Path) -> None:
    fields = [
        "algorithm",
        "final_return",
        "final_success_rate",
        "eval_success_rate",
        "eval_average_steps",
    ]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for algorithm, values in summary["algorithms"].items():
            writer.writerow({"algorithm": algorithm, **values})


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "gridworld.txt").write_text(default_gridworld().render() + "\n")

    results = run_tabular_comparison(
        episodes=args.episodes,
        runs=args.runs,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    plot_metric(
        results,
        "return",
        "Gridworld Return",
        "Episode return",
        RESULTS_DIR / "return.png",
    )
    plot_metric(
        results,
        "success_rate",
        "Gridworld Success Rate",
        "Fraction of successful episodes",
        RESULTS_DIR / "success_rate.png",
    )

    summary = {
        "episodes": args.episodes,
        "runs": args.runs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "algorithms": summarize(results),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_summary_csv(summary, RESULTS_DIR / "summary.csv")

    print(f"saved results to {RESULTS_DIR.relative_to(ROOT)}")
    for name, values in summary["algorithms"].items():
        success = values["eval_success_rate"]
        steps = values["eval_average_steps"]
        print(f"{name}: eval_success={success:.3f}, eval_steps={steps:.1f}")


if __name__ == "__main__":
    main()
