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

from ml_systems_lab.policy_gradients.agent import PolicyGradientAgent  # noqa: E402
from ml_systems_lab.policy_gradients.train import (  # noqa: E402
    evaluate_policy,
    policy_action_probabilities,
    train_reinforce,
)
from ml_systems_lab.tabular.experiment import moving_average  # noqa: E402
from ml_systems_lab.tabular.gridworld import default_gridworld  # noqa: E402


RESULTS_DIR = ROOT / "results" / "policy_gradients"
ACTION_NAMES = ["up", "right", "down", "left"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small REINFORCE Gridworld experiment.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=19)
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


def policy_rows(probabilities: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for state, action_probs in enumerate(probabilities):
        rows.append(
            {
                "state": state,
                "greedy_action": ACTION_NAMES[int(np.argmax(action_probs))],
                "probabilities": {
                    action: round(float(probability), 4)
                    for action, probability in zip(ACTION_NAMES, action_probs, strict=True)
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = default_gridworld()
    (RESULTS_DIR / "gridworld.txt").write_text(env.render() + "\n")

    agent = PolicyGradientAgent(
        state_dim=env.n_states,
        n_actions=env.n_actions,
        hidden_dim=32,
        learning_rate=5e-3,
        seed=args.seed,
    )
    history = train_reinforce(env, agent, episodes=args.episodes, max_steps=args.max_steps, discount=0.95)
    evaluation = evaluate_policy(env, agent, episodes=20, max_steps=args.max_steps)
    probabilities = policy_action_probabilities(env, agent)

    plot_curve(history["returns"], "REINFORCE Gridworld Return", "Episode return", RESULTS_DIR / "return.png")
    plot_curve(
        history["successes"].astype(float),
        "REINFORCE Gridworld Success Rate",
        "Smoothed success",
        RESULTS_DIR / "success_rate.png",
    )
    plot_curve(history["losses"], "REINFORCE Loss", "Policy loss", RESULTS_DIR / "loss.png")

    summary = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "final_success_rate": round(float(np.mean(history["successes"][-50:])), 4),
        "final_return": round(float(np.mean(history["returns"][-50:])), 4),
        "eval_success_rate": round(float(evaluation["success_rate"]), 4),
        "eval_average_steps": round(float(evaluation["average_steps"]), 4),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "policy.json").write_text(json.dumps(policy_rows(probabilities), indent=2) + "\n")

    print(f"saved results to {RESULTS_DIR.relative_to(ROOT)}")
    print(
        "reinforce: "
        f"final_success={summary['final_success_rate']:.3f}, "
        f"eval_success={summary['eval_success_rate']:.3f}, "
        f"eval_steps={summary['eval_average_steps']:.1f}"
    )


if __name__ == "__main__":
    main()
