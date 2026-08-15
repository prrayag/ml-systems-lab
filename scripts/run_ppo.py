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

from ml_systems_lab.policy_gradients.train import run_ppo_training  # noqa: E402
from ml_systems_lab.tabular.experiment import moving_average  # noqa: E402
from ml_systems_lab.tabular.gridworld import default_gridworld  # noqa: E402


RESULTS_DIR = ROOT / "results" / "ppo"
ACTION_NAMES = ["up", "right", "down", "left"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small PPO Gridworld experiment.")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def plot_curve(values: np.ndarray, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(moving_average(values, window=5))
    plt.title(title)
    plt.xlabel("Update")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def policy_rows(agent) -> list[dict[str, object]]:
    rows = []
    for state in range(agent.state_dim):
        state_tensor = agent.encode_states(np.array([state]))
        distribution, _ = agent.distribution_and_value(state_tensor)
        probabilities = distribution.probs.detach().numpy().squeeze(0)
        rows.append(
            {
                "state": state,
                "greedy_action": ACTION_NAMES[int(np.argmax(probabilities))],
                "probabilities": {
                    action: round(float(probability), 4)
                    for action, probability in zip(ACTION_NAMES, probabilities, strict=True)
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = default_gridworld()
    (RESULTS_DIR / "gridworld.txt").write_text(env.render() + "\n")

    history, evaluation, agent = run_ppo_training(
        updates=args.updates,
        rollout_steps=args.rollout_steps,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    plot_curve(history["returns"], "PPO Gridworld Return", "Mean episode return", RESULTS_DIR / "return.png")
    plot_curve(history["successes"], "PPO Gridworld Success Rate", "Mean success", RESULTS_DIR / "success_rate.png")
    plot_curve(history["policy_losses"], "PPO Policy Loss", "Policy loss", RESULTS_DIR / "policy_loss.png")
    plot_curve(history["value_losses"], "PPO Value Loss", "Value loss", RESULTS_DIR / "value_loss.png")
    plot_curve(history["entropies"], "PPO Policy Entropy", "Entropy", RESULTS_DIR / "entropy.png")

    summary = {
        "updates": args.updates,
        "rollout_steps": args.rollout_steps,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "final_success_rate": round(float(np.mean(history["successes"][-10:])), 4),
        "final_return": round(float(np.mean(history["returns"][-10:])), 4),
        "eval_success_rate": round(float(evaluation["success_rate"]), 4),
        "eval_average_steps": round(float(evaluation["average_steps"]), 4),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "policy.json").write_text(json.dumps(policy_rows(agent), indent=2) + "\n")

    print(f"saved results to {RESULTS_DIR.relative_to(ROOT)}")
    print(
        "ppo: "
        f"final_success={summary['final_success_rate']:.3f}, "
        f"eval_success={summary['eval_success_rate']:.3f}, "
        f"eval_steps={summary['eval_average_steps']:.1f}"
    )


if __name__ == "__main__":
    main()
