from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def require(text: str, expected: str) -> None:
    if expected not in text:
        raise SystemExit(f"README is missing expected result: {expected}")


def main() -> None:
    readme = (ROOT / "README.md").read_text()
    bandits = load_json("results/bandits/summary.json")
    tabular = load_json("results/tabular/summary.json")
    dqn = load_json("results/dqn/summary.json")
    reinforce = load_json("results/policy_gradients/summary.json")
    ppo = load_json("results/ppo/summary.json")

    bandit_labels = {
        "random": "Random",
        "epsilon-greedy (0.1)": "Epsilon-greedy (0.1)",
        "optimistic values": "Optimistic values",
        "ucb1": "UCB1",
        "thompson sampling": "Thompson Sampling",
    }

    for name, metrics in bandits["standard_comparison"].items():
        require(readme, f"| {bandit_labels[name]}")
        require(readme, f"{metrics['final_average_reward']:.4f}")
        require(readme, f"{metrics['final_cumulative_regret']:.4f}")

    for name, metrics in tabular["algorithms"].items():
        require(readme, f"{metrics['eval_success_rate']:.4f}")
        require(readme, f"{metrics['eval_average_steps']:.4f}")

    require(readme, f"| DQN | {dqn['episodes']} | {dqn['final_success_rate']:.4f} | {dqn['eval_success_rate']:.4f} | {dqn['eval_average_steps']:.4f} |")
    require(readme, f"| REINFORCE | {reinforce['episodes']} | {reinforce['final_success_rate']:.4f} | {reinforce['eval_success_rate']:.4f} | {reinforce['eval_average_steps']:.4f} |")
    require(readme, f"| PPO | {ppo['updates']} | {ppo['rollout_steps']} | {ppo['final_success_rate']:.4f} | {ppo['eval_success_rate']:.4f} | {ppo['eval_average_steps']:.4f} |")

    print("README results match saved summaries")


if __name__ == "__main__":
    main()
