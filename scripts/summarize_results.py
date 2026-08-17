from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "summary.md"


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def line(title: str, value: str) -> str:
    return f"| {title} | {value} |"


def main() -> None:
    bandits = load_json("results/bandits/summary.json")
    tabular = load_json("results/tabular/summary.json")
    dqn = load_json("results/dqn/summary.json")
    reinforce = load_json("results/policy_gradients/summary.json")
    ppo = load_json("results/ppo/summary.json")

    best_bandit = min(
        bandits["standard_comparison"].items(),
        key=lambda item: item[1]["final_cumulative_regret"],
    )

    rows = [
        line("Bandits best final regret", f"{best_bandit[0]}: {best_bandit[1]['final_cumulative_regret']:.4f}"),
        line("Q-learning eval success", f"{tabular['algorithms']['q-learning']['eval_success_rate']:.4f}"),
        line("SARSA eval success", f"{tabular['algorithms']['sarsa']['eval_success_rate']:.4f}"),
        line("DQN eval success", f"{dqn['eval_success_rate']:.4f} in {dqn['eval_average_steps']:.1f} steps"),
        line("REINFORCE eval success", f"{reinforce['eval_success_rate']:.4f} in {reinforce['eval_average_steps']:.1f} steps"),
        line("PPO eval success", f"{ppo['eval_success_rate']:.4f} in {ppo['eval_average_steps']:.1f} steps"),
    ]

    text = "\n".join(
        [
            "# Result Summary",
            "",
            "Generated from saved experiment summaries.",
            "",
            "| Result | Value |",
            "| --- | ---: |",
            *rows,
            "",
            "The Gridworld results use a small deterministic environment, so they should be read as implementation checks rather than benchmark claims.",
            "",
        ]
    )

    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
