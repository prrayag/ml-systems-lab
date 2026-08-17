import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_gridworld_summaries_have_eval_metrics():
    summaries = [
        load_json("results/dqn/summary.json"),
        load_json("results/policy_gradients/summary.json"),
        load_json("results/ppo/summary.json"),
    ]

    for summary in summaries:
        assert summary["eval_success_rate"] >= 0.8
        assert summary["eval_average_steps"] > 0


def test_saved_policy_tables_match_gridworld_size():
    for path in ["results/policy_gradients/policy.json", "results/ppo/policy.json"]:
        policy_rows = load_json(path)

        assert len(policy_rows) == 16
        assert {row["state"] for row in policy_rows} == set(range(16))


def test_result_summary_uses_saved_values():
    summary_text = (ROOT / "results" / "summary.md").read_text()
    ppo = load_json("results/ppo/summary.json")

    assert "thompson sampling: 20.8390" in summary_text
    assert f"PPO eval success | {ppo['eval_success_rate']:.4f}" in summary_text
