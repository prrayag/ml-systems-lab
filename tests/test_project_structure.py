from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_modules_exist():
    modules = [
        "src/ml_systems_lab/bandits",
        "src/ml_systems_lab/tabular",
        "src/ml_systems_lab/dqn",
        "src/ml_systems_lab/policy_gradients",
    ]

    for module in modules:
        assert (ROOT / module / "__init__.py").exists()


def test_expected_scripts_exist():
    scripts = [
        "run_bandits.py",
        "run_tabular_rl.py",
        "run_dqn.py",
        "run_policy_gradients.py",
        "run_ppo.py",
        "run_all_experiments.py",
        "check_all_results.py",
        "summarize_results.py",
    ]

    for script in scripts:
        assert (ROOT / "scripts" / script).exists()


def test_expected_docs_exist():
    docs = [
        "bandits.md",
        "tabular_rl.md",
        "dqn.md",
        "policy_gradients.md",
        "ppo.md",
        "interview_guide.md",
        "reproducibility.md",
    ]

    for doc in docs:
        assert (ROOT / "docs" / doc).exists()
