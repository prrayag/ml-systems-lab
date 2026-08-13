from __future__ import annotations

import numpy as np
import torch

from ml_systems_lab.dqn.agent import DQNAgent
from ml_systems_lab.dqn.replay import ReplayBuffer
from ml_systems_lab.tabular.gridworld import GridWorld, default_gridworld
from ml_systems_lab.tabular.train import evaluate_greedy_policy


def train_dqn(
    env: GridWorld,
    agent: DQNAgent,
    episodes: int = 300,
    max_steps: int = 50,
    batch_size: int = 32,
    replay_capacity: int = 2000,
    min_replay_size: int = 64,
    target_update_interval: int = 20,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    if batch_size <= 0 or min_replay_size <= 0:
        raise ValueError("batch_size and min_replay_size must be positive")

    buffer = ReplayBuffer(replay_capacity, env.n_states, seed=seed)
    returns = np.zeros(episodes, dtype=float)
    lengths = np.zeros(episodes, dtype=int)
    successes = np.zeros(episodes, dtype=bool)
    losses: list[float] = []

    for episode in range(episodes):
        state = env.reset()

        for step in range(max_steps):
            action = agent.select_action(state)
            result = env.step(action)
            buffer.add(
                agent.encode_state(state),
                action,
                result.reward,
                agent.encode_state(result.state),
                result.done,
            )

            if len(buffer) >= min_replay_size:
                loss = agent.train_on_batch(buffer.sample(batch_size))
                losses.append(loss)

            returns[episode] += result.reward
            state = result.state
            if result.done:
                successes[episode] = True
                lengths[episode] = step + 1
                break
        else:
            lengths[episode] = max_steps

        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        if (episode + 1) % target_update_interval == 0:
            agent.sync_target()

    return {
        "returns": returns,
        "lengths": lengths,
        "successes": successes,
        "losses": np.asarray(losses, dtype=float),
    }


def run_dqn_training(
    episodes: int = 300,
    max_steps: int = 50,
    seed: int = 7,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = default_gridworld()
    agent = DQNAgent(
        state_dim=env.n_states,
        n_actions=env.n_actions,
        hidden_dim=32,
        learning_rate=1e-3,
        discount=0.95,
        epsilon=0.2,
        seed=seed,
    )
    history = train_dqn(env, agent, episodes=episodes, max_steps=max_steps, seed=seed)
    evaluation = evaluate_greedy_policy(env, agent, episodes=20, max_steps=max_steps)
    return history, evaluation

