"""玩家 Q-learning 训练循环。 / Training loop for player Q-learning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from config import get_train_defaults
from domain.enemies import create_enemy
from domain.game.env import GameEnv
from domain.models import LinearQModel
from domain.train.artifacts import (
    TRAINING_STATUS_COMPLETED,
    TRAINING_STATUS_INCOMPLETE,
    complete_training_artifact,
    initial_training_model_path,
    resolve_training_output,
    resume_training_context,
    training_run_log_path,
)
from domain.train.looping import has_remaining_episodes, resolve_episode_limit, scheduled_epsilon

PLAYER_Q_DEFAULTS = get_train_defaults("player_q")


@dataclass(frozen=True)
class TrainingSummary:
    """玩家 Q-learning 训练完成后的指标摘要。 / Metrics summary after player Q-learning training."""
    episodes: int
    output_path: Path
    average_score: float
    average_max_tile: float
    best_max_tile: int
    info_path: Path | None = None
    status: str = TRAINING_STATUS_COMPLETED
    target_episodes: int | None = None
    completed_episodes: int = 0
    reference_model_path: Path | None = None
    resume_run_path: Path | None = None
    run_log_path: Path | None = None


def player_reward(previous_state, next_state) -> float:
    """计算玩家训练使用的奖励。 / Compute the reward used for player training."""
    score_delta = next_state.score - previous_state.score
    reward = float(score_delta)
    if next_state.max_tile > previous_state.max_tile:
        reward += next_state.max_tile.bit_length() * 8.0
    if next_state.done:
        reward -= 100.0
    return reward


def train_q_player(
    episodes: int | None = None,
    enemy_type: str | None = None,
    seed: int | None = None,
    output: str | Path | None = None,
    learning_rate: float | None = None,
    gamma: float | None = None,
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
    max_steps: int | None = None,
    reference_model_path: str | Path | None = None,
    resume_run_path: str | Path | None = None,
    stop_event=None,
    progress_callback=None,
) -> TrainingSummary:
    """训练玩家 Q-learning 模型并保存产物。 / Train the player Q-learning model and save artifacts."""
    episode_limit = resolve_episode_limit(episodes, PLAYER_Q_DEFAULTS["episodes"])
    target_episodes = episode_limit.target_episodes
    enemy_type = enemy_type or PLAYER_Q_DEFAULTS["enemy"]
    seed = seed if seed is not None else PLAYER_Q_DEFAULTS.get("seed")
    output_path, run_directory = resolve_training_output(
        "player_q",
        output,
    )
    start_completed, resume_path, selected_reference, resume_info = resume_training_context(
        "player_q",
        resume_run_path,
        target_episodes,
        reference_model_path,
    )
    initial_model_path = initial_training_model_path(selected_reference, resume_info)
    remaining_episodes = episode_limit.remaining_after(start_completed)
    learning_rate = float(learning_rate if learning_rate is not None else PLAYER_Q_DEFAULTS["learning_rate"])
    gamma = float(gamma if gamma is not None else PLAYER_Q_DEFAULTS["gamma"])
    epsilon_start = float(epsilon_start if epsilon_start is not None else PLAYER_Q_DEFAULTS["epsilon_start"])
    epsilon_end = float(epsilon_end if epsilon_end is not None else PLAYER_Q_DEFAULTS["epsilon_end"])
    max_steps = int(max_steps if max_steps is not None else PLAYER_Q_DEFAULTS["max_steps"])

    rng = random.Random(seed)
    model = LinearQModel.load(initial_model_path) if initial_model_path is not None else LinearQModel.create(rng=rng)
    scores: list[int] = []
    max_tiles: list[int] = []
    stopped = False

    local_episode = 0
    try:
        while has_remaining_episodes(local_episode, remaining_episodes):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            local_episode += 1
            episode = start_completed + local_episode
            # 线性退火探索率：无限训练会在默认训练局数窗口内退火，之后固定在末值。
            # Linearly anneal epsilon; unlimited runs use the default episode window then hold.
            epsilon = scheduled_epsilon(
                episode=episode,
                limit=episode_limit,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
            )

            episode_seed = None if seed is None else seed + episode - 1
            enemy = create_enemy(enemy_type, rng=rng)
            env = GameEnv(enemy=enemy, seed=episode_seed)
            state = env.reset()

            while not state.done and state.steps < max_steps:
                legal_actions = env.get_legal_actions()
                action = model.epsilon_greedy_action(state.board, legal_actions, epsilon=epsilon, rng=rng)
                if action is None:
                    break

                previous_state = state
                state = env.step(action)
                reward = player_reward(previous_state, state)
                target = reward
                if not state.done:
                    # Q-learning 目标 = 当前奖励 + 折扣后的下一状态最佳估值。
                    # Q-learning target = immediate reward plus discounted best next-state value.
                    target += gamma * model.max_next_q(state.board)
                model.update(previous_state.board, action, target=target, learning_rate=learning_rate)

            scores.append(state.score)
            max_tiles.append(state.max_tile)
            if progress_callback is not None:
                progress_callback(episode, target_episodes, state, epsilon)
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
    except KeyboardInterrupt:
        stopped = True

    output_path = model.save(output_path)
    completed_episodes = start_completed + len(scores)
    status = (
        TRAINING_STATUS_INCOMPLETE
        if stopped and target_episodes is not None and completed_episodes < target_episodes
        else TRAINING_STATUS_COMPLETED
    )
    run_log_path = training_run_log_path(run_directory, output_path)
    parameters = {
        "episodes": target_episodes,
        "target_episodes": target_episodes,
        "completed_episodes": completed_episodes,
        "enemy_type": enemy_type,
        "seed": seed,
        "output": output_path,
        "reference_model_path": selected_reference,
        "resume_run_path": resume_path,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "max_steps": max_steps,
    }
    summary = TrainingSummary(
        episodes=completed_episodes,
        output_path=output_path,
        average_score=sum(scores) / len(scores) if scores else 0.0,
        average_max_tile=sum(max_tiles) / len(max_tiles) if max_tiles else 0.0,
        best_max_tile=max(max_tiles) if max_tiles else 0,
        info_path=run_directory / "info.json" if run_directory is not None else None,
        status=status,
        target_episodes=target_episodes,
        completed_episodes=completed_episodes,
        reference_model_path=selected_reference,
        resume_run_path=resume_path,
        run_log_path=run_log_path,
    )
    return complete_training_artifact(
        training_type="player_q",
        model_path=output_path,
        run_directory=run_directory,
        parameters=parameters,
        summary=summary,
        status=status,
        target_episodes=target_episodes,
        completed_episodes=completed_episodes,
        reference_model_path=selected_reference,
        resume_run_path=resume_path,
        run_log_path=run_log_path,
    )
