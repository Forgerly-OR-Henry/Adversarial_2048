"""敌人 Q-learning 训练循环。 / Training loop for enemy Q-learning."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

from config import get_train_defaults
from domain.game.board import get_empty_cells, place_tile
from domain.game.env import GameEnv
from domain.game.rules import evaluate_badness, is_game_over, move
from domain.game.state import GameState
from domain.models import EnemyQModel, action_to_spawn, get_legal_spawn_actions
from domain.players import create_player
from domain.train.artifacts import (
    TRAINING_STATUS_COMPLETED,
    TRAINING_STATUS_INCOMPLETE,
    cleanup_resumed_incomplete_run,
    finalize_training_artifact,
    initial_training_model_path,
    resolve_training_output,
    resume_training_context,
    training_run_log_path,
)
from utils.training_log import log_training_result

ENEMY_Q_DEFAULTS = get_train_defaults("enemy_q")


@dataclass(frozen=True)
class EnemyTrainingSummary:
    """敌人 Q-learning 训练完成后的指标摘要。 / Metrics summary after enemy Q-learning training."""
    episodes: int
    output_path: Path
    average_player_score: float
    average_player_max_tile: float
    best_suppressed_max_tile: int
    latest_output_path: Path | None = None
    info_path: Path | None = None
    status: str = TRAINING_STATUS_COMPLETED
    target_episodes: int = 0
    completed_episodes: int = 0
    reference_model_path: Path | None = None
    resume_run_path: Path | None = None
    run_log_path: Path | None = None


def enemy_reward(board_after_player_move: list[list[int]], next_state: GameState, score_delta: int) -> float:
    """计算敌人训练使用的奖励。 / Compute the reward used for enemy training."""
    # 奖励衡量“出块后比玩家刚移动后更糟多少”，并惩罚给玩家带来的合并收益。
    # Reward measures how much worse the spawn made the board, penalizing merge score given to the player.
    badness_reward = evaluate_badness(next_state.board) / 40.0
    merge_penalty = score_delta * 0.05
    max_tile_penalty = next_state.max_tile.bit_length() * 0.4
    game_over_bonus = 80.0 if next_state.done else 0.0
    empty_bonus = (16 - len(get_empty_cells(next_state.board))) * 0.2
    previous_badness = evaluate_badness(board_after_player_move) / 40.0
    return badness_reward - previous_badness - merge_penalty - max_tile_penalty + game_over_bonus + empty_bonus


def train_q_enemy(
    episodes: int | None = None,
    player_type: str | None = None,
    seed: int | None = None,
    output: str | Path | None = None,
    learning_rate: float | None = None,
    gamma: float | None = None,
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
    max_steps: int | None = None,
    publish_latest: bool = True,
    reference_model_path: str | Path | None = None,
    resume_run_path: str | Path | None = None,
    stop_event=None,
    progress_callback=None,
) -> EnemyTrainingSummary:
    """训练敌人 Q-learning 模型并保存产物。 / Train the enemy Q-learning model and save artifacts."""
    target_episodes = int(episodes if episodes is not None else ENEMY_Q_DEFAULTS["episodes"])
    player_type = player_type or ENEMY_Q_DEFAULTS["player"]
    seed = seed if seed is not None else ENEMY_Q_DEFAULTS.get("seed")
    output_path, run_directory, latest_path = resolve_training_output(
        "enemy_q",
        output,
    )
    start_completed, resume_path, selected_reference, resume_info = resume_training_context(
        "enemy_q",
        resume_run_path,
        target_episodes,
        reference_model_path,
    )
    initial_model_path = initial_training_model_path(selected_reference, resume_info)
    remaining_episodes = target_episodes - start_completed
    learning_rate = float(learning_rate if learning_rate is not None else ENEMY_Q_DEFAULTS["learning_rate"])
    gamma = float(gamma if gamma is not None else ENEMY_Q_DEFAULTS["gamma"])
    epsilon_start = float(epsilon_start if epsilon_start is not None else ENEMY_Q_DEFAULTS["epsilon_start"])
    epsilon_end = float(epsilon_end if epsilon_end is not None else ENEMY_Q_DEFAULTS["epsilon_end"])
    max_steps = int(max_steps if max_steps is not None else ENEMY_Q_DEFAULTS["max_steps"])

    rng = random.Random(seed)
    model = EnemyQModel.load(initial_model_path) if initial_model_path is not None else EnemyQModel.create(rng=rng)
    scores: list[int] = []
    max_tiles: list[int] = []
    stopped = False

    for local_episode in range(1, remaining_episodes + 1):
        episode = start_completed + local_episode
        if target_episodes == 1:
            epsilon = epsilon_end
        else:
            progress = (episode - 1) / (target_episodes - 1)
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        episode_seed = None if seed is None else seed + episode - 1
        player = create_player(player_type, rng=rng)
        env = GameEnv(seed=episode_seed)
        state = env.reset()

        while not state.done and state.steps < max_steps:
            legal_player_actions = env.get_legal_actions()
            player_action = player.select_action(state, legal_player_actions)
            if player_action is None:
                break

            moved = move(state.board, player_action)
            if not moved.moved:
                break

            board_after_player = moved.board
            score_after_player = state.score + moved.score_delta
            steps_after_player = state.steps + 1

            if not get_empty_cells(board_after_player):
                # 玩家移动后若没有空格，敌人没有出块动作，只需同步终局状态。
                # If the player move leaves no empty cells, the enemy has no spawn action to apply.
                state = GameState(
                    board=board_after_player,
                    score=score_after_player,
                    steps=steps_after_player,
                    done=is_game_over(board_after_player),
                )
                break

            legal_enemy_actions = get_legal_spawn_actions(board_after_player)
            enemy_action = model.epsilon_greedy_action(
                board_after_player,
                legal_enemy_actions,
                epsilon=epsilon,
                rng=rng,
            )
            if enemy_action is None:
                break

            row, col, value = action_to_spawn(enemy_action)
            next_board = place_tile(board_after_player, row, col, value)
            next_state = GameState(
                board=next_board,
                score=score_after_player,
                steps=steps_after_player,
                done=is_game_over(next_board),
            )
            reward = enemy_reward(board_after_player, next_state, moved.score_delta)
            target = reward
            if not next_state.done:
                # 敌人 Q 值基于“玩家移动后的棋盘”，因为敌人只学习出块选择。
                # Enemy Q values are based on the post-player board because it learns spawn choices only.
                target += gamma * model.max_next_q(next_state.board)
            model.update(board_after_player, enemy_action, target=target, learning_rate=learning_rate)
            env.set_board(next_state.board, score=next_state.score, steps=next_state.steps)
            state = env.snapshot()

        scores.append(state.score)
        max_tiles.append(state.max_tile)
        if progress_callback is not None:
            progress_callback(episode, target_episodes, state, epsilon)
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break

    output_path = model.save(output_path)
    completed_episodes = start_completed + len(scores)
    status = (
        TRAINING_STATUS_INCOMPLETE
        if stopped and completed_episodes < target_episodes
        else TRAINING_STATUS_COMPLETED
    )
    run_log_path = training_run_log_path(run_directory, output_path)
    publish = publish_latest and run_directory is not None and status == TRAINING_STATUS_COMPLETED
    parameters = {
        "episodes": target_episodes,
        "target_episodes": target_episodes,
        "completed_episodes": completed_episodes,
        "player_type": player_type,
        "seed": seed,
        "output": output_path,
        "reference_model_path": selected_reference,
        "resume_run_path": resume_path,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "max_steps": max_steps,
        "publish_latest": publish,
    }
    summary = EnemyTrainingSummary(
        episodes=completed_episodes,
        output_path=output_path,
        average_player_score=sum(scores) / len(scores) if scores else 0.0,
        average_player_max_tile=sum(max_tiles) / len(max_tiles) if max_tiles else 0.0,
        best_suppressed_max_tile=min(max_tiles) if max_tiles else 0,
        latest_output_path=latest_path if publish else None,
        info_path=run_directory / "info.json" if run_directory is not None else None,
        status=status,
        target_episodes=target_episodes,
        completed_episodes=completed_episodes,
        reference_model_path=selected_reference,
        resume_run_path=resume_path,
        run_log_path=run_log_path,
    )
    info_path, published_path = finalize_training_artifact(
        training_type="enemy_q",
        model_path=output_path,
        run_directory=run_directory,
        latest_path=latest_path,
        parameters=parameters,
        summary=summary,
        publish=publish,
        status=status,
        target_episodes=target_episodes,
        completed_episodes=completed_episodes,
        reference_model_path=selected_reference,
        resume_run_path=resume_path,
        run_log_path=run_log_path,
    )
    if info_path is not None or published_path is not None:
        summary = replace(summary, latest_output_path=published_path, info_path=info_path)
    log_training_result(
        "enemy_q",
        parameters,
        summary,
        path=run_log_path or output_path.with_name("log.jsonl"),
        event="training_stopped" if status == TRAINING_STATUS_INCOMPLETE else "training_completed",
    )
    cleanup_resumed_incomplete_run(resume_path, run_directory)
    return summary
