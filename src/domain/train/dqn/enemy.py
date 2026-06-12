"""敌人 DQN 训练循环。 / Training loop for the enemy DQN."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from config import get_train_defaults
from domain.game.board import get_empty_cells, place_tile
from domain.game.env import GameEnv
from domain.game.rules import is_game_over, move
from domain.game.state import GameState
from domain.models import ENEMY_ACTIONS, action_to_spawn, get_legal_spawn_actions
from domain.models.dqn_network import DQNNetwork, batch_boards_to_tensor
from domain.models.torch_utils import get_torch_device, require_torch
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
from domain.train.dqn.checkpoints import (
    checkpoint_path,
    clone_state_dict,
    load_dqn_checkpoint_into_model,
    load_state_dict_to_device,
    save_dqn_checkpoint,
)
from domain.train.dqn.common import masked_next_values
from domain.train.dqn.replay_buffer import ReplayBuffer, Transition
from domain.train.dqn.stability import (
    StabilityConfig,
    StabilityController,
    set_optimizer_learning_rate,
    stability_config_from_mapping,
)
from domain.train.q_learning.enemy import enemy_reward
from utils.training_log import log_training_result

ENEMY_DQN_DEFAULTS = get_train_defaults("enemy_dqn")


@dataclass(frozen=True)
class EnemyDQNTrainingSummary:
    """敌人 DQN 训练完成后的指标摘要。 / Metrics summary after enemy DQN training."""
    episodes: int
    output_path: Path
    average_player_score: float
    average_player_max_tile: float
    best_suppressed_max_tile: int
    device: str
    best_checkpoint_path: Path | None = None
    best_episode: int = 0
    final_learning_rate: float = 0.0
    final_epsilon: float = 0.0
    latest_output_path: Path | None = None
    info_path: Path | None = None
    status: str = TRAINING_STATUS_COMPLETED
    target_episodes: int = 0
    completed_episodes: int = 0
    reference_model_path: Path | None = None
    resume_run_path: Path | None = None
    run_log_path: Path | None = None


def train_dqn_enemy(
    episodes: int | None = None,
    player_type: str | None = None,
    seed: int | None = None,
    output: str | Path | None = None,
    learning_rate: float | None = None,
    gamma: float | None = None,
    epsilon_start: float | None = None,
    epsilon_end: float | None = None,
    batch_size: int | None = None,
    replay_capacity: int | None = None,
    min_replay_size: int | None = None,
    target_update_interval: int | None = None,
    max_steps: int | None = None,
    device: str | None = None,
    stability: StabilityConfig | dict[str, Any] | None = None,
    publish_latest: bool = True,
    reference_model_path: str | Path | None = None,
    resume_run_path: str | Path | None = None,
    stop_event=None,
    progress_callback=None,
) -> EnemyDQNTrainingSummary:
    """训练敌人 DQN 并保存产物。 / Train the enemy DQN and save artifacts."""
    target_episodes = int(episodes if episodes is not None else ENEMY_DQN_DEFAULTS["episodes"])
    player_type = player_type or ENEMY_DQN_DEFAULTS["player"]
    seed = seed if seed is not None else ENEMY_DQN_DEFAULTS.get("seed")
    output_path, run_directory, latest_path = resolve_training_output(
        "enemy_dqn",
        output,
    )
    start_completed, resume_path, selected_reference, resume_info = resume_training_context(
        "enemy_dqn",
        resume_run_path,
        target_episodes,
        reference_model_path,
    )
    initial_model_path = initial_training_model_path(selected_reference, resume_info)
    remaining_episodes = target_episodes - start_completed
    learning_rate = float(learning_rate if learning_rate is not None else ENEMY_DQN_DEFAULTS["learning_rate"])
    gamma = float(gamma if gamma is not None else ENEMY_DQN_DEFAULTS["gamma"])
    epsilon_start = float(epsilon_start if epsilon_start is not None else ENEMY_DQN_DEFAULTS["epsilon_start"])
    epsilon_end = float(epsilon_end if epsilon_end is not None else ENEMY_DQN_DEFAULTS["epsilon_end"])
    batch_size = int(batch_size if batch_size is not None else ENEMY_DQN_DEFAULTS["batch_size"])
    replay_capacity = int(replay_capacity if replay_capacity is not None else ENEMY_DQN_DEFAULTS["replay_capacity"])
    min_replay_size = int(min_replay_size if min_replay_size is not None else ENEMY_DQN_DEFAULTS["min_replay_size"])
    target_update_interval = int(
        target_update_interval
        if target_update_interval is not None
        else ENEMY_DQN_DEFAULTS["target_update_interval"]
    )
    max_steps = int(max_steps if max_steps is not None else ENEMY_DQN_DEFAULTS["max_steps"])
    device = device if device is not None else ENEMY_DQN_DEFAULTS.get("device")
    if stability is None:
        stability_config = stability_config_from_mapping(ENEMY_DQN_DEFAULTS.get("stability"))
    elif isinstance(stability, StabilityConfig):
        stability_config = stability
    else:
        stability_config = stability_config_from_mapping(stability)

    torch = require_torch()
    device = device or get_torch_device()
    rng = random.Random(seed)
    model = DQNNetwork(output_size=len(ENEMY_ACTIONS)).to(device)
    if initial_model_path is not None:
        load_dqn_checkpoint_into_model(torch, model, initial_model_path, device)
    target_model = DQNNetwork(output_size=len(ENEMY_ACTIONS)).to(device)
    target_model.load_state_dict(model.state_dict())
    controller = StabilityController(
        config=stability_config,
        learning_rate=learning_rate,
        epsilon=epsilon_start,
        maximize=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=controller.learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()
    replay = ReplayBuffer(replay_capacity, rng=rng)
    scores: list[int] = []
    max_tiles: list[int] = []
    best_checkpoint_path = checkpoint_path(output_path, "best")
    rolling_checkpoint_path = checkpoint_path(output_path, "checkpoint")
    stopped = False

    for local_episode in range(1, remaining_episodes + 1):
        episode = start_completed + local_episode
        progress = 0 if target_episodes == 1 else (episode - 1) / (target_episodes - 1)
        scheduled_epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        epsilon = controller.episode_epsilon(scheduled_epsilon)
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
                state = GameState(
                    board=board_after_player,
                    score=score_after_player,
                    steps=steps_after_player,
                    done=is_game_over(board_after_player),
                )
                break

            legal_actions = get_legal_spawn_actions(board_after_player)
            legal_indexes = [ENEMY_ACTIONS.index(action) for action in legal_actions]
            # 敌人探索也限制在合法空格上，否则经验回放会混入不可执行动作。
            # Enemy exploration is also limited to legal empty cells to keep replay executable.
            if rng.random() < epsilon:
                action_index = rng.choice(legal_indexes)
            else:
                with torch.no_grad():
                    q_values = model(batch_boards_to_tensor([board_after_player], device))[0]
                action_index = max(legal_indexes, key=lambda index: float(q_values[index].item()))

            row, col, value = action_to_spawn(ENEMY_ACTIONS[action_index])
            next_board = place_tile(board_after_player, row, col, value)
            next_state = GameState(
                board=next_board,
                score=score_after_player,
                steps=steps_after_player,
                done=is_game_over(next_board),
            )
            reward = enemy_reward(board_after_player, next_state, moved.score_delta)
            replay.push(
                Transition(
                    state=board_after_player,
                    action=action_index,
                    reward=reward,
                    next_state=next_state.board,
                    done=next_state.done,
                    legal_next_actions=[
                        ENEMY_ACTIONS.index(action)
                        for action in get_legal_spawn_actions(next_state.board)
                    ],
                )
            )

            if len(replay) >= max(batch_size, min_replay_size):
                # 这里的 state 是“玩家移动后的棋盘”，因为敌人只决定随后生成哪个方块。
                # Here state means the post-player board because the enemy only chooses the following spawn.
                batch = replay.sample(batch_size)
                states = batch_boards_to_tensor([item.state for item in batch], device)
                next_states = batch_boards_to_tensor([item.next_state for item in batch], device)
                actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
                dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=device)
                current = model(states).gather(1, actions).squeeze(1)
                next_values = masked_next_values(
                    torch,
                    target_model,
                    next_states,
                    [item.legal_next_actions for item in batch],
                    device,
                )
                target = rewards + gamma * next_values * (1.0 - dones)
                loss = loss_fn(current, target)
                optimizer.zero_grad()
                loss.backward()
                if stability_config.enabled and stability_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), stability_config.max_grad_norm)
                optimizer.step()

            env.set_board(next_state.board, score=next_state.score, steps=next_state.steps)
            state = env.snapshot()

        if episode % target_update_interval == 0:
            # 目标网络按 episode 同步，降低敌人训练中的 bootstrap 震荡。
            # Sync the target network by episode to reduce bootstrap oscillation in enemy training.
            target_model.load_state_dict(model.state_dict())

        scores.append(state.score)
        max_tiles.append(state.max_tile)
        decision = controller.observe(
            episode=episode,
            metric=float(state.score),
            model_state=clone_state_dict(model),
        )
        if decision.learning_rate is not None:
            set_optimizer_learning_rate(optimizer, decision.learning_rate)
        if decision.rollback and controller.best_state is not None:
            # 回滚后同步目标网络，防止下个 batch 仍使用旧目标估值。
            # Sync the target network after rollback so the next batch does not use stale target values.
            load_state_dict_to_device(model, controller.best_state, device)
            target_model.load_state_dict(model.state_dict())
        if decision.improved and stability_config.keep_best_model:
            save_dqn_checkpoint(
                torch,
                best_checkpoint_path,
                model,
                episode,
                opponent_key="player_type",
                opponent_type=player_type,
                device=device,
                metadata={
                    "best_metric": controller.best_metric,
                    "best_episode": controller.best_episode,
                    "learning_rate": controller.learning_rate,
                    "epsilon": controller.epsilon,
                    "reason": decision.reason,
                },
            )
        if stability_config.enabled and stability_config.checkpoint_interval > 0 and episode % stability_config.checkpoint_interval == 0:
            save_dqn_checkpoint(
                torch,
                rolling_checkpoint_path,
                model,
                episode,
                opponent_key="player_type",
                opponent_type=player_type,
                device=device,
                metadata={
                    "best_metric": controller.best_metric,
                    "best_episode": controller.best_episode,
                    "learning_rate": controller.learning_rate,
                    "epsilon": controller.epsilon,
                    "reason": decision.reason,
                },
            )
        if progress_callback is not None:
            progress_callback(episode, target_episodes, state, epsilon, device)
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break

    completed_episodes = start_completed + len(scores)
    status = (
        TRAINING_STATUS_INCOMPLETE
        if stopped and completed_episodes < target_episodes
        else TRAINING_STATUS_COMPLETED
    )
    if (
        stability_config.enabled
        and stability_config.restore_best_on_finish
        and controller.best_state is not None
        and status == TRAINING_STATUS_COMPLETED
    ):
        # 敌人以压低玩家表现为目标，收尾时恢复窗口内最优压制模型。
        # The enemy minimizes player performance, so finishing restores the best suppressing model.
        load_state_dict_to_device(model, controller.best_state, device)
    save_dqn_checkpoint(
        torch,
        output_path,
        model,
        completed_episodes,
        opponent_key="player_type",
        opponent_type=player_type,
        device=device,
        metadata={
            "best_metric": controller.best_metric,
            "best_episode": controller.best_episode,
            "learning_rate": controller.learning_rate,
            "epsilon": controller.epsilon,
            "status": status,
            "target_episodes": target_episodes,
            "completed_episodes": completed_episodes,
            "reference_model_path": selected_reference,
            "resume_run_path": resume_path,
            "restored_best_on_finish": (
                stability_config.enabled
                and stability_config.restore_best_on_finish
                and status == TRAINING_STATUS_COMPLETED
            ),
        },
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
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "min_replay_size": min_replay_size,
        "target_update_interval": target_update_interval,
        "max_steps": max_steps,
        "device": device,
        "best_checkpoint_path": best_checkpoint_path if best_checkpoint_path.exists() else None,
        "rolling_checkpoint_path": rolling_checkpoint_path,
        "publish_latest": publish,
    }
    summary = EnemyDQNTrainingSummary(
        episodes=completed_episodes,
        output_path=output_path,
        average_player_score=sum(scores) / len(scores) if scores else 0.0,
        average_player_max_tile=sum(max_tiles) / len(max_tiles) if max_tiles else 0.0,
        best_suppressed_max_tile=min(max_tiles) if max_tiles else 0,
        device=device,
        best_checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else None,
        best_episode=controller.best_episode,
        final_learning_rate=controller.learning_rate,
        final_epsilon=controller.epsilon,
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
        training_type="enemy_dqn",
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
        "enemy_dqn",
        parameters,
        summary,
        path=run_log_path or output_path.with_name("log.jsonl"),
        event="training_stopped" if status == TRAINING_STATUS_INCOMPLETE else "training_completed",
    )
    cleanup_resumed_incomplete_run(resume_path, run_directory)
    return summary
