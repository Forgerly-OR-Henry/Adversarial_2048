"""玩家 DQN 训练循环。 / Training loop for the player DQN."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import get_train_defaults
from domain.enemies import create_enemy
from domain.game.constants import ACTIONS
from domain.game.env import GameEnv
from domain.models.dqn.network import DQNNetwork, batch_boards_to_tensor
from domain.models.torch_utils import get_torch_device, require_torch
from domain.train.artifacts import (
    TRAINING_STATUS_COMPLETED,
    TRAINING_STATUS_INCOMPLETE,
    complete_training_artifact,
    initial_training_model_path,
    resolve_training_output,
    resume_training_context,
    training_run_log_path,
)
from domain.train.dqn.checkpoints import (
    checkpoint_path,
    load_dqn_checkpoint_into_model,
)
from domain.train.dqn.common import (
    apply_dqn_stability,
    optimize_dqn_batch,
    resolve_dqn_stability_config,
    save_final_dqn_checkpoint,
    sync_target_model_if_due,
)
from domain.train.looping import resolve_episode_limit, run_training_episode_loop
from domain.train.dqn.replay_buffer import ReplayBuffer, Transition
from domain.train.dqn.stability import (
    StabilityConfig,
    StabilityController,
)
from domain.train.q_learning.player import player_reward

PLAYER_DQN_DEFAULTS = get_train_defaults("player_dqn")


@dataclass(frozen=True)
class DQNTrainingSummary:
    """玩家 DQN 训练完成后的指标摘要。 / Metrics summary after player DQN training."""
    episodes: int
    output_path: Path
    average_score: float
    average_max_tile: float
    best_max_tile: int
    device: str
    best_checkpoint_path: Path | None = None
    best_episode: int = 0
    final_learning_rate: float = 0.0
    final_epsilon: float = 0.0
    info_path: Path | None = None
    status: str = TRAINING_STATUS_COMPLETED
    target_episodes: int | None = None
    completed_episodes: int = 0
    reference_model_path: Path | None = None
    resume_run_path: Path | None = None
    run_log_path: Path | None = None


def train_dqn_player(
    episodes: int | None = None,
    enemy_type: str | None = None,
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
    reference_model_path: str | Path | None = None,
    resume_run_path: str | Path | None = None,
    stop_event=None,
    progress_callback=None,
) -> DQNTrainingSummary:
    """训练玩家 DQN 并保存产物。 / Train the player DQN and save artifacts."""
    episode_limit = resolve_episode_limit(episodes, PLAYER_DQN_DEFAULTS["episodes"])
    target_episodes = episode_limit.target_episodes
    enemy_type = enemy_type or PLAYER_DQN_DEFAULTS["enemy"]
    seed = seed if seed is not None else PLAYER_DQN_DEFAULTS.get("seed")
    output_path, run_directory = resolve_training_output(
        "player_dqn",
        output,
    )
    start_completed, resume_path, selected_reference, resume_info = resume_training_context(
        "player_dqn",
        resume_run_path,
        target_episodes,
        reference_model_path,
    )
    initial_model_path = initial_training_model_path(selected_reference, resume_info)
    remaining_episodes = episode_limit.remaining_after(start_completed)
    learning_rate = float(learning_rate if learning_rate is not None else PLAYER_DQN_DEFAULTS["learning_rate"])
    gamma = float(gamma if gamma is not None else PLAYER_DQN_DEFAULTS["gamma"])
    epsilon_start = float(epsilon_start if epsilon_start is not None else PLAYER_DQN_DEFAULTS["epsilon_start"])
    epsilon_end = float(epsilon_end if epsilon_end is not None else PLAYER_DQN_DEFAULTS["epsilon_end"])
    batch_size = int(batch_size if batch_size is not None else PLAYER_DQN_DEFAULTS["batch_size"])
    replay_capacity = int(replay_capacity if replay_capacity is not None else PLAYER_DQN_DEFAULTS["replay_capacity"])
    min_replay_size = int(min_replay_size if min_replay_size is not None else PLAYER_DQN_DEFAULTS["min_replay_size"])
    target_update_interval = int(
        target_update_interval
        if target_update_interval is not None
        else PLAYER_DQN_DEFAULTS["target_update_interval"]
    )
    max_steps = int(max_steps if max_steps is not None else PLAYER_DQN_DEFAULTS["max_steps"])
    device = device if device is not None else PLAYER_DQN_DEFAULTS.get("device")
    stability_config = resolve_dqn_stability_config(PLAYER_DQN_DEFAULTS, stability)

    torch = require_torch()
    device = device or get_torch_device()
    rng = random.Random(seed)
    model = DQNNetwork(output_size=len(ACTIONS)).to(device)
    if initial_model_path is not None:
        load_dqn_checkpoint_into_model(torch, model, initial_model_path, device)
    target_model = DQNNetwork(output_size=len(ACTIONS)).to(device)
    target_model.load_state_dict(model.state_dict())
    controller = StabilityController(
        config=stability_config,
        learning_rate=learning_rate,
        epsilon=epsilon_start,
        maximize=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=controller.learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()
    replay = ReplayBuffer(replay_capacity, rng=rng)
    best_checkpoint_path = checkpoint_path(output_path, "best")
    rolling_checkpoint_path = checkpoint_path(output_path, "checkpoint")

    def run_episode(episode: int, episode_seed: int | None, epsilon: float):
        env = GameEnv(enemy=create_enemy(enemy_type, rng=rng), seed=episode_seed)
        state = env.reset()

        while not state.done and state.steps < max_steps:
            legal_actions = env.get_legal_actions()
            legal_indexes = [ACTIONS.index(action) for action in legal_actions]
            if not legal_indexes:
                break
            # DQN 只在合法动作集合内做 epsilon-greedy，保证探索也不会产生非法移动。
            # DQN performs epsilon-greedy only within legal actions, keeping exploration valid.
            if rng.random() < epsilon:
                action_index = rng.choice(legal_indexes)
            else:
                with torch.no_grad():
                    q_values = model(batch_boards_to_tensor([state.board], device))[0]
                action_index = max(legal_indexes, key=lambda index: float(q_values[index].item()))

            previous_state = state
            state = env.step(ACTIONS[action_index])
            reward = player_reward(previous_state, state)
            replay.push(
                Transition(
                    state=previous_state.board,
                    action=action_index,
                    reward=reward,
                    next_state=state.board,
                    done=state.done,
                    legal_next_actions=[ACTIONS.index(action) for action in env.get_legal_actions()],
                )
            )
            optimize_dqn_batch(
                torch=torch,
                model=model,
                target_model=target_model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                replay=replay,
                batch_size=batch_size,
                min_replay_size=min_replay_size,
                gamma=gamma,
                device=device,
                stability_config=stability_config,
            )

        sync_target_model_if_due(model, target_model, episode, target_update_interval)
        apply_dqn_stability(
            torch=torch,
            model=model,
            target_model=target_model,
            optimizer=optimizer,
            controller=controller,
            stability_config=stability_config,
            episode=episode,
            metric=float(state.score),
            best_checkpoint_path=best_checkpoint_path,
            rolling_checkpoint_path=rolling_checkpoint_path,
            opponent_key="enemy_type",
            opponent_type=enemy_type,
            device=device,
        )
        return state

    def report_progress(episode: int, total: int | None, state, epsilon: float) -> None:
        if progress_callback is not None:
            progress_callback(episode, total, state, epsilon, device)

    loop_result = run_training_episode_loop(
        start_completed=start_completed,
        remaining_episodes=remaining_episodes,
        limit=episode_limit,
        seed=seed,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_resolver=controller.episode_epsilon,
        episode_runner=run_episode,
        stop_event=stop_event,
        progress_callback=report_progress if progress_callback is not None else None,
    )
    scores = loop_result.scores
    max_tiles = loop_result.max_tiles
    stopped = loop_result.stopped

    completed_episodes = start_completed + loop_result.episodes_run
    status = (
        TRAINING_STATUS_INCOMPLETE
        if stopped and target_episodes is not None and completed_episodes < target_episodes
        else TRAINING_STATUS_COMPLETED
    )
    save_final_dqn_checkpoint(
        torch=torch,
        output_path=output_path,
        model=model,
        completed_episodes=completed_episodes,
        opponent_key="enemy_type",
        opponent_type=enemy_type,
        device=device,
        controller=controller,
        stability_config=stability_config,
        status=status,
        target_episodes=target_episodes,
        reference_model_path=selected_reference,
        resume_run_path=resume_path,
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
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "min_replay_size": min_replay_size,
        "target_update_interval": target_update_interval,
        "max_steps": max_steps,
        "device": device,
        "best_checkpoint_path": best_checkpoint_path if best_checkpoint_path.exists() else None,
        "rolling_checkpoint_path": rolling_checkpoint_path,
    }
    summary = DQNTrainingSummary(
        episodes=completed_episodes,
        output_path=output_path,
        average_score=sum(scores) / len(scores) if scores else 0.0,
        average_max_tile=sum(max_tiles) / len(max_tiles) if max_tiles else 0.0,
        best_max_tile=max(max_tiles) if max_tiles else 0,
        device=device,
        best_checkpoint_path=best_checkpoint_path if best_checkpoint_path.exists() else None,
        best_episode=controller.best_episode,
        final_learning_rate=controller.learning_rate,
        final_epsilon=controller.epsilon,
        info_path=run_directory / "info.json" if run_directory is not None else None,
        status=status,
        target_episodes=target_episodes,
        completed_episodes=completed_episodes,
        reference_model_path=selected_reference,
        resume_run_path=resume_path,
        run_log_path=run_log_path,
    )
    return complete_training_artifact(
        training_type="player_dqn",
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
