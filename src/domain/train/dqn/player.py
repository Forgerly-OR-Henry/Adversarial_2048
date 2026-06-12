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
    clone_state_dict,
    load_dqn_checkpoint_into_model,
    load_state_dict_to_device,
    save_dqn_checkpoint,
)
from domain.train.dqn.common import masked_next_values
from domain.train.looping import has_remaining_episodes, resolve_episode_limit, scheduled_epsilon
from domain.train.dqn.replay_buffer import ReplayBuffer, Transition
from domain.train.dqn.stability import (
    StabilityConfig,
    StabilityController,
    set_optimizer_learning_rate,
    stability_config_from_mapping,
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
    if stability is None:
        stability_config = stability_config_from_mapping(PLAYER_DQN_DEFAULTS.get("stability"))
    elif isinstance(stability, StabilityConfig):
        stability_config = stability
    else:
        stability_config = stability_config_from_mapping(stability)

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
    scores: list[int] = []
    max_tiles: list[int] = []
    best_checkpoint_path = checkpoint_path(output_path, "best")
    rolling_checkpoint_path = checkpoint_path(output_path, "checkpoint")
    stopped = False

    local_episode = 0
    try:
        while has_remaining_episodes(local_episode, remaining_episodes):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            local_episode += 1
            episode = start_completed + local_episode
            scheduled = scheduled_epsilon(
                episode=episode,
                limit=episode_limit,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
            )
            epsilon = controller.episode_epsilon(scheduled)
            episode_seed = None if seed is None else seed + episode - 1
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

                if len(replay) >= max(batch_size, min_replay_size):
                    # 经验回放打散相邻局面相关性，目标网络提供更稳定的 bootstrap 估值。
                    # Replay breaks adjacent-state correlation; the target network gives stabler bootstrap values.
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

            if episode % target_update_interval == 0:
                # 周期性同步目标网络，避免每个梯度步都追逐正在变化的在线网络。
                # Periodically sync the target network instead of chasing the changing online network each step.
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
                # 稳定性控制触发回滚时，在线网络和目标网络必须保持一致。
                # When stability rolls back, keep the online and target networks in sync.
                load_state_dict_to_device(model, controller.best_state, device)
                target_model.load_state_dict(model.state_dict())
            if decision.improved and stability_config.keep_best_model:
                save_dqn_checkpoint(
                    torch,
                    best_checkpoint_path,
                    model,
                    episode,
                    opponent_key="enemy_type",
                    opponent_type=enemy_type,
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
                    opponent_key="enemy_type",
                    opponent_type=enemy_type,
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
    except KeyboardInterrupt:
        stopped = True

    completed_episodes = start_completed + len(scores)
    status = (
        TRAINING_STATUS_INCOMPLETE
        if stopped and target_episodes is not None and completed_episodes < target_episodes
        else TRAINING_STATUS_COMPLETED
    )
    if (
        stability_config.enabled
        and stability_config.restore_best_on_finish
        and controller.best_state is not None
        and status == TRAINING_STATUS_COMPLETED
    ):
        # 训练结束可恢复历史最佳权重，避免最终几局退化污染输出模型。
        # Finishing can restore the best weights so late degradation does not pollute the output model.
        load_state_dict_to_device(model, controller.best_state, device)
    save_dqn_checkpoint(
        torch,
        output_path,
        model,
        completed_episodes,
        opponent_key="enemy_type",
        opponent_type=enemy_type,
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
