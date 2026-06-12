"""DQN 训练共享计算。 / Shared DQN training calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from domain.models.dqn.network import batch_boards_to_tensor
from domain.train.artifacts import TRAINING_STATUS_COMPLETED
from domain.train.dqn.checkpoints import (
    clone_state_dict,
    load_state_dict_to_device,
    save_dqn_checkpoint,
)
from domain.train.dqn.stability import (
    StabilityConfig,
    StabilityController,
    StabilityDecision,
    set_optimizer_learning_rate,
    stability_config_from_mapping,
)


def masked_next_values(torch, target_model, next_states, legal_next_actions_batch, device: str):
    """只在下一状态的合法动作中取最大 Q 值。 / Maximize next-state Q values over legal actions only."""
    with torch.no_grad():
        q_values = target_model(next_states)
        values = []
        for row, legal_indexes in zip(q_values, legal_next_actions_batch):
            if legal_indexes:
                indexes = torch.tensor(legal_indexes, dtype=torch.long, device=device)
                values.append(row.index_select(0, indexes).max())
            else:
                values.append(torch.tensor(0.0, device=device))
        return torch.stack(values)


def resolve_dqn_stability_config(
    defaults: dict[str, Any],
    stability: StabilityConfig | dict[str, Any] | None,
) -> StabilityConfig:
    """解析 DQN 稳定性配置。 / Resolve the DQN stability configuration."""
    if stability is None:
        return stability_config_from_mapping(defaults.get("stability"))
    if isinstance(stability, StabilityConfig):
        return stability
    return stability_config_from_mapping(stability)


def optimize_dqn_batch(
    *,
    torch,
    model,
    target_model,
    loss_fn,
    optimizer,
    replay,
    batch_size: int,
    min_replay_size: int,
    gamma: float,
    device: str,
    stability_config: StabilityConfig,
) -> bool:
    """从回放池抽样并执行一次 DQN 梯度更新。 / Run one sampled DQN optimization step."""
    if len(replay) < max(batch_size, min_replay_size):
        return False

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
    return True


def sync_target_model_if_due(model, target_model, episode: int, interval: int) -> None:
    """按 episode 间隔同步目标网络。 / Sync the target model on episode intervals."""
    if episode % interval == 0:
        target_model.load_state_dict(model.state_dict())


def apply_dqn_stability(
    *,
    torch,
    model,
    target_model,
    optimizer,
    controller: StabilityController,
    stability_config: StabilityConfig,
    episode: int,
    metric: float,
    best_checkpoint_path: Path,
    rolling_checkpoint_path: Path,
    opponent_key: str,
    opponent_type: str,
    device: str,
) -> StabilityDecision:
    """应用 episode 末尾的稳定性决策和 checkpoint 策略。 / Apply stability and checkpoint policy."""
    decision = controller.observe(
        episode=episode,
        metric=metric,
        model_state=clone_state_dict(model),
    )
    if decision.learning_rate is not None:
        set_optimizer_learning_rate(optimizer, decision.learning_rate)
    if decision.rollback and controller.best_state is not None:
        load_state_dict_to_device(model, controller.best_state, device)
        target_model.load_state_dict(model.state_dict())
    metadata = {
        "best_metric": controller.best_metric,
        "best_episode": controller.best_episode,
        "learning_rate": controller.learning_rate,
        "epsilon": controller.epsilon,
        "reason": decision.reason,
    }
    if decision.improved and stability_config.keep_best_model:
        save_dqn_checkpoint(
            torch,
            best_checkpoint_path,
            model,
            episode,
            opponent_key=opponent_key,
            opponent_type=opponent_type,
            device=device,
            metadata=metadata,
        )
    if (
        stability_config.enabled
        and stability_config.checkpoint_interval > 0
        and episode % stability_config.checkpoint_interval == 0
    ):
        save_dqn_checkpoint(
            torch,
            rolling_checkpoint_path,
            model,
            episode,
            opponent_key=opponent_key,
            opponent_type=opponent_type,
            device=device,
            metadata=metadata,
        )
    return decision


def save_final_dqn_checkpoint(
    *,
    torch,
    output_path: Path,
    model,
    completed_episodes: int,
    opponent_key: str,
    opponent_type: str,
    device: str,
    controller: StabilityController,
    stability_config: StabilityConfig,
    status: str,
    target_episodes: int | None,
    reference_model_path: str | Path | None,
    resume_run_path: str | Path | None,
) -> None:
    """保存最终 DQN checkpoint，并按需恢复最佳权重。 / Save the final DQN checkpoint."""
    restore_best_on_finish = (
        stability_config.enabled
        and stability_config.restore_best_on_finish
        and status == TRAINING_STATUS_COMPLETED
    )
    if restore_best_on_finish and controller.best_state is not None:
        load_state_dict_to_device(model, controller.best_state, device)
    save_dqn_checkpoint(
        torch,
        output_path,
        model,
        completed_episodes,
        opponent_key=opponent_key,
        opponent_type=opponent_type,
        device=device,
        metadata={
            "best_metric": controller.best_metric,
            "best_episode": controller.best_episode,
            "learning_rate": controller.learning_rate,
            "epsilon": controller.epsilon,
            "status": status,
            "target_episodes": target_episodes,
            "completed_episodes": completed_episodes,
            "reference_model_path": reference_model_path,
            "resume_run_path": resume_run_path,
            "restored_best_on_finish": restore_best_on_finish,
        },
    )


__all__ = [
    "apply_dqn_stability",
    "masked_next_values",
    "optimize_dqn_batch",
    "resolve_dqn_stability_config",
    "save_final_dqn_checkpoint",
    "sync_target_model_if_due",
]
