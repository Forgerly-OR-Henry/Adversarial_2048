"""训练局数上限和探索率调度。 / Episode limits and epsilon scheduling for training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeLimit:
    """一次训练的局数约束。 / Episode limit for one training run."""

    target_episodes: int | None
    epsilon_window: int

    @property
    def unlimited(self) -> bool:
        """是否无限训练。 / Whether the run is unlimited."""
        return self.target_episodes is None

    def remaining_after(self, completed_episodes: int) -> int | None:
        """返回剩余局数；无限训练返回 None。 / Return remaining episodes; unlimited runs return None."""
        if self.target_episodes is None:
            return None
        remaining = self.target_episodes - completed_episodes
        if remaining < 0:
            raise ValueError("继续训练总局数不能低于当前已训练局数。")
        return remaining


def resolve_episode_limit(episodes: int | None, default_episodes: int) -> EpisodeLimit:
    """解析训练局数；None 表示无限训练。 / Resolve episode count; None means unlimited."""
    epsilon_window = max(1, int(default_episodes))
    if episodes is None:
        return EpisodeLimit(target_episodes=None, epsilon_window=epsilon_window)
    target_episodes = int(episodes)
    if target_episodes < 1:
        raise ValueError("训练局数必须至少为 1，或留空表示无限训练。")
    return EpisodeLimit(target_episodes=target_episodes, epsilon_window=epsilon_window)


def has_remaining_episodes(local_episode_count: int, remaining_episodes: int | None) -> bool:
    """判断训练循环是否还应继续。 / Return whether the training loop should continue."""
    return remaining_episodes is None or local_episode_count < remaining_episodes


def scheduled_epsilon(
    *,
    episode: int,
    limit: EpisodeLimit,
    epsilon_start: float,
    epsilon_end: float,
) -> float:
    """按有限目标或无限训练窗口计算探索率。 / Compute epsilon from a finite target or unlimited window."""
    schedule_episodes = limit.target_episodes if limit.target_episodes is not None else limit.epsilon_window
    if schedule_episodes <= 1:
        return epsilon_end
    progress = min(1.0, max(0.0, (episode - 1) / (schedule_episodes - 1)))
    return epsilon_start + (epsilon_end - epsilon_start) * progress


__all__ = [
    "EpisodeLimit",
    "has_remaining_episodes",
    "resolve_episode_limit",
    "scheduled_epsilon",
]
