"""DQN 稳定性控制、学习率调整和回滚策略。 / DQN stability controls, learning-rate adjustment, and rollback policy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StabilityConfig:
    """DQN 自增长稳定控制参数。 / Configuration for DQN self-growth stability controls."""
    enabled: bool = True
    eval_interval: int = 20
    eval_window: int = 20
    improve_tolerance: float = 0.01
    drop_tolerance: float = 0.20
    bad_eval_patience: int = 2
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 3e-4
    lr_decay: float = 0.5
    min_epsilon: float = 0.05
    max_epsilon: float = 1.0
    epsilon_decay_on_improve: float = 0.98
    epsilon_boost_on_drop: float = 1.2
    max_grad_norm: float = 1.0
    checkpoint_interval: int = 50
    keep_best_model: bool = True
    rollback_on_drop: bool = True
    restore_best_on_finish: bool = True


@dataclass(frozen=True)
class StabilityDecision:
    """一次稳定性检查产生的训练调整。 / Training adjustment produced by one stability check."""
    episode: int
    metric: float
    improved: bool = False
    degraded: bool = False
    rollback: bool = False
    learning_rate: float | None = None
    epsilon: float | None = None
    reason: str = "stable"


@dataclass
class StabilityController:
    """跟踪最佳模型并决定学习率、探索率和回滚。 / Tracks the best model and decides learning-rate, epsilon, and rollback updates."""
    config: StabilityConfig
    learning_rate: float
    epsilon: float
    maximize: bool = True
    metrics: deque[float] = field(init=False)
    best_metric: float | None = None
    best_state: dict[str, Any] | None = None
    best_episode: int = 0
    bad_eval_count: int = 0

    def __post_init__(self) -> None:
        self.metrics = deque(maxlen=max(1, self.config.eval_window))
        self.learning_rate = clamp(self.learning_rate, self.config.min_learning_rate, self.config.max_learning_rate)
        self.epsilon = clamp(self.epsilon, self.config.min_epsilon, self.config.max_epsilon)

    def episode_epsilon(self, scheduled_epsilon: float) -> float:
        if not self.config.enabled:
            return scheduled_epsilon
        self.epsilon = clamp(
            min(self.epsilon, scheduled_epsilon),
            self.config.min_epsilon,
            self.config.max_epsilon,
        )
        return self.epsilon

    def observe(
        self,
        episode: int,
        metric: float,
        model_state: dict[str, Any] | None = None,
    ) -> StabilityDecision:
        self.metrics.append(metric)
        if not self.config.enabled or episode % max(1, self.config.eval_interval) != 0:
            return StabilityDecision(episode=episode, metric=metric, learning_rate=self.learning_rate, epsilon=self.epsilon)

        window_metric = sum(self.metrics) / len(self.metrics)
        # 用滑动窗口而不是单局结果判断进退步，减少 2048 随机出块带来的误判。
        # Use a moving window instead of one episode to reduce noise from random 2048 spawns.
        improved = self._is_improved(window_metric)
        degraded = self._is_degraded(window_metric)
        rollback = False
        reason = "stable"

        if improved:
            self.best_metric = window_metric
            self.best_episode = episode
            if model_state is not None:
                self.best_state = model_state
            self.bad_eval_count = 0
            self.epsilon = clamp(
                self.epsilon * self.config.epsilon_decay_on_improve,
                self.config.min_epsilon,
                self.config.max_epsilon,
            )
            reason = "improved"
        elif degraded:
            self.bad_eval_count += 1
            # 退化时同时降低学习率并恢复部分探索，让模型先稳定再重新搜索。
            # On degradation, lower the learning rate and restore exploration before searching again.
            self.learning_rate = clamp(
                self.learning_rate * self.config.lr_decay,
                self.config.min_learning_rate,
                self.config.max_learning_rate,
            )
            self.epsilon = clamp(
                self.epsilon * self.config.epsilon_boost_on_drop,
                self.config.min_epsilon,
                self.config.max_epsilon,
            )
            rollback = (
                self.config.rollback_on_drop
                and self.best_state is not None
                and self.bad_eval_count >= self.config.bad_eval_patience
            )
            reason = "rollback" if rollback else "degraded"
        else:
            self.bad_eval_count = 0

        return StabilityDecision(
            episode=episode,
            metric=window_metric,
            improved=improved,
            degraded=degraded,
            rollback=rollback,
            learning_rate=self.learning_rate,
            epsilon=self.epsilon,
            reason=reason,
        )

    def _is_improved(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.maximize:
            return metric > self.best_metric * (1.0 + self.config.improve_tolerance)
        return metric < self.best_metric * (1.0 - self.config.improve_tolerance)

    def _is_degraded(self, metric: float) -> bool:
        if self.best_metric is None:
            return False
        if self.maximize:
            return metric < self.best_metric * (1.0 - self.config.drop_tolerance)
        return metric > self.best_metric * (1.0 + self.config.drop_tolerance)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """把数值限制在上下界内。 / Clamp a value into inclusive bounds."""
    return min(max(value, minimum), maximum)


def stability_config_from_mapping(mapping: dict[str, Any] | None) -> StabilityConfig:
    """从配置字典构造稳定性配置。 / Build stability config from a mapping."""
    if not mapping:
        return StabilityConfig()
    allowed = StabilityConfig.__dataclass_fields__
    values = {key: value for key, value in mapping.items() if key in allowed}
    return StabilityConfig(**values)


def set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    """同步修改优化器所有参数组的学习率。 / Update the learning rate for all optimizer parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
