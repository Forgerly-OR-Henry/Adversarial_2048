"""DQN 训练入口和支撑工具。 / DQN training entrypoints and support tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.train.dqn.enemy import EnemyDQNTrainingSummary
    from domain.train.dqn.player import DQNTrainingSummary


def train_dqn_enemy(*args, **kwargs):
    """训练敌人 DQN 并保存产物。 / Train the enemy DQN and save artifacts."""
    from domain.train.dqn.enemy import train_dqn_enemy as _train_dqn_enemy

    return _train_dqn_enemy(*args, **kwargs)


def train_dqn_player(*args, **kwargs):
    """训练玩家 DQN 并保存产物。 / Train the player DQN and save artifacts."""
    from domain.train.dqn.player import train_dqn_player as _train_dqn_player

    return _train_dqn_player(*args, **kwargs)


def __getattr__(name: str):
    if name == "DQNTrainingSummary":
        from domain.train.dqn.player import DQNTrainingSummary

        return DQNTrainingSummary
    if name == "EnemyDQNTrainingSummary":
        from domain.train.dqn.enemy import EnemyDQNTrainingSummary

        return EnemyDQNTrainingSummary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DQNTrainingSummary",
    "EnemyDQNTrainingSummary",
    "train_dqn_enemy",
    "train_dqn_player",
]
