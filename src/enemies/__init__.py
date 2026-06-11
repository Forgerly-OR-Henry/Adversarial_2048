"""敌人策略工厂和公开类型入口。 / Enemy strategy factory and public type entrypoint."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from enemies.base_enemy import BaseEnemy
from enemies.greedy_enemy import GreedyEnemy
from enemies.q_enemy import QEnemy
from enemies.random_enemy import RandomEnemy

if TYPE_CHECKING:
    from enemies.dqn_enemy import DQNEnemy


def create_enemy(
    name: str,
    rng: random.Random | None = None,
    model_path: str | Path | None = None,
) -> BaseEnemy:
    """按类型创建敌人策略实例。 / Create an enemy strategy instance by type."""
    normalized = name.lower()
    if normalized == "random":
        return RandomEnemy(rng=rng)
    if normalized == "greedy":
        return GreedyEnemy()
    if normalized == "q_enemy":
        if model_path is None:
            return QEnemy(rng=rng)
        return QEnemy(model_path=model_path, rng=rng)
    if normalized == "dqn_enemy":
        from enemies.dqn_enemy import DQNEnemy

        if model_path is None:
            return DQNEnemy(rng=rng)
        return DQNEnemy(model_path=model_path, rng=rng)
    raise ValueError(f"Unknown enemy: {name}")


def __getattr__(name: str):
    if name == "DQNEnemy":
        from enemies.dqn_enemy import DQNEnemy

        return DQNEnemy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseEnemy",
    "RandomEnemy",
    "GreedyEnemy",
    "QEnemy",
    "DQNEnemy",
    "create_enemy",
]
