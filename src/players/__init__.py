"""玩家策略工厂和公开类型入口。 / Player strategy factory and public type entrypoint."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from players.base_player import BasePlayer
from players.heuristic_player import HeuristicPlayer
from players.human_player import HumanPlayer
from players.q_player import QPlayer
from players.random_player import RandomPlayer

if TYPE_CHECKING:
    from players.dqn_player import DQNPlayer


def create_player(
    name: str,
    rng: random.Random | None = None,
    model_path: str | Path | None = None,
) -> BasePlayer:
    """按类型创建玩家策略实例。 / Create a player strategy instance by type."""
    normalized = name.lower()
    if normalized == "human":
        return HumanPlayer()
    if normalized == "random":
        return RandomPlayer(rng=rng)
    if normalized == "heuristic":
        return HeuristicPlayer()
    if normalized == "q_ai":
        if model_path is None:
            return QPlayer(rng=rng)
        return QPlayer(model_path=model_path, rng=rng)
    if normalized == "dqn_player":
        from players.dqn_player import DQNPlayer

        if model_path is None:
            return DQNPlayer(rng=rng)
        return DQNPlayer(model_path=model_path, rng=rng)
    raise ValueError(f"Unknown player: {name}")


def __getattr__(name: str):
    if name == "DQNPlayer":
        from players.dqn_player import DQNPlayer

        return DQNPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BasePlayer",
    "HumanPlayer",
    "RandomPlayer",
    "HeuristicPlayer",
    "QPlayer",
    "DQNPlayer",
    "create_player",
]
