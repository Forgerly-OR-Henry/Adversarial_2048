"""敌人策略接口定义。 / Interface definition for enemy strategies."""

from __future__ import annotations

from game.state import GameState

Spawn = tuple[int, int, int]


class BaseEnemy:
    """所有敌人策略的最小接口。 / Minimal interface shared by all enemy strategies."""
    name = "base"

    def select_spawn(self, state: GameState) -> Spawn:
        raise NotImplementedError
