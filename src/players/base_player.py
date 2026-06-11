"""玩家策略接口定义。 / Interface definition for player strategies."""

from __future__ import annotations

from game.state import GameState


class BasePlayer:
    """所有玩家策略的最小接口。 / Minimal interface shared by all player strategies."""
    name = "base"

    def select_action(self, state: GameState, legal_actions: list[str]) -> str | None:
        raise NotImplementedError
