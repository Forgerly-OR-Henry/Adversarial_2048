"""随机合法动作玩家策略。 / Player strategy that chooses random legal moves."""

from __future__ import annotations

import random

from domain.game.state import GameState
from domain.players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    """随机选择一个合法移动。 / Chooses one legal move at random."""
    name = "random"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def select_action(self, state: GameState, legal_actions: list[str]) -> str | None:
        if not legal_actions:
            return None
        return self.rng.choice(legal_actions)
