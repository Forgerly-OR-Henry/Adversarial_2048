"""标准随机出块敌人策略。 / Standard random tile-spawning enemy strategy."""

from __future__ import annotations

import random

from domain.enemies.base_enemy import BaseEnemy, Spawn
from domain.game.board import get_empty_cells
from domain.game.constants import STANDARD_FOUR_PROBABILITY
from domain.game.state import GameState


class RandomEnemy(BaseEnemy):
    """按 2048 标准概率随机选择空格和值。 / Randomly selects an empty cell and value using standard 2048 probabilities."""
    name = "random"

    def __init__(self, p_four: float = STANDARD_FOUR_PROBABILITY, rng: random.Random | None = None):
        self.p_four = p_four
        self.rng = rng or random.Random()

    def select_spawn(self, state: GameState) -> Spawn:
        empty_cells = get_empty_cells(state.board)
        if not empty_cells:
            raise ValueError("No empty cells available.")
        row, col = self.rng.choice(empty_cells)
        value = 4 if self.rng.random() < self.p_four else 2
        return row, col, value
