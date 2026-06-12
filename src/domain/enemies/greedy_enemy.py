"""枚举最坏出块的贪心敌人策略。 / Greedy enemy strategy that enumerates worst tile spawns."""

from __future__ import annotations

from domain.enemies.base_enemy import BaseEnemy, Spawn
from domain.game.board import get_empty_cells, place_tile
from domain.game.rules import evaluate_badness
from domain.game.state import GameState


class GreedyEnemy(BaseEnemy):
    """选择让玩家局面评分最低的出块。 / Chooses the spawn that minimizes the player's board score."""
    name = "greedy"

    def select_spawn(self, state: GameState) -> Spawn:
        empty_cells = get_empty_cells(state.board)
        if not empty_cells:
            raise ValueError("No empty cells available.")

        best_spawn: Spawn | None = None
        best_key: tuple[float, int, int, int] | None = None
        for row, col in empty_cells:
            for value in (2, 4):
                candidate = place_tile(state.board, row, col, value)
                badness = evaluate_badness(candidate)
                key = (badness, value, row, col)
                if best_key is None or key > best_key:
                    best_key = key
                    best_spawn = (row, col, value)

        if best_spawn is None:
            raise ValueError("No legal spawn found.")
        return best_spawn
