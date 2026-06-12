"""基于局面启发式评分的贪心玩家。 / Greedy player based on heuristic board evaluation."""

from __future__ import annotations

from domain.game.rules import evaluate_player_board, move
from domain.game.state import GameState
from domain.players.base_player import BasePlayer


class HeuristicPlayer(BasePlayer):
    """选择即时启发式评分最高的移动。 / Chooses the move with the best immediate heuristic score."""
    name = "heuristic"

    def select_action(self, state: GameState, legal_actions: list[str]) -> str | None:
        if not legal_actions:
            return None

        best_action = legal_actions[0]
        best_score = float("-inf")
        for action in legal_actions:
            result = move(state.board, action)
            if not result.moved:
                continue
            score = evaluate_player_board(result.board, score_delta=result.score_delta)
            if score > best_score:
                best_action = action
                best_score = score
        return best_action
