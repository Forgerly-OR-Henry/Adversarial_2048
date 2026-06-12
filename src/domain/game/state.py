"""不可变对局状态快照。 / Immutable game-state snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from domain.game.board import Board, copy_board, get_max_tile


@dataclass(frozen=True)
class GameState:
    """保存一局游戏的当前棋盘、分数和终止状态。 / Stores the current board, score, and terminal state for an episode."""
    board: Board
    score: int
    steps: int
    done: bool

    @property
    def max_tile(self) -> int:
        return get_max_tile(self.board)

    def copy(self) -> "GameState":
        return GameState(
            board=copy_board(self.board),
            score=self.score,
            steps=self.steps,
            done=self.done,
        )
