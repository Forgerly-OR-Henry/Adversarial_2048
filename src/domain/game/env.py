"""2048 对局环境，负责移动、出块和状态推进。 / 2048 game environment responsible for moves, spawns, and state transitions."""

from __future__ import annotations

import random
from typing import Optional

from domain.game.board import Board, copy_board, create_empty_board, get_empty_cells, place_tile
from domain.game.constants import DEFAULT_INITIAL_TILES
from domain.game.rules import get_legal_actions, is_game_over, move
from domain.game.state import GameState


class GameEnv:
    """封装 2048 对局生命周期和敌人出块。 / Encapsulates the 2048 episode lifecycle and enemy spawns."""
    def __init__(self, enemy=None, seed: Optional[int] = None, initial_tiles: int = DEFAULT_INITIAL_TILES):
        self.rng = random.Random(seed)
        self.seed = seed
        self.initial_tiles = initial_tiles
        self.enemy = enemy
        self.board = create_empty_board()
        self.score = 0
        self.steps = 0
        self.done = False

    def _enemy(self):
        if self.enemy is None:
            from domain.enemies.random_enemy import RandomEnemy

            self.enemy = RandomEnemy(rng=self.rng)
        return self.enemy

    def reset(self) -> GameState:
        self.board = create_empty_board()
        self.score = 0
        self.steps = 0
        self.done = False
        for _ in range(self.initial_tiles):
            self.spawn_enemy_tile()
        return self.snapshot()

    def snapshot(self) -> GameState:
        return GameState(
            board=copy_board(self.board),
            score=self.score,
            steps=self.steps,
            done=self.done,
        )

    def get_legal_actions(self) -> list[str]:
        """返回当前棋盘所有合法移动。 / Return all legal moves for the current board."""
        return get_legal_actions(self.board)

    def is_game_over(self) -> bool:
        """判断棋盘是否没有空格且无合法移动。 / Check whether the board has no empty cells and no legal moves."""
        return is_game_over(self.board)

    def spawn_enemy_tile(self) -> bool:
        if not get_empty_cells(self.board):
            return False
        row, col, value = self._enemy().select_spawn(self.snapshot())
        self.board = place_tile(self.board, row, col, value)
        return True

    def step(self, action: str) -> GameState:
        if self.done:
            return self.snapshot()

        result = move(self.board, action)
        if not result.moved:
            return self.snapshot()

        self.board = result.board
        self.score += result.score_delta
        self.steps += 1

        if get_empty_cells(self.board):
            self.spawn_enemy_tile()

        self.done = is_game_over(self.board)
        return self.snapshot()

    def set_board(self, board: Board, score: int = 0, steps: int = 0) -> GameState:
        self.board = copy_board(board)
        self.score = score
        self.steps = steps
        self.done = is_game_over(self.board)
        return self.snapshot()
