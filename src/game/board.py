"""2048 棋盘数据结构和基础操作。 / 2048 board data structure helpers and primitive operations."""

from __future__ import annotations

from typing import Iterable

from game.constants import BOARD_SIZE, EMPTY

Board = list[list[int]]
Cell = tuple[int, int]


def create_empty_board(size: int = BOARD_SIZE) -> Board:
    """创建空 2048 棋盘。 / Create an empty 2048 board."""
    return [[EMPTY for _ in range(size)] for _ in range(size)]


def copy_board(board: Board) -> Board:
    """复制棋盘，避免调用方共享内部列表。 / Copy a board so callers do not share internal lists."""
    return [row[:] for row in board]


def get_empty_cells(board: Board) -> list[Cell]:
    """列出所有空格坐标。 / List coordinates of all empty cells."""
    return [
        (row_index, col_index)
        for row_index, row in enumerate(board)
        for col_index, value in enumerate(row)
        if value == EMPTY
    ]


def get_max_tile(board: Board) -> int:
    """返回棋盘上的最大方块值。 / Return the largest tile value on the board."""
    return max(max(row) for row in board)


def place_tile(board: Board, row: int, col: int, value: int) -> Board:
    """在指定位置放置新方块并返回新棋盘。 / Place a tile at a coordinate and return a new board."""
    if board[row][col] != EMPTY:
        raise ValueError(f"Cell ({row}, {col}) is not empty.")
    if value not in (2, 4):
        raise ValueError("Only 2 or 4 can be spawned.")
    new_board = copy_board(board)
    new_board[row][col] = value
    return new_board


def boards_equal(left: Board, right: Board) -> bool:
    """比较两个棋盘是否完全一致。 / Compare whether two boards are identical."""
    return left == right


def iter_cells(board: Board) -> Iterable[tuple[int, int, int]]:
    """按行列顺序遍历棋盘单元格。 / Iterate board cells in row-column order."""
    for row_index, row in enumerate(board):
        for col_index, value in enumerate(row):
            yield row_index, col_index, value


def format_board(board: Board) -> str:
    """生成命令行可读的棋盘文本。 / Build a terminal-readable board string."""
    width = max(4, len(str(get_max_tile(board))))
    line = "+" + "+".join("-" * (width + 2) for _ in board[0]) + "+"
    rows = [line]
    for row in board:
        values = [str(value if value else ".").rjust(width) for value in row]
        rows.append("| " + " | ".join(values) + " |")
        rows.append(line)
    return "\n".join(rows)
