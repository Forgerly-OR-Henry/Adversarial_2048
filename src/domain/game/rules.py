"""2048 移动合并规则和启发式局面评分。 / 2048 merge rules and heuristic board evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from domain.game.board import Board, copy_board, get_empty_cells, get_max_tile
from domain.game.constants import ACTIONS, DOWN, LEFT, RIGHT, UP


@dataclass(frozen=True)
class MoveResult:
    """一次移动后的棋盘、得分增量和是否移动。 / Board, score delta, and movement flag after one move."""
    board: Board
    score_delta: int
    moved: bool


def merge_line_left(line: list[int]) -> tuple[list[int], int]:
    """按 2048 规则把一行向左压缩并合并。 / Compress and merge one row to the left using 2048 rules."""
    # 2048 的一次合并只能消费相邻两个相同值；合并后继续扫描剩余值。
    # One 2048 merge consumes only two adjacent equal values; scanning then continues.
    values = [value for value in line if value != 0]
    merged: list[int] = []
    score_delta = 0
    index = 0
    while index < len(values):
        if index + 1 < len(values) and values[index] == values[index + 1]:
            new_value = values[index] * 2
            merged.append(new_value)
            score_delta += new_value
            index += 2
        else:
            merged.append(values[index])
            index += 1
    merged.extend([0] * (len(line) - len(merged)))
    return merged, score_delta


def transpose(board: Board) -> Board:
    """转置棋盘，用于复用横向移动逻辑。 / Transpose a board to reuse horizontal move logic."""
    return [list(row) for row in zip(*board)]


def reverse_rows(board: Board) -> Board:
    """反转每一行，用于复用向左合并逻辑。 / Reverse every row to reuse left-merge logic."""
    return [list(reversed(row)) for row in board]


def move(board: Board, action: str) -> MoveResult:
    """执行一个方向移动并返回移动结果。 / Apply one directional move and return the result."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")

    working = copy_board(board)
    restore_reverse = False
    restore_transpose = False

    # 先把任意方向转换为“向左合并”，再按相反顺序还原棋盘方向。
    # Normalize every move into a left merge, then restore board orientation in reverse order.
    if action in (UP, DOWN):
        working = transpose(working)
        restore_transpose = True
    if action in (RIGHT, DOWN):
        working = reverse_rows(working)
        restore_reverse = True

    score_delta = 0
    moved_rows: Board = []
    for row in working:
        merged, row_score = merge_line_left(row)
        moved_rows.append(merged)
        score_delta += row_score

    if restore_reverse:
        moved_rows = reverse_rows(moved_rows)
    if restore_transpose:
        moved_rows = transpose(moved_rows)

    return MoveResult(
        board=moved_rows,
        score_delta=score_delta,
        moved=moved_rows != board,
    )


def can_move(board: Board, action: str) -> bool:
    """判断指定方向是否能改变棋盘。 / Check whether a direction changes the board."""
    return move(board, action).moved


def get_legal_actions(board: Board) -> list[str]:
    """返回当前棋盘所有合法移动。 / Return all legal moves for the current board."""
    return [action for action in ACTIONS if can_move(board, action)]


def is_game_over(board: Board) -> bool:
    """判断棋盘是否没有空格且无合法移动。 / Check whether the board has no empty cells and no legal moves."""
    return not get_empty_cells(board) and not get_legal_actions(board)


def count_merge_pairs(board: Board) -> int:
    """统计相邻且可合并的方块对。 / Count adjacent pairs that can merge."""
    pairs = 0
    size = len(board)
    for row in range(size):
        for col in range(size):
            value = board[row][col]
            if value == 0:
                continue
            if col + 1 < size and board[row][col + 1] == value:
                pairs += 1
            if row + 1 < size and board[row + 1][col] == value:
                pairs += 1
    return pairs


def max_tile_in_corner(board: Board) -> bool:
    """判断最大块是否位于角落。 / Check whether the largest tile is in a corner."""
    max_tile = get_max_tile(board)
    last = len(board) - 1
    return any(board[row][col] == max_tile for row, col in ((0, 0), (0, last), (last, 0), (last, last)))


def monotonicity_score(board: Board) -> int:
    """计算行列单调趋势得分。 / Compute a score for monotonic row and column trends."""
    total = 0
    lines = board + transpose(board)
    for line in lines:
        increasing = 0
        decreasing = 0
        values = [value for value in line if value != 0]
        for left, right in zip(values, values[1:]):
            if left <= right:
                increasing += 1
            if left >= right:
                decreasing += 1
        total += max(increasing, decreasing)
    return total


def evaluate_player_board(board: Board, score_delta: int = 0) -> float:
    """从玩家角度给棋盘打启发式分。 / Score a board heuristically from the player's perspective."""
    empty_cells = len(get_empty_cells(board))
    merge_pairs = count_merge_pairs(board)
    max_tile = get_max_tile(board)
    corner_bonus = 300 if max_tile_in_corner(board) else 0
    # 玩家评分偏好“空间、可合并性、单调布局、最大块和当步得分”。
    # Player scoring favors space, mergeability, monotonic layout, max tile, and immediate score.
    return (
        empty_cells * 120
        + merge_pairs * 60
        + monotonicity_score(board) * 35
        + max_tile * 1.5
        + score_delta
        + corner_bonus
    )


def evaluate_badness(board: Board) -> float:
    """从敌人角度评估棋盘对玩家的压迫程度。 / Evaluate how punishing a board is from the enemy perspective."""
    empty_cells = len(get_empty_cells(board))
    merge_pairs = count_merge_pairs(board)
    corner_penalty = 0 if max_tile_in_corner(board) else 180
    # 敌人评分与玩家偏好相反：更满、更乱、可合并机会更少的棋盘更“坏”。
    # Enemy scoring inverts player preferences: fuller, messier boards with fewer merges are worse.
    return (
        (16 - empty_cells) * 140
        + (12 - monotonicity_score(board)) * 30
        + (8 - merge_pairs) * 65
        + corner_penalty
    )
