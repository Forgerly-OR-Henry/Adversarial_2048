"""Q-learning 棋盘特征编码。 / Board feature encoding for Q-learning."""

from __future__ import annotations

import math

from domain.game.rules import count_merge_pairs, max_tile_in_corner, monotonicity_score


def tile_log2(value: int) -> float:
    """把方块值转换为稳定的 log2 特征。 / Convert a tile value into a stable log2 feature."""
    if value <= 0:
        return 0.0
    return math.log2(value)


def encode_board(board: list[list[int]]) -> list[float]:
    """把棋盘转换为线性 Q 模型特征。 / Convert a board into features for the linear Q model."""
    # 4x4 原始方块使用 log2 归一化，避免大块数值支配线性模型。
    # Raw 4x4 tiles use normalized log2 values so large tiles do not dominate the linear model.
    flat = [tile_log2(value) / 16.0 for row in board for value in row]
    empty_count = sum(1 for row in board for value in row if value == 0)
    max_tile = max(max(row) for row in board)
    features = flat + [
        empty_count / 16.0,
        count_merge_pairs(board) / 24.0,
        monotonicity_score(board) / 24.0,
        1.0 if max_tile_in_corner(board) else 0.0,
        tile_log2(max_tile) / 16.0,
        1.0,
    ]
    return features
