"""敌人 Q-learning 动作空间和线性模型。 / Enemy Q-learning action space and linear model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from config import get_model_path
from domain.game.board import get_empty_cells
from domain.models.q_learning.linear import LinearFeatureQModel

DEFAULT_ENEMY_MODEL_PATH = get_model_path("q_learning_enemy")

ENEMY_ACTIONS = tuple(
    f"{row},{col},{value}"
    for row in range(4)
    for col in range(4)
    for value in (2, 4)
)


def spawn_to_action(row: int, col: int, value: int) -> str:
    """把敌人出块三元组转换为动作索引。 / Convert an enemy spawn tuple into an action index."""
    return f"{row},{col},{value}"


def action_to_spawn(action: str) -> tuple[int, int, int]:
    """把敌人动作索引还原为出块三元组。 / Convert an enemy action index back to a spawn tuple."""
    row, col, value = action.split(",")
    return int(row), int(col), int(value)


def get_legal_spawn_actions(board: list[list[int]]) -> list[str]:
    """列出当前棋盘所有合法敌人出块动作。 / List all legal enemy spawn actions for the board."""
    # 敌人动作空间固定为 16 个格子 x 2/4 两种值，训练时再按空格过滤非法动作。
    # The enemy action space is fixed at 16 cells x values 2/4, then filtered by empty cells.
    return [
        spawn_to_action(row, col, value)
        for row, col in get_empty_cells(board)
        for value in (2, 4)
    ]


@dataclass
class EnemyQModel(LinearFeatureQModel):
    """面向 32 个出块动作的线性 Q 模型。 / Linear Q model over the 32 enemy spawn actions."""

    model_type: ClassVar[str] = "enemy_linear_q"
    default_model_path: ClassVar = DEFAULT_ENEMY_MODEL_PATH
    default_actions: ClassVar[tuple[str, ...]] = ENEMY_ACTIONS

    weights: list[list[float]]
    actions: tuple[str, ...] = ENEMY_ACTIONS

    @classmethod
    def legal_actions(cls, board: list[list[int]]) -> list[str]:
        return get_legal_spawn_actions(board)
