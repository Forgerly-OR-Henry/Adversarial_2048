"""玩家 Q-learning 的线性特征模型。 / Linear feature model for player Q-learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from config import get_model_path
from domain.game.constants import ACTIONS
from domain.game.rules import get_legal_actions
from domain.models.q_learning.linear import LinearFeatureQModel
from domain.models.q_learning.player_features import encode_board, tile_log2

DEFAULT_MODEL_PATH = get_model_path("q_learning_player")


@dataclass
class LinearQModel(LinearFeatureQModel):
    """使用线性权重估算玩家动作价值。 / Estimates player action values with linear weights."""

    model_type: ClassVar[str] = "linear_q"
    default_model_path: ClassVar = DEFAULT_MODEL_PATH
    default_actions: ClassVar[tuple[str, ...]] = ACTIONS

    weights: list[list[float]]
    actions: tuple[str, ...] = ACTIONS

    @classmethod
    def legal_actions(cls, board: list[list[int]]) -> list[str]:
        return get_legal_actions(board)
