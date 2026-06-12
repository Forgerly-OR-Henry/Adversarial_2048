"""敌人 Q-learning 动作空间和线性模型。 / Enemy Q-learning action space and linear model."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from config import get_model_path
from domain.game.board import get_empty_cells
from domain.models.q_learning.player import encode_board

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
class EnemyQModel:
    """面向 32 个出块动作的线性 Q 模型。 / Linear Q model over the 32 enemy spawn actions."""
    weights: list[list[float]]
    actions: tuple[str, ...] = ENEMY_ACTIONS

    @classmethod
    def create(cls, rng: random.Random | None = None) -> "EnemyQModel":
        rng = rng or random.Random()
        feature_count = len(encode_board([[0, 0, 0, 0] for _ in range(4)]))
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
            for _ in ENEMY_ACTIONS
        ]
        return cls(weights=weights)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ENEMY_MODEL_PATH) -> "EnemyQModel":
        model_path = Path(path)
        with model_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(weights=data["weights"], actions=tuple(data.get("actions", ENEMY_ACTIONS)))

    @classmethod
    def load_or_create(cls, path: str | Path = DEFAULT_ENEMY_MODEL_PATH, rng: random.Random | None = None) -> "EnemyQModel":
        model_path = Path(path)
        if model_path.exists():
            return cls.load(model_path)
        return cls.create(rng=rng)

    def save(self, path: str | Path = DEFAULT_ENEMY_MODEL_PATH) -> Path:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": "enemy_linear_q",
            "version": 1,
            "actions": list(self.actions),
            "weights": self.weights,
        }
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return model_path

    def q_values(self, board: list[list[int]]) -> dict[str, float]:
        features = encode_board(board)
        return {
            action: sum(weight * feature for weight, feature in zip(action_weights, features))
            for action, action_weights in zip(self.actions, self.weights)
        }

    def best_action(self, board: list[list[int]], legal_actions: list[str]) -> str | None:
        if not legal_actions:
            return None
        q_values = self.q_values(board)
        return max(legal_actions, key=lambda action: q_values[action])

    def epsilon_greedy_action(
        self,
        board: list[list[int]],
        legal_actions: list[str],
        epsilon: float,
        rng: random.Random,
    ) -> str | None:
        if not legal_actions:
            return None
        if rng.random() < epsilon:
            return rng.choice(legal_actions)
        return self.best_action(board, legal_actions)

    def update(
        self,
        board: list[list[int]],
        action: str,
        target: float,
        learning_rate: float,
        error_clip: float = 25.0,
    ) -> float:
        action_index = self.actions.index(action)
        features = encode_board(board)
        prediction = sum(weight * feature for weight, feature in zip(self.weights[action_index], features))
        # 敌人复用玩家棋盘特征，但目标值来自“让局面变差”的奖励函数。
        # The enemy reuses player board features, but targets come from the badness reward.
        error = max(-error_clip, min(error_clip, target - prediction))
        for index, feature in enumerate(features):
            self.weights[action_index][index] += learning_rate * error * feature
        return error

    def max_next_q(self, board: list[list[int]]) -> float:
        legal_actions = get_legal_spawn_actions(board)
        if not legal_actions:
            return 0.0
        q_values = self.q_values(board)
        return max(q_values[action] for action in legal_actions)
