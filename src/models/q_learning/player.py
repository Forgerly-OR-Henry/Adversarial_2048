"""玩家 Q-learning 的线性特征模型。 / Linear feature model for player Q-learning."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from config import get_model_path
from game.constants import ACTIONS
from game.rules import count_merge_pairs, get_legal_actions, max_tile_in_corner, monotonicity_score

DEFAULT_MODEL_PATH = get_model_path("q_learning_player")


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


@dataclass
class LinearQModel:
    """使用线性权重估算玩家动作价值。 / Estimates player action values with linear weights."""
    weights: list[list[float]]
    actions: tuple[str, ...] = ACTIONS

    @classmethod
    def create(cls, rng: random.Random | None = None) -> "LinearQModel":
        rng = rng or random.Random()
        feature_count = len(encode_board([[0, 0, 0, 0] for _ in range(4)]))
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
            for _ in ACTIONS
        ]
        return cls(weights=weights)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "LinearQModel":
        model_path = Path(path)
        with model_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(weights=data["weights"], actions=tuple(data.get("actions", ACTIONS)))

    @classmethod
    def load_or_create(cls, path: str | Path = DEFAULT_MODEL_PATH, rng: random.Random | None = None) -> "LinearQModel":
        model_path = Path(path)
        if model_path.exists():
            return cls.load(model_path)
        return cls.create(rng=rng)

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": "linear_q",
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
        # 裁剪 TD 误差，防止单局异常奖励把轻量线性模型推得过远。
        # Clip TD error so one unusual reward does not push the lightweight linear model too far.
        error = max(-error_clip, min(error_clip, target - prediction))
        for index, feature in enumerate(features):
            self.weights[action_index][index] += learning_rate * error * feature
        return error

    def max_next_q(self, board: list[list[int]]) -> float:
        legal_actions = get_legal_actions(board)
        if not legal_actions:
            return 0.0
        q_values = self.q_values(board)
        return max(q_values[action] for action in legal_actions)
