"""Q-learning 线性模型共享实现。 / Shared implementation for linear Q-learning models."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import ClassVar

from domain.models.q_learning.player_features import encode_board


class LinearFeatureQModel:
    """基于棋盘特征和动作权重的线性 Q 模型。 / Linear Q model over board features and action weights."""

    model_type: ClassVar[str]
    default_model_path: ClassVar[Path]
    default_actions: ClassVar[tuple[str, ...]]

    weights: list[list[float]]
    actions: tuple[str, ...]

    @classmethod
    def create(cls, rng: random.Random | None = None):
        """创建随机初始化的线性模型。 / Create a randomly initialized linear model."""
        rng = rng or random.Random()
        feature_count = len(encode_board([[0, 0, 0, 0] for _ in range(4)]))
        weights = [
            [rng.uniform(-0.01, 0.01) for _ in range(feature_count)]
            for _ in cls.default_actions
        ]
        return cls(weights=weights)

    @classmethod
    def load(cls, path: str | Path | None = None):
        """从 JSON 文件加载模型。 / Load the model from a JSON file."""
        model_path = Path(path) if path is not None else cls.default_model_path
        with model_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(weights=data["weights"], actions=tuple(data.get("actions", cls.default_actions)))

    @classmethod
    def load_or_create(cls, path: str | Path | None = None, rng: random.Random | None = None):
        """存在则加载，否则创建新模型。 / Load an existing model or create a new one."""
        model_path = Path(path) if path is not None else cls.default_model_path
        if model_path.exists():
            return cls.load(model_path)
        return cls.create(rng=rng)

    def save(self, path: str | Path | None = None) -> Path:
        """把模型保存为 JSON。 / Save the model as JSON."""
        model_path = Path(path) if path is not None else self.default_model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": self.model_type,
            "version": 1,
            "actions": list(self.actions),
            "weights": self.weights,
        }
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        return model_path

    def q_values(self, board: list[list[int]]) -> dict[str, float]:
        """计算每个动作的 Q 值。 / Compute Q values for every action."""
        features = encode_board(board)
        return {
            action: sum(weight * feature for weight, feature in zip(action_weights, features))
            for action, action_weights in zip(self.actions, self.weights)
        }

    def best_action(self, board: list[list[int]], legal_actions: list[str]) -> str | None:
        """从合法动作中选择当前估值最高的动作。 / Select the best-valued legal action."""
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
        """按 epsilon-greedy 在合法动作中选择。 / Select among legal actions with epsilon-greedy."""
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
        """用 TD 目标更新指定动作权重。 / Update weights for one action from a TD target."""
        action_index = self.actions.index(action)
        features = encode_board(board)
        prediction = sum(weight * feature for weight, feature in zip(self.weights[action_index], features))
        error = max(-error_clip, min(error_clip, target - prediction))
        for index, feature in enumerate(features):
            self.weights[action_index][index] += learning_rate * error * feature
        return error

    def max_next_q(self, board: list[list[int]]) -> float:
        """返回下一局面合法动作的最大 Q 值。 / Return the maximum legal next-state Q value."""
        legal_actions = self.legal_actions(board)
        if not legal_actions:
            return 0.0
        q_values = self.q_values(board)
        return max(q_values[action] for action in legal_actions)

    @classmethod
    def legal_actions(cls, board: list[list[int]]) -> list[str]:
        """列出当前棋盘合法动作。 / List legal actions for the board."""
        raise NotImplementedError
