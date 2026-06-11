"""基于轻量 Q-learning 模型的玩家策略。 / Player strategy backed by the lightweight Q-learning model."""

from __future__ import annotations

import random
from pathlib import Path

from game.state import GameState
from models import DEFAULT_MODEL_PATH, LinearQModel
from players.base_player import BasePlayer


class QPlayer(BasePlayer):
    """使用轻量 Q 模型选择玩家移动。 / Uses the lightweight Q model to select player moves."""
    name = "q_ai"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
    ):
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.model = LinearQModel.load_or_create(model_path, rng=self.rng)

    def select_action(self, state: GameState, legal_actions: list[str]) -> str | None:
        return self.model.epsilon_greedy_action(
            state.board,
            legal_actions,
            epsilon=self.epsilon,
            rng=self.rng,
        )
