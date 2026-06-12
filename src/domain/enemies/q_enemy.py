"""基于轻量 Q-learning 模型的敌人策略。 / Enemy strategy backed by the lightweight Q-learning model."""

from __future__ import annotations

import random
from pathlib import Path

from domain.enemies.base_enemy import BaseEnemy, Spawn
from domain.game.state import GameState
from domain.models import DEFAULT_ENEMY_MODEL_PATH, EnemyQModel, action_to_spawn, get_legal_spawn_actions


class QEnemy(BaseEnemy):
    """使用轻量 Q 模型选择敌人出块。 / Uses the lightweight Q model to select enemy spawns."""
    name = "q_enemy"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ENEMY_MODEL_PATH,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
    ):
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.model = EnemyQModel.load_or_create(model_path, rng=self.rng)

    def select_spawn(self, state: GameState) -> Spawn:
        legal_actions = get_legal_spawn_actions(state.board)
        action = self.model.epsilon_greedy_action(
            state.board,
            legal_actions,
            epsilon=self.epsilon,
            rng=self.rng,
        )
        if action is None:
            raise ValueError("No legal spawn actions available.")
        return action_to_spawn(action)
