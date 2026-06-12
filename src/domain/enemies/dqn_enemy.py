"""基于 PyTorch DQN 权重的敌人策略。 / Enemy strategy backed by PyTorch DQN weights."""

from __future__ import annotations

import random
from pathlib import Path

from config import get_model_path
from domain.enemies.base_enemy import BaseEnemy, Spawn
from domain.game.state import GameState
from domain.models import ENEMY_ACTIONS, action_to_spawn, get_legal_spawn_actions
from domain.models.dqn_policy import best_legal_dqn_action, load_dqn_policy_model
from domain.models.torch_utils import get_torch_device, require_torch

DEFAULT_DQN_ENEMY_PATH = get_model_path("dqn_enemy")


class DQNEnemy(BaseEnemy):
    """加载 DQN 权重并选择敌人出块动作。 / Loads DQN weights and selects enemy spawn actions."""
    name = "dqn_enemy"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_DQN_ENEMY_PATH,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
        device: str | None = None,
    ):
        self.torch = require_torch()
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.device = device or get_torch_device()
        self.model = load_dqn_policy_model(self.torch, model_path, output_size=len(ENEMY_ACTIONS), device=self.device)

    def select_spawn(self, state: GameState) -> Spawn:
        legal_actions = get_legal_spawn_actions(state.board)
        if not legal_actions:
            raise ValueError("No legal spawn actions available.")
        if self.rng.random() < self.epsilon:
            return action_to_spawn(self.rng.choice(legal_actions))
        action = best_legal_dqn_action(self.torch, self.model, state.board, self.device, ENEMY_ACTIONS, legal_actions)
        if action is None:
            raise ValueError("No legal spawn actions available.")
        return action_to_spawn(action)
