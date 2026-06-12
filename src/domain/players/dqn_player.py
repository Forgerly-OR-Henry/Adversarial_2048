"""基于 PyTorch DQN 权重的玩家策略。 / Player strategy backed by PyTorch DQN weights."""

from __future__ import annotations

import random
from pathlib import Path

from config import get_model_path
from domain.game.constants import ACTIONS
from domain.game.state import GameState
from domain.models.dqn_policy import best_legal_dqn_action, load_dqn_policy_model
from domain.models.torch_utils import get_torch_device, require_torch
from domain.players.base_player import BasePlayer

DEFAULT_DQN_PLAYER_PATH = get_model_path("dqn_player")


class DQNPlayer(BasePlayer):
    """加载 DQN 权重并选择玩家移动。 / Loads DQN weights and selects player moves."""
    name = "dqn_player"

    def __init__(
        self,
        model_path: str | Path = DEFAULT_DQN_PLAYER_PATH,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
        device: str | None = None,
    ):
        self.torch = require_torch()
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.device = device or get_torch_device()
        self.model = load_dqn_policy_model(self.torch, model_path, output_size=len(ACTIONS), device=self.device)

    def select_action(self, state: GameState, legal_actions: list[str]) -> str | None:
        if not legal_actions:
            return None
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)
        return best_legal_dqn_action(self.torch, self.model, state.board, self.device, ACTIONS, legal_actions)
