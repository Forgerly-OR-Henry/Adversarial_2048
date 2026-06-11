"""基于 PyTorch DQN 权重的敌人策略。 / Enemy strategy backed by PyTorch DQN weights."""

from __future__ import annotations

import random
from pathlib import Path

from config import get_model_path
from enemies.base_enemy import BaseEnemy, Spawn
from game.state import GameState
from models import ENEMY_ACTIONS, action_to_spawn, get_legal_spawn_actions
from models.dqn_network import DQNNetwork, board_to_tensor
from models.torch_utils import get_torch_device, load_torch_checkpoint, require_torch

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
        self.model = DQNNetwork(output_size=len(ENEMY_ACTIONS)).to(self.device)
        model_path = Path(model_path)
        if model_path.exists():
            checkpoint = load_torch_checkpoint(self.torch, model_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def select_spawn(self, state: GameState) -> Spawn:
        legal_actions = get_legal_spawn_actions(state.board)
        if not legal_actions:
            raise ValueError("No legal spawn actions available.")
        if self.rng.random() < self.epsilon:
            return action_to_spawn(self.rng.choice(legal_actions))
        legal_indexes = [ENEMY_ACTIONS.index(action) for action in legal_actions]
        with self.torch.no_grad():
            q_values = self.model(board_to_tensor(state.board, self.device))[0]
        best_index = max(legal_indexes, key=lambda index: float(q_values[index].item()))
        return action_to_spawn(ENEMY_ACTIONS[best_index])
