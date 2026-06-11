"""基于 PyTorch DQN 权重的玩家策略。 / Player strategy backed by PyTorch DQN weights."""

from __future__ import annotations

import random
from pathlib import Path

from config import get_model_path
from game.constants import ACTIONS
from game.state import GameState
from models.dqn_network import DQNNetwork, board_to_tensor
from models.torch_utils import get_torch_device, load_torch_checkpoint, require_torch
from players.base_player import BasePlayer

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
        self.model = DQNNetwork(output_size=len(ACTIONS)).to(self.device)
        model_path = Path(model_path)
        if model_path.exists():
            checkpoint = load_torch_checkpoint(self.torch, model_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def select_action(self, state: GameState, legal_actions: list[str]) -> str | None:
        if not legal_actions:
            return None
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)
        legal_indexes = [ACTIONS.index(action) for action in legal_actions]
        with self.torch.no_grad():
            q_values = self.model(board_to_tensor(state.board, self.device))[0]
        best_index = max(legal_indexes, key=lambda index: float(q_values[index].item()))
        return ACTIONS[best_index]
