"""PyTorch DQN 网络和棋盘张量编码。 / PyTorch DQN network and board tensor encoding."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from domain.models.q_learning import encode_board
from domain.models.torch_utils import TORCH_INSTALL_HINT


def _ensure_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError(TORCH_INSTALL_HINT)


class DQNNetwork(nn.Module if nn is not None else object):
    """用于玩家或敌人的小型 MLP DQN 网络。 / Small MLP DQN network used by either player or enemy."""
    def __init__(self, input_size: int = 22, output_size: int = 4, hidden_size: int = 128):
        _ensure_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def board_to_tensor(board: list[list[int]], device: str = "cpu"):
    """把棋盘编码为神经网络输入张量。 / Encode one board as a neural-network input tensor."""
    _ensure_torch()
    return torch.tensor([encode_board(board)], dtype=torch.float32, device=device)


def batch_boards_to_tensor(boards: list[list[list[int]]], device: str = "cpu"):
    """批量编码棋盘为张量。 / Encode a batch of boards as a tensor."""
    _ensure_torch()
    return torch.tensor([encode_board(board) for board in boards], dtype=torch.float32, device=device)
