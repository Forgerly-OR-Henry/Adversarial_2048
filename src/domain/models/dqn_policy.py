"""DQN 策略运行时共享工具。 / Shared runtime helpers for DQN policies."""

from __future__ import annotations

from pathlib import Path

from domain.models.dqn_network import DQNNetwork, board_to_tensor
from domain.models.torch_utils import load_torch_checkpoint


def load_dqn_policy_model(torch, model_path: str | Path, output_size: int, device: str) -> DQNNetwork:
    """加载 DQN 策略网络；文件不存在时保留随机初始化权重。 / Load a DQN policy network or keep initialized weights."""
    model = DQNNetwork(output_size=output_size).to(device)
    path = Path(model_path)
    if path.exists():
        checkpoint = load_torch_checkpoint(torch, path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    model.eval()
    return model


def best_legal_dqn_action(
    torch,
    model: DQNNetwork,
    board: list[list[int]],
    device: str,
    actions: tuple[str, ...],
    legal_actions: list[str],
) -> str | None:
    """从合法动作中选择 DQN 估值最高的动作。 / Select the highest-valued legal action from a DQN."""
    if not legal_actions:
        return None
    legal_indexes = [actions.index(action) for action in legal_actions]
    with torch.no_grad():
        q_values = model(board_to_tensor(board, device))[0]
    best_index = max(legal_indexes, key=lambda index: float(q_values[index].item()))
    return actions[best_index]
