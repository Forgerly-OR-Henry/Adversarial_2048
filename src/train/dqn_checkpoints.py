"""DQN checkpoint 保存、加载和 state_dict 工具。 / DQN checkpoint save, load, and state_dict helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models.torch_utils import load_torch_checkpoint
from utils.serialization import checkpoint_metadata


def clone_state_dict(model) -> dict[str, Any]:
    """复制模型参数到 CPU，供稳定性控制器保存。 / Clone model parameters to CPU for stability tracking."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_state_dict_to_device(model, state_dict: dict[str, Any], device: str) -> None:
    """把参数加载到指定设备上的模型。 / Load parameters into a model on the selected device."""
    model.load_state_dict({key: value.to(device) for key, value in state_dict.items()})


def checkpoint_path(output_path: Path, suffix: str) -> Path:
    """根据最终模型路径生成旁路 checkpoint 路径。 / Build a sibling checkpoint path from the final model path."""
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def save_dqn_checkpoint(
    torch,
    path: Path,
    model,
    episodes: int,
    *,
    opponent_key: str,
    opponent_type: str,
    device: str,
    metadata: dict[str, Any],
) -> None:
    """保存 DQN 权重和简单元数据。 / Save DQN weights with simple metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "episodes": episodes,
            opponent_key: opponent_type,
            "device": device,
            "stability": checkpoint_metadata(metadata),
        },
        path,
    )


def load_dqn_checkpoint_into_model(torch, model, path: Path, device: str) -> None:
    """安全加载本项目 DQN checkpoint 到模型。 / Safely load a project DQN checkpoint into a model."""
    checkpoint = load_torch_checkpoint(torch, path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
