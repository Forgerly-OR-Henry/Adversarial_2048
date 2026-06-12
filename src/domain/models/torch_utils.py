"""PyTorch 可用性检查和设备选择工具。 / PyTorch availability checks and device selection helpers."""

from __future__ import annotations

from pathlib import Path

from utils.serialization import checkpoint_metadata


TORCH_INSTALL_HINT = (
    "PyTorch is required for deep reinforcement learning. "
    "Install it first, for example: pip install torch"
)


def require_torch():
    """导入 PyTorch，缺失时给出明确错误。 / Import PyTorch and raise a clear error when it is missing."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(TORCH_INSTALL_HINT) from exc
    return torch


def get_torch_device(prefer_cuda: bool = True) -> str:
    """选择可用的 CUDA 或 CPU 训练设备。 / Select an available CUDA or CPU training device."""
    torch = require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_torch_checkpoint(torch, path: str | Path, map_location: str | None = None):
    """按安全 weights_only 模式加载本项目 DQN checkpoint。 / Load project DQN checkpoints in safe weights_only mode."""
    return torch.load(path, map_location=map_location, weights_only=True)


__all__ = [
    "TORCH_INSTALL_HINT",
    "checkpoint_metadata",
    "get_torch_device",
    "load_torch_checkpoint",
    "require_torch",
]
