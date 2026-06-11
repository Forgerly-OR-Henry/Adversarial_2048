"""JSON 序列化与项目内路径清洗工具。 / JSON serialization and project-path sanitizing helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT


def project_relative_path(value: Path) -> str:
    """把项目内路径写成相对路径。 / Store project-local paths as relative paths."""
    try:
        return value.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        if value.is_absolute():
            return value.name
        return value.as_posix()


def sanitize_string(value: str) -> str:
    """移除日志和 info 中泄露的本机项目绝对路径。 / Remove local absolute project paths from logs and info."""
    project_windows = str(PROJECT_ROOT)
    project_posix = PROJECT_ROOT.as_posix()
    cleaned = value.replace(project_windows, ".").replace(project_posix, ".")
    if re.fullmatch(r"[A-Za-z]:[\\/].+", cleaned):
        return Path(cleaned).name
    return cleaned


def json_ready(value: Any, *, sanitize_paths: bool = False) -> Any:
    """转换为 JSON 兼容值，可选清洗本机路径。 / Convert a value into JSON-compatible data."""
    if isinstance(value, Path):
        return project_relative_path(value) if sanitize_paths else str(value)
    if isinstance(value, str):
        return sanitize_string(value) if sanitize_paths else value
    if is_dataclass(value):
        return json_ready(asdict(value), sanitize_paths=sanitize_paths)
    if isinstance(value, dict):
        return {str(key): json_ready(item, sanitize_paths=sanitize_paths) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item, sanitize_paths=sanitize_paths) for item in value]
    return value


def checkpoint_metadata(value: Any) -> Any:
    """转换为 PyTorch weights_only 可安全读取的元数据。 / Convert metadata for PyTorch weights_only checkpoints."""
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str):
        return value
    if is_dataclass(value):
        return checkpoint_metadata(asdict(value))
    if isinstance(value, dict):
        return {str(key): checkpoint_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [checkpoint_metadata(item) for item in value]
    return value
