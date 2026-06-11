"""命令行输出格式化工具。 / Formatting helpers for CLI output."""

from __future__ import annotations

from typing import Any

from utils.serialization import json_ready as _json_ready


def json_ready(value: Any) -> Any:
    """把路径和 dataclass 等对象转换为可 JSON 序列化的值。 / Convert paths and dataclasses into JSON-serializable values."""
    return _json_ready(value)
