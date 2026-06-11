"""集中加载 YAML 配置并提供路径与默认值访问函数。 / Centralized YAML config loading with accessors for paths and defaults."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"

MODEL_FILENAMES = {
    "q_learning_player": "player_q_model.json",
    "q_learning_enemy": "enemy_q_model.json",
    "dqn_player": "player_dqn_model.pt",
    "dqn_enemy": "enemy_dqn_model.pt",
}


def _parse_scalar(value: str) -> Any:
    normalized = value.strip()
    if normalized in ("", "null", "Null", "NULL", "~"):
        return None
    if normalized in ('""', "''"):
        return ""
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in ("'", '"')
    ):
        return normalized[1:-1]
    if normalized.lower() == "true":
        return True
    if normalized.lower() == "false":
        return False
    try:
        return int(normalized)
    except ValueError:
        pass
    try:
        return float(normalized)
    except ValueError:
        return normalized


def _next_content_indent(lines: list[str], start_index: int) -> int | None:
    for raw_line in lines[start_index + 1 :]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return len(raw_line) - len(raw_line.lstrip(" "))
    return None


def _load_simple_yaml(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line}")

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        # 这个 fallback 只支持当前配置需要的“缩进字典 + 标量”，不是完整 YAML 实现。
        # This fallback supports only the current configs' indented mappings and scalars, not full YAML.
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_indent = _next_content_indent(lines, index)
        if next_indent is not None and next_indent > indent:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = None

    return root


@lru_cache(maxsize=None)
def _load_config_cached(relative_path: str) -> dict[str, Any]:
    config_path = CONFIG_ROOT / relative_path
    with config_path.open("r", encoding="utf-8") as handle:
        text = handle.read()

    try:
        import yaml
    except ModuleNotFoundError:
        # 薄环境没有 PyYAML 时仍可跑测试；生产环境优先使用 yaml.safe_load。
        # Tests can still run without PyYAML; production paths prefer yaml.safe_load.
        data = _load_simple_yaml(text)
    else:
        loaded = yaml.safe_load(text)
        data = loaded if loaded is not None else {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def load_config(relative_path: str) -> dict[str, Any]:
    """读取配置文件并返回可安全修改的深拷贝。 / Load a config file and return a safely mutable deep copy."""
    return copy.deepcopy(_load_config_cached(relative_path))


def get_config_value(relative_path: str, *keys: str, default: Any = None) -> Any:
    """按键路径读取配置值，缺失时返回默认值。 / Read a nested config value by key path, falling back to a default."""
    value: Any = load_config(relative_path)
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return copy.deepcopy(value)


def get_model_directory(name: str) -> Path:
    """返回指定模型类型的目录路径。 / Return the directory path for a model type."""
    value = get_config_value("paths.yaml", "models", name)
    if value is None:
        raise KeyError(f"Missing model directory config: {name}")
    return Path(value)


def get_model_path(name: str) -> Path:
    """返回指定模型类型的默认模型文件路径。 / Return the default model file path for a model type."""
    if name not in MODEL_FILENAMES:
        raise KeyError(f"Missing model filename mapping: {name}")
    return get_model_directory(name) / MODEL_FILENAMES[name]


def get_log_directory() -> Path:
    """返回系统日志根目录。 / Return the root directory for system logs."""
    value = get_config_value("paths.yaml", "logs", "directory", default="logs")
    return Path(value)


def get_experiment_directory() -> Path:
    """返回评估实验 CSV 输出目录。 / Return the output directory for evaluation CSV files."""
    value = get_config_value("paths.yaml", "outputs", "experiments_directory", default="outputs/experiments")
    return Path(value)


def get_training_results_log_path() -> Path:
    """返回训练成果 JSONL 日志路径。 / Return the JSONL log path for training results."""
    return get_log_directory() / "system" / "training_log.jsonl"


def get_evaluation_results_log_path() -> Path:
    """返回评估成果 JSONL 日志路径。 / Return the JSONL log path for evaluation results."""
    return get_log_directory() / "system" / "evaluation_log.jsonl"


def get_error_log_path() -> Path:
    """返回异常日志路径。 / Return the error-log path."""
    directory = get_config_value("paths.yaml", "logs", "errors_directory")
    if directory is not None:
        return Path(directory) / "log.jsonl"
    return get_log_directory() / "errors" / "log.jsonl"


def get_train_defaults(name: str) -> dict[str, Any]:
    """读取指定训练类型的默认参数。 / Load default parameters for a training type."""
    return load_config(f"train/{name}.yaml")


def get_tuning_defaults() -> dict[str, Any]:
    """读取自动调参默认参数。 / Load default parameters for auto-tuning."""
    return load_config("train/tuning.yaml")


def get_evaluate_defaults() -> dict[str, Any]:
    """读取评估默认参数。 / Load default parameters for evaluation."""
    return load_config("evaluate/default.yaml")


def get_ui_defaults() -> dict[str, Any]:
    """读取 GUI 默认参数。 / Load default parameters for the GUI."""
    return load_config("ui/default.yaml")
