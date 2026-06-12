"""集中加载 YAML 配置并提供路径与默认值访问函数。 / Centralized YAML config loading with accessors for paths and defaults."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"

MODEL_FILENAMES = {
    "q_learning_player": "player_q_model.json",
    "q_learning_enemy": "enemy_q_model.json",
    "dqn_player": "player_dqn_model.pt",
    "dqn_enemy": "enemy_dqn_model.pt",
}


@lru_cache(maxsize=None)
def _load_config_cached(relative_path: str) -> dict[str, Any]:
    config_path = CONFIG_ROOT / relative_path
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
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
