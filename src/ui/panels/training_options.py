"""训练面板的模型选项和输出路径解析。 / Model options and output-path parsing for the training panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import get_train_defaults
from train.artifacts import (
    TRAINING_MODEL_FILENAMES,
    default_training_output_path,
    latest_training_output_path,
    list_incomplete_training_artifacts,
    list_training_artifacts,
    model_path_from_info,
    resolve_info_path,
    training_info_status,
)
from ui.settings.options import TRAINING_TYPE_LABELS

NO_REFERENCE_LABEL = "不使用参考模型"
NO_RESUME_LABEL = "不继续训练"
REFERENCE_TYPE_INITIAL_WEIGHTS = "起始权重"
# REFERENCE_TYPE_DISTILLATION = "蒸馏"  # 预留扩展：蒸馏训练实现后再启用。
REFERENCE_TYPE_OPTIONS = (REFERENCE_TYPE_INITIAL_WEIGHTS,)

TRAIN_DEFAULTS = {
    "player_q": get_train_defaults("player_q"),
    "enemy_q": get_train_defaults("enemy_q"),
    "player_dqn": get_train_defaults("player_dqn"),
    "enemy_dqn": get_train_defaults("enemy_dqn"),
}


def default_training_output_directory(training_type: str) -> Path:
    """返回 GUI 显示和可编辑的训练输出目录。 / Return the editable GUI output directory."""
    return default_training_output_path(training_type).parent


def training_output_model_path(training_type: str, output_directory: str | Path) -> Path:
    """把 GUI 输出目录解析为固定模型文件路径。 / Resolve GUI output directory to the fixed model path."""
    text = str(output_directory).strip()
    if not text:
        raise ValueError("输出目录不能为空。")
    directory = Path(text)
    filename = TRAINING_MODEL_FILENAMES[training_type]
    model_suffixes = {Path(name).suffix for name in TRAINING_MODEL_FILENAMES.values()}
    if directory.name == filename or directory.suffix in model_suffixes:
        raise ValueError("输出目录不能填写模型文件名，模型、info 和日志文件名由系统生成。")
    if directory.name == "latest":
        raise ValueError("latest 目录由系统发布管理，请选择普通输出目录。")
    return directory / filename


def build_training_reference_options(training_type: str) -> dict[str, Path | None]:
    """构造参考模型下拉选项。 / Build reference-model options."""
    options: dict[str, Path | None] = {NO_REFERENCE_LABEL: None}
    latest_path = latest_training_output_path(training_type)
    if latest_path.exists():
        _append_model_option(options, training_type, "latest", latest_path, status="latest")

    for artifact in list_training_artifacts(include_incomplete=False):
        if str(artifact.get("training_type", "")) != training_type:
            continue
        status = training_info_status(artifact)
        if status != "completed":
            continue
        model_path = model_path_from_info(artifact)
        if model_path is None or not model_path.exists():
            continue
        created_at = str(artifact.get("created_at") or model_path.parent.name)
        _append_model_option(options, training_type, created_at, model_path, status=status)
    return options


def build_training_resume_options(training_type: str) -> dict[str, dict[str, Any] | None]:
    """构造继续训练下拉选项。 / Build resume-run options."""
    options: dict[str, dict[str, Any] | None] = {NO_RESUME_LABEL: None}
    for artifact in list_incomplete_training_artifacts(training_type):
        info_path = resolve_info_path(artifact.get("info_path"))
        if info_path is None:
            continue
        created_at = str(artifact.get("created_at") or info_path.parent.name)
        completed = int(artifact.get("completed_episodes") or 0)
        target = int(artifact.get("target_episodes") or completed)
        label = _unique_option_label(
            options,
            f"{TRAINING_TYPE_LABELS.get(training_type, training_type)} | 未完成 {completed}/{target} | {_display_timestamp(created_at)}",
        )
        options[label] = artifact
    return options


def default_enemy_type_for_algorithm(algorithm_key: str) -> str:
    """返回玩家训练默认敌人。 / Return the default enemy for player training."""
    return TRAIN_DEFAULTS[f"player_{algorithm_key}"]["enemy"]


def default_player_type_for_algorithm(algorithm_key: str) -> str:
    """返回敌人训练默认玩家。 / Return the default player for enemy training."""
    return TRAIN_DEFAULTS[f"enemy_{algorithm_key}"]["player"]


def _append_model_option(
    options: dict[str, Path | None],
    training_type: str,
    created_at: str,
    model_path: Path,
    *,
    status: str,
) -> None:
    suffix = "latest" if status == "latest" else _display_timestamp(created_at)
    label = _unique_option_label(
        options,
        f"{TRAINING_TYPE_LABELS.get(training_type, training_type)} | {suffix}",
    )
    options[label] = model_path


def _unique_option_label(options: dict[str, Any], base_label: str) -> str:
    if base_label not in options:
        return base_label
    index = 2
    while f"{base_label} #{index}" in options:
        index += 1
    return f"{base_label} #{index}"


def _display_timestamp(value: str) -> str:
    return value.replace("T", " ") if value else "unknown"
