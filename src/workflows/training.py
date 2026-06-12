"""训练工作流参数、产物选项和输出路径解析。 / Training workflow parameters, artifacts, and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import get_train_defaults
from domain.train.artifacts import (
    TRAINING_MODEL_FILENAMES,
    default_training_output_path,
    latest_training_output_path,
    list_incomplete_training_artifacts,
    list_training_artifacts,
    model_path_from_info,
    resolve_info_path,
    training_info_status,
)

REFERENCE_TYPE_INITIAL_WEIGHTS = "initial_weights"
# REFERENCE_TYPE_DISTILLATION = "distillation"  # Reserved until distillation training is implemented.
REFERENCE_TYPE_OPTIONS = (REFERENCE_TYPE_INITIAL_WEIGHTS,)

TRAIN_DEFAULTS = {
    "player_q": get_train_defaults("player_q"),
    "enemy_q": get_train_defaults("enemy_q"),
    "player_dqn": get_train_defaults("player_dqn"),
    "enemy_dqn": get_train_defaults("enemy_dqn"),
}


@dataclass(frozen=True)
class TrainingReferenceOption:
    """可作为训练起始权重的模型。 / Model selectable as initial training weights."""

    training_type: str
    path: Path
    created_at: str
    status: str


@dataclass(frozen=True)
class TrainingResumeOption:
    """可继续训练的未完成产物。 / Incomplete artifact selectable for continued training."""

    training_type: str
    artifact: dict[str, Any]
    created_at: str
    completed_episodes: int
    target_episodes: int


def default_training_output_directory(training_type: str) -> Path:
    """返回可编辑的训练输出目录。 / Return the editable training output directory."""
    return default_training_output_path(training_type).parent


def training_output_model_path(training_type: str, output_directory: str | Path) -> Path:
    """把输出目录解析为固定模型文件路径。 / Resolve an output directory to the fixed model path."""
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


def build_training_reference_options(training_type: str) -> list[TrainingReferenceOption]:
    """构造可作为参考模型的训练产物。 / Build artifacts selectable as reference models."""
    options: list[TrainingReferenceOption] = []
    latest_path = latest_training_output_path(training_type)
    if latest_path.exists():
        options.append(
            TrainingReferenceOption(
                training_type=training_type,
                path=latest_path,
                created_at="latest",
                status="latest",
            )
        )

    for artifact in list_training_artifacts(include_incomplete=False):
        if str(artifact.get("training_type", "")) != training_type:
            continue
        status = training_info_status(artifact)
        if status != "completed":
            continue
        model_path = model_path_from_info(artifact)
        if model_path is None or not model_path.exists():
            continue
        options.append(
            TrainingReferenceOption(
                training_type=training_type,
                path=model_path,
                created_at=str(artifact.get("created_at") or model_path.parent.name),
                status=status,
            )
        )
    return options


def build_training_resume_options(training_type: str) -> list[TrainingResumeOption]:
    """构造可继续训练的未完成产物。 / Build incomplete artifacts selectable for continued training."""
    options: list[TrainingResumeOption] = []
    for artifact in list_incomplete_training_artifacts(training_type):
        info_path = resolve_info_path(artifact.get("info_path"))
        if info_path is None:
            continue
        completed = int(artifact.get("completed_episodes") or 0)
        target = int(artifact.get("target_episodes") or completed)
        options.append(
            TrainingResumeOption(
                training_type=training_type,
                artifact=artifact,
                created_at=str(artifact.get("created_at") or info_path.parent.name),
                completed_episodes=completed,
                target_episodes=target,
            )
        )
    return options


def default_enemy_type_for_algorithm(algorithm_key: str) -> str:
    """返回玩家训练默认敌人。 / Return the default enemy for player training."""
    return TRAIN_DEFAULTS[f"player_{algorithm_key}"]["enemy"]


def default_player_type_for_algorithm(algorithm_key: str) -> str:
    """返回敌人训练默认玩家。 / Return the default player for enemy training."""
    return TRAIN_DEFAULTS[f"enemy_{algorithm_key}"]["player"]


__all__ = [
    "REFERENCE_TYPE_INITIAL_WEIGHTS",
    "REFERENCE_TYPE_OPTIONS",
    "TrainingReferenceOption",
    "TrainingResumeOption",
    "build_training_reference_options",
    "build_training_resume_options",
    "default_enemy_type_for_algorithm",
    "default_player_type_for_algorithm",
    "default_training_output_directory",
    "training_output_model_path",
]
