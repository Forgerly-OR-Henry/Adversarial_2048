"""训练面板下拉选项映射。 / Dropdown option mapping for the training panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ui.panels.shared import display_timestamp, unique_label
from ui.settings.options import NO_REFERENCE_LABEL, NO_RESUME_LABEL, TRAINING_TYPE_LABELS
from workflows.training import build_training_reference_options, build_training_resume_options


def reference_option_map(training_type: str) -> dict[str, Path | None]:
    """构造参考模型下拉标签到模型路径的映射。 / Map reference labels to model paths."""
    options: dict[str, Path | None] = {NO_REFERENCE_LABEL: None}
    for option in build_training_reference_options(training_type):
        suffix = "latest" if option.status == "latest" else display_timestamp(option.created_at)
        label = unique_label(
            options,
            f"{TRAINING_TYPE_LABELS.get(option.training_type, option.training_type)} | {suffix}",
        )
        options[label] = option.path
    return options


def resume_option_map(training_type: str) -> dict[str, dict[str, Any] | None]:
    """构造继续训练下拉标签到未完成产物的映射。 / Map resume labels to incomplete artifacts."""
    options: dict[str, dict[str, Any] | None] = {NO_RESUME_LABEL: None}
    for option in build_training_resume_options(training_type):
        label = unique_label(
            options,
            (
                f"{TRAINING_TYPE_LABELS.get(option.training_type, option.training_type)} | "
                f"未完成 {option.completed_episodes}/{option.target_episodes} | "
                f"{display_timestamp(option.created_at)}"
            ),
        )
        options[label] = option.artifact
    return options


__all__ = ["reference_option_map", "resume_option_map"]
