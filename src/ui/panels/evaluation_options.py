"""单项评估面板的选项、请求和输出路径解析。 / Options, requests, and output paths for evaluation UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation import default_experiment_path, experiment_csv_filename
from train.artifacts import TRAINING_MODEL_FILENAMES, latest_training_output_path, list_training_artifacts
from ui.settings.options import ENEMY_LABELS, ENEMY_TYPES_BY_LABEL, PLAYER_LABELS, PLAYER_TYPES_BY_LABEL, TRAINING_TYPE_LABELS

NO_MODEL_LABEL = "暂无可用模型"

EVALUATION_TARGET_LABELS = {
    "auto_player": "自动玩家",
    "player": "玩家模型",
    "auto_enemy": "自动敌人",
    "enemy": "敌对模型",
}
EVALUATION_TARGETS_BY_LABEL = {label: key for key, label in EVALUATION_TARGET_LABELS.items()}
AUTO_PLAYER_TYPES = ("heuristic", "q_ai", "dqn_player")
AUTO_ENEMY_TYPES = tuple(ENEMY_LABELS)
TRAINING_TYPE_TO_EVALUATION_TYPE = {
    "player_q": ("player", "q_ai"),
    "player_dqn": ("player", "dqn_player"),
    "enemy_q": ("enemy", "q_enemy"),
    "enemy_dqn": ("enemy", "dqn_enemy"),
}
EVALUATION_OUTPUT_FILE_SUFFIXES = {".csv", ".jsonl"}


@dataclass(frozen=True)
class EvaluationModelOption:
    """评估面板中的可选模型项。 / Selectable model item in the evaluation panel."""
    label: str
    path: Path | None
    training_type: str
    role: str
    evaluation_type: str
    created_at: str


@dataclass(frozen=True)
class SingleEvaluationRequest:
    """一次单项评估任务的完整参数。 / Complete parameters for one single-model evaluation task."""
    player_type: str
    enemy_type: str
    player_model_path: Path | None
    enemy_model_path: Path | None


def default_single_evaluation_output_directory(player_type: str, enemy_type: str) -> Path:
    """返回 GUI 显示和可编辑的评估输出目录。 / Return the editable GUI evaluation output directory."""
    return default_experiment_path(player_type, enemy_type).parent


def single_evaluation_output_csv_path(player_type: str, enemy_type: str, output_directory: str | Path) -> Path:
    """把 GUI 输出目录解析为固定 CSV 路径。 / Resolve a GUI output directory to the fixed CSV path."""
    text = str(output_directory).strip()
    if not text:
        raise ValueError("输出目录不能为空。")
    directory = Path(text)
    if directory.suffix.lower() in EVALUATION_OUTPUT_FILE_SUFFIXES:
        raise ValueError("输出目录不能填写 CSV 或日志文件名，CSV 和日志文件名由系统生成。")
    return directory / experiment_csv_filename(player_type, enemy_type)


def default_evaluation_pair_for_empty_selection(role: str, opponent_label: str) -> tuple[str, str]:
    """在下拉选项尚未构造时推导默认评估双方。 / Infer default sides before options are built."""
    if role == "player":
        return "selected_player_model", ENEMY_TYPES_BY_LABEL[opponent_label]
    if role == "auto_player":
        return AUTO_PLAYER_TYPES[0], ENEMY_TYPES_BY_LABEL[opponent_label]
    if role == "enemy":
        return PLAYER_TYPES_BY_LABEL[opponent_label], "selected_enemy_model"
    if role == "auto_enemy":
        return PLAYER_TYPES_BY_LABEL[opponent_label], AUTO_ENEMY_TYPES[0]
    raise ValueError(f"Unsupported evaluation role: {role}")


def training_type_role(training_type: str) -> str:
    """返回训练类型在评估中的玩家或敌人角色。 / Return whether a training type acts as player or enemy in evaluation."""
    return TRAINING_TYPE_TO_EVALUATION_TYPE[training_type][0]


def training_type_evaluation_type(training_type: str) -> str:
    """返回训练类型对应的评估策略类型。 / Return the evaluation strategy type for a training type."""
    return TRAINING_TYPE_TO_EVALUATION_TYPE[training_type][1]


def build_evaluation_model_options() -> list[EvaluationModelOption]:
    """从训练产物构造评估下拉选项。 / Build evaluation dropdown options from training artifacts."""
    options: list[EvaluationModelOption] = []
    seen: set[tuple[str, str]] = set()
    # latest 模型排在历史产物前面，方便评估当前默认加载的 AI。
    # latest models are added before historical artifacts for quick evaluation of default AIs.
    for training_type in TRAINING_MODEL_FILENAMES:
        latest_path = latest_training_output_path(training_type)
        if latest_path.exists():
            _append_model_option(
                options,
                seen,
                training_type=training_type,
                model_path=latest_path,
                created_at="latest",
            )

    for artifact in list_training_artifacts():
        training_type = str(artifact.get("training_type", ""))
        if training_type not in TRAINING_TYPE_TO_EVALUATION_TYPE:
            continue
        model_path = _artifact_model_path(artifact)
        if model_path is None or not model_path.exists():
            continue
        _append_model_option(
            options,
            seen,
            training_type=training_type,
            model_path=model_path,
            created_at=str(artifact.get("created_at") or model_path.parent.name),
        )
    return options


def build_automatic_player_options() -> list[EvaluationModelOption]:
    """构造和对局自动玩家一致的评估选项。 / Build evaluation options matching gameplay auto players."""
    return [
        EvaluationModelOption(
            label=PLAYER_LABELS[player_type],
            path=None,
            training_type=player_type,
            role="auto_player",
            evaluation_type=player_type,
            created_at="builtin",
        )
        for player_type in AUTO_PLAYER_TYPES
    ]


def build_automatic_enemy_options() -> list[EvaluationModelOption]:
    """构造和对局敌人一致的评估选项。 / Build evaluation options matching gameplay enemies."""
    return [
        EvaluationModelOption(
            label=ENEMY_LABELS[enemy_type],
            path=None,
            training_type=enemy_type,
            role="auto_enemy",
            evaluation_type=enemy_type,
            created_at="builtin",
        )
        for enemy_type in AUTO_ENEMY_TYPES
    ]


def resolve_single_evaluation_request(
    option: EvaluationModelOption,
    opponent_label: str,
) -> SingleEvaluationRequest:
    """把界面输入解析为可运行评估请求。 / Resolve panel inputs into an executable evaluation request."""
    if option.role == "player":
        return SingleEvaluationRequest(
            player_type=option.evaluation_type,
            enemy_type=ENEMY_TYPES_BY_LABEL[opponent_label],
            player_model_path=option.path,
            enemy_model_path=None,
        )
    if option.role == "enemy":
        return SingleEvaluationRequest(
            player_type=PLAYER_TYPES_BY_LABEL[opponent_label],
            enemy_type=option.evaluation_type,
            player_model_path=None,
            enemy_model_path=option.path,
        )
    if option.role == "auto_player":
        return SingleEvaluationRequest(
            player_type=option.evaluation_type,
            enemy_type=ENEMY_TYPES_BY_LABEL[opponent_label],
            player_model_path=None,
            enemy_model_path=None,
        )
    if option.role == "auto_enemy":
        return SingleEvaluationRequest(
            player_type=PLAYER_TYPES_BY_LABEL[opponent_label],
            enemy_type=option.evaluation_type,
            player_model_path=None,
            enemy_model_path=None,
        )
    raise ValueError(f"Unsupported evaluation option role: {option.role}")


def _append_model_option(
    options: list[EvaluationModelOption],
    seen: set[tuple[str, str]],
    *,
    training_type: str,
    model_path: Path,
    created_at: str,
) -> None:
    role, evaluation_type = TRAINING_TYPE_TO_EVALUATION_TYPE[training_type]
    key = (training_type, str(model_path))
    if key in seen:
        return
    seen.add(key)
    label = _unique_model_label(
        options,
        f"{TRAINING_TYPE_LABELS[training_type]} | {_display_timestamp(created_at)}",
    )
    options.append(
        EvaluationModelOption(
            label=label,
            path=model_path,
            training_type=training_type,
            role=role,
            evaluation_type=evaluation_type,
            created_at=created_at,
        )
    )


def _artifact_model_path(artifact: dict[str, Any]) -> Path | None:
    value = artifact.get("model_path")
    if not value:
        return None
    model_path = Path(str(value))
    if model_path.exists():
        return model_path
    info_path = artifact.get("info_path")
    if info_path:
        sibling = Path(str(info_path)).parent / model_path.name
        if sibling.exists():
            return sibling
    return model_path


def _display_timestamp(value: str) -> str:
    return value.replace("T", " ") if value else "unknown"


def _unique_model_label(options: list[EvaluationModelOption], base_label: str) -> str:
    existing = {option.label for option in options}
    if base_label not in existing:
        return base_label
    index = 2
    while f"{base_label} #{index}" in existing:
        index += 1
    return f"{base_label} #{index}"
