"""单项评估工作流选项、请求和输出路径解析。 / Single-evaluation workflow options, requests, and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from domain.evaluation import default_experiment_path, experiment_csv_filename
from domain.train.artifacts import (
    TRAINING_MODEL_FILENAMES,
    latest_training_output_path,
    list_training_artifacts,
    model_path_from_info,
)

EVALUATION_TARGETS = ("auto_player", "player", "auto_enemy", "enemy")
AUTO_PLAYER_TYPES = ("heuristic", "q_ai", "dqn_player")
AUTO_ENEMY_TYPES = ("random", "greedy", "q_enemy", "dqn_enemy")
TRAINING_TYPE_TO_EVALUATION_TYPE = {
    "player_q": ("player", "q_ai"),
    "player_dqn": ("player", "dqn_player"),
    "enemy_q": ("enemy", "q_enemy"),
    "enemy_dqn": ("enemy", "dqn_enemy"),
}
EVALUATION_OUTPUT_FILE_SUFFIXES = {".csv", ".jsonl"}


@dataclass(frozen=True)
class EvaluationModelOption:
    """可转换为一次单项评估请求的模型或内置策略。 / Model or built-in strategy for one evaluation."""

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
    """返回可编辑的评估输出目录。 / Return the editable evaluation output directory."""
    return default_experiment_path(player_type, enemy_type).parent


def single_evaluation_output_csv_path(player_type: str, enemy_type: str, output_directory: str | Path) -> Path:
    """把输出目录解析为固定 CSV 路径。 / Resolve an output directory to the fixed CSV path."""
    text = str(output_directory).strip()
    if not text:
        raise ValueError("输出目录不能为空。")
    directory = Path(text)
    if directory.suffix.lower() in EVALUATION_OUTPUT_FILE_SUFFIXES:
        raise ValueError("输出目录不能填写 CSV 或日志文件名，CSV 和日志文件名由系统生成。")
    return directory / experiment_csv_filename(player_type, enemy_type)


def default_evaluation_pair_for_empty_selection(role: str, opponent_type: str) -> tuple[str, str]:
    """在可选项尚未构造时推导默认评估双方。 / Infer default sides before options are built."""
    if role == "player":
        return "selected_player_model", opponent_type
    if role == "auto_player":
        return AUTO_PLAYER_TYPES[0], opponent_type
    if role == "enemy":
        return opponent_type, "selected_enemy_model"
    if role == "auto_enemy":
        return opponent_type, AUTO_ENEMY_TYPES[0]
    raise ValueError(f"Unsupported evaluation role: {role}")


def training_type_role(training_type: str) -> str:
    """返回训练类型在评估中的玩家或敌人角色。 / Return whether a training type acts as player or enemy."""
    return TRAINING_TYPE_TO_EVALUATION_TYPE[training_type][0]


def training_type_evaluation_type(training_type: str) -> str:
    """返回训练类型对应的评估策略类型。 / Return the evaluation strategy type for a training type."""
    return TRAINING_TYPE_TO_EVALUATION_TYPE[training_type][1]


def build_evaluation_model_options() -> list[EvaluationModelOption]:
    """从训练产物构造可评估模型选项。 / Build model options from training artifacts."""
    options: list[EvaluationModelOption] = []
    seen: set[tuple[str, str]] = set()
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
        model_path = model_path_from_info(artifact)
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
    """构造内置自动玩家评估选项。 / Build built-in automatic player evaluation options."""
    return [
        EvaluationModelOption(
            path=None,
            training_type=player_type,
            role="auto_player",
            evaluation_type=player_type,
            created_at="builtin",
        )
        for player_type in AUTO_PLAYER_TYPES
    ]


def build_automatic_enemy_options() -> list[EvaluationModelOption]:
    """构造内置自动敌人评估选项。 / Build built-in automatic enemy evaluation options."""
    return [
        EvaluationModelOption(
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
    opponent_type: str,
) -> SingleEvaluationRequest:
    """把选项和对手类型解析为可运行评估请求。 / Resolve an option and opponent type into a request."""
    if option.role == "player":
        return SingleEvaluationRequest(
            player_type=option.evaluation_type,
            enemy_type=opponent_type,
            player_model_path=option.path,
            enemy_model_path=None,
        )
    if option.role == "enemy":
        return SingleEvaluationRequest(
            player_type=opponent_type,
            enemy_type=option.evaluation_type,
            player_model_path=None,
            enemy_model_path=option.path,
        )
    if option.role == "auto_player":
        return SingleEvaluationRequest(
            player_type=option.evaluation_type,
            enemy_type=opponent_type,
            player_model_path=None,
            enemy_model_path=None,
        )
    if option.role == "auto_enemy":
        return SingleEvaluationRequest(
            player_type=opponent_type,
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
    options.append(
        EvaluationModelOption(
            path=model_path,
            training_type=training_type,
            role=role,
            evaluation_type=evaluation_type,
            created_at=created_at,
        )
    )


__all__ = [
    "AUTO_ENEMY_TYPES",
    "AUTO_PLAYER_TYPES",
    "EVALUATION_TARGETS",
    "EvaluationModelOption",
    "SingleEvaluationRequest",
    "build_automatic_enemy_options",
    "build_automatic_player_options",
    "build_evaluation_model_options",
    "default_evaluation_pair_for_empty_selection",
    "default_single_evaluation_output_directory",
    "resolve_single_evaluation_request",
    "single_evaluation_output_csv_path",
    "training_type_evaluation_type",
    "training_type_role",
]
