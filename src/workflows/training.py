"""训练工作流参数、产物选项和输出路径解析。 / Training workflow parameters, artifacts, and paths."""

from __future__ import annotations

from collections.abc import Callable
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


@dataclass(frozen=True)
class TrainingRunRequest:
    """一次 GUI 训练任务的规范化请求。 / Normalized request for one GUI training run."""

    target: str
    algorithm: str
    enemy_type: str
    player_type: str
    episodes: int | None
    seed: int | None
    output: str | Path | None
    reference_model_path: Path | None
    resume_run_path: Path | None
    stop_event: Any = None

    @property
    def training_type(self) -> str:
        """返回训练产物类型键。 / Return the training artifact type key."""
        if self.target not in ("player", "enemy"):
            raise ValueError(f"Unsupported training target: {self.target}")
        if self.algorithm not in ("q", "dqn"):
            raise ValueError(f"Unsupported training algorithm: {self.algorithm}")
        return f"{self.target}_{self.algorithm}"


@dataclass(frozen=True)
class TrainingProgress:
    """后台训练进度事件。 / Background training progress event."""

    target: str
    current: int
    total: int | None
    max_tile: int
    score: int
    epsilon: float
    device: str
    board: list[list[int]]
    steps: int


TrainingProgressCallback = Callable[[TrainingProgress], None]


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


def parse_training_episodes(value: str) -> int | None:
    """解析 GUI 输入的训练局数；空值表示无限训练。 / Parse GUI episode input; blank means unlimited."""
    text = value.strip()
    if not text:
        return None
    try:
        episodes = int(text)
    except (TypeError, ValueError):
        raise ValueError("训练局数必须是正整数，或留空表示无限训练。") from None
    if episodes < 1:
        raise ValueError("训练局数至少为 1，或留空表示无限训练。")
    return episodes


def parse_training_seed(value: str) -> int | None:
    """解析 GUI 输入的随机种子。 / Parse GUI seed input."""
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError("随机种子必须留空或填写整数。") from None


def validate_reference_type(reference_type: str) -> None:
    """校验当前支持的参考类型。 / Validate the currently supported reference type."""
    if reference_type != REFERENCE_TYPE_INITIAL_WEIGHTS:
        raise ValueError("当前仅支持使用参考模型作为起始权重。")


def validate_resume_episodes(episodes: int | None, resume_artifact: dict[str, Any] | None) -> None:
    """校验继续训练的目标局数。 / Validate total episodes for a resumed run."""
    if resume_artifact is None:
        return
    completed = int(resume_artifact.get("completed_episodes") or 0)
    if episodes is not None and episodes < completed:
        raise ValueError(f"继续训练总局数不能低于当前已训练局数 {completed}。")


def training_request_log_parameters(request: TrainingRunRequest) -> dict[str, Any]:
    """返回错误日志可安全记录的训练请求字段。 / Return log-safe training request fields."""
    return {
        "training_type": request.training_type,
        "target": request.target,
        "algorithm": request.algorithm,
        "enemy_type": request.enemy_type,
        "player_type": request.player_type,
        "episodes": request.episodes,
        "seed": request.seed,
        "output": request.output,
        "reference_model_path": request.reference_model_path,
        "resume_run_path": request.resume_run_path,
    }


def run_training_request(
    request: TrainingRunRequest,
    *,
    progress_callback: TrainingProgressCallback | None = None,
) -> Any:
    """执行一次训练请求，并把底层进度回调归一化。 / Run a training request with normalized progress events."""
    # Keep imports lazy so DQN/Torch paths are loaded only when a DQN run is actually executed.
    from domain.train import train_dqn_enemy, train_dqn_player, train_q_enemy, train_q_player

    request.training_type
    q_progress = _q_progress_callback(request, progress_callback)
    dqn_progress = _dqn_progress_callback(request, progress_callback)
    if request.target == "enemy" and request.algorithm == "dqn":
        return train_dqn_enemy(
            episodes=request.episodes,
            player_type=request.player_type,
            seed=request.seed,
            output=request.output,
            reference_model_path=request.reference_model_path,
            resume_run_path=request.resume_run_path,
            stop_event=request.stop_event,
            progress_callback=dqn_progress,
        )
    if request.target == "enemy":
        return train_q_enemy(
            episodes=request.episodes,
            player_type=request.player_type,
            seed=request.seed,
            output=request.output,
            reference_model_path=request.reference_model_path,
            resume_run_path=request.resume_run_path,
            stop_event=request.stop_event,
            progress_callback=q_progress,
        )
    if request.algorithm == "dqn":
        return train_dqn_player(
            episodes=request.episodes,
            enemy_type=request.enemy_type,
            seed=request.seed,
            output=request.output,
            reference_model_path=request.reference_model_path,
            resume_run_path=request.resume_run_path,
            stop_event=request.stop_event,
            progress_callback=dqn_progress,
        )
    return train_q_player(
        episodes=request.episodes,
        enemy_type=request.enemy_type,
        seed=request.seed,
        output=request.output,
        reference_model_path=request.reference_model_path,
        resume_run_path=request.resume_run_path,
        stop_event=request.stop_event,
        progress_callback=q_progress,
    )


def _q_progress_callback(
    request: TrainingRunRequest,
    progress_callback: TrainingProgressCallback | None,
):
    if progress_callback is None:
        return None

    def callback(current: int, total: int | None, state: Any, epsilon: float) -> None:
        _emit_training_progress(request, progress_callback, current, total, state, epsilon, "cpu")

    return callback


def _dqn_progress_callback(
    request: TrainingRunRequest,
    progress_callback: TrainingProgressCallback | None,
):
    if progress_callback is None:
        return None

    def callback(current: int, total: int | None, state: Any, epsilon: float, device: str) -> None:
        _emit_training_progress(request, progress_callback, current, total, state, epsilon, device)

    return callback


def _emit_training_progress(
    request: TrainingRunRequest,
    progress_callback: TrainingProgressCallback,
    current: int,
    total: int | None,
    state: Any,
    epsilon: float,
    device: str,
) -> None:
    progress_callback(
        TrainingProgress(
            target=request.target,
            current=current,
            total=total,
            max_tile=state.max_tile,
            score=state.score,
            epsilon=epsilon,
            device=device,
            board=[row[:] for row in state.board],
            steps=state.steps,
        )
    )


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
    "TrainingProgress",
    "TrainingProgressCallback",
    "TrainingRunRequest",
    "build_training_reference_options",
    "build_training_resume_options",
    "default_enemy_type_for_algorithm",
    "default_player_type_for_algorithm",
    "default_training_output_directory",
    "parse_training_episodes",
    "parse_training_seed",
    "run_training_request",
    "training_request_log_parameters",
    "training_output_model_path",
    "validate_reference_type",
    "validate_resume_episodes",
]
