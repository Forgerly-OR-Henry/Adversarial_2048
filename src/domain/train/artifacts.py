"""训练产物路径、信息文件和查询工具。 / Training artifact paths, info files, and lookup helpers."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, get_train_defaults
from utils.serialization import json_ready, project_relative_path
from utils.training_log import log_training_result


TRAINING_MODEL_FILENAMES = {
    "player_q": "player_q_model.json",
    "enemy_q": "enemy_q_model.json",
    "player_dqn": "player_dqn_model.pt",
    "enemy_dqn": "enemy_dqn_model.pt",
}

TRAINING_STATUS_COMPLETED = "completed"
TRAINING_STATUS_INCOMPLETE = "incomplete"
TRAINING_INFO_FILENAME = "info.json"
TRAINING_RUN_LOG_FILENAME = "log.jsonl"


def training_timestamp() -> str:
    """生成训练产物目录使用的时间戳。 / Generate the timestamp used for training artifact directories."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_training_output_path(training_type: str) -> Path:
    """生成训练类型的默认历史输出路径。 / Build the default historical output path for a training type."""
    defaults = get_train_defaults(training_type)
    run_directory = _next_available_run_directory(Path(defaults["runs_directory"]))
    return run_directory / TRAINING_MODEL_FILENAMES[training_type]


def training_run_log_path(run_directory: Path | None, model_path: Path | None = None) -> Path | None:
    """返回一次训练产物旁的 JSONL 日志路径。 / Return the per-run JSONL log path."""
    if run_directory is not None:
        return run_directory / TRAINING_RUN_LOG_FILENAME
    if model_path is not None:
        return model_path.parent / TRAINING_RUN_LOG_FILENAME
    return None


def training_info_status(info: dict[str, Any] | None) -> str:
    """读取训练信息状态；缺少状态时默认视为完成。 / Read info status; missing statuses are completed."""
    if not isinstance(info, dict):
        return TRAINING_STATUS_COMPLETED
    status = str(info.get("status") or TRAINING_STATUS_COMPLETED)
    if status == TRAINING_STATUS_INCOMPLETE:
        return TRAINING_STATUS_INCOMPLETE
    if _info_has_unfinished_progress(info):
        return TRAINING_STATUS_INCOMPLETE
    if status == TRAINING_STATUS_COMPLETED:
        return TRAINING_STATUS_COMPLETED
    return TRAINING_STATUS_COMPLETED


def is_incomplete_training_info(info: dict[str, Any] | None) -> bool:
    """判断训练信息是否为未完成状态。 / Return whether training info is incomplete."""
    return training_info_status(info) == TRAINING_STATUS_INCOMPLETE


def _info_has_unfinished_progress(info: dict[str, Any]) -> bool:
    try:
        completed = int(info.get("completed_episodes") or 0)
        target = int(info.get("target_episodes") or 0)
    except (TypeError, ValueError):
        return False
    return target > 0 and completed < target


def _next_available_run_directory(runs_directory: Path) -> Path:
    while True:
        run_directory = runs_directory / training_timestamp()
        if not run_directory.exists():
            return run_directory
        time.sleep(0.05)


def latest_training_output_path(training_type: str) -> Path:
    """返回训练类型的 latest 输出路径。 / Return the latest output path for a training type."""
    defaults = get_train_defaults(training_type)
    filename = TRAINING_MODEL_FILENAMES[training_type]
    directory = defaults.get("latest_output_directory")
    if directory is None:
        raise KeyError(f"Missing latest_output_directory config: {training_type}")
    return Path(directory) / filename


def resolve_training_output(
    training_type: str,
    output: str | Path | None,
) -> tuple[Path, Path | None]:
    """解析显式输出或默认训练输出路径。 / Resolve explicit or default training output paths."""
    if output is not None and str(output).strip():
        output_path = Path(output)
        _ensure_not_latest_output(training_type, output_path)
        return output_path, _run_directory_for_explicit_output(training_type, output_path)

    output_path = default_training_output_path(training_type)
    return output_path, output_path.parent


def _ensure_not_latest_output(training_type: str, output_path: Path) -> None:
    """禁止训练直接写 latest 路径。 / Prevent training from writing directly to latest paths."""
    latest_path = latest_training_output_path(training_type)
    latest_directory = latest_path.parent
    try:
        resolved_output = output_path.resolve()
        resolved_latest_directory = latest_directory.resolve()
        resolved_output.relative_to(resolved_latest_directory)
    except (OSError, ValueError):
        return
    raise ValueError("latest 目录只能在结果管理中手动指定。")


def _run_directory_for_explicit_output(training_type: str, output_path: Path) -> Path | None:
    """自动生成的时间戳输出即使作为文本传入，也应写入 info。 / Keep timestamp outputs info-backed."""
    runs_directory = Path(get_train_defaults(training_type)["runs_directory"])
    parent = output_path.parent
    try:
        resolved_parent = parent.resolve()
        resolved_runs = runs_directory.resolve()
        resolved_parent.relative_to(resolved_runs)
    except (OSError, ValueError):
        return None
    if resolved_parent == resolved_runs or parent.name == "latest":
        return None
    return parent


def write_training_info(
    *,
    training_type: str,
    model_path: Path,
    run_directory: Path,
    parameters: dict[str, Any],
    summary: Any,
    source: str = "training",
    extra: dict[str, Any] | None = None,
    status: str = TRAINING_STATUS_COMPLETED,
    target_episodes: int | None = None,
    completed_episodes: int | None = None,
    reference_model_path: str | Path | None = None,
    resume_run_path: str | Path | None = None,
    run_log_path: str | Path | None = None,
    include_target_episodes: bool = False,
) -> Path:
    """写出训练产物信息文件。 / Write an info file for a training artifact."""
    run_directory.mkdir(parents=True, exist_ok=True)
    info_path = run_directory / TRAINING_INFO_FILENAME
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "training_type": training_type,
        "status": status,
        "model_path": model_path,
        "parameters": parameters,
        "summary": summary,
    }
    if target_episodes is not None or include_target_episodes:
        payload["target_episodes"] = target_episodes
    if completed_episodes is not None:
        payload["completed_episodes"] = completed_episodes
    if reference_model_path is not None:
        payload["reference_model_path"] = reference_model_path
    if resume_run_path is not None:
        payload["resume_run_path"] = resume_run_path
    if run_log_path is not None:
        payload["run_log_path"] = run_log_path
    if extra:
        payload["extra"] = extra
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload, sanitize_paths=True), handle, ensure_ascii=False, indent=2, sort_keys=True)
    return info_path


def replace_latest_model(source_model_path: Path, latest_model_path: Path) -> bool:
    """替换 latest 目录中的模型文件。 / Replace the model file inside a latest directory."""
    latest_directory = latest_model_path.parent
    try:
        source_resolved = source_model_path.resolve()
        latest_resolved = latest_model_path.resolve()
    except OSError:
        source_resolved = source_model_path
        latest_resolved = latest_model_path

    if source_resolved == latest_resolved:
        return False

    removed_previous_latest = latest_directory.exists()
    if removed_previous_latest:
        shutil.rmtree(latest_directory) if latest_directory.is_dir() else latest_directory.unlink()

    latest_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_model_path, latest_model_path)
    return removed_previous_latest


def finalize_training_artifact(
    *,
    training_type: str,
    model_path: Path,
    run_directory: Path | None,
    parameters: dict[str, Any],
    summary: Any,
    source: str = "training",
    extra: dict[str, Any] | None = None,
    status: str = TRAINING_STATUS_COMPLETED,
    target_episodes: int | None = None,
    completed_episodes: int | None = None,
    reference_model_path: str | Path | None = None,
    resume_run_path: str | Path | None = None,
    run_log_path: str | Path | None = None,
) -> Path | None:
    """写出训练产物信息文件。 / Write the training artifact info file."""
    info_path = None
    if run_directory is not None:
        info_path = write_training_info(
            training_type=training_type,
            model_path=model_path,
            run_directory=run_directory,
            parameters=parameters,
            summary=summary,
            source=source,
            extra=extra,
            status=status,
            target_episodes=target_episodes,
            completed_episodes=completed_episodes,
            reference_model_path=reference_model_path,
            resume_run_path=resume_run_path,
            run_log_path=run_log_path,
            include_target_episodes=source == "training",
        )
    return info_path


def complete_training_artifact(
    *,
    training_type: str,
    model_path: Path,
    run_directory: Path | None,
    parameters: dict[str, Any],
    summary: Any,
    status: str,
    target_episodes: int | None,
    completed_episodes: int,
    reference_model_path: str | Path | None,
    resume_run_path: str | Path | None,
    run_log_path: Path | None,
) -> Any:
    """完成训练产物收尾：写 info、记日志并清理旧续训目录。 / Finish info, logs, and resume cleanup."""
    info_path = finalize_training_artifact(
        training_type=training_type,
        model_path=model_path,
        run_directory=run_directory,
        parameters=parameters,
        summary=summary,
        status=status,
        target_episodes=target_episodes,
        completed_episodes=completed_episodes,
        reference_model_path=reference_model_path,
        resume_run_path=resume_run_path,
        run_log_path=run_log_path,
    )
    if info_path is not None:
        summary = replace(summary, info_path=info_path)
    log_training_result(
        training_type,
        parameters,
        summary,
        path=run_log_path or model_path.with_name(TRAINING_RUN_LOG_FILENAME),
        event="training_stopped" if status == TRAINING_STATUS_INCOMPLETE else "training_completed",
    )
    cleanup_resumed_incomplete_run(resume_run_path, run_directory)
    return summary


def load_training_info(path: str | Path) -> dict[str, Any] | None:
    """读取训练产物信息文件。 / Load a training artifact info file."""
    candidate = Path(path)
    run_directory = candidate if candidate.is_dir() else candidate.parent
    info_path = run_directory / TRAINING_INFO_FILENAME
    if not info_path.exists():
        return None
    with info_path.open("r", encoding="utf-8-sig") as handle:
        loaded = json.load(handle)
    if isinstance(loaded, dict):
        loaded.setdefault("info_path", project_relative_path(info_path))
    return loaded if isinstance(loaded, dict) else None


def resolve_info_path(value: Any) -> Path | None:
    """把信息文件中保存的路径恢复为项目内 Path。 / Resolve a path stored in info."""
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_path_from_info(info: dict[str, Any]) -> Path | None:
    """从训练信息读取模型路径。 / Read the model path from training info."""
    model_path = resolve_info_path(info.get("model_path"))
    info_path = resolve_info_path(info.get("info_path"))
    training_type = str(info.get("training_type", ""))
    filename = TRAINING_MODEL_FILENAMES.get(training_type)
    if model_path is not None and model_path.exists():
        return model_path
    if info_path is not None:
        if model_path is not None:
            sibling = info_path.parent / model_path.name
            if sibling.exists():
                return sibling
        if filename:
            return info_path.parent / filename
    if model_path is not None:
        return model_path
    return None


def resume_training_context(
    training_type: str,
    resume_run_path: str | Path | None,
    target_episodes: int | None,
    reference_model_path: str | Path | None,
) -> tuple[int, Path | None, Path | None, dict[str, Any] | None]:
    """解析续训来源并返回已完成局数和已选参考模型。 / Resolve resume metadata and selected reference."""
    if resume_run_path is None:
        return 0, None, Path(reference_model_path) if reference_model_path is not None else None, None

    resume_path = Path(resume_run_path)
    info = load_training_info(resume_path)
    if info is None:
        raise FileNotFoundError(f"Cannot find resume info: {resume_run_path}")
    if str(info.get("training_type")) != training_type:
        raise ValueError(f"Cannot resume {training_type} from {info.get('training_type')}")
    if not is_incomplete_training_info(info):
        raise ValueError("Only incomplete training runs can be resumed.")

    completed = int(info.get("completed_episodes") or 0)
    if target_episodes is not None and target_episodes < completed:
        raise ValueError("继续训练总局数不能低于当前已训练局数。")

    selected_reference = Path(reference_model_path) if reference_model_path is not None else None
    if selected_reference is None:
        previous_reference = resolve_info_path(info.get("reference_model_path"))
        if previous_reference is not None and previous_reference.exists():
            selected_reference = previous_reference
    return completed, resume_path, selected_reference, info


def initial_training_model_path(reference_model_path: Path | None, resume_info: dict[str, Any] | None) -> Path | None:
    """返回本次实际加载的初始权重。续训模型不是参考模型。 / Return initial weights; resume source is not a reference."""
    if reference_model_path is not None:
        return reference_model_path
    if resume_info is None:
        return None
    return model_path_from_info(resume_info)


def cleanup_resumed_incomplete_run(resume_run_path: str | Path | None, replacement_run_directory: Path | None) -> bool:
    """新续训目录保存成功后删除旧未完成目录。 / Remove the old incomplete run after replacement is saved."""
    if resume_run_path is None:
        return False
    source = Path(resume_run_path)
    source_run_directory = source if source.is_dir() else source.parent
    if replacement_run_directory is not None:
        try:
            if source_run_directory.resolve() == replacement_run_directory.resolve():
                return False
        except OSError:
            return False

    info = load_training_info(source_run_directory)
    if info is None or not is_incomplete_training_info(info):
        return False
    training_type = str(info.get("training_type", ""))
    if training_type not in TRAINING_MODEL_FILENAMES:
        return False

    allowed_root = Path(get_train_defaults(training_type)["runs_directory"]).resolve()
    try:
        resolved_source = source_run_directory.resolve()
        resolved_source.relative_to(allowed_root)
    except (OSError, ValueError):
        return False
    if resolved_source == allowed_root:
        return False

    shutil.rmtree(source_run_directory)
    return True


def list_training_artifacts(root: str | Path | None = None, *, include_incomplete: bool = True) -> list[dict[str, Any]]:
    """列出历史训练产物。 / List historical training artifacts."""
    artifacts: list[dict[str, Any]] = []
    roots = [Path(root)] if root is not None else [
        Path(get_train_defaults(training_type)["runs_directory"])
        for training_type in TRAINING_MODEL_FILENAMES
    ]
    for root_path in roots:
        if not root_path.exists():
            continue
        run_directories = _candidate_run_directories(root_path)
        for run_directory in run_directories:
            if run_directory.name == "latest":
                continue
            info = load_training_info(run_directory)
            if not isinstance(info, dict):
                continue
            if not include_incomplete and is_incomplete_training_info(info):
                continue
            info["status"] = training_info_status(info)
            info["info_path"] = project_relative_path(run_directory / TRAINING_INFO_FILENAME)
            info["run_directory"] = project_relative_path(run_directory)
            artifacts.append(info)
    return artifacts


def list_incomplete_training_artifacts(training_type: str | None = None) -> list[dict[str, Any]]:
    """列出未完成训练记录。 / List incomplete training artifacts."""
    artifacts = [
        artifact
        for artifact in list_training_artifacts()
        if is_incomplete_training_info(artifact)
    ]
    if training_type is not None:
        artifacts = [artifact for artifact in artifacts if str(artifact.get("training_type")) == training_type]
    return artifacts


def _candidate_run_directories(root_path: Path) -> list[Path]:
    if (root_path / TRAINING_INFO_FILENAME).exists():
        return [root_path]
    return sorted(child for child in root_path.iterdir() if child.is_dir())
