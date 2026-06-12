"""训练、评估和错误日志产物的统一管理。 / Unified management for training, evaluation, and error-log artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from config import (
    PROJECT_ROOT,
    get_error_log_path,
    get_evaluation_results_log_path,
    get_experiment_directory,
    get_log_directory,
    get_train_defaults,
    get_training_results_log_path,
)
from domain.train.artifacts import TRAINING_MODEL_FILENAMES
from domain.train.artifacts import (
    TRAINING_STATUS_COMPLETED,
    TRAINING_STATUS_INCOMPLETE,
    list_training_artifacts,
    load_training_info,
    model_path_from_info,
    replace_latest_model,
    training_info_status,
)


@dataclass(frozen=True)
class ManagedResult:
    """结果管理界面展示的一条产物记录。 / One artifact row shown in result management."""
    result_type: str
    path: Path
    display_path: Path
    created_at: str
    size_bytes: int
    is_latest: bool = False
    training_type: str | None = None
    status: str = TRAINING_STATUS_COMPLETED
    log_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeleteManagedResultsSummary:
    """删除产物后同步清理日志的汇总。 / Summary of artifact deletion and related log cleanup."""
    deleted_paths: tuple[Path, ...]
    removed_training_log_rows: int
    removed_evaluation_log_rows: int
    removed_error_log_rows: int


@dataclass(frozen=True)
class CompactManagedLogsSummary:
    """压缩本地产物日志后的汇总。 / Summary of compacting local artifact logs."""
    compacted_paths: tuple[Path, ...]
    removed_log_rows: int


@dataclass(frozen=True)
class CompactSystemLogsSummary:
    """压缩系统和错误日志后的汇总。 / Summary of compacting system and error logs."""
    compacted_paths: tuple[Path, ...]
    removed_log_rows: int
    kept_rows_per_file: int


@dataclass(frozen=True)
class LatestPromotionSummary:
    """将历史训练产物设为 latest 的结果。 / Summary of promoting a historical model to latest."""
    training_type: str
    source_model_path: Path
    latest_directory: Path
    latest_model_path: Path
    removed_previous_latest: bool


def list_managed_results(
    *,
    training_roots: dict[str, Path] | None = None,
    latest_roots: dict[str, Path] | None = None,
    experiment_directory: Path | None = None,
) -> list[ManagedResult]:
    """列出可管理的训练和评估产物。 / List manageable training and evaluation artifacts."""
    results: list[ManagedResult] = []
    training_roots = training_roots or _configured_training_roots("runs_directory")
    latest_roots = latest_roots or _configured_training_roots("latest_output_directory")
    experiment_directory = experiment_directory or get_experiment_directory()

    for training_type, root in training_roots.items():
        if not root.exists():
            continue
        for info in list_training_artifacts(root):
            run_directory = _run_directory_from_info(info, root)
            if run_directory.name == "latest":
                continue
            model_path = model_path_from_info(info) if isinstance(info, dict) else None
            display_path = model_path if model_path is not None else run_directory
            status = training_info_status(info)
            results.append(
                ManagedResult(
                    result_type="training",
                    path=run_directory,
                    display_path=display_path,
                    created_at=_created_at(info, run_directory),
                    size_bytes=_path_size(run_directory),
                    is_latest=False,
                    training_type=str(info.get("training_type", training_type)) if isinstance(info, dict) else training_type,
                    status=status,
                    log_hints=_log_hints(info, run_directory),
                )
            )

    for training_type, latest_root in latest_roots.items():
        if not latest_root.exists():
            continue
        filename = TRAINING_MODEL_FILENAMES.get(training_type)
        model_path = latest_root / filename if filename else latest_root
        display_path = model_path if model_path.exists() else latest_root
        results.append(
            ManagedResult(
                result_type="training",
                path=latest_root,
                display_path=display_path,
                created_at=_mtime_text(latest_root),
                size_bytes=_path_size(latest_root),
                is_latest=True,
                training_type=training_type,
                status="latest",
                log_hints=_path_tokens(latest_root) + _path_tokens(display_path),
            )
        )

    if experiment_directory.exists():
        for csv_path in sorted(experiment_directory.glob("*.csv")):
            results.append(
                ManagedResult(
                    result_type="evaluation",
                    path=csv_path,
                    display_path=csv_path,
                    created_at=_mtime_text(csv_path),
                    size_bytes=_path_size(csv_path),
                    log_hints=_path_tokens(csv_path),
                )
            )
        for run_directory in sorted(child for child in experiment_directory.iterdir() if child.is_dir()):
            for csv_path in sorted(run_directory.glob("*.csv")):
                results.append(
                    ManagedResult(
                        result_type="evaluation",
                        path=run_directory,
                        display_path=csv_path,
                        created_at=_mtime_text(run_directory),
                        size_bytes=_path_size(run_directory),
                        log_hints=_path_tokens(run_directory) + _path_tokens(csv_path),
                    )
                )

    return sorted(results, key=lambda item: (item.result_type, item.created_at, item.path.as_posix()), reverse=True)


def delete_managed_results(
    results: Iterable[ManagedResult | str | Path],
    *,
    allow_latest: bool = False,
    training_log_path: Path | None = None,
    evaluation_log_path: Path | None = None,
    error_log_path: Path | None = None,
    training_roots: dict[str, Path] | None = None,
    latest_roots: dict[str, Path] | None = None,
    experiment_directory: Path | None = None,
) -> DeleteManagedResultsSummary:
    """安全删除选中产物并清理相关日志行。 / Safely delete selected artifacts and clean related log rows."""
    items = [_coerce_result(item) for item in results]
    if any(item.is_latest for item in items) and not allow_latest:
        raise PermissionError("Deleting latest/default models requires explicit confirmation.")

    training_roots = training_roots or _configured_training_roots("runs_directory")
    latest_roots = latest_roots or _configured_training_roots("latest_output_directory")
    experiment_directory = experiment_directory or get_experiment_directory()
    # allowed_roots 定义“能删哪里”，protected_roots 定义“根目录本身不能被删”。
    # allowed_roots defines where deletion may happen; protected_roots keeps root folders intact.
    allowed_roots = tuple(training_roots.values()) + tuple(latest_roots.values()) + (experiment_directory,)
    protected_roots = tuple(training_roots.values()) + (experiment_directory,)

    deleted: list[Path] = []
    all_tokens: set[str] = set()
    for item in items:
        _ensure_safe_delete_path(item.path, allowed_roots, protected_roots, allow_latest=allow_latest)
        # 日志里可能记录模型路径、目录路径或展示路径，清理时统一用 token 匹配。
        # Logs may mention model paths, run directories, or display paths, so cleanup matches all tokens.
        all_tokens.update(item.log_hints)
        all_tokens.update(_path_tokens(item.path))
        all_tokens.update(_path_tokens(item.display_path))
        if item.path.exists():
            if item.path.is_dir():
                shutil.rmtree(item.path)
            else:
                item.path.unlink()
            deleted.append(item.path)

    target_paths = tuple(item.path for item in items) + tuple(item.display_path for item in items)
    removed_training = _rewrite_jsonl_without_references(
        training_log_path or get_training_results_log_path(),
        target_paths,
        all_tokens,
    )
    removed_evaluation = _rewrite_jsonl_without_references(
        evaluation_log_path or get_evaluation_results_log_path(),
        target_paths,
        all_tokens,
    )
    removed_error = _rewrite_jsonl_without_references(
        error_log_path or get_error_log_path(),
        target_paths,
        all_tokens,
    )
    return DeleteManagedResultsSummary(
        deleted_paths=tuple(deleted),
        removed_training_log_rows=removed_training,
        removed_evaluation_log_rows=removed_evaluation,
        removed_error_log_rows=removed_error,
    )


def compact_managed_result_logs(
    results: Iterable[ManagedResult | str | Path],
    *,
    training_roots: dict[str, Path] | None = None,
    latest_roots: dict[str, Path] | None = None,
    experiment_directory: Path | None = None,
) -> CompactManagedLogsSummary:
    """将选中结果的本地日志压缩为仅保留最近一条。 / Keep only the latest row in selected local logs."""
    items = [_coerce_result(item) for item in results]
    training_roots = training_roots or _configured_training_roots("runs_directory")
    latest_roots = latest_roots or _configured_training_roots("latest_output_directory")
    experiment_directory = experiment_directory or get_experiment_directory()
    allowed_roots = tuple(training_roots.values()) + tuple(latest_roots.values()) + (experiment_directory,)

    compacted: list[Path] = []
    removed_rows = 0
    seen_logs: set[Path] = set()
    for item in items:
        _ensure_managed_result_path(item.path, allowed_roots)
        for log_path in _local_log_paths_for_result(item):
            try:
                resolved_log = log_path.resolve()
            except OSError:
                resolved_log = _resolve_existing_or_parent(log_path)
            if resolved_log in seen_logs:
                continue
            seen_logs.add(resolved_log)
            _ensure_managed_result_path(log_path, allowed_roots)
            removed = _compact_jsonl_to_recent_rows(log_path, keep=1)
            if removed:
                compacted.append(log_path)
                removed_rows += removed
    return CompactManagedLogsSummary(compacted_paths=tuple(compacted), removed_log_rows=removed_rows)


def compact_system_logs(
    *,
    keep: int = 10,
    log_directory: Path | None = None,
    training_log_path: Path | None = None,
    evaluation_log_path: Path | None = None,
    error_log_path: Path | None = None,
) -> CompactSystemLogsSummary:
    """压缩系统和错误日志，每个 JSONL 保留最近 keep 条。 / Keep recent rows in system and error JSONL logs."""
    if keep < 1:
        raise ValueError("保留日志条数必须至少为 1。")
    log_root = log_directory or get_log_directory()
    compacted: list[Path] = []
    removed_rows = 0
    seen_logs: set[Path] = set()
    for log_path in _system_log_paths_for_cleanup(
        log_directory=log_root,
        training_log_path=training_log_path,
        evaluation_log_path=evaluation_log_path,
        error_log_path=error_log_path,
    ):
        try:
            resolved_log = log_path.resolve()
        except OSError:
            resolved_log = _resolve_existing_or_parent(log_path)
        if resolved_log in seen_logs:
            continue
        seen_logs.add(resolved_log)
        _ensure_project_log_path(log_path, log_root)
        removed = _compact_jsonl_to_recent_rows(log_path, keep=keep)
        if removed:
            compacted.append(log_path)
            removed_rows += removed
    return CompactSystemLogsSummary(
        compacted_paths=tuple(compacted),
        removed_log_rows=removed_rows,
        kept_rows_per_file=keep,
    )


def promote_training_result_to_latest(
    result: ManagedResult | str | Path,
    *,
    training_roots: dict[str, Path] | None = None,
    latest_roots: dict[str, Path] | None = None,
) -> LatestPromotionSummary:
    """把一个历史训练模型设为对应类型的 latest。 / Promote one historical training model to latest."""
    item = _coerce_result(result)
    if item.result_type == "evaluation":
        raise ValueError("Only training results can be promoted to latest.")
    if item.is_latest:
        raise ValueError("The selected result is already latest.")
    if item.status == TRAINING_STATUS_INCOMPLETE:
        raise ValueError("未完成训练不能设为 latest。")
    info = _info_for_result(item)
    if training_info_status(info) == TRAINING_STATUS_INCOMPLETE:
        raise ValueError("未完成训练不能设为 latest。")

    training_roots = training_roots or _configured_training_roots("runs_directory")
    latest_roots = latest_roots or _configured_training_roots("latest_output_directory")
    training_type = _training_type_for_result(item)
    if training_type not in TRAINING_MODEL_FILENAMES:
        raise ValueError(f"Unsupported training type: {training_type}")
    if training_type not in latest_roots:
        raise KeyError(f"Missing latest root for training type: {training_type}")

    source_model_path = _source_model_path_for_result(item)
    _ensure_source_model_path(source_model_path, tuple(training_roots.values()))

    latest_directory = latest_roots[training_type]
    latest_model_path = latest_directory / TRAINING_MODEL_FILENAMES[training_type]
    allowed_roots = tuple(training_roots.values()) + tuple(latest_roots.values())
    protected_roots = tuple(training_roots.values())
    removed_previous_latest = latest_directory.exists()
    if removed_previous_latest:
        _ensure_safe_delete_path(latest_directory, allowed_roots, protected_roots, allow_latest=True)
    removed_previous_latest = replace_latest_model(source_model_path, latest_model_path)
    return LatestPromotionSummary(
        training_type=training_type,
        source_model_path=source_model_path,
        latest_directory=latest_directory,
        latest_model_path=latest_model_path,
        removed_previous_latest=removed_previous_latest,
    )


def _configured_training_roots(key: str) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for training_type in TRAINING_MODEL_FILENAMES:
        value = get_train_defaults(training_type).get(key)
        if value is not None:
            roots[training_type] = Path(value)
    return roots


def _coerce_result(value: ManagedResult | str | Path) -> ManagedResult:
    if isinstance(value, ManagedResult):
        return value
    path = Path(value)
    return ManagedResult(
        result_type="unknown",
        path=path,
        display_path=path,
        created_at=_mtime_text(path) if path.exists() else "",
        size_bytes=_path_size(path) if path.exists() else 0,
        is_latest=path.name == "latest",
        status="latest" if path.name == "latest" else TRAINING_STATUS_COMPLETED,
        log_hints=_path_tokens(path),
    )


def _training_type_for_result(item: ManagedResult) -> str:
    if item.training_type:
        return item.training_type
    info = _info_for_result(item)
    if isinstance(info, dict) and info.get("training_type"):
        return str(info["training_type"])
    raise ValueError("Selected result does not include a training type.")


def _source_model_path_for_result(item: ManagedResult) -> Path:
    candidates: list[Path] = []
    info = _info_for_result(item)
    if isinstance(info, dict):
        info_model_path = model_path_from_info(info)
        if info_model_path is not None:
            candidates.append(info_model_path)
    candidates.append(item.display_path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find a model file for selected result: {item.path}")


def _info_for_result(item: ManagedResult) -> dict[str, Any] | None:
    return load_training_info(item.path)


def _ensure_source_model_path(path: Path, training_roots: tuple[Path, ...]) -> None:
    resolved = _resolve_existing_or_parent(path)
    project_root = PROJECT_ROOT.resolve()
    if not _is_relative_to(resolved, project_root):
        raise ValueError(f"Refusing to promote a model outside the project root: {path}")
    resolved_roots = tuple(_resolve_existing_or_parent(root) for root in training_roots)
    if not any(_is_relative_to(resolved, root) for root in resolved_roots):
        raise ValueError(f"Refusing to promote a model outside managed training directories: {path}")


def _ensure_safe_delete_path(
    path: Path,
    allowed_roots: tuple[Path, ...],
    protected_roots: tuple[Path, ...],
    *,
    allow_latest: bool,
) -> None:
    resolved = _resolve_existing_or_parent(path)
    project_root = PROJECT_ROOT.resolve()
    # 删除前始终解析绝对路径，防止相对路径或不存在的目标越过项目边界。
    # Always resolve absolute paths before deletion to keep relative or missing targets inside the project.
    if resolved == project_root:
        raise ValueError("Refusing to delete the project root.")
    if not _is_relative_to(resolved, project_root):
        raise ValueError(f"Refusing to delete outside the project root: {path}")

    resolved_allowed = tuple(_resolve_existing_or_parent(root) for root in allowed_roots)
    if not any(_is_relative_to(resolved, root) for root in resolved_allowed):
        raise ValueError(f"Refusing to delete outside managed result directories: {path}")

    for root in protected_roots:
        root_resolved = _resolve_existing_or_parent(root)
        if resolved == root_resolved:
            raise ValueError(f"Refusing to delete a managed root directory: {path}")
    if resolved.name == "latest" and not allow_latest:
        raise PermissionError("Deleting latest/default models requires explicit confirmation.")


def _ensure_managed_result_path(path: Path, allowed_roots: tuple[Path, ...]) -> None:
    resolved = _resolve_existing_or_parent(path)
    project_root = PROJECT_ROOT.resolve()
    if not _is_relative_to(resolved, project_root):
        raise ValueError(f"Refusing to clean logs outside the project root: {path}")
    resolved_allowed = tuple(_resolve_existing_or_parent(root) for root in allowed_roots)
    if not any(_is_relative_to(resolved, root) for root in resolved_allowed):
        raise ValueError(f"Refusing to clean logs outside managed result directories: {path}")


def _ensure_project_log_path(path: Path, log_root: Path) -> None:
    resolved = _resolve_existing_or_parent(path)
    project_root = PROJECT_ROOT.resolve()
    if not _is_relative_to(resolved, project_root):
        raise ValueError(f"Refusing to clean logs outside the project root: {path}")
    resolved_log_root = _resolve_existing_or_parent(log_root)
    if not _is_relative_to(resolved, resolved_log_root):
        raise ValueError(f"Refusing to clean logs outside the configured log directory: {path}")


def _resolve_existing_or_parent(path: Path) -> Path:
    if path.exists():
        return path.resolve()
    parent = path.parent if path.parent != Path("") else Path(".")
    return (parent.resolve() / path.name)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _created_at(info: dict[str, Any] | None, fallback: Path) -> str:
    if isinstance(info, dict) and info.get("created_at"):
        return str(info["created_at"])
    return _mtime_text(fallback)


def _run_directory_from_info(info: dict[str, Any], fallback_root: Path) -> Path:
    run_directory = _path_from_info_value(info.get("run_directory"))
    if run_directory is not None:
        return run_directory
    info_path = _path_from_info_value(info.get("info_path"))
    if info_path is not None:
        return info_path.parent
    model_path = _path_from_info_value(info.get("model_path"))
    if model_path is not None:
        return model_path.parent
    return fallback_root


def _path_from_info_value(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


def _mtime_text(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
    return total


def _local_log_paths_for_result(item: ManagedResult) -> tuple[Path, ...]:
    directory = item.path if item.path.is_dir() else item.path.parent
    candidates: list[Path] = []
    standard_log = directory / "log.jsonl"
    if standard_log.exists():
        candidates.append(standard_log)
    candidates.extend(sorted(path for path in directory.glob("*_log.jsonl") if path.is_file()))
    return tuple(candidates)


def _system_log_paths_for_cleanup(
    *,
    log_directory: Path,
    training_log_path: Path | None,
    evaluation_log_path: Path | None,
    error_log_path: Path | None,
) -> tuple[Path, ...]:
    error_path = error_log_path or get_error_log_path()
    paths: set[Path] = {
        training_log_path or get_training_results_log_path(),
        evaluation_log_path or get_evaluation_results_log_path(),
        error_path,
    }
    for directory in (log_directory / "system", error_path.parent):
        if directory.exists():
            paths.update(path for path in directory.glob("*.jsonl") if path.is_file())
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _resolve_project_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    text = str(value)
    if not _looks_like_path(text):
        return None
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _log_hints(info: dict[str, Any] | None, run_directory: Path) -> tuple[str, ...]:
    hints: set[str] = set(_path_tokens(run_directory))
    if isinstance(info, dict):
        for value in _json_strings(info):
            candidate = _resolve_project_path(value)
            if candidate is not None:
                hints.update(_path_tokens(candidate))
    return tuple(sorted(item for item in hints if item))


def _looks_like_path(value: str) -> bool:
    if "/" in value or "\\" in value:
        return True
    if len(value) >= 3 and value[1] == ":":
        return True
    return bool(Path(value).suffix)


def _path_tokens(path: Path) -> tuple[str, ...]:
    tokens: set[str] = set()
    tokens.add(str(path))
    tokens.add(path.as_posix())
    try:
        resolved = path.resolve()
    except OSError:
        resolved = PROJECT_ROOT.resolve() / path
    tokens.add(str(resolved))
    tokens.add(resolved.as_posix())
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        pass
    else:
        tokens.add(str(relative))
        tokens.add(relative.as_posix())
    return tuple(sorted(item for item in tokens if item))


def _json_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for key, item in value.items():
            strings.extend(_json_strings(key))
            strings.extend(_json_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(_json_strings(item))
        return strings
    return []


def _rewrite_jsonl_without_references(path: Path, target_paths: tuple[Path, ...], tokens: set[str]) -> int:
    if not path.exists():
        return 0
    kept: list[str] = []
    removed = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            referenced = _raw_text_references(stripped, tokens)
        else:
            referenced = _payload_references(payload, target_paths, tokens)
        if referenced:
            removed += 1
        else:
            kept.append(line)
    if removed:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.writelines(kept)
        temp_path.replace(path)
    return removed


def _compact_jsonl_to_recent_rows(path: Path, *, keep: int) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as handle:
        rows = [line.rstrip("\r\n") for line in handle if line.strip()]
    if len(rows) <= keep:
        return 0
    removed = len(rows) - keep
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows[-keep:]:
            handle.write(row)
            handle.write("\n")
    temp_path.replace(path)
    return removed


def _payload_references(payload: Any, target_paths: tuple[Path, ...], tokens: set[str]) -> bool:
    values = _json_strings(payload)
    return any(_value_references(value, target_paths, tokens) for value in values)


def _value_references(value: str, target_paths: tuple[Path, ...], tokens: set[str]) -> bool:
    if _raw_text_references(value, tokens):
        return True
    candidate = _resolve_project_path(value)
    if candidate is None:
        return False
    candidate_resolved = _resolve_existing_or_parent(candidate)
    for target in target_paths:
        target_resolved = _resolve_existing_or_parent(target)
        if candidate_resolved == target_resolved or _is_relative_to(candidate_resolved, target_resolved):
            return True
    return False


def _raw_text_references(text: str, tokens: set[str]) -> bool:
    return any(token and token in text for token in tokens)


__all__ = [
    "CompactManagedLogsSummary",
    "CompactSystemLogsSummary",
    "DeleteManagedResultsSummary",
    "LatestPromotionSummary",
    "ManagedResult",
    "compact_managed_result_logs",
    "compact_system_logs",
    "delete_managed_results",
    "list_managed_results",
    "promote_training_result_to_latest",
]
