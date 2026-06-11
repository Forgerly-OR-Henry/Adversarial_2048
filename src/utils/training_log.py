"""训练、评估和错误事件的 JSONL 日志写入。 / JSONL logging for training, evaluation, and error events."""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_error_log_path
from utils.serialization import json_ready


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload, sanitize_paths=True), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def log_training_result(
    training_type: str,
    parameters: dict[str, Any],
    summary: Any,
    *,
    path: str | Path,
    event: str = "training_completed",
) -> Path:
    """追加一条训练结果日志。 / Append one training-result log row."""
    return _append_jsonl(
        Path(path),
        {
            "timestamp": _timestamp(),
            "event": event,
            "training_type": training_type,
            "parameters": parameters,
            "summary": summary,
        },
    )


def log_evaluation_result(parameters: dict[str, Any], summary: dict[str, Any], *, path: str | Path) -> Path:
    """追加一条评估完成日志。 / Append one evaluation-completed log row."""
    return _append_jsonl(
        Path(path),
        {
            "timestamp": _timestamp(),
            "event": "evaluation_completed",
            "parameters": parameters,
            "summary": summary,
        },
    )


def log_error(context: str, error: BaseException, parameters: dict[str, Any] | None = None) -> Path:
    """追加一条错误日志。 / Append one error log row."""
    return _append_jsonl(
        get_error_log_path(),
        {
            "timestamp": _timestamp(),
            "event": "error",
            "context": context,
            "parameters": parameters or {},
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exception(type(error), error, error.__traceback__),
        },
    )
