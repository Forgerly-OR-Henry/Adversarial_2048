"""Q-learning 训练产物合并和发布。 / Q-learning training artifact merging and publishing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.compare import identify_training_artifact
from train.artifacts import finalize_training_artifact, resolve_training_output
from models.torch_utils import load_torch_checkpoint, require_torch


@dataclass(frozen=True)
class MergeSummary:
    """Q-learning 模型合并后的结果摘要。 / Result summary after merging Q-learning models."""
    training_type: str
    output_path: Path
    artifact_a: Path
    artifact_b: Path
    weight_a: float
    weight_b: float
    latest_output_path: Path | None = None
    info_path: Path | None = None


def merge_training_artifacts(
    artifact_a: str | Path,
    artifact_b: str | Path,
    output: str | Path | None = None,
    weight_a: float = 0.5,
    publish_latest: bool = False,
) -> MergeSummary:
    """合并两个兼容的 Q-learning 训练产物。 / Merge two compatible Q-learning training artifacts."""
    if not 0.0 <= weight_a <= 1.0:
        raise ValueError("weight_a must be between 0 and 1.")
    info_a = identify_training_artifact(artifact_a)
    info_b = identify_training_artifact(artifact_b)
    if info_a.training_type != info_b.training_type:
        raise ValueError("Can only merge artifacts with the same training type.")

    output_path, run_directory, latest_path = resolve_training_output(info_a.training_type, output)
    if output is not None and str(output).strip():
        run_directory = output_path.parent
    weight_b = 1.0 - weight_a
    if info_a.algorithm == "q_learning":
        _merge_q_models(info_a.path, info_b.path, output_path, weight_a, weight_b)
    else:
        _merge_dqn_models(info_a.path, info_b.path, output_path, weight_a, weight_b, info_a.training_type)

    summary = MergeSummary(
        training_type=info_a.training_type,
        output_path=output_path,
        artifact_a=info_a.path,
        artifact_b=info_b.path,
        weight_a=weight_a,
        weight_b=weight_b,
        latest_output_path=latest_path if publish_latest else None,
        info_path=run_directory / "info.json" if run_directory is not None else None,
    )
    info_path, published_path = finalize_training_artifact(
        training_type=info_a.training_type,
        model_path=output_path,
        run_directory=run_directory,
        latest_path=latest_path,
        parameters={
            "artifact_a": info_a.path,
            "artifact_b": info_b.path,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "publish_latest": publish_latest,
        },
        summary=summary,
        publish=publish_latest,
        source="merge",
    )
    return MergeSummary(
        training_type=summary.training_type,
        output_path=summary.output_path,
        artifact_a=summary.artifact_a,
        artifact_b=summary.artifact_b,
        weight_a=summary.weight_a,
        weight_b=summary.weight_b,
        latest_output_path=published_path,
        info_path=info_path,
    )


def _merge_q_models(path_a: Path, path_b: Path, output_path: Path, weight_a: float, weight_b: float) -> None:
    data_a = _load_json_model(path_a)
    data_b = _load_json_model(path_b)
    for key in ("model_type", "actions"):
        if data_a.get(key) != data_b.get(key):
            raise ValueError(f"Q model mismatch: {key}")
    weights_a = data_a.get("weights")
    weights_b = data_b.get("weights")
    if not _same_matrix_shape(weights_a, weights_b):
        raise ValueError("Q model weights must have the same shape.")
    merged_weights = [
        [
            float(left) * weight_a + float(right) * weight_b
            for left, right in zip(row_a, row_b)
        ]
        for row_a, row_b in zip(weights_a, weights_b)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": data_a["model_type"],
                "version": data_a.get("version", 1),
                "actions": data_a["actions"],
                "weights": merged_weights,
                "merged_from": [str(path_a), str(path_b)],
                "merge_weights": [weight_a, weight_b],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _merge_dqn_models(
    path_a: Path,
    path_b: Path,
    output_path: Path,
    weight_a: float,
    weight_b: float,
    training_type: str,
) -> None:
    torch = require_torch()
    checkpoint_a = load_torch_checkpoint(torch, path_a, map_location="cpu")
    checkpoint_b = load_torch_checkpoint(torch, path_b, map_location="cpu")
    state_a = checkpoint_a["model_state_dict"] if isinstance(checkpoint_a, dict) and "model_state_dict" in checkpoint_a else checkpoint_a
    state_b = checkpoint_b["model_state_dict"] if isinstance(checkpoint_b, dict) and "model_state_dict" in checkpoint_b else checkpoint_b
    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError("DQN state_dict keys must match.")
    merged_state: dict[str, Any] = {}
    for key in state_a:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"DQN tensor shape mismatch: {key}")
        if torch.is_floating_point(tensor_a):
            merged_state[key] = tensor_a * weight_a + tensor_b * weight_b
        elif torch.equal(tensor_a, tensor_b):
            merged_state[key] = tensor_a.clone()
        else:
            raise ValueError(f"DQN non-floating tensor mismatch: {key}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": merged_state,
            "training_type": training_type,
            "merged_from": [str(path_a), str(path_b)],
            "merge_weights": [weight_a, weight_b],
        },
        output_path,
    )


def _load_json_model(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Model file must contain an object: {path}")
    return data


def _same_matrix_shape(left: Any, right: Any) -> bool:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(
        isinstance(row_left, list)
        and isinstance(row_right, list)
        and len(row_left) == len(row_right)
        for row_left, row_right in zip(left, right)
    )
