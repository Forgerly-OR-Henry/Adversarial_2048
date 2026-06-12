"""训练产物识别和横向评估比较。 / Training artifact identification and side-by-side evaluation comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.evaluation.run_episode import run_episode
from domain.train.artifacts import load_training_info


@dataclass(frozen=True)
class TrainingArtifactInfo:
    """可评估训练产物的归一化描述。 / Normalized description of a trainable artifact that can be evaluated."""
    path: Path
    training_type: str
    role: str
    algorithm: str


@dataclass(frozen=True)
class EvaluationStats:
    """评估结果的汇总指标。 / Aggregate metrics from evaluation runs."""
    model_path: Path
    episodes: int
    average_score: float
    average_max_tile: float
    average_steps: float
    best_max_tile: int


@dataclass(frozen=True)
class TrainingComparison:
    """两个训练产物的横向比较结果。 / Side-by-side comparison result for two training artifacts."""
    artifact_a: TrainingArtifactInfo
    artifact_b: TrainingArtifactInfo
    stats_a: EvaluationStats
    stats_b: EvaluationStats
    winner: str
    score_delta: float
    max_tile_delta: float
    steps_delta: float
    recommendation: str


def identify_training_artifact(path: str | Path) -> TrainingArtifactInfo:
    """识别模型路径对应的训练类型和加载信息。 / Identify a model path's training type and loading details."""
    model_path = Path(path)
    artifact_info = load_training_info(model_path)
    if artifact_info and artifact_info.get("training_type"):
        return _info_from_training_type(model_path, str(artifact_info["training_type"]))

    if model_path.suffix.lower() == ".json":
        with model_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        model_type = data.get("model_type")
        if model_type == "linear_q":
            return _info_from_training_type(model_path, "player_q")
        if model_type == "enemy_linear_q":
            return _info_from_training_type(model_path, "enemy_q")

    lowered = model_path.as_posix().lower()
    if "enemy" in lowered:
        return _info_from_training_type(model_path, "enemy_dqn")
    return _info_from_training_type(model_path, "player_dqn")


def _info_from_training_type(path: Path, training_type: str) -> TrainingArtifactInfo:
    mapping = {
        "player_q": ("player", "q_learning"),
        "enemy_q": ("enemy", "q_learning"),
        "player_dqn": ("player", "dqn"),
        "enemy_dqn": ("enemy", "dqn"),
    }
    if training_type not in mapping:
        raise ValueError(f"Unsupported training type: {training_type}")
    role, algorithm = mapping[training_type]
    return TrainingArtifactInfo(path=path, training_type=training_type, role=role, algorithm=algorithm)


def compare_training_artifacts(
    artifact_a: str | Path,
    artifact_b: str | Path,
    episodes: int,
    seed: int | None = None,
    player_type: str = "heuristic",
    enemy_type: str = "random",
) -> TrainingComparison:
    """用相同设置评估并比较两个训练产物。 / Evaluate and compare two training artifacts under the same settings."""
    info_a = identify_training_artifact(artifact_a)
    info_b = identify_training_artifact(artifact_b)
    if info_a.role != info_b.role:
        raise ValueError("Can only compare artifacts with the same role.")
    if info_a.algorithm != info_b.algorithm:
        raise ValueError("Can only compare artifacts trained by the same algorithm.")

    stats_a = evaluate_training_artifact(info_a, episodes, seed, player_type, enemy_type)
    stats_b = evaluate_training_artifact(info_b, episodes, seed, player_type, enemy_type)
    if info_a.role == "enemy":
        better_a = (stats_a.average_score, stats_a.average_max_tile) < (stats_b.average_score, stats_b.average_max_tile)
        score_delta = stats_b.average_score - stats_a.average_score
        max_tile_delta = stats_b.average_max_tile - stats_a.average_max_tile
    else:
        better_a = (stats_a.average_score, stats_a.average_max_tile) > (stats_b.average_score, stats_b.average_max_tile)
        score_delta = stats_a.average_score - stats_b.average_score
        max_tile_delta = stats_a.average_max_tile - stats_b.average_max_tile
    winner = "A" if better_a else "B"
    steps_delta = stats_a.average_steps - stats_b.average_steps
    recommendation = _recommend(info_a.role, winner, score_delta, max_tile_delta)
    return TrainingComparison(
        artifact_a=info_a,
        artifact_b=info_b,
        stats_a=stats_a,
        stats_b=stats_b,
        winner=winner,
        score_delta=score_delta,
        max_tile_delta=max_tile_delta,
        steps_delta=steps_delta,
        recommendation=recommendation,
    )


def evaluate_training_artifact(
    info: TrainingArtifactInfo,
    episodes: int,
    seed: int | None,
    player_type: str,
    enemy_type: str,
) -> EvaluationStats:
    """对单个训练产物运行多局评估。 / Run multi-episode evaluation for one training artifact."""
    scores: list[int] = []
    max_tiles: list[int] = []
    steps: list[int] = []
    for episode in range(1, episodes + 1):
        episode_seed = None if seed is None else seed + episode - 1
        if info.role == "player":
            state = run_episode(
                player_type="dqn_player" if info.algorithm == "dqn" else "q_ai",
                enemy_type=enemy_type,
                seed=episode_seed,
                player_model_path=info.path,
            )
        else:
            state = run_episode(
                player_type=player_type,
                enemy_type="dqn_enemy" if info.algorithm == "dqn" else "q_enemy",
                seed=episode_seed,
                enemy_model_path=info.path,
            )
        scores.append(state.score)
        max_tiles.append(state.max_tile)
        steps.append(state.steps)
    return EvaluationStats(
        model_path=info.path,
        episodes=episodes,
        average_score=sum(scores) / len(scores),
        average_max_tile=sum(max_tiles) / len(max_tiles),
        average_steps=sum(steps) / len(steps),
        best_max_tile=max(max_tiles),
    )


def comparison_to_dict(comparison: TrainingComparison) -> dict[str, Any]:
    """把比较结果转换为日志和 CLI 可用字典。 / Convert a comparison result into a dict for logs and CLI output."""
    return {
        "winner": comparison.winner,
        "score_delta": comparison.score_delta,
        "max_tile_delta": comparison.max_tile_delta,
        "steps_delta": comparison.steps_delta,
        "recommendation": comparison.recommendation,
        "a": comparison.stats_a,
        "b": comparison.stats_b,
    }


def _recommend(role: str, winner: str, score_delta: float, max_tile_delta: float) -> str:
    if abs(score_delta) < 1e-9 and abs(max_tile_delta) < 1e-9:
        return "两次训练表现接近，建议增加评估局数后再判断。"
    if role == "enemy":
        return f"{winner} 对固定玩家压制更强，可作为合并或后续调参基线。"
    return f"{winner} 玩家表现更强，可作为合并或后续调参基线。"
