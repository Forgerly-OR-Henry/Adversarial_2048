"""短轮自动调参候选生成和评估排序。 / Short-run auto-tuning candidate generation and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import get_train_defaults, get_tuning_defaults
from domain.evaluation.compare import EvaluationStats, evaluate_training_artifact, identify_training_artifact
from domain.train.q_learning.enemy import train_q_enemy
from domain.train.q_learning.player import train_q_player


@dataclass(frozen=True)
class TuningCandidate:
    """一次自动调参试跑的参数候选。 / Parameter candidate for one auto-tuning trial."""
    name: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TuningResult:
    """自动调参试跑后的排序结果。 / Ranked result after an auto-tuning trial."""
    candidate: TuningCandidate
    output_path: Path
    stats: EvaluationStats
    rank_score: tuple[float, float]


def generate_tuning_candidates(
    target: str,
    algorithm: str,
    count: int | None = None,
) -> list[TuningCandidate]:
    """生成自动调参候选参数集。 / Generate candidate parameter sets for auto-tuning."""
    training_type = _training_type(target, algorithm)
    defaults = get_train_defaults(training_type)
    tuning_defaults = get_tuning_defaults()
    count = int(count if count is not None else tuning_defaults["candidates"])
    variants = [
        ("baseline", 1.0, 1.0, 1.0, 1.0),
        ("faster", 1.35, 0.99, 0.95, 0.95),
        ("explore", 0.75, 1.01, 1.10, 1.10),
    ]
    candidates: list[TuningCandidate] = []
    for name, lr_scale, gamma_scale, eps_start_scale, eps_end_scale in variants[:count]:
        parameters = {
            "learning_rate": _clamp_positive(float(defaults["learning_rate"]) * lr_scale),
            "gamma": _clamp(float(defaults["gamma"]) * gamma_scale, 0.50, 0.999),
            "epsilon_start": _clamp(float(defaults["epsilon_start"]) * eps_start_scale, 0.01, 1.0),
            "epsilon_end": _clamp(float(defaults["epsilon_end"]) * eps_end_scale, 0.01, 1.0),
        }
        if algorithm == "dqn" and "stability" in defaults:
            stability = dict(defaults["stability"])
            if name == "faster":
                stability["lr_decay"] = max(0.1, float(stability.get("lr_decay", 0.5)) * 0.9)
            elif name == "explore":
                stability["epsilon_boost_on_drop"] = min(
                    2.0,
                    float(stability.get("epsilon_boost_on_drop", 1.2)) * 1.1,
                )
            parameters["stability"] = stability
        candidates.append(TuningCandidate(name=name, parameters=parameters))
    return candidates


def run_auto_tuning(
    target: str,
    algorithm: str,
    candidates: int | None = None,
    training_episodes: int | None = None,
    evaluation_episodes: int | None = None,
    seed: int | None = None,
) -> list[TuningResult]:
    """运行短轮调参并按效果排序。 / Run short tuning trials and rank them by outcome."""
    tuning_defaults = get_tuning_defaults()
    training_type = _training_type(target, algorithm)
    defaults = get_train_defaults(training_type)
    training_episodes = int(training_episodes if training_episodes is not None else tuning_defaults["training_episodes"])
    evaluation_episodes = int(evaluation_episodes if evaluation_episodes is not None else tuning_defaults["evaluation_episodes"])
    seed = seed if seed is not None else tuning_defaults.get("seed")
    results: list[TuningResult] = []

    for index, candidate in enumerate(generate_tuning_candidates(target, algorithm, candidates)):
        candidate_seed = None if seed is None else seed + index * 1000
        summary = _train_candidate(
            target=target,
            algorithm=algorithm,
            defaults=defaults,
            episodes=training_episodes,
            seed=candidate_seed,
            candidate=candidate,
        )
        info = identify_training_artifact(summary.output_path)
        stats = evaluate_training_artifact(
            info,
            episodes=evaluation_episodes,
            seed=candidate_seed,
            player_type=tuning_defaults["enemy_evaluation_player"],
            enemy_type=tuning_defaults["player_evaluation_enemy"],
        )
        rank_score = _rank_score(target, stats)
        results.append(
            TuningResult(
                candidate=candidate,
                output_path=summary.output_path,
                stats=stats,
                rank_score=rank_score,
            )
        )
    return sorted(results, key=lambda item: item.rank_score, reverse=True)


def _train_candidate(
    *,
    target: str,
    algorithm: str,
    defaults: dict[str, Any],
    episodes: int,
    seed: int | None,
    candidate: TuningCandidate,
):
    parameters = dict(candidate.parameters)
    if target == "player" and algorithm == "q_learning":
        return train_q_player(
            episodes=episodes,
            enemy_type=defaults["enemy"],
            seed=seed,
            output=None,
            max_steps=defaults["max_steps"],
            publish_latest=False,
            **parameters,
        )
    if target == "enemy" and algorithm == "q_learning":
        return train_q_enemy(
            episodes=episodes,
            player_type=defaults["player"],
            seed=seed,
            output=None,
            max_steps=defaults["max_steps"],
            publish_latest=False,
            **parameters,
        )
    if target == "player":
        from domain.train.dqn.player import train_dqn_player

        return train_dqn_player(
            episodes=episodes,
            enemy_type=defaults["enemy"],
            seed=seed,
            output=None,
            max_steps=defaults["max_steps"],
            publish_latest=False,
            **parameters,
        )
    from domain.train.dqn.enemy import train_dqn_enemy

    return train_dqn_enemy(
        episodes=episodes,
        player_type=defaults["player"],
        seed=seed,
        output=None,
        max_steps=defaults["max_steps"],
        publish_latest=False,
        **parameters,
    )


def _training_type(target: str, algorithm: str) -> str:
    normalized_algorithm = "dqn" if algorithm == "dqn" else "q"
    if target not in ("player", "enemy"):
        raise ValueError("target must be player or enemy.")
    return f"{target}_{normalized_algorithm}"


def _rank_score(target: str, stats: EvaluationStats) -> tuple[float, float]:
    if target == "enemy":
        return (-stats.average_score, -stats.average_max_tile)
    return (stats.average_score, stats.average_max_tile)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _clamp_positive(value: float) -> float:
    return max(value, 1e-9)
