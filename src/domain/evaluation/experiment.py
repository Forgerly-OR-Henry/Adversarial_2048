"""多局自动实验与 CSV 记录。 / Multi-episode automated experiments with CSV recording."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_experiment_directory
from domain.evaluation.run_episode import run_episode
from utils import EpisodeRecord, ExperimentRecorder, log_evaluation_result

ProgressCallback = Callable[[int, int, EpisodeRecord], None]
StateCallback = Callable[[int, int, EpisodeRecord, Any], None]


def experiment_csv_filename(player_type: str, enemy_type: str) -> str:
    """生成实验 CSV 文件名。 / Build the experiment CSV filename."""
    return f"{player_type}_vs_{enemy_type}.csv"


def default_experiment_path(player_type: str, enemy_type: str) -> Path:
    """生成默认实验 CSV 输出路径。 / Build the default CSV output path for an experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_experiment_directory() / timestamp / experiment_csv_filename(player_type, enemy_type)


def run_experiment(
    player_type: str,
    enemy_type: str,
    episodes: int,
    seed: int | None = None,
    output: str | Path | None = None,
    player_model_path: str | Path | None = None,
    enemy_model_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    state_callback: StateCallback | None = None,
) -> Path:
    """运行多局实验并写出 CSV 记录。 / Run multiple episodes and write CSV records."""
    recorder = ExperimentRecorder()
    for episode in range(1, episodes + 1):
        episode_seed = None if seed is None else seed + episode - 1
        state = run_episode(
            player_type=player_type,
            enemy_type=enemy_type,
            seed=episode_seed,
            player_model_path=player_model_path,
            enemy_model_path=enemy_model_path,
        )
        record = EpisodeRecord(
            episode=episode,
            max_tile=state.max_tile,
            score=state.score,
            steps=state.steps,
            player_type=player_type,
            enemy_type=enemy_type,
            seed=episode_seed,
        )
        recorder.log_episode(record)
        if progress_callback is not None:
            progress_callback(episode, episodes, record)
        if state_callback is not None:
            state_callback(episode, episodes, record, state)

    output_path = Path(output) if output is not None else default_experiment_path(player_type, enemy_type)
    saved_path = recorder.save_csv(output_path)
    log_evaluation_result(
        {
            "player_type": player_type,
            "enemy_type": enemy_type,
            "episodes": episodes,
            "seed": seed,
            "output": saved_path,
            "player_model_path": player_model_path,
            "enemy_model_path": enemy_model_path,
        },
        {
            "output_path": saved_path,
            "episodes": episodes,
            "player_type": player_type,
            "enemy_type": enemy_type,
        },
        path=saved_path.parent / "log.jsonl",
    )
    return saved_path
