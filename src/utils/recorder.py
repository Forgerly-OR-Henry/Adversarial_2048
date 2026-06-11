"""实验记录数据结构和 CSV 写入器。 / Experiment record data structures and CSV writer."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpisodeRecord:
    """一局实验的 CSV 记录。 / CSV record for one experiment episode."""
    episode: int
    max_tile: int
    score: int
    steps: int
    player_type: str
    enemy_type: str
    seed: int | None


class ExperimentRecorder:
    """把多局实验记录写入 CSV。 / Writes multi-episode experiment records to CSV."""
    field_keys = ["episode", "max_tile", "score", "steps", "player_type", "enemy_type", "seed"]
    fieldnames = [
        "episode (局数)",
        "max_tile (最大块)",
        "score (分数)",
        "steps (步数)",
        "player_type (玩家)",
        "enemy_type (敌人)",
        "seed (随机种子)",
    ]
    field_labels = dict(zip(field_keys, fieldnames, strict=True))
    player_labels = {
        "random": "随机玩家",
        "heuristic": "启发式玩家",
        "q_ai": "轻量 Q 玩家",
        "dqn_player": "深度 DQN 玩家",
    }
    enemy_labels = {
        "random": "随机敌人",
        "greedy": "贪心敌人",
        "q_enemy": "轻量 Q 敌人",
        "dqn_enemy": "深度 DQN 敌人",
    }

    def __init__(self):
        self.records: list[EpisodeRecord] = []

    def log_episode(self, record: EpisodeRecord) -> None:
        self.records.append(record)

    def save_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(self._display_row(record))
        return output_path

    def _display_row(self, record: EpisodeRecord) -> dict[str, int | str | None]:
        row: dict[str, int | str | None] = {
            "episode": record.episode,
            "max_tile": record.max_tile,
            "score": record.score,
            "steps": record.steps,
            "player_type": self._display_type(self.player_labels, record.player_type),
            "enemy_type": self._display_type(self.enemy_labels, record.enemy_type),
            "seed": record.seed,
        }
        return {self.field_labels[key]: row[key] for key in self.field_keys}

    @staticmethod
    def _display_type(labels: dict[str, str], value: str) -> str:
        label = labels.get(value)
        if label is None:
            return value
        return f"{value} ({label})"
