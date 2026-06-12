"""训练包导出 Q-learning、DQN、合并和调参入口。 / Training package exports Q-learning, DQN, merge, and tuning entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.train.q_learning.enemy import EnemyTrainingSummary, train_q_enemy
from domain.train.q_learning.player import TrainingSummary, train_q_player

if TYPE_CHECKING:
    from domain.train.dqn.enemy import EnemyDQNTrainingSummary
    from domain.train.dqn.player import DQNTrainingSummary
    from domain.train.merge import MergeSummary
    from domain.train.tuning import TuningCandidate, TuningResult


def list_training_artifacts(*args, **kwargs):
    """列出历史训练产物。 / List historical training artifacts."""
    from domain.train.artifacts import list_training_artifacts as _list_training_artifacts

    return _list_training_artifacts(*args, **kwargs)


def merge_training_artifacts(*args, **kwargs):
    """合并两个兼容的训练产物。 / Merge two compatible training artifacts."""
    from domain.train.merge import merge_training_artifacts as _merge_training_artifacts

    return _merge_training_artifacts(*args, **kwargs)


def generate_tuning_candidates(*args, **kwargs):
    """生成自动调参候选参数集。 / Generate candidate parameter sets for auto-tuning."""
    from domain.train.tuning import generate_tuning_candidates as _generate_tuning_candidates

    return _generate_tuning_candidates(*args, **kwargs)


def run_auto_tuning(*args, **kwargs):
    """运行短轮调参并按效果排序。 / Run short tuning trials and rank them by outcome."""
    from domain.train.tuning import run_auto_tuning as _run_auto_tuning

    return _run_auto_tuning(*args, **kwargs)


def train_dqn_enemy(*args, **kwargs):
    """训练敌人 DQN 并保存产物。 / Train the enemy DQN and save artifacts."""
    from domain.train.dqn.enemy import train_dqn_enemy as _train_dqn_enemy

    return _train_dqn_enemy(*args, **kwargs)


def train_dqn_player(*args, **kwargs):
    """训练玩家 DQN 并保存产物。 / Train the player DQN and save artifacts."""
    from domain.train.dqn.player import train_dqn_player as _train_dqn_player

    return _train_dqn_player(*args, **kwargs)


def __getattr__(name: str):
    if name == "DQNTrainingSummary":
        from domain.train.dqn.player import DQNTrainingSummary

        return DQNTrainingSummary
    if name == "EnemyDQNTrainingSummary":
        from domain.train.dqn.enemy import EnemyDQNTrainingSummary

        return EnemyDQNTrainingSummary
    if name == "MergeSummary":
        from domain.train.merge import MergeSummary

        return MergeSummary
    if name == "TuningCandidate":
        from domain.train.tuning import TuningCandidate

        return TuningCandidate
    if name == "TuningResult":
        from domain.train.tuning import TuningResult

        return TuningResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DQNTrainingSummary",
    "EnemyDQNTrainingSummary",
    "EnemyTrainingSummary",
    "MergeSummary",
    "TuningCandidate",
    "TuningResult",
    "TrainingSummary",
    "generate_tuning_candidates",
    "list_training_artifacts",
    "merge_training_artifacts",
    "run_auto_tuning",
    "train_dqn_enemy",
    "train_dqn_player",
    "train_q_enemy",
    "train_q_player",
]
