"""主功能面板。 / Main feature panels."""

from ui.panels.evaluation import EvaluationPanel, build_evaluation_panel
from ui.panels.game import GamePanel, build_game_panel
from ui.panels.training import TrainingPanel, build_training_panel
from ui.panels.training_platform import TrainingPlatformPanel, build_training_platform_panel

__all__ = [
    "EvaluationPanel",
    "GamePanel",
    "TrainingPanel",
    "TrainingPlatformPanel",
    "build_evaluation_panel",
    "build_game_panel",
    "build_training_panel",
    "build_training_platform_panel",
]
