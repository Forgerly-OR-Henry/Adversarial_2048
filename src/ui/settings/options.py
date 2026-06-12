"""GUI 展示标签与内部策略类型映射。 / GUI display labels and internal strategy type mapping."""

from __future__ import annotations

from workflows.evaluation import EVALUATION_TARGETS
from workflows.training import REFERENCE_TYPE_INITIAL_WEIGHTS

PLAYER_LABELS = {
    "random": "随机玩家",
    "heuristic": "启发式玩家",
    "q_ai": "轻量 Q 玩家",
    "dqn_player": "深度 DQN 玩家",
}
ENEMY_LABELS = {
    "random": "随机敌人",
    "greedy": "贪心敌人",
    "q_enemy": "轻量 Q 敌人",
    "dqn_enemy": "深度 DQN 敌人",
}
PLAYER_TYPES_BY_LABEL = {label: key for key, label in PLAYER_LABELS.items()}
ENEMY_TYPES_BY_LABEL = {label: key for key, label in ENEMY_LABELS.items()}
TRAINING_TARGET_LABELS = {
    "player": "玩家 AI",
    "enemy": "敌对 AI",
}
TRAINING_ALGORITHM_LABELS = {
    "q_learning": "轻量 Q-learning",
    "dqn": "深度 DQN",
}
TRAINING_TYPE_LABELS = {
    "player_q": "玩家 Q-learning",
    "player_dqn": "玩家 DQN",
    "enemy_q": "敌人 Q-learning",
    "enemy_dqn": "敌人 DQN",
}

NO_REFERENCE_LABEL = "不使用参考模型"
NO_RESUME_LABEL = "不继续训练"
REFERENCE_TYPE_LABELS = {
    REFERENCE_TYPE_INITIAL_WEIGHTS: "起始权重",
}
# 蒸馏训练实现前不进入下拉选项。 / Distillation stays hidden until implemented.
REFERENCE_TYPE_OPTIONS = tuple(REFERENCE_TYPE_LABELS.values())
REFERENCE_TYPES_BY_LABEL = {label: key for key, label in REFERENCE_TYPE_LABELS.items()}

NO_MODEL_LABEL = "暂无可用模型"
EVALUATION_TARGET_LABELS = {
    "auto_player": "自动玩家",
    "player": "玩家模型",
    "auto_enemy": "自动敌人",
    "enemy": "敌对模型",
}
EVALUATION_TARGET_OPTIONS = tuple(EVALUATION_TARGET_LABELS[target] for target in EVALUATION_TARGETS)
EVALUATION_TARGETS_BY_LABEL = {label: key for key, label in EVALUATION_TARGET_LABELS.items()}
