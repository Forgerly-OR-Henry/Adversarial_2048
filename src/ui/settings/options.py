"""GUI 展示标签与内部策略类型映射。 / GUI display labels and internal strategy type mapping."""

from __future__ import annotations

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
