"""单局玩家与敌人的自动对战流程。 / Single-episode automated matchup flow between player and enemy."""

from __future__ import annotations

import random
from pathlib import Path

from config import get_evaluate_defaults
from enemies import create_enemy
from game.env import GameEnv
from game.state import GameState
from players import create_player

EVALUATE_DEFAULTS = get_evaluate_defaults()


def run_episode(
    player_type: str | None = None,
    enemy_type: str | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    player_model_path: str | Path | None = None,
    enemy_model_path: str | Path | None = None,
) -> GameState:
    """运行一局自动对战并返回记录。 / Run one automated episode and return its record."""
    player_type = player_type or EVALUATE_DEFAULTS["player"]
    enemy_type = enemy_type or EVALUATE_DEFAULTS["enemy"]
    seed = seed if seed is not None else EVALUATE_DEFAULTS.get("seed")
    max_steps = int(max_steps if max_steps is not None else EVALUATE_DEFAULTS["max_steps"])

    rng = random.Random(seed)
    enemy = create_enemy(enemy_type, rng=rng, model_path=enemy_model_path)
    player = create_player(player_type, rng=rng, model_path=player_model_path)
    env = GameEnv(enemy=enemy, seed=seed)
    state = env.reset()

    while not state.done and state.steps < max_steps:
        legal_actions = env.get_legal_actions()
        action = player.select_action(state, legal_actions)
        if action is None:
            break
        state = env.step(action)
    return state
