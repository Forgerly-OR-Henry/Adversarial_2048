from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tests._path  # noqa: F401

from enemies import create_enemy
from game.env import GameEnv
from models import EnemyQModel, get_legal_spawn_actions
from train import train_q_enemy


class QEnemyTest(unittest.TestCase):
    def test_enemy_q_model_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "enemy_model.json"
            model = EnemyQModel.create()
            model.save(path)
            loaded = EnemyQModel.load(path)
            self.assertEqual(len(loaded.weights), 32)
            self.assertEqual(len(loaded.weights[0]), len(model.weights[0]))

    def test_q_enemy_returns_legal_spawn(self):
        env = GameEnv(seed=17)
        state = env.reset()
        enemy = create_enemy("q_enemy")
        spawn = enemy.select_spawn(state)
        self.assertIn(f"{spawn[0]},{spawn[1]},{spawn[2]}", get_legal_spawn_actions(state.board))

    def test_train_q_enemy_writes_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "trained_enemy.json"
            summary = train_q_enemy(
                episodes=2,
                player_type="random",
                seed=19,
                output=output,
                max_steps=100,
            )
            self.assertEqual(summary.output_path, output)
            self.assertTrue(output.exists())
            self.assertEqual(summary.episodes, 2)


if __name__ == "__main__":
    unittest.main()
