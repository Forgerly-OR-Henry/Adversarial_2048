from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tests._path  # noqa: F401

from game.env import GameEnv
from models import LinearQModel
from players import create_player
from train import train_q_player


class QPlayerTest(unittest.TestCase):
    def test_linear_q_model_saves_and_loads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.json"
            model = LinearQModel.create()
            model.save(path)
            loaded = LinearQModel.load(path)
            self.assertEqual(len(loaded.weights), 4)
            self.assertEqual(len(loaded.weights[0]), len(model.weights[0]))

    def test_q_player_returns_legal_action(self):
        env = GameEnv(seed=3)
        state = env.reset()
        player = create_player("q_ai")
        action = player.select_action(state, env.get_legal_actions())
        self.assertIn(action, env.get_legal_actions())

    def test_train_q_player_writes_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "trained.json"
            summary = train_q_player(
                episodes=2,
                enemy_type="random",
                seed=11,
                output=output,
                max_steps=100,
            )
            self.assertEqual(summary.output_path, output)
            self.assertTrue(output.exists())
            self.assertEqual(summary.episodes, 2)


if __name__ == "__main__":
    unittest.main()
