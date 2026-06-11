from __future__ import annotations

import random
import unittest

import tests._path  # noqa: F401

from enemies.greedy_enemy import GreedyEnemy
from enemies import create_enemy
from enemies.q_enemy import QEnemy
from enemies.random_enemy import RandomEnemy
from game.board import get_empty_cells
from game.env import GameEnv
from players import create_player
from players.heuristic_player import HeuristicPlayer
from players.random_player import RandomPlayer


class StrategyTest(unittest.TestCase):
    def setUp(self):
        self.board = [
            [2, 2, 4, 8],
            [0, 4, 8, 16],
            [0, 0, 16, 32],
            [0, 0, 0, 64],
        ]
        self.env = GameEnv(seed=7)
        self.state = self.env.set_board(self.board)
        self.legal_actions = self.env.get_legal_actions()

    def test_players_return_legal_actions(self):
        random_player = RandomPlayer(rng=random.Random(1))
        heuristic_player = HeuristicPlayer()

        self.assertIn(random_player.select_action(self.state, self.legal_actions), self.legal_actions)
        self.assertIn(heuristic_player.select_action(self.state, self.legal_actions), self.legal_actions)

    def test_enemies_spawn_only_on_empty_cells_with_valid_values(self):
        empty_cells = set(get_empty_cells(self.board))
        enemies = [
            RandomEnemy(rng=random.Random(1)),
            GreedyEnemy(),
            QEnemy(rng=random.Random(1)),
        ]

        for enemy in enemies:
            row, col, value = enemy.select_spawn(self.state)
            self.assertIn((row, col), empty_cells)
            self.assertIn(value, (2, 4))

    def test_greedy_enemy_is_deterministic(self):
        enemy = GreedyEnemy()
        first = enemy.select_spawn(self.state)
        second = enemy.select_spawn(self.state)
        self.assertEqual(first, second)

    def test_legacy_strategy_aliases_are_rejected(self):
        for name in ("ai", "model", "dqn_ai", "deep_ai"):
            with self.subTest(player=name):
                with self.assertRaises(ValueError):
                    create_player(name)
        for name in ("enemy_ai", "rl_enemy", "deep_enemy"):
            with self.subTest(enemy=name):
                with self.assertRaises(ValueError):
                    create_enemy(name)


if __name__ == "__main__":
    unittest.main()
