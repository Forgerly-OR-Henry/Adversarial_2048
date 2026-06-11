from __future__ import annotations

import unittest

import tests._path  # noqa: F401

from game.constants import LEFT, RIGHT
from game.rules import get_legal_actions, is_game_over, move


class RulesTest(unittest.TestCase):
    def test_left_merge_three_tiles(self):
        board = [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        result = move(board, LEFT)
        self.assertTrue(result.moved)
        self.assertEqual(result.board[0], [4, 2, 0, 0])
        self.assertEqual(result.score_delta, 4)

    def test_left_merge_four_tiles(self):
        board = [
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        result = move(board, LEFT)
        self.assertEqual(result.board[0], [4, 4, 0, 0])
        self.assertEqual(result.score_delta, 8)

    def test_invalid_move_does_not_change_board(self):
        board = [
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0],
        ]
        result = move(board, LEFT)
        self.assertFalse(result.moved)
        self.assertEqual(result.board, board)
        self.assertIn(RIGHT, get_legal_actions(board))

    def test_game_over_when_full_and_no_merges(self):
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
        self.assertTrue(is_game_over(board))
        self.assertEqual(get_legal_actions(board), [])


if __name__ == "__main__":
    unittest.main()
