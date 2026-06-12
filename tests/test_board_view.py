from __future__ import annotations

import unittest

import tests._path  # noqa: F401

from ui.components.board_view import BoardView


class BoardViewTest(unittest.TestCase):
    def test_tiles_above_2048_have_distinct_colors(self):
        self.assertNotEqual(BoardView.tile_color(4096), BoardView.tile_color(2048))
        self.assertNotEqual(BoardView.tile_color(8192), BoardView.tile_color(4096))
        self.assertEqual(BoardView.tile_color(4096)[1], "#f9f6f2")

    def test_very_high_tiles_reuse_high_tile_palette(self):
        self.assertIn(BoardView.tile_color(131072), BoardView.high_tile_palette)
        self.assertNotEqual(BoardView.tile_color(131072), ("#3c3a32", "#f9f6f2"))


if __name__ == "__main__":
    unittest.main()
