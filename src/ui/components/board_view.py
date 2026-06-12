"""2048 棋盘 Canvas 绘制和动画。 / Canvas rendering and animation for the 2048 board."""

from __future__ import annotations

import tkinter as tk

from domain.game.constants import DOWN, LEFT, RIGHT, UP
from ui.settings.theme import BOARD_BG, CELL_BG, TILE_FONT_FAMILY


class BoardView(tk.Canvas):
    """绘制棋盘并处理方块动画的 Canvas 组件。 / Canvas component that draws the board and tile animations."""
    tile_size = 130
    tile_gap = 16
    board_size = tile_size * 4 + tile_gap * 5
    animation_frames = 9
    animation_delay_ms = 16

    tile_colors = {
        0: ("#cdc1b4", "#776e65"),
        2: ("#eee4da", "#776e65"),
        4: ("#ede0c8", "#776e65"),
        8: ("#f2b179", "#f9f6f2"),
        16: ("#f59563", "#f9f6f2"),
        32: ("#f67c5f", "#f9f6f2"),
        64: ("#f65e3b", "#f9f6f2"),
        128: ("#edcf72", "#f9f6f2"),
        256: ("#edcc61", "#f9f6f2"),
        512: ("#edc850", "#f9f6f2"),
        1024: ("#edc53f", "#f9f6f2"),
        2048: ("#edc22e", "#f9f6f2"),
        4096: ("#3c91e6", "#f9f6f2"),
        8192: ("#2a9d8f", "#f9f6f2"),
        16384: ("#8e5ea2", "#f9f6f2"),
        32768: ("#d1495b", "#f9f6f2"),
        65536: ("#6d597a", "#f9f6f2"),
    }
    high_tile_palette = (
        ("#3c91e6", "#f9f6f2"),
        ("#2a9d8f", "#f9f6f2"),
        ("#8e5ea2", "#f9f6f2"),
        ("#d1495b", "#f9f6f2"),
        ("#6d597a", "#f9f6f2"),
    )

    def __init__(self, master: tk.Misc):
        super().__init__(
            master,
            width=self.board_size,
            height=self.board_size,
            bg=BOARD_BG,
            highlightthickness=0,
        )
        self.animating = False
        self._draw_background()

    @property
    def animation_total_ms(self) -> int:
        return self.animation_frames * self.animation_delay_ms

    def clear_animation(self) -> None:
        self.animating = False
        self.delete("moving")

    def render(self, board: list[list[int]]) -> None:
        self.delete("tile")
        for row in range(4):
            for col in range(4):
                self._draw_tile(row, col, board[row][col])

    def animate_move(
        self,
        old_board: list[list[int]],
        final_board: list[list[int]],
        action: str,
        on_complete=None,
    ) -> None:
        self._animate_move(old_board, final_board, action, on_complete=on_complete)

    def _cell_bounds(self, row: int, col: int) -> tuple[int, int, int, int]:
        x1 = self.tile_gap + col * (self.tile_size + self.tile_gap)
        y1 = self.tile_gap + row * (self.tile_size + self.tile_gap)
        x2 = x1 + self.tile_size
        y2 = y1 + self.tile_size
        return x1, y1, x2, y2

    def _cell_center(self, row: int, col: int) -> tuple[int, int]:
        x1, y1, x2, y2 = self._cell_bounds(row, col)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _tile_font_size(self, value: int) -> int:
        digits = len(str(value))
        if digits >= 7:
            return 30
        if digits >= 6:
            return 34
        if digits >= 5:
            return 38
        if digits >= 4:
            return 46
        return 60

    @classmethod
    def tile_color(cls, value: int) -> tuple[str, str]:
        """返回方块颜色，2048 之后继续按高阶调色板循环。 / Return tile colors beyond 2048 too."""
        if value in cls.tile_colors:
            return cls.tile_colors[value]
        if value > 2048:
            palette_index = max(0, value.bit_length() - 13) % len(cls.high_tile_palette)
            return cls.high_tile_palette[palette_index]
        return "#3c3a32", "#f9f6f2"

    def _draw_background(self) -> None:
        self.delete("cell")
        for row in range(4):
            for col in range(4):
                self.create_rectangle(
                    *self._cell_bounds(row, col),
                    fill=CELL_BG,
                    outline="",
                    tags=("cell",),
                )

    def _draw_tile(self, row: int, col: int, value: int, tag: str = "tile") -> tuple[int, int] | None:
        if value == 0:
            return None
        bg, fg = self.tile_color(value)
        rect = self.create_rectangle(
            *self._cell_bounds(row, col),
            fill=bg,
            outline="",
            tags=(tag,),
        )
        text = self.create_text(
            *self._cell_center(row, col),
            text=str(value),
            fill=fg,
            font=(TILE_FONT_FAMILY, self._tile_font_size(value), "bold"),
            tags=(tag,),
        )
        return rect, text

    def _line_positions(self, action: str, index: int) -> list[tuple[int, int]]:
        if action == LEFT:
            return [(index, col) for col in range(4)]
        if action == RIGHT:
            return [(index, col) for col in range(3, -1, -1)]
        if action == UP:
            return [(row, index) for row in range(4)]
        if action == DOWN:
            return [(row, index) for row in range(3, -1, -1)]
        raise ValueError(f"Unknown action: {action}")

    def _movement_tracks(self, old_board: list[list[int]], action: str) -> list[tuple[int, int, int, int, int]]:
        tracks: list[tuple[int, int, int, int, int]] = []
        for index in range(4):
            positions = self._line_positions(action, index)
            values = [(row, col, old_board[row][col]) for row, col in positions if old_board[row][col] != 0]
            source_index = 0
            target_index = 0
            while source_index < len(values):
                target_row, target_col = positions[target_index]
                row, col, value = values[source_index]
                if source_index + 1 < len(values) and values[source_index + 1][2] == value:
                    next_row, next_col, next_value = values[source_index + 1]
                    tracks.append((row, col, target_row, target_col, value))
                    tracks.append((next_row, next_col, target_row, target_col, next_value))
                    source_index += 2
                else:
                    tracks.append((row, col, target_row, target_col, value))
                    source_index += 1
                target_index += 1
        return tracks

    def _animate_move(
        self,
        old_board: list[list[int]],
        final_board: list[list[int]],
        action: str,
        frame: int = 0,
        items: list[tuple[int, int, tuple[int, int], tuple[int, int]]] | None = None,
        on_complete=None,
    ) -> None:
        if items is None:
            self.animating = True
            self.delete("tile")
            items = []
            for start_row, start_col, end_row, end_col, value in self._movement_tracks(old_board, action):
                tile_items = self._draw_tile(start_row, start_col, value, tag="moving")
                if tile_items is None:
                    continue
                items.append(
                    (
                        tile_items[0],
                        tile_items[1],
                        self._cell_center(start_row, start_col),
                        self._cell_center(end_row, end_col),
                    )
                )

        if frame >= self.animation_frames:
            self.delete("moving")
            self.animating = False
            self.render(final_board)
            if on_complete is not None:
                on_complete()
            return

        progress = (frame + 1) / self.animation_frames
        for rect, text, start, end in items:
            current_x = start[0] + (end[0] - start[0]) * progress
            current_y = start[1] + (end[1] - start[1]) * progress
            half = self.tile_size / 2
            self.coords(rect, current_x - half, current_y - half, current_x + half, current_y + half)
            self.coords(text, current_x, current_y)

        self.after(
            self.animation_delay_ms,
            lambda: self._animate_move(old_board, final_board, action, frame + 1, items, on_complete),
        )
