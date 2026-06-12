"""全局 GUI 布局基础尺寸。 / Global GUI layout base dimensions."""

from __future__ import annotations

import tkinter as tk

WINDOW_WIDTH = 1660
WINDOW_HEIGHT = 800

OUTER_PADDING = 18
SIDE_PANEL_PADDING = 18
BOARD_SIZE = 600
BOARD_SCORE_HEIGHT = 112
BOARD_SCORE_BOTTOM_GAP = 14
BOARD_GROUP_WIDTH = BOARD_SIZE
BOARD_GROUP_HEIGHT = BOARD_SCORE_HEIGHT + BOARD_SCORE_BOTTOM_GAP + BOARD_SIZE
BOARD_TO_SIDE_GAP = 26

MAIN_WIDTH = WINDOW_WIDTH - OUTER_PADDING * 2
MAIN_HEIGHT = WINDOW_HEIGHT - OUTER_PADDING * 2
SIDE_PANEL_WIDTH = MAIN_WIDTH - BOARD_GROUP_WIDTH - BOARD_TO_SIDE_GAP
SIDE_PANEL_HEIGHT = MAIN_HEIGHT
SIDE_CONTENT_WIDTH = SIDE_PANEL_WIDTH - SIDE_PANEL_PADDING * 2
SIDE_CONTENT_HEIGHT = SIDE_PANEL_HEIGHT - SIDE_PANEL_PADDING * 2

HEADER_HEIGHT = 64
HEADER_BOTTOM_GAP = 16
PANEL_HOST_HEIGHT = SIDE_CONTENT_HEIGHT - HEADER_HEIGHT - HEADER_BOTTOM_GAP

FORM_WIDTH = SIDE_CONTENT_WIDTH
FORM_HEIGHT = PANEL_HOST_HEIGHT


def lock_widget_size(widget: tk.Misc, *, width: int | None = None, height: int | None = None) -> None:
    """固定控件请求尺寸并关闭子控件反向撑开。 / Fix requested size and disable propagation."""
    options: dict[str, int] = {}
    if width is not None:
        options["width"] = width
    if height is not None:
        options["height"] = height
    if options:
        widget.configure(**options)
    try:
        widget.grid_propagate(False)
    except tk.TclError:
        pass
    try:
        widget.pack_propagate(False)
    except tk.TclError:
        pass


__all__ = [
    "BOARD_SIZE",
    "BOARD_GROUP_HEIGHT",
    "BOARD_GROUP_WIDTH",
    "BOARD_SCORE_BOTTOM_GAP",
    "BOARD_SCORE_HEIGHT",
    "BOARD_TO_SIDE_GAP",
    "FORM_HEIGHT",
    "FORM_WIDTH",
    "HEADER_BOTTOM_GAP",
    "HEADER_HEIGHT",
    "MAIN_HEIGHT",
    "MAIN_WIDTH",
    "OUTER_PADDING",
    "PANEL_HOST_HEIGHT",
    "SIDE_CONTENT_WIDTH",
    "SIDE_PANEL_HEIGHT",
    "SIDE_PANEL_PADDING",
    "SIDE_PANEL_WIDTH",
    "WINDOW_HEIGHT",
    "WINDOW_WIDTH",
    "lock_widget_size",
]
