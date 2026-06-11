"""Tkinter UI 包导出应用入口和棋盘视图。 / Tkinter UI package exports app entrypoint and board view."""

from ui.app import Adversarial2048App, run_gui
from ui.components.board_view import BoardView

__all__ = ["Adversarial2048App", "BoardView", "run_gui"]
