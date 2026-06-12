"""Tkinter 主窗口、面板切换和全局应用状态。 / Tkinter main window, panel switching, and global app state."""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import messagebox, ttk

from config import get_ui_defaults
from domain.game.constants import DOWN, LEFT, RIGHT, UP
from domain.results import compact_system_logs
from ui.components import BoardView, create_action_button, create_select
from ui.panels.evaluation import build_evaluation_panel
from ui.panels.game import build_game_panel
from ui.panels.training import build_training_panel
from ui.panels.training_platform import build_training_platform_panel
from ui.settings.layout.base import (
    BOARD_GROUP_HEIGHT,
    BOARD_GROUP_WIDTH,
    BOARD_SCORE_BOTTOM_GAP,
    BOARD_SCORE_HEIGHT,
    BOARD_SIZE,
    BOARD_TO_SIDE_GAP,
    HEADER_BOTTOM_GAP,
    HEADER_HEIGHT,
    MAIN_HEIGHT,
    OUTER_PADDING,
    PANEL_HOST_HEIGHT,
    SIDE_CONTENT_WIDTH,
    SIDE_PANEL_HEIGHT,
    SIDE_PANEL_PADDING,
    SIDE_PANEL_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    lock_widget_size,
)
from ui.settings.theme import APP_BG, SCORE_FONT, TEXT, apply_theme, enable_high_dpi
from ui.windows.result_manager import open_result_manager_window

UI_DEFAULTS = get_ui_defaults()


class Adversarial2048App:
    """Tkinter 应用根对象和共享状态容器。 / Root Tkinter application object and shared state container."""
    def __init__(self, root: tk.Tk, enemy_type: str | None = None):
        self.root = root
        self.root.title("敌对版 2048")
        apply_theme(self.root)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.root.maxsize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.root.resizable(False, False)

        self.ui_defaults = UI_DEFAULTS
        self.initial_enemy_type = enemy_type or self.ui_defaults["enemy"]
        self.active_panel = tk.StringVar(value="对局")
        self.result_manager_window = None
        self.rng = random.Random()

        self._build_layout()
        self.env = self.game_panel.new_env()
        self.state = self.env.reset()
        self._bind_keys()
        self.render()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        frame = ttk.Frame(self.root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, padding=OUTER_PADDING)
        lock_widget_size(frame, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=0, minsize=BOARD_GROUP_WIDTH)
        frame.columnconfigure(1, weight=0, minsize=SIDE_PANEL_WIDTH)
        frame.rowconfigure(0, weight=0, minsize=MAIN_HEIGHT)

        board_group = ttk.Frame(frame, width=BOARD_GROUP_WIDTH, height=BOARD_GROUP_HEIGHT)
        lock_widget_size(board_group, width=BOARD_GROUP_WIDTH, height=BOARD_GROUP_HEIGHT)
        board_group.grid(row=0, column=0, sticky="", padx=(0, BOARD_TO_SIDE_GAP))
        board_group.columnconfigure(0, weight=0, minsize=BOARD_GROUP_WIDTH)
        board_group.rowconfigure(0, weight=0, minsize=BOARD_SCORE_HEIGHT)
        board_group.rowconfigure(1, weight=0, minsize=BOARD_SCORE_BOTTOM_GAP)
        board_group.rowconfigure(2, weight=0, minsize=BOARD_SIZE)

        score_host = tk.Frame(board_group, bg=APP_BG, width=BOARD_GROUP_WIDTH, height=BOARD_SCORE_HEIGHT)
        lock_widget_size(score_host, width=BOARD_GROUP_WIDTH, height=BOARD_SCORE_HEIGHT)
        score_host.grid(row=0, column=0, sticky="ew")
        score_host.columnconfigure(0, weight=0, minsize=BOARD_GROUP_WIDTH)
        score_line_height = BOARD_SCORE_HEIGHT // 3
        self.score_labels: dict[str, tk.Label] = {}
        for row, key in enumerate(("score", "steps", "max_tile")):
            score_host.rowconfigure(row, weight=0, minsize=score_line_height)
            label = tk.Label(
                score_host,
                bg=APP_BG,
                fg=TEXT,
                font=SCORE_FONT,
                anchor="w",
            )
            label.grid(row=row, column=0, sticky="w")
            self.score_labels[key] = label
        self.score_label = self.score_labels["score"]

        self.board_view = BoardView(board_group)
        self.board_view.grid(row=2, column=0, sticky="")

        side_panel = ttk.Frame(
            frame,
            style="Panel.TFrame",
            width=SIDE_PANEL_WIDTH,
            height=SIDE_PANEL_HEIGHT,
            padding=SIDE_PANEL_PADDING,
        )
        lock_widget_size(side_panel, width=SIDE_PANEL_WIDTH, height=SIDE_PANEL_HEIGHT)
        side_panel.grid(row=0, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=0, minsize=SIDE_CONTENT_WIDTH)

        header = ttk.Frame(side_panel, style="Panel.TFrame", width=SIDE_CONTENT_WIDTH, height=HEADER_HEIGHT)
        lock_widget_size(header, width=SIDE_CONTENT_WIDTH, height=HEADER_HEIGHT)
        header.grid(row=0, column=0, sticky="ew", pady=(0, HEADER_BOTTOM_GAP))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0, minsize=360)
        header.rowconfigure(0, weight=0, minsize=HEADER_HEIGHT)

        function_toolbar = ttk.Frame(header, style="Panel.TFrame", width=410, height=HEADER_HEIGHT)
        lock_widget_size(function_toolbar, width=410, height=HEADER_HEIGHT)
        function_toolbar.grid(row=0, column=0, sticky="w")
        function_toolbar.columnconfigure(0, minsize=96)
        function_toolbar.columnconfigure(1, minsize=300)
        function_toolbar.rowconfigure(0, minsize=HEADER_HEIGHT)

        ttk.Label(function_toolbar, text="当前功能").grid(row=0, column=0, sticky="w", padx=(0, 8))
        create_select(
            function_toolbar,
            self.active_panel,
            ("对局", "模型训练", "单项评估平台", "训练评估平台"),
            command=lambda _: self._show_active_panel(),
            width=16,
        ).grid(row=0, column=1, sticky="ew")

        action_toolbar = ttk.Frame(header, style="Panel.TFrame", width=360, height=HEADER_HEIGHT)
        lock_widget_size(action_toolbar, width=360, height=HEADER_HEIGHT)
        action_toolbar.grid(row=0, column=1, sticky="e")
        action_toolbar.columnconfigure(0, minsize=145)
        action_toolbar.columnconfigure(1, minsize=205)
        action_toolbar.rowconfigure(0, minsize=HEADER_HEIGHT)
        create_action_button(
            action_toolbar,
            text="结果管理",
            command=lambda: open_result_manager_window(self),
        ).grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        create_action_button(
            action_toolbar,
            text="清除系统日志",
            command=self.clear_system_logs,
        ).grid(row=0, column=1, sticky="nsew")

        side_panel.rowconfigure(0, weight=0, minsize=HEADER_HEIGHT)
        side_panel.rowconfigure(1, weight=0, minsize=PANEL_HOST_HEIGHT)
        panel_host = ttk.Frame(
            side_panel,
            style="Panel.TFrame",
            width=SIDE_CONTENT_WIDTH,
            height=PANEL_HOST_HEIGHT,
        )
        lock_widget_size(panel_host, width=SIDE_CONTENT_WIDTH, height=PANEL_HOST_HEIGHT)
        panel_host.grid(row=1, column=0, sticky="nsew")
        panel_host.columnconfigure(0, weight=0, minsize=SIDE_CONTENT_WIDTH)
        panel_host.rowconfigure(0, weight=0, minsize=PANEL_HOST_HEIGHT)

        # 多个功能面板叠放在同一网格位置，通过 tkraise 切换可见面板。
        # Feature panels share one grid cell and use tkraise to switch the visible panel.
        self.mode_panels: dict[str, ttk.Frame] = {}
        game_frame = ttk.Frame(panel_host, style="Panel.TFrame", width=SIDE_CONTENT_WIDTH, height=PANEL_HOST_HEIGHT)
        evaluation_frame = ttk.Frame(panel_host, style="Panel.TFrame", width=SIDE_CONTENT_WIDTH, height=PANEL_HOST_HEIGHT)
        training_frame = ttk.Frame(panel_host, style="Panel.TFrame", width=SIDE_CONTENT_WIDTH, height=PANEL_HOST_HEIGHT)
        training_platform_frame = ttk.Frame(panel_host, style="Panel.TFrame", width=SIDE_CONTENT_WIDTH, height=PANEL_HOST_HEIGHT)
        for panel in (game_frame, evaluation_frame, training_frame, training_platform_frame):
            lock_widget_size(panel, width=SIDE_CONTENT_WIDTH, height=PANEL_HOST_HEIGHT)
            panel.grid(row=0, column=0, sticky="nsew")
            panel.columnconfigure(0, weight=0, minsize=SIDE_CONTENT_WIDTH)
        self.mode_panels["对局"] = game_frame
        self.mode_panels["模型训练"] = training_frame
        self.mode_panels["单项评估平台"] = evaluation_frame
        self.mode_panels["训练评估平台"] = training_platform_frame

        self.game_panel = build_game_panel(self, game_frame)
        self.evaluation_panel = build_evaluation_panel(self, evaluation_frame)
        self.training_panel = build_training_panel(self, training_frame)
        self.training_platform_panel = build_training_platform_panel(self, training_platform_frame)
        self._show_active_panel()

    def _bind_keys(self) -> None:
        key_map = {
            "<Up>": UP,
            "<Down>": DOWN,
            "<Left>": LEFT,
            "<Right>": RIGHT,
            "w": UP,
            "s": DOWN,
            "a": LEFT,
            "d": RIGHT,
        }
        for key, action in key_map.items():
            self.root.bind(key, lambda event, selected=action: self._apply_game_action(selected))

    def _show_active_panel(self) -> None:
        panel = self.mode_panels[self.active_panel.get()]
        panel.tkraise()

    def clear_system_logs(self) -> None:
        if not messagebox.askyesno(
            "清除系统日志",
            "将清理系统日志和错误日志，每个日志文件仅保留最近 10 条。是否继续？",
            parent=self.root,
        ):
            return
        try:
            summary = compact_system_logs(keep=10)
        except Exception as exc:
            messagebox.showerror("清除系统日志失败", str(exc), parent=self.root)
            return
        messagebox.showinfo(
            "清除系统日志",
            f"已清理 {len(summary.compacted_paths)} 个日志文件，删除旧日志 {summary.removed_log_rows} 行，"
            f"每个日志保留最近 {summary.kept_rows_per_file} 条。",
            parent=self.root,
        )

    def _apply_game_action(self, action: str) -> None:
        if self.active_panel.get() != "对局":
            return
        # 方向键只在对局面板生效，避免训练/评估输入框获得焦点时误触棋盘。
        # Direction keys affect the board only in gameplay mode to avoid accidental moves in forms.
        self.game_panel.apply_action(action)

    def render(self) -> None:
        self.board_view.render(self.state.board)
        self.game_panel.render_status()

    def update_score_display(
        self,
        score: int,
        steps: int,
        max_tile: int,
        _source: str | None = None,
    ) -> None:
        """纵向刷新棋盘上方计分信息。 / Refresh board score information vertically."""
        self.score_labels["score"].configure(text=f"分数：{score}")
        self.score_labels["steps"].configure(text=f"步数：{steps}")
        self.score_labels["max_tile"].configure(text=f"最大块：{max_tile}")

    def render_preview(self, board: list[list[int]], score: int, steps: int, max_tile: int, source: str) -> None:
        self.board_view.render(board)
        self.update_score_display(score, steps, max_tile, source)


def run_gui(enemy_type: str | None = None) -> None:
    """启动 Tkinter GUI 主循环。 / Start the Tkinter GUI main loop."""
    enable_high_dpi()
    root = tk.Tk()
    Adversarial2048App(root, enemy_type=enemy_type)
    root.mainloop()
