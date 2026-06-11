"""对局控制面板和玩家动作调度。 / Gameplay control panel and player-action dispatching."""

from __future__ import annotations

import tkinter as tk
from typing import Any
from tkinter import ttk

from enemies import create_enemy
from game.env import GameEnv
from game.rules import move
from players import create_player
from ui.components.controls import create_action_button, create_message_area, create_select, set_button_visual
from ui.settings.layout import (
    BUTTON_BAR_HEIGHT,
    FIELD_ROW_HEIGHT,
    FORM_WIDTH,
    GAME_CONTROLS_HEIGHT,
    GAME_STATUS_HEIGHT,
    lock_widget_size,
)
from ui.settings.options import ENEMY_LABELS, ENEMY_TYPES_BY_LABEL, PLAYER_LABELS, PLAYER_TYPES_BY_LABEL
from ui.settings.theme import BUTTON_NORMAL, BUTTON_PRESSED


class GamePanel:
    """对局模式的界面状态和操作控制器。 / UI state and action controller for gameplay mode."""
    def __init__(self, app: Any, parent: ttk.Frame, enemy_type: str):
        self.app = app
        self.enemy_type = tk.StringVar(value=ENEMY_LABELS[enemy_type])
        self.status = tk.StringVar(value="")
        self.ai_player_type = tk.StringVar(value=PLAYER_LABELS["heuristic"])
        self.autoplaying = False
        self.player = create_player("heuristic", rng=self.app.rng)

        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        controls = ttk.LabelFrame(parent, text="对局控制", width=FORM_WIDTH, height=GAME_CONTROLS_HEIGHT, padding=16)
        lock_widget_size(controls, width=FORM_WIDTH, height=GAME_CONTROLS_HEIGHT)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(2, weight=1)
        controls.rowconfigure(0, weight=0, minsize=FIELD_ROW_HEIGHT)
        controls.rowconfigure(1, weight=0, minsize=FIELD_ROW_HEIGHT)
        controls.rowconfigure(2, weight=0, minsize=BUTTON_BAR_HEIGHT)

        ttk.Label(controls, text="本局敌人").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=(0, 12))
        enemy_menu = create_select(
            controls,
            self.enemy_type,
            tuple(ENEMY_LABELS.values()),
            command=lambda _: self.restart(),
        )
        enemy_menu.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 12))

        ttk.Label(controls, text="自动玩家").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=(0, 14))
        create_select(
            controls,
            self.ai_player_type,
            (PLAYER_LABELS["heuristic"], PLAYER_LABELS["q_ai"], PLAYER_LABELS["dqn_player"]),
            command=lambda _: self.reload_ai_player(),
        ).grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0, 14))

        button_bar = ttk.Frame(controls, style="Panel.TFrame", height=BUTTON_BAR_HEIGHT)
        lock_widget_size(button_bar, height=BUTTON_BAR_HEIGHT)
        button_bar.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(4, 14))
        button_bar.rowconfigure(0, weight=1)
        for column in range(3):
            button_bar.columnconfigure(column, weight=1, uniform="game_actions")

        self.restart_button = create_action_button(button_bar, text="重新开始", command=self.restart)
        self.restart_button.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.ai_step_button = create_action_button(button_bar, text="AI 单步", command=self.ai_step)
        self.ai_step_button.grid(row=0, column=1, sticky="nsew", padx=8)
        self.auto_button = create_action_button(button_bar, text="自动播放：关", command=self.toggle_auto)
        self.auto_button.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        create_message_area(parent, self.status, height=GAME_STATUS_HEIGHT).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(18, 0),
        )

    def new_env(self) -> GameEnv:
        try:
            enemy = create_enemy(ENEMY_TYPES_BY_LABEL[self.enemy_type.get()], rng=self.app.rng)
        except ModuleNotFoundError as exc:
            self.enemy_type.set(ENEMY_LABELS["random"])
            self.status.set(str(exc))
            enemy = create_enemy("random", rng=self.app.rng)
        return GameEnv(enemy=enemy)

    def reload_ai_player(self) -> None:
        try:
            self.player = create_player(PLAYER_TYPES_BY_LABEL[self.ai_player_type.get()], rng=self.app.rng)
        except ModuleNotFoundError as exc:
            self.ai_player_type.set(PLAYER_LABELS["q_ai"])
            self.player = create_player("q_ai", rng=self.app.rng)
            self.status.set(str(exc))

    def restart(self) -> None:
        self._flash_button(self.restart_button)
        self.autoplaying = False
        self._refresh_auto_button()
        self.app.board_view.clear_animation()
        self.app.env = self.new_env()
        self.reload_ai_player()
        self.app.state = self.app.env.reset()
        self.app.render()

    def apply_action(self, action: str) -> None:
        if self.app.state.done or self.app.board_view.animating:
            return
        if action not in self.app.env.get_legal_actions():
            self.status.set("这一步不能移动。")
            return
        old_board = [row[:] for row in self.app.state.board]
        moved = move(old_board, action)
        self.app.state = self.app.env.step(action)
        if moved.moved:
            self.render_status()
            self.app.board_view.animate_move(
                old_board,
                self.app.state.board,
                action,
                on_complete=self.render_status,
            )
        else:
            self.app.render()

    def ai_step(self) -> None:
        if self.app.state.done or self.app.board_view.animating:
            return
        self._flash_button(self.ai_step_button)
        action = self.player.select_action(self.app.state, self.app.env.get_legal_actions())
        if action is not None:
            self.apply_action(action)

    def toggle_auto(self) -> None:
        self.autoplaying = not self.autoplaying
        self._refresh_auto_button()
        if self.autoplaying:
            self._auto_tick()

    def render_status(self) -> None:
        self.app.update_score_display(
            self.app.state.score,
            self.app.state.steps,
            self.app.state.max_tile,
        )
        if self.app.state.done:
            self.status.set(f"游戏结束，当前敌人：{self.enemy_type.get()}。")
        else:
            self.status.set(f"使用方向键或 WASD 移动。AI 单步和自动播放会使用{self.ai_player_type.get()}。")

    def _auto_tick(self) -> None:
        if not self.autoplaying or self.app.state.done:
            self.autoplaying = False
            self._refresh_auto_button()
            self.app.render()
            return
        self.ai_step()
        self.app.root.after(self.app.board_view.animation_total_ms + 80, self._auto_tick)

    def _refresh_auto_button(self) -> None:
        if self.autoplaying:
            self.auto_button.configure(text="自动播放：开")
            set_button_visual(self.auto_button, BUTTON_PRESSED)
        else:
            self.auto_button.configure(text="自动播放：关")
            set_button_visual(self.auto_button, BUTTON_NORMAL)

    def _flash_button(self, button: tk.Button) -> None:
        set_button_visual(button, BUTTON_PRESSED)
        self.app.root.after(120, lambda: set_button_visual(button, BUTTON_NORMAL))


def build_game_panel(app: Any, parent: ttk.Frame) -> GamePanel:
    """创建对局控制面板。 / Build the gameplay control panel."""
    return GamePanel(app, parent, app.initial_enemy_type)


__all__ = ["GamePanel", "build_game_panel"]
