"""对局控制面板和玩家动作调度。 / Gameplay control panel and player-action dispatching."""

from __future__ import annotations

import tkinter as tk
from typing import Any
from tkinter import ttk

from domain.enemies import create_enemy
from domain.game.env import GameEnv
from domain.game.rules import move
from domain.players import create_player
from ui.components import (
    GRID_CONTROL_OPTIONS,
    create_action_button,
    create_message_area,
    create_select,
    set_button_visual,
)
from ui.panels.shared import make_grid_placer
from ui.settings.layout.grid import create_area_panel
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
        controls, area = create_area_panel(parent, "对局控制")
        place = make_grid_placer(area)

        place(ttk.Label(controls, text="本局敌人"), 0, 0, colspan=3, sticky="w")
        enemy_menu = create_select(
            controls,
            self.enemy_type,
            tuple(ENEMY_LABELS.values()),
            command=lambda _: self.restart(),
            **GRID_CONTROL_OPTIONS,
        )
        place(enemy_menu, 0, 3, colspan=17)

        place(ttk.Label(controls, text="自动玩家"), 1, 0, colspan=3, sticky="w")
        place(
            create_select(
                controls,
                self.ai_player_type,
                (PLAYER_LABELS["heuristic"], PLAYER_LABELS["q_ai"], PLAYER_LABELS["dqn_player"]),
                command=lambda _: self.reload_ai_player(),
                **GRID_CONTROL_OPTIONS,
            ),
            1,
            3,
            colspan=17,
        )

        self.restart_button = create_action_button(controls, text="重新开始", command=self.restart)
        place(self.restart_button, 2, 0, colspan=6)
        self.ai_step_button = create_action_button(controls, text="AI 单步", command=self.ai_step)
        place(self.ai_step_button, 2, 7, colspan=6)
        self.auto_button = create_action_button(controls, text="自动播放：关", command=self.toggle_auto)
        place(self.auto_button, 2, 14, colspan=6)

        place(
            create_message_area(controls, self.status, **GRID_CONTROL_OPTIONS),
            3,
            0,
            rowspan=2,
            colspan=20,
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
