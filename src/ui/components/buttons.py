"""按钮控件和按钮状态样式。 / Button widgets and button-state styling."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from ui.settings.theme import BASE_FONT_BOLD, BUTTON_FONT, BUTTON_NORMAL


def create_action_button(
    parent: tk.Misc,
    text: str,
    command: Callable[[], None],
    *,
    compact: bool = False,
) -> tk.Button:
    """创建统一样式的操作按钮。 / Create a consistently styled action button."""
    button = tk.Button(
        parent,
        text=text,
        command=command,
        font=BASE_FONT_BOLD if compact else BUTTON_FONT,
        borderwidth=0,
        highlightthickness=0,
        padx=12 if compact else 18,
        pady=2 if compact else 6,
        anchor=tk.CENTER,
        cursor="hand2",
    )
    set_button_visual(button, BUTTON_NORMAL)
    return button


def set_button_visual(button: tk.Button, visual: dict[str, object]) -> None:
    """根据按钮状态切换强调样式。 / Switch emphasized styling based on button state."""
    button.configure(**visual)


__all__ = ["create_action_button", "set_button_visual"]
