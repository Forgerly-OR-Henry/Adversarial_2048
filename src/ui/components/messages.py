"""消息和结果文本区域控件。 / Message and result text-area widgets."""

from __future__ import annotations

import tkinter as tk

from ui.settings.theme import BASE_FONT, MUTED, PANEL_BG

MESSAGE_WRAP_LENGTH = 820
STATUS_AREA_HEIGHT = 88
RESULT_AREA_HEIGHT = 126


def create_message_area(
    parent: tk.Misc,
    variable: tk.StringVar,
    *,
    height: int = STATUS_AREA_HEIGHT,
    wraplength: int = MESSAGE_WRAP_LENGTH,
    fixed_height: bool = True,
) -> tk.Frame:
    """创建提示文本区域。 / Create a message text area."""
    area = tk.Frame(parent, bg=PANEL_BG)
    if fixed_height:
        area.configure(height=height)
    area.grid_propagate(False)
    area.columnconfigure(0, weight=1)
    area.rowconfigure(0, weight=1)

    label = tk.Label(
        area,
        textvariable=variable,
        bg=PANEL_BG,
        fg=MUTED,
        font=BASE_FONT,
        anchor="nw",
        justify="left",
        wraplength=wraplength,
    )
    label.grid(row=0, column=0, sticky="nsew")
    return area


__all__ = [
    "MESSAGE_WRAP_LENGTH",
    "RESULT_AREA_HEIGHT",
    "STATUS_AREA_HEIGHT",
    "create_message_area",
]
