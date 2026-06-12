"""UI 面板共享标签工具。 / Shared label helpers for UI panels."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Container

from ui.settings.layout.grid import AreaGridSpec


def unique_label(existing: Container[str], base_label: str) -> str:
    """避免下拉选项标签重复。 / Return a dropdown label that does not duplicate existing labels."""
    if base_label not in existing:
        return base_label
    index = 2
    while f"{base_label} #{index}" in existing:
        index += 1
    return f"{base_label} #{index}"


def display_timestamp(value: str) -> str:
    """把 ISO 时间转为界面展示文本。 / Convert an ISO timestamp into display text."""
    return value.replace("T", " ") if value else "unknown"


def make_grid_placer(area: AreaGridSpec):
    """绑定区域栅格，返回面板内的快捷布局函数。 / Bind an area grid into a panel-local placement helper."""
    def place(
        widget: tk.Misc,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
        *,
        sticky: str = "nsew",
        padx: int = 6,
        pady: int | None = None,
    ) -> None:
        area.grid_widget(widget, row, col, rowspan, colspan, sticky=sticky, padx=padx, pady=pady)

    return place
