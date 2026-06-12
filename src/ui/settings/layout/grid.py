"""区域级栅格布局模板。 / Area-level grid layout template."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

from ui.settings.layout.base import FORM_HEIGHT, FORM_WIDTH, lock_widget_size

# 区域面板使用 20 列虚拟网格：行内比例可直接换算为 colspan。
# 例如 3/6/3/8 表示左标签 3 列、左输入 6 列、右标签 3 列、右输入 8 列；
# 对应坐标是 col=0/3/9/12，四段 colspan 相加必须等于 20。
# Area panels use a 20-column virtual grid; ratios map directly to colspan.
AREA_GRID_COLUMNS = 20
AREA_GRID_ROWS = 9
AREA_PANEL_PADDING = 16
AREA_PANEL_BOTTOM_GAP = 8
AREA_GRID_WIDTH_INSET = AREA_PANEL_PADDING * 2
AREA_GRID_HEIGHT_INSET = AREA_PANEL_PADDING * 4
AREA_WIDGET_PADX = 6
AREA_WIDGET_PADDING_RATIO = 0.1


@dataclass(frozen=True)
class AreaGridSpec:
    """区域级栅格规格。 / Area-level grid specification."""

    columns: int
    rows: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.columns < 1:
            raise ValueError("columns must be at least 1.")
        if self.rows < 1:
            raise ValueError("rows must be at least 1.")
        if self.width < 1:
            raise ValueError("width must be at least 1.")
        if self.height < 1:
            raise ValueError("height must be at least 1.")

    def place(self, row: int, col: int, rowspan: int = 1, colspan: int = 1) -> dict[str, int]:
        """返回 Tk grid 坐标。 / Return Tk grid coordinates."""
        if rowspan < 1 or colspan < 1:
            raise ValueError("rowspan and colspan must be at least 1.")
        self._validate_range(row, col, row + rowspan, col + colspan)
        return {"row": row, "column": col, "rowspan": rowspan, "columnspan": colspan}

    def place_from_to(self, row: int, col: int, to_row: int, to_col: int) -> dict[str, int]:
        """用右下角独占坐标返回 Tk grid 坐标。 / Return Tk grid coordinates from exclusive bounds."""
        if to_row <= row or to_col <= col:
            raise ValueError("to_row and to_col must be greater than row and col.")
        self._validate_range(row, col, to_row, to_col)
        return self.place(row, col, to_row - row, to_col - col)

    def column_sizes(self) -> tuple[int, ...]:
        """按总宽度分配列宽。 / Allocate column widths from the total width."""
        return _distribute_pixels(self.width, self.columns)

    def row_sizes(self) -> tuple[int, ...]:
        """按总高度分配行高。 / Allocate row heights from the total height."""
        return _distribute_pixels(self.height, self.rows)

    def padding(self, span_width: int, span_height: int, ratio: float = 0.1, minimum: int = 0) -> tuple[int, int]:
        """返回按单元格区域缩进的内边距。 / Return inset padding derived from cell area."""
        if span_width < 1 or span_height < 1:
            raise ValueError("span_width and span_height must be at least 1.")
        if ratio < 0:
            raise ValueError("ratio must be non-negative.")
        col_width = self.width / self.columns
        row_height = self.height / self.rows
        pad_x = max(minimum, int(col_width * span_width * ratio))
        pad_y = max(minimum, int(row_height * span_height * ratio))
        return pad_x, pad_y

    def configure(self, widget: tk.Misc) -> None:
        """把区域规格应用到 Tk 容器。 / Apply the area grid to a Tk container."""
        for column, minsize in enumerate(self.column_sizes()):
            widget.columnconfigure(column, weight=0, minsize=minsize)
        for row, minsize in enumerate(self.row_sizes()):
            widget.rowconfigure(row, weight=0, minsize=minsize)

    def grid_widget(
        self,
        widget: tk.Misc,
        row: int,
        col: int,
        rowspan: int = 1,
        colspan: int = 1,
        *,
        sticky: str = "nsew",
        padx: int = AREA_WIDGET_PADX,
        pady: int | None = None,
        padding_ratio: float = AREA_WIDGET_PADDING_RATIO,
    ) -> None:
        """按区域栅格放置控件。 / Place a widget inside the area grid."""
        grid_pady = self.padding(colspan, rowspan, ratio=padding_ratio)[1] if pady is None else pady
        widget.grid(**self.place(row, col, rowspan, colspan), sticky=sticky, padx=padx, pady=grid_pady)

    def _validate_range(self, row: int, col: int, to_row: int, to_col: int) -> None:
        if row < 0 or col < 0:
            raise ValueError("row and col must be non-negative.")
        if to_row > self.rows or to_col > self.columns:
            raise ValueError("grid placement is outside the area.")


def create_area_panel(parent: tk.Misc, title: str) -> tuple[ttk.LabelFrame, AreaGridSpec]:
    """创建统一的 20x9 区域栅格面板。 / Create a shared 20x9 area-grid panel."""
    form_height = FORM_HEIGHT - AREA_PANEL_BOTTOM_GAP
    panel = ttk.LabelFrame(parent, text=title, width=FORM_WIDTH, height=form_height, padding=AREA_PANEL_PADDING)
    lock_widget_size(panel, width=FORM_WIDTH, height=form_height)
    panel.grid(row=0, column=0, sticky="new", pady=(0, AREA_PANEL_BOTTOM_GAP))
    area = AreaGridSpec(
        columns=AREA_GRID_COLUMNS,
        rows=AREA_GRID_ROWS,
        width=FORM_WIDTH - AREA_GRID_WIDTH_INSET,
        height=form_height - AREA_GRID_HEIGHT_INSET,
    )
    area.configure(panel)
    return panel, area


def _distribute_pixels(total: int, parts: int) -> tuple[int, ...]:
    base, remainder = divmod(total, parts)
    return tuple(base + (1 if index < remainder else 0) for index in range(parts))


__all__ = [
    "AREA_GRID_COLUMNS",
    "AREA_GRID_HEIGHT_INSET",
    "AREA_GRID_ROWS",
    "AREA_GRID_WIDTH_INSET",
    "AREA_PANEL_BOTTOM_GAP",
    "AREA_PANEL_PADDING",
    "AREA_WIDGET_PADDING_RATIO",
    "AREA_WIDGET_PADX",
    "AreaGridSpec",
    "create_area_panel",
]
