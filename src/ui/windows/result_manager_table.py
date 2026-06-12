"""结果管理表格和展示辅助工具。 / Result-manager table and presentation helpers."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

from domain.results import ManagedResult
from domain.train.artifacts import TRAINING_STATUS_INCOMPLETE
from ui.settings.theme import BORDER, FONT_FAMILY, MUTED, TEXT


class ResultTable:
    """一个结果管理分页中的 Treeview 表格。 / Treeview table inside one result-manager tab."""

    def __init__(
        self,
        notebook: ttk.Notebook,
        *,
        title: str,
        checkbox_images: dict[str, tk.PhotoImage],
        on_toggle: Callable[[ManagedResult], None],
    ):
        self.checkbox_images = checkbox_images
        self.on_toggle = on_toggle
        self.results: list[ManagedResult] = []
        self.item_results: dict[str, ManagedResult] = {}
        self.tree = self._create_tree(notebook, title)

    def render(self, results: list[ManagedResult], selected_paths: set[str]) -> None:
        """重绘表格行。 / Redraw the table rows."""
        self.results = list(results)
        self.item_results.clear()
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        for index, result in enumerate(self.results):
            item_id = f"{result.result_type}-{index}"
            self.item_results[item_id] = result
            selected = result_key(result.path) in selected_paths
            if result.is_latest:
                tags = ("latest",)
            elif result.status == TRAINING_STATUS_INCOMPLETE:
                tags = ("incomplete",)
            else:
                tags = ("normal",)
            self.tree.insert(
                "",
                "end",
                iid=item_id,
                image=self.checkbox_images["checked" if selected else "unchecked"],
                values=(
                    kind_label(result),
                    result.created_at,
                    format_size(result.size_bytes),
                    status_label(result),
                    display_path(result.display_path),
                ),
                tags=tags,
            )

    def _create_tree(self, notebook: ttk.Notebook, title: str) -> ttk.Treeview:
        frame = ttk.Frame(notebook, style="Panel.TFrame", padding=14)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        notebook.add(frame, text=title)

        columns = ("kind", "created", "size", "latest", "path")
        tree = ttk.Treeview(
            frame,
            columns=columns,
            show="tree headings",
            selectmode="browse",
            height=15,
            style="Result.Treeview",
        )
        tree.heading("#0", text="选择")
        tree.heading("kind", text="类型")
        tree.heading("created", text="时间")
        tree.heading("size", text="大小")
        tree.heading("latest", text="状态")
        tree.heading("path", text="路径")
        tree.column("#0", width=78, minwidth=78, anchor="center", stretch=False)
        tree.column("kind", width=150, minwidth=140, anchor="w", stretch=False)
        tree.column("created", width=205, minwidth=190, anchor="w", stretch=False)
        tree.column("size", width=100, minwidth=90, anchor="e", stretch=False)
        tree.column("latest", width=95, minwidth=88, anchor="center", stretch=False)
        tree.column("path", width=860, minwidth=620, anchor="w", stretch=True)
        tree.grid(row=0, column=0, sticky="nsew")

        y_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=y_scrollbar.set)
        tree.bind("<ButtonRelease-1>", self._toggle_from_event)
        tree.bind("<space>", self._toggle_keyboard)
        tree.tag_configure("latest", foreground="#8a4f12")
        tree.tag_configure("incomplete", foreground="#9b2f2f")
        tree.tag_configure("normal", foreground=TEXT)
        return tree

    def _toggle_from_event(self, event: tk.Event) -> None:
        item_id = self.tree.identify_row(event.y)
        if item_id:
            self._toggle_item(item_id)

    def _toggle_keyboard(self, _event: tk.Event) -> str:
        for item_id in self.tree.focus(), *self.tree.selection():
            if item_id:
                self._toggle_item(item_id)
                break
        return "break"

    def _toggle_item(self, item_id: str) -> None:
        result = self.item_results.get(item_id)
        if result is not None:
            self.on_toggle(result)


def apply_result_tree_style(window: tk.Misc) -> None:
    """注册结果表格 Treeview 样式。 / Register result table Treeview styles."""
    style = ttk.Style(window)
    style.configure(
        "Result.Treeview",
        background="#ffffff",
        fieldbackground="#ffffff",
        foreground=TEXT,
        bordercolor=BORDER,
        rowheight=38,
        font=(FONT_FAMILY, 12),
    )
    style.configure(
        "Result.Treeview.Heading",
        background="#f0ece4",
        foreground=TEXT,
        font=(FONT_FAMILY, 13, "bold"),
        padding=(8, 8),
    )
    style.map(
        "Result.Treeview",
        background=[("selected", "#efe4d1")],
        foreground=[("selected", TEXT)],
    )


def create_checkbox_images() -> dict[str, tk.PhotoImage]:
    """创建结果表格复选框图片。 / Create checkbox images for result tables."""
    return {
        "unchecked": _create_checkbox_image(checked=False),
        "checked": _create_checkbox_image(checked=True),
    }


def display_path(path: Path) -> str:
    """显示项目相对路径，失败时保留原路径。 / Display a project-relative path when possible."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def format_size(size: int) -> str:
    """格式化字节大小。 / Format a byte size."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def kind_label(result: ManagedResult) -> str:
    """返回结果类型展示文案。 / Return the display label for the result kind."""
    if result.result_type == "evaluation":
        return "评估 CSV"
    return result.training_type or "训练模型"


def status_label(result: ManagedResult) -> str:
    """返回结果状态展示文案。 / Return the display label for the result status."""
    if result.is_latest:
        return "latest"
    if result.status == TRAINING_STATUS_INCOMPLETE:
        return "未完成"
    return ""


def result_key(path: Path) -> str:
    """返回稳定选择 key。 / Return a stable selection key."""
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _create_checkbox_image(*, checked: bool) -> tk.PhotoImage:
    size = 24
    image = tk.PhotoImage(width=size, height=size)
    image.put("#ffffff", to=(0, 0, size, size))
    image.put("#ffffff", to=(3, 3, size - 3, size - 3))
    image.put(MUTED, to=(3, 3, size - 3, 5))
    image.put(MUTED, to=(3, size - 5, size - 3, size - 3))
    image.put(MUTED, to=(3, 3, 5, size - 3))
    image.put(MUTED, to=(size - 5, 3, size - 3, size - 3))
    if checked:
        _draw_check_mark(image)
    return image


def _draw_check_mark(image: tk.PhotoImage) -> None:
    for index in range(5):
        _draw_square(image, 7 + index, 12 + index, "#2f8f5b")
    for index in range(9):
        _draw_square(image, 11 + index, 16 - index, "#2f8f5b")


def _draw_square(image: tk.PhotoImage, x: int, y: int, color: str) -> None:
    image.put(color, to=(x - 1, y - 1, x + 2, y + 2))


__all__ = [
    "ResultTable",
    "apply_result_tree_style",
    "create_checkbox_images",
    "display_path",
    "format_size",
    "kind_label",
    "result_key",
    "status_label",
]
