"""训练和评估结果的图形化管理窗口。 / Graphical management window for training and evaluation results."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from domain.results import (
    ManagedResult,
    compact_managed_result_logs,
    delete_managed_results,
    list_managed_results,
    promote_training_result_to_latest,
)
from domain.train.artifacts import TRAINING_STATUS_INCOMPLETE
from ui.components import create_action_button
from ui.settings.theme import MUTED, PANEL_BG
from ui.windows.result_manager_table import (
    ResultTable,
    apply_result_tree_style,
    create_checkbox_images,
    display_path,
    result_key,
)


class ResultManagerWindow:
    """用于浏览和删除结果产物的窗口。 / Window for browsing and deleting result artifacts."""
    def __init__(self, app: Any):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("训练结果管理")
        self.window.geometry("1600x760")
        self.window.minsize(1600, 760)
        self.window.maxsize(1600, 760)
        self.window.resizable(False, False)
        self.window.configure(bg=PANEL_BG)
        self.window.transient(app.root)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.status = tk.StringVar(value="请选择要清理的训练模型或评估结果。")
        self.training_results: list[ManagedResult] = []
        self.evaluation_results: list[ManagedResult] = []
        self.selected_paths: set[str] = set()
        self.checkbox_images = create_checkbox_images()

        apply_result_tree_style(self.window)
        self._build()
        self.refresh()

    def focus(self) -> None:
        if self.window.winfo_exists():
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()

    def close(self) -> None:
        self.app.result_manager_window = None
        self.window.destroy()

    def _build(self) -> None:
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        header = ttk.Frame(self.window, style="Panel.TFrame", padding=(20, 16, 20, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="训练结果管理", font=("Microsoft YaHei UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status, foreground=MUTED, style="Muted.TLabel").grid(
            row=1,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        actions = ttk.Frame(header, style="Panel.TFrame")
        actions.grid(row=0, column=1, rowspan=2, sticky="e")
        self.refresh_button = create_action_button(actions, text="刷新", command=self.refresh)
        self.select_button = create_action_button(actions, text="全选", command=self.select_visible)
        self.clear_button = create_action_button(actions, text="取消", command=self.clear_visible)
        self.promote_button = create_action_button(actions, text="设为 latest", command=self.promote_selected_to_latest)
        self.compact_logs_button = create_action_button(actions, text="清理日志", command=self.compact_selected_logs)
        self.delete_button = create_action_button(actions, text="删除所选", command=self.delete_selected)
        self.refresh_button.grid(row=0, column=0, padx=(0, 10))
        self.select_button.grid(row=0, column=1, padx=(0, 10))
        self.clear_button.grid(row=0, column=2, padx=(0, 10))
        self.promote_button.grid(row=0, column=3, padx=(0, 10))
        self.compact_logs_button.grid(row=0, column=4, padx=(0, 10))
        self.delete_button.grid(row=0, column=5)

        self.notebook = ttk.Notebook(self.window)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=20, pady=(8, 20))
        self.notebook.bind("<<NotebookTabChanged>>", lambda _event: self._refresh_status())

        self.training_table = ResultTable(
            self.notebook,
            title="训练模型",
            checkbox_images=self.checkbox_images,
            on_toggle=self._toggle_result,
        )
        self.evaluation_table = ResultTable(
            self.notebook,
            title="评估结果",
            checkbox_images=self.checkbox_images,
            on_toggle=self._toggle_result,
        )

    def refresh(self) -> None:
        results = list_managed_results()
        self.training_results = [item for item in results if item.result_type == "training"]
        self.evaluation_results = [item for item in results if item.result_type == "evaluation"]
        existing_paths = {result_key(item.path) for item in results}
        self.selected_paths.intersection_update(existing_paths)
        self.training_table.render(self.training_results, self.selected_paths)
        self.evaluation_table.render(self.evaluation_results, self.selected_paths)
        self._refresh_status()

    def select_visible(self) -> None:
        for item in self._visible_results():
            self.selected_paths.add(result_key(item.path))
        self._render_all()

    def clear_visible(self) -> None:
        for item in self._visible_results():
            self.selected_paths.discard(result_key(item.path))
        self._render_all()

    def promote_selected_to_latest(self) -> None:
        selected = self._selected_results()
        if len(selected) != 1:
            self.status.set("请只勾选一个训练模型来设为 latest。")
            return
        result = selected[0]
        if result.result_type != "training":
            self.status.set("只有训练模型可以设为 latest。")
            return
        if result.is_latest:
            self.status.set("所选模型已经是 latest。")
            return
        if result.status == TRAINING_STATUS_INCOMPLETE:
            self.status.set("未完成训练不能设为 latest。")
            return
        if not messagebox.askyesno(
            "设为 latest",
            (
                f"将 {display_path(result.display_path)} 设为 {result.training_type} 的 latest，"
                "并移除旧 latest。是否继续？"
            ),
            parent=self.window,
        ):
            return
        try:
            summary = promote_training_result_to_latest(result)
        except Exception as exc:
            messagebox.showerror("设为 latest 失败", str(exc), parent=self.window)
            self.status.set(f"设为 latest 失败：{exc}")
            return

        self.selected_paths.discard(result_key(result.path))
        self.app.game_panel.reload_ai_player()
        if hasattr(self.app, "training_platform_panel"):
            self.app.training_platform_panel._reload_selects()
        self.refresh()
        self.status.set(
            f"已将 {display_path(summary.source_model_path)} 设为 {summary.training_type} latest："
            f"{display_path(summary.latest_model_path)}"
        )

    def compact_selected_logs(self) -> None:
        selected = self._selected_results()
        if not selected:
            self.status.set("请先勾选要清理日志的结果。")
            return
        try:
            summary = compact_managed_result_logs(selected)
        except Exception as exc:
            messagebox.showerror("清理日志失败", str(exc), parent=self.window)
            self.status.set(f"清理日志失败：{exc}")
            return

        self.refresh()
        self.status.set(
            f"已清理 {len(summary.compacted_paths)} 个日志文件，删除旧日志 {summary.removed_log_rows} 行；"
            "每个日志仅保留最近一条。"
        )

    def delete_selected(self) -> None:
        selected = self._selected_results()
        if not selected:
            self.status.set("请先勾选要删除的结果。")
            return
        latest_items = [item for item in selected if item.is_latest]
        if not messagebox.askyesno(
            "确认删除",
            f"将删除 {len(selected)} 项结果，包括模型/CSV、信息文件和关联日志。是否继续？",
            parent=self.window,
        ):
            return
        allow_latest = False
        if latest_items:
            # latest 模型是各 AI 默认加载入口，删除前需要单独确认。
            # latest models are default AI loading entrypoints, so deletion needs separate confirmation.
            names = "\n".join(display_path(item.path) for item in latest_items[:6])
            if len(latest_items) > 6:
                names += f"\n...以及 {len(latest_items) - 6} 项"
            allow_latest = messagebox.askyesno(
                "确认删除 latest 模型",
                f"所选内容包含当前默认/latest 模型：\n{names}\n\n删除后对应 AI 可能无法加载。仍要删除吗？",
                parent=self.window,
            )
            if not allow_latest:
                self.status.set("已取消删除 latest 模型。")
                return
        try:
            summary = delete_managed_results(selected, allow_latest=allow_latest)
        except Exception as exc:
            messagebox.showerror("删除失败", str(exc), parent=self.window)
            self.status.set(f"删除失败：{exc}")
            return

        for item in selected:
            self.selected_paths.discard(result_key(item.path))
        if any(item.result_type == "training" for item in selected):
            self.app.game_panel.reload_ai_player()
            if hasattr(self.app, "training_platform_panel"):
                self.app.training_platform_panel._reload_selects()
        self.refresh()
        self.status.set(
            "已删除 "
            f"{len(summary.deleted_paths)} 项结果及其 info/本地日志；清理训练日志 {summary.removed_training_log_rows} 行，"
            f"评估日志 {summary.removed_evaluation_log_rows} 行，错误日志 {summary.removed_error_log_rows} 行。"
        )

    def _render_all(self) -> None:
        self.training_table.render(self.training_results, self.selected_paths)
        self.evaluation_table.render(self.evaluation_results, self.selected_paths)
        self._refresh_status()

    def _toggle_result(self, result: ManagedResult) -> None:
        key = result_key(result.path)
        if key in self.selected_paths:
            self.selected_paths.remove(key)
        else:
            self.selected_paths.add(key)
        self._render_all()

    def _visible_results(self) -> list[ManagedResult]:
        current = self.notebook.index(self.notebook.select())
        return self.training_table.results if current == 0 else self.evaluation_table.results

    def _selected_results(self) -> list[ManagedResult]:
        return [
            item
            for item in self.training_table.results + self.evaluation_table.results
            if result_key(item.path) in self.selected_paths
        ]

    def _refresh_status(self) -> None:
        selected = len(self._selected_results())
        self.status.set(
            f"训练模型 {len(self.training_results)} 项，评估结果 {len(self.evaluation_results)} 项，已选择 {selected} 项。"
        )


def open_result_manager_window(app: Any) -> ResultManagerWindow:
    """打开或聚焦结果管理窗口。 / Open or focus the result manager window."""
    existing = getattr(app, "result_manager_window", None)
    if existing is not None and existing.window.winfo_exists():
        existing.focus()
        return existing
    window = ResultManagerWindow(app)
    app.result_manager_window = window
    return window


__all__ = ["ResultManagerWindow", "open_result_manager_window"]
