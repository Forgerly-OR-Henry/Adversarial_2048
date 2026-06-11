"""训练评估平台面板，支持比较、合并和自动调参。 / Training platform panel for comparison, merging, and auto-tuning."""

from __future__ import annotations

import threading
import tkinter as tk
from queue import Empty, Queue
from typing import Any
from tkinter import ttk

from evaluation import compare_training_artifacts
from train import list_training_artifacts, merge_training_artifacts, run_auto_tuning
from train.artifacts import TRAINING_MODEL_FILENAMES, latest_training_output_path
from ui.components.controls import (
    create_action_button,
    create_message_area,
    create_select,
    create_stepper,
    set_button_visual,
)
from ui.settings.layout import (
    BUTTON_BAR_HEIGHT,
    FIELD_ROW_HEIGHT,
    FORM_HEIGHT,
    FORM_FIELD_WIDTH,
    FORM_LABEL_WIDTH,
    FORM_SECOND_FIELD_WIDTH,
    FORM_SECOND_LABEL_WIDTH,
    FORM_WIDTH,
    TRAINING_PLATFORM_RESULT_HEIGHT,
    lock_widget_size,
)
from ui.settings.options import TRAINING_ALGORITHM_LABELS, TRAINING_TARGET_LABELS, TRAINING_TYPE_LABELS
from ui.settings.theme import BUTTON_BUSY, BUTTON_NORMAL
from utils.training_log import log_error

NO_ARTIFACT_LABEL = "暂无训练成果"


def build_training_artifact_labels() -> dict[str, str]:
    """构造训练评估平台的成果下拉选项。 / Build artifact dropdown options for the training platform."""
    labels: dict[str, str] = {}
    for training_type in TRAINING_MODEL_FILENAMES:
        latest_path = latest_training_output_path(training_type)
        if latest_path.exists():
            _append_artifact_label(labels, training_type, "latest", str(latest_path))

    for artifact in list_training_artifacts():
        model_path = _artifact_model_path(artifact)
        if model_path is None:
            continue
        training_type = str(artifact.get("training_type", "unknown"))
        created_at = str(artifact.get("created_at", ""))
        _append_artifact_label(labels, training_type, created_at, str(model_path))
    return labels


def _append_artifact_label(
    labels: dict[str, str],
    training_type: str,
    created_at: str,
    model_path: str,
) -> None:
    label = _unique_artifact_label(
        labels,
        f"{TRAINING_TYPE_LABELS.get(training_type, training_type)} | {_display_timestamp(created_at)}",
    )
    labels[label] = model_path


def _artifact_model_path(artifact: dict[str, Any]) -> str | None:
    value = artifact.get("model_path")
    if not value:
        return None
    return str(value)


def _unique_artifact_label(labels: dict[str, str], base_label: str) -> str:
    if base_label not in labels:
        return base_label
    index = 2
    while f"{base_label} #{index}" in labels:
        index += 1
    return f"{base_label} #{index}"


def _display_timestamp(value: str) -> str:
    return value.replace("T", " ") if value else "unknown"


class TrainingPlatformPanel:
    """训练评估平台的产物选择和操作控制器。 / Artifact selection and operation controller for the training platform."""
    def __init__(self, app: Any, parent: ttk.Frame):
        self.app = app
        self.artifact_labels: dict[str, str] = {}
        self.artifact_a = tk.StringVar()
        self.artifact_b = tk.StringVar()
        self.episodes = tk.IntVar(value=30)
        self.seed = tk.StringVar(value="")
        self.target = tk.StringVar(value=TRAINING_TARGET_LABELS["player"])
        self.algorithm = tk.StringVar(value=TRAINING_ALGORITHM_LABELS["q_learning"])
        self.status = tk.StringVar(value="已准备好评估训练成果。")
        self.result = tk.StringVar(value="")
        self.progress = tk.IntVar(value=0)
        self.running = False
        self.queue: Queue[tuple[str, object]] = Queue()
        self.buttons: list[tk.Button] = []

        self._refresh_artifacts()
        self._build(parent)

    def _build(self, parent: ttk.Frame) -> None:
        self.platform = ttk.LabelFrame(
            parent,
            text="训练智能评估平台",
            width=FORM_WIDTH,
            height=FORM_HEIGHT,
            padding=16,
        )
        lock_widget_size(self.platform, width=FORM_WIDTH, height=FORM_HEIGHT)
        self.platform.grid(row=0, column=0, sticky="nsew")
        self.platform.columnconfigure(0, weight=0, minsize=FORM_LABEL_WIDTH)
        self.platform.columnconfigure(1, weight=0, minsize=FORM_FIELD_WIDTH)
        self.platform.columnconfigure(2, weight=0, minsize=FORM_SECOND_LABEL_WIDTH)
        self.platform.columnconfigure(3, weight=0, minsize=FORM_SECOND_FIELD_WIDTH)
        for row in range(4):
            self.platform.rowconfigure(row, weight=0, minsize=FIELD_ROW_HEIGHT)
        self.platform.rowconfigure(4, weight=0, minsize=BUTTON_BAR_HEIGHT + 16)
        self.platform.rowconfigure(5, weight=0, minsize=34)
        self.platform.rowconfigure(6, weight=0, minsize=64)
        self.platform.rowconfigure(7, weight=0, minsize=TRAINING_PLATFORM_RESULT_HEIGHT)

        ttk.Label(self.platform, text="训练成果 A").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=7)
        self.artifact_a_select = create_select(self.platform, self.artifact_a, tuple(self._artifact_values()))
        self.artifact_a_select.grid(row=0, column=1, columnspan=3, sticky="ew", pady=7)

        ttk.Label(self.platform, text="训练成果 B").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=7)
        self.artifact_b_select = create_select(self.platform, self.artifact_b, tuple(self._artifact_values()))
        self.artifact_b_select.grid(row=1, column=1, columnspan=3, sticky="ew", pady=7)

        ttk.Label(self.platform, text="评估局数").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=7)
        create_stepper(self.platform, self.episodes, from_=1, to=100000, width=10).grid(
            row=2,
            column=1,
            sticky="ew",
            pady=7,
        )
        ttk.Label(self.platform, text="随机种子").grid(row=2, column=2, sticky="w", padx=(16, 10), pady=7)
        ttk.Entry(self.platform, textvariable=self.seed).grid(row=2, column=3, sticky="ew", pady=7, ipady=2)

        ttk.Label(self.platform, text="调参对象").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=7)
        create_select(self.platform, self.target, tuple(TRAINING_TARGET_LABELS.values())).grid(
            row=3,
            column=1,
            sticky="ew",
            pady=7,
        )
        ttk.Label(self.platform, text="调参算法").grid(row=3, column=2, sticky="w", padx=(16, 10), pady=7)
        create_select(self.platform, self.algorithm, tuple(TRAINING_ALGORITHM_LABELS.values())).grid(
            row=3,
            column=3,
            sticky="ew",
            pady=7,
        )

        button_bar = ttk.Frame(self.platform, style="Panel.TFrame", height=BUTTON_BAR_HEIGHT)
        lock_widget_size(button_bar, height=BUTTON_BAR_HEIGHT)
        button_bar.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        button_bar.rowconfigure(0, weight=1)
        for column in range(4):
            button_bar.columnconfigure(column, weight=1, uniform="training_platform_actions")

        compare_button = create_action_button(button_bar, text="比较 A/B", command=lambda: self.start("compare"))
        merge_button = create_action_button(button_bar, text="合并 A/B", command=lambda: self.start("merge"))
        tune_button = create_action_button(button_bar, text="自动调参试跑", command=lambda: self.start("tune"))
        refresh_button = create_action_button(button_bar, text="刷新成果", command=self._reload_selects)
        self.buttons = [compare_button, merge_button, tune_button, refresh_button]
        compare_button.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        merge_button.grid(row=0, column=1, sticky="nsew", padx=6)
        tune_button.grid(row=0, column=2, sticky="nsew", padx=6)
        refresh_button.grid(row=0, column=3, sticky="nsew", padx=(6, 0))

        self.progress_bar = ttk.Progressbar(self.platform, variable=self.progress, maximum=100, mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(14, 0))
        create_message_area(self.platform, self.status, height=58).grid(
            row=6,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(12, 0),
        )
        create_message_area(self.platform, self.result, height=TRAINING_PLATFORM_RESULT_HEIGHT).grid(
            row=7,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(10, 0),
        )

    def start(self, action: str) -> None:
        if self.running:
            return
        seed = self._parse_seed()
        if seed == "invalid":
            self.status.set("随机种子必须留空或填写整数。")
            return
        try:
            episodes = int(self.episodes.get())
        except (TypeError, ValueError, tk.TclError):
            self.status.set("评估局数必须是正整数。")
            return
        if episodes < 1:
            self.status.set("评估局数至少为 1。")
            return

        path_a = self.artifact_labels.get(self.artifact_a.get())
        path_b = self.artifact_labels.get(self.artifact_b.get())
        if action in ("compare", "merge") and (not path_a or not path_b):
            self.status.set("请先选择两个训练成果。")
            return

        self.running = True
        self.progress.set(10)
        self.result.set("")
        self._refresh_buttons()
        self.status.set("任务运行中...")
        worker = threading.Thread(
            target=self._worker,
            args=(action, path_a, path_b, episodes, seed),
            daemon=True,
        )
        worker.start()
        self.app.root.after(100, self._poll_queue)

    def _worker(
        self,
        action: str,
        path_a: str | None,
        path_b: str | None,
        episodes: int,
        seed: int | None,
    ) -> None:
        try:
            if action == "compare":
                comparison = compare_training_artifacts(path_a, path_b, episodes=episodes, seed=seed)
                self.queue.put(("done", self._format_comparison(comparison)))
            elif action == "merge":
                summary = merge_training_artifacts(path_a, path_b)
                self.queue.put(("done", f"合并完成：{summary.output_path}"))
            else:
                results = run_auto_tuning(
                    target=self._target_key(),
                    algorithm=self._algorithm_key(),
                    evaluation_episodes=episodes,
                    seed=seed,
                )
                self.queue.put(("done", self._format_tuning(results)))
        except Exception as exc:  # pragma: no cover - surfaced through GUI.
            log_error(
                "gui_training_platform_worker",
                exc,
                {
                    "action": action,
                    "artifact_a": path_a,
                    "artifact_b": path_b,
                    "episodes": episodes,
                    "seed": seed,
                },
            )
            self.queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        while True:
            try:
                event, payload = self.queue.get_nowait()
            except Empty:
                break
            if event == "done":
                self.running = False
                self.progress.set(100)
                self.status.set("任务完成。")
                self.result.set(str(payload))
                self._reload_selects()
                self._refresh_buttons()
            elif event == "error":
                self.running = False
                self.progress.set(0)
                self.status.set(f"任务失败：{payload}")
                self._refresh_buttons()
        if self.running:
            self.progress.set(min(95, self.progress.get() + 1))
            self.app.root.after(300, self._poll_queue)

    def _refresh_buttons(self) -> None:
        for button in self.buttons:
            if self.running:
                button.configure(state=tk.DISABLED)
                set_button_visual(button, BUTTON_BUSY)
            else:
                button.configure(state=tk.NORMAL)
                set_button_visual(button, BUTTON_NORMAL)

    def _refresh_artifacts(self) -> None:
        self.artifact_labels = build_training_artifact_labels()
        values = self._artifact_values()
        if values:
            if self.artifact_a.get() not in self.artifact_labels:
                self.artifact_a.set(values[0])
            if self.artifact_b.get() not in self.artifact_labels:
                self.artifact_b.set(values[min(1, len(values) - 1)])
        else:
            self.artifact_a.set(NO_ARTIFACT_LABEL)
            self.artifact_b.set(NO_ARTIFACT_LABEL)

    def _reload_selects(self) -> None:
        self._refresh_artifacts()
        self.artifact_a_select.destroy()
        self.artifact_b_select.destroy()
        self.artifact_a_select = create_select(self.platform, self.artifact_a, tuple(self._artifact_values()))
        self.artifact_a_select.grid(row=0, column=1, columnspan=3, sticky="ew", pady=7)
        self.artifact_b_select = create_select(self.platform, self.artifact_b, tuple(self._artifact_values()))
        self.artifact_b_select.grid(row=1, column=1, columnspan=3, sticky="ew", pady=7)

    def _artifact_values(self) -> list[str]:
        return list(self.artifact_labels.keys()) or [NO_ARTIFACT_LABEL]

    def _parse_seed(self) -> int | None | str:
        text = self.seed.get().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return "invalid"

    def _target_key(self) -> str:
        return "enemy" if self.target.get() == TRAINING_TARGET_LABELS["enemy"] else "player"

    def _algorithm_key(self) -> str:
        return "dqn" if self.algorithm.get() == TRAINING_ALGORITHM_LABELS["dqn"] else "q_learning"

    def _format_comparison(self, comparison) -> str:
        return (
            f"胜出：{comparison.winner} | "
            f"A 平均分 {comparison.stats_a.average_score:.1f} / 平均最大块 {comparison.stats_a.average_max_tile:.1f}；"
            f"B 平均分 {comparison.stats_b.average_score:.1f} / 平均最大块 {comparison.stats_b.average_max_tile:.1f}。"
            f"{comparison.recommendation}"
        )

    def _format_tuning(self, results) -> str:
        if not results:
            return "没有生成调参结果。"
        best = results[0]
        return (
            f"最佳候选：{best.candidate.name} | 输出 {best.output_path} | "
            f"平均分 {best.stats.average_score:.1f} | 平均最大块 {best.stats.average_max_tile:.1f}"
        )


def build_training_platform_panel(app: Any, parent: ttk.Frame) -> TrainingPlatformPanel:
    """创建训练评估平台面板。 / Build the training platform panel."""
    return TrainingPlatformPanel(app, parent)


__all__ = ["TrainingPlatformPanel", "build_training_artifact_labels", "build_training_platform_panel"]
