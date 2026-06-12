"""训练评估平台面板，支持比较、合并和自动调参。 / Training platform panel for comparison, merging, and auto-tuning."""

from __future__ import annotations

import threading
import tkinter as tk
from queue import Empty, Queue
from typing import Any
from tkinter import ttk

from domain.evaluation import compare_training_artifacts
from domain.train import list_training_artifacts, merge_training_artifacts, run_auto_tuning
from domain.train.artifacts import TRAINING_MODEL_FILENAMES, latest_training_output_path, model_path_from_info
from ui.components import (
    GRID_CONTROL_OPTIONS,
    create_action_button,
    create_message_area,
    create_select,
    create_stepper,
    create_text_entry,
    set_button_visual,
)
from ui.panels.shared import create_field_label, display_timestamp, make_grid_placer, unique_label
from ui.settings.layout.grid import create_area_panel
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
        model_path = model_path_from_info(artifact)
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
    label = unique_label(
        labels,
        f"{TRAINING_TYPE_LABELS.get(training_type, training_type)} | {display_timestamp(created_at)}",
    )
    labels[label] = model_path


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
        self.platform, self._area_grid = create_area_panel(parent, "训练智能评估平台")
        place = make_grid_placer(self._area_grid)

        place(create_field_label(self.platform, text="训练成果 A"), 0, 0, colspan=3)
        self.artifact_a_select = create_select(
            self.platform,
            self.artifact_a,
            tuple(self._artifact_values()),
            **GRID_CONTROL_OPTIONS,
        )
        place(self.artifact_a_select, 0, 3, colspan=17)

        place(create_field_label(self.platform, text="训练成果 B"), 1, 0, colspan=3)
        self.artifact_b_select = create_select(
            self.platform,
            self.artifact_b,
            tuple(self._artifact_values()),
            **GRID_CONTROL_OPTIONS,
        )
        place(self.artifact_b_select, 1, 3, colspan=17)

        place(create_field_label(self.platform, text="评估局数"), 2, 0, colspan=3)
        place(
            create_stepper(self.platform, self.episodes, from_=1, to=100000, width=10, **GRID_CONTROL_OPTIONS),
            2,
            3,
            colspan=7,
        )
        place(create_field_label(self.platform, text="随机种子"), 2, 10, colspan=3)
        place(create_text_entry(self.platform, self.seed, **GRID_CONTROL_OPTIONS), 2, 13, colspan=7)

        place(create_field_label(self.platform, text="调参对象"), 3, 0, colspan=3)
        place(
            create_select(self.platform, self.target, tuple(TRAINING_TARGET_LABELS.values()), **GRID_CONTROL_OPTIONS),
            3,
            3,
            colspan=7,
        )
        place(create_field_label(self.platform, text="调参算法"), 3, 10, colspan=3)
        place(
            create_select(
                self.platform,
                self.algorithm,
                tuple(TRAINING_ALGORITHM_LABELS.values()),
                **GRID_CONTROL_OPTIONS,
            ),
            3,
            13,
            colspan=7,
        )

        compare_button = create_action_button(self.platform, text="比较 A/B", command=lambda: self.start("compare"))
        merge_button = create_action_button(self.platform, text="合并 A/B", command=lambda: self.start("merge"))
        tune_button = create_action_button(self.platform, text="自动调参试跑", command=lambda: self.start("tune"))
        refresh_button = create_action_button(self.platform, text="刷新成果", command=self._reload_selects)
        self.buttons = [compare_button, merge_button, tune_button, refresh_button]
        place(compare_button, 4, 0, colspan=5)
        place(merge_button, 4, 5, colspan=5)
        place(tune_button, 4, 10, colspan=5)
        place(refresh_button, 4, 15, colspan=5)

        self.progress_bar = ttk.Progressbar(self.platform, variable=self.progress, maximum=100, mode="determinate")
        place(self.progress_bar, 5, 0, colspan=20, sticky="ew")
        place(create_message_area(self.platform, self.status, **GRID_CONTROL_OPTIONS), 6, 0, colspan=20)
        place(create_message_area(self.platform, self.result, **GRID_CONTROL_OPTIONS), 7, 0, rowspan=2, colspan=20)

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
        self.artifact_a_select = create_select(
            self.platform,
            self.artifact_a,
            tuple(self._artifact_values()),
            **GRID_CONTROL_OPTIONS,
        )
        self._area_grid.grid_widget(self.artifact_a_select, 0, 3, colspan=17)
        self.artifact_b_select = create_select(
            self.platform,
            self.artifact_b,
            tuple(self._artifact_values()),
            **GRID_CONTROL_OPTIONS,
        )
        self._area_grid.grid_widget(self.artifact_b_select, 1, 3, colspan=17)

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
