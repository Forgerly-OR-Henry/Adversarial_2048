"""单项模型评估面板和后台评估任务。 / Single-model evaluation panel and background evaluation tasks."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from queue import Empty, Queue
from tkinter import ttk
from typing import Any

from domain.evaluation import run_experiment
from ui.components import (
    GRID_CONTROL_OPTIONS,
    create_action_button,
    create_message_area,
    create_select,
    create_stepper,
    create_text_entry,
    set_button_visual,
)
from ui.settings.layout.grid import create_area_panel
from ui.settings.options import (
    EVALUATION_TARGET_LABELS,
    EVALUATION_TARGET_OPTIONS,
    EVALUATION_TARGETS_BY_LABEL,
    ENEMY_LABELS,
    ENEMY_TYPES_BY_LABEL,
    NO_MODEL_LABEL,
    PLAYER_LABELS,
    PLAYER_TYPES_BY_LABEL,
    TRAINING_TYPE_LABELS,
)
from ui.settings.theme import BUTTON_BUSY, BUTTON_NORMAL
from workflows.evaluation import (
    EvaluationModelOption,
    SingleEvaluationRequest,
    build_automatic_enemy_options,
    build_automatic_player_options,
    build_evaluation_model_options,
    default_evaluation_pair_for_empty_selection,
    default_single_evaluation_output_directory,
    resolve_single_evaluation_request,
    single_evaluation_output_csv_path,
    training_type_evaluation_type,
    training_type_role,
)
from utils.training_log import log_error

PREVIEW_REFRESH_MS = 5000


class EvaluationPanel:
    """单项评估平台的界面状态和后台任务控制器。 / UI state and background-task controller for single evaluation."""
    def __init__(self, app: Any, parent: ttk.Frame, defaults: dict[str, Any], enemy_type: str):
        self.app = app
        self.target_type = tk.StringVar(value=EVALUATION_TARGET_LABELS["auto_player"])
        self.fixed_player_type = tk.StringVar(value=PLAYER_LABELS[defaults["player"]])
        self.fixed_enemy_type = tk.StringVar(value=ENEMY_LABELS[enemy_type])
        self.opponent_type = tk.StringVar(value=self.fixed_enemy_type.get())
        self.opponent_label = tk.StringVar(value="固定敌人")
        self.model_label = tk.StringVar(value="模型成果")
        self.model_artifact = tk.StringVar(value=NO_MODEL_LABEL)
        self.model_options: dict[str, EvaluationModelOption] = {}
        self.model_select: tk.Frame | None = None
        self.opponent_select: tk.Frame | None = None
        self.episodes = tk.IntVar(value=defaults["episodes"])
        self.seed = tk.StringVar(value=defaults["seed"] or "")
        configured_output = defaults["output"] or ""
        self._auto_output = not bool(configured_output)
        self._updating_output = False
        self.output = tk.StringVar(value=configured_output)
        if self._auto_output:
            self._set_output(self._default_output())
        self.output.trace_add("write", self._mark_custom_output)
        self.status = tk.StringVar(value="已准备好运行实验。")
        self.progress = tk.IntVar(value=0)
        self.running = False
        self.queue: Queue[tuple[str, object]] = Queue()
        self.latest_preview: tuple[list[list[int]], int, int, int] | None = None

        self._build(parent)
        self.refresh_model_options()

    def _build(self, parent: ttk.Frame) -> None:
        experiment, area = create_area_panel(parent, "单项评估平台设置")

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

        place(ttk.Label(experiment, text="评估对象"), 0, 0, colspan=3, sticky="w")
        place(
            create_select(
                experiment,
                self.target_type,
                EVALUATION_TARGET_OPTIONS,
                command=lambda _: self.refresh_model_options(),
                **GRID_CONTROL_OPTIONS,
            ),
            0,
            3,
            colspan=7,
        )

        place(ttk.Label(experiment, textvariable=self.opponent_label), 0, 10, colspan=3, sticky="w")
        self.opponent_select_host = ttk.Frame(experiment, style="Panel.TFrame")
        self.opponent_select_host.columnconfigure(0, weight=1)
        self.opponent_select_host.rowconfigure(0, weight=1)
        place(self.opponent_select_host, 0, 13, colspan=7)

        place(ttk.Label(experiment, textvariable=self.model_label), 1, 0, colspan=3, sticky="w")
        self.model_select_host = ttk.Frame(experiment, style="Panel.TFrame")
        self.model_select_host.columnconfigure(0, weight=1)
        self.model_select_host.columnconfigure(1, weight=0, minsize=96)
        self.model_select_host.rowconfigure(0, weight=1)
        place(self.model_select_host, 1, 3, colspan=17)

        refresh_button = create_action_button(
            self.model_select_host,
            text="刷新",
            command=self.refresh_model_options,
            compact=True,
        )
        refresh_button.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        place(ttk.Label(experiment, text="局数"), 2, 0, colspan=3, sticky="w")
        place(
            create_stepper(
                experiment,
                self.episodes,
                from_=1,
                to=100000,
                width=10,
                **GRID_CONTROL_OPTIONS,
            ),
            2,
            3,
            colspan=7,
        )

        place(ttk.Label(experiment, text="随机种子"), 2, 10, colspan=3, sticky="w")
        place(create_text_entry(experiment, self.seed, **GRID_CONTROL_OPTIONS), 2, 13, colspan=7)

        place(ttk.Label(experiment, text="输出目录"), 3, 0, colspan=3, sticky="w")
        place(create_text_entry(experiment, self.output, **GRID_CONTROL_OPTIONS), 3, 3, colspan=17)

        self.button = create_action_button(experiment, text="运行单项评估", command=self.start)
        place(self.button, 4, 0, colspan=5)
        self.progress_bar = ttk.Progressbar(
            experiment,
            variable=self.progress,
            maximum=100,
            mode="determinate",
        )
        place(self.progress_bar, 4, 5, colspan=15, sticky="ew")
        place(create_message_area(experiment, self.status, **GRID_CONTROL_OPTIONS), 5, 0, rowspan=2, colspan=20)

    def start(self) -> None:
        if self.running:
            return

        try:
            episodes = int(self.episodes.get())
        except (TypeError, ValueError, tk.TclError):
            self.status.set("局数必须是正整数。")
            return
        if episodes < 1:
            self.status.set("局数至少为 1。")
            return

        seed_text = self.seed.get().strip()
        if seed_text:
            try:
                seed: int | None = int(seed_text)
            except ValueError:
                self.status.set("随机种子必须留空或填写整数。")
                return
        else:
            seed = None

        selected_model = self._selected_model_option()
        if selected_model is None:
            self.status.set("请先选择一个可用的训练模型。")
            return
        try:
            request = resolve_single_evaluation_request(selected_model, self._opponent_type(selected_model.role))
        except KeyError:
            self.status.set("固定对手必须是完整算法或可默认加载的模型。")
            return

        output_directory = self.output.get().strip() or self._default_output()
        try:
            output_path = single_evaluation_output_csv_path(
                request.player_type,
                request.enemy_type,
                output_directory,
            )
        except ValueError as exc:
            self.status.set(str(exc))
            return

        self.running = True
        self.progress.set(0)
        self._refresh_button()
        self.status.set(f"正在运行 {episodes} 局实验...")
        self.latest_preview = None

        # Tkinter 只能在主线程更新界面，后台线程把结果放进 queue 后由 after 轮询消费。
        # Tkinter UI updates must stay on the main thread; the worker posts results into a queued poll.
        worker = threading.Thread(
            target=self._worker,
            args=(request, episodes, seed, output_path),
            daemon=True,
        )
        worker.start()
        self.app.root.after(100, self._poll_queue)
        self.app.root.after(PREVIEW_REFRESH_MS, self._refresh_preview)

    def _worker(
        self,
        request: SingleEvaluationRequest,
        episodes: int,
        seed: int | None,
        output: Path,
    ) -> None:
        try:
            output_path = run_experiment(
                player_type=request.player_type,
                enemy_type=request.enemy_type,
                episodes=episodes,
                seed=seed,
                output=output,
                player_model_path=request.player_model_path,
                enemy_model_path=request.enemy_model_path,
                progress_callback=lambda current, total, record: self.queue.put(
                    ("progress", (current, total, record))
                ),
                state_callback=lambda current, total, record, state: self.queue.put(
                    (
                        "preview",
                        ([row[:] for row in state.board], state.score, state.steps, state.max_tile),
                    )
                ),
            )
        except Exception as exc:  # pragma: no cover - surfaced through the GUI.
            log_error(
                "gui_experiment_worker",
                exc,
                {
                    "player_type": request.player_type,
                    "enemy_type": request.enemy_type,
                    "episodes": episodes,
                    "seed": seed,
                    "output": output,
                    "player_model_path": request.player_model_path,
                    "enemy_model_path": request.enemy_model_path,
                },
            )
            self.queue.put(("error", str(exc)))
            return
        self.queue.put(("done", str(output_path)))

    def _poll_queue(self) -> None:
        while True:
            try:
                event, payload = self.queue.get_nowait()
            except Empty:
                break

            if event == "progress":
                current, total, record = payload
                self.progress.set(int(current * 100 / total))
                self.status.set(
                    f"{current}/{total} | 最大块 {record.max_tile} | 分数 {record.score} | 步数 {record.steps}"
                )
            elif event == "done":
                self.running = False
                self._refresh_button()
                self.progress.set(100)
                self._render_latest_preview()
                if self._auto_output:
                    self.refresh_output()
                self.status.set(f"结果已保存到 {payload}")
            elif event == "error":
                self.running = False
                self._refresh_button()
                self.status.set(f"实验失败：{payload}")
            elif event == "preview":
                self.latest_preview = payload

        if self.running:
            self.app.root.after(100, self._poll_queue)

    def _refresh_preview(self) -> None:
        if not self.running:
            return
        self._render_latest_preview()
        self.app.root.after(PREVIEW_REFRESH_MS, self._refresh_preview)

    def _render_latest_preview(self) -> None:
        if self.latest_preview is None:
            return
        board, score, steps, max_tile = self.latest_preview
        self.app.render_preview(board, score, steps, max_tile, "评估")

    def _refresh_button(self) -> None:
        if self.running:
            self.button.configure(text="评估运行中...", state=tk.DISABLED)
            set_button_visual(self.button, BUTTON_BUSY)
        else:
            self.button.configure(text="运行单项评估", state=tk.NORMAL)
            set_button_visual(self.button, BUTTON_NORMAL)

    def refresh_model_options(self) -> None:
        role = EVALUATION_TARGETS_BY_LABEL[self.target_type.get()]
        self._sync_opponent_select(role)
        if role == "auto_player":
            self.model_label.set("玩家模型")
            options = build_automatic_player_options()
        elif role == "auto_enemy":
            self.model_label.set("敌人模型")
            options = build_automatic_enemy_options()
        else:
            self.model_label.set("模型成果")
            options = [option for option in build_evaluation_model_options() if option.role == role]
        self.model_options = _model_option_map(options)
        labels = tuple(self.model_options) or (NO_MODEL_LABEL,)
        if self.model_artifact.get() not in labels:
            self.model_artifact.set(labels[0])
        if self.model_select is not None:
            self.model_select.destroy()
        self.model_select = create_select(
            self.model_select_host,
            self.model_artifact,
            labels,
            command=lambda _: self.refresh_output(),
            **GRID_CONTROL_OPTIONS,
        )
        self.model_select.grid(row=0, column=0, sticky="nsew")
        self.refresh_output()

    def _sync_opponent_select(self, role: str) -> None:
        if role in ("player", "auto_player"):
            self.opponent_label.set("固定敌人")
            self.opponent_type.set(self.fixed_enemy_type.get())
            values = tuple(ENEMY_LABELS.values())
            command = self._set_fixed_enemy
        else:
            self.opponent_label.set("固定玩家")
            self.opponent_type.set(self.fixed_player_type.get())
            values = tuple(PLAYER_LABELS.values())
            command = self._set_fixed_player

        if self.opponent_select is not None:
            self.opponent_select.destroy()
        self.opponent_select = create_select(
            self.opponent_select_host,
            self.opponent_type,
            values,
            command=command,
            **GRID_CONTROL_OPTIONS,
        )
        self.opponent_select.grid(row=0, column=0, sticky="nsew")

    def _set_fixed_enemy(self, value: str) -> None:
        self.fixed_enemy_type.set(value)
        self.refresh_output()

    def _set_fixed_player(self, value: str) -> None:
        self.fixed_player_type.set(value)
        self.refresh_output()

    def refresh_output(self) -> None:
        if self._auto_output:
            self._set_output(self._default_output())

    def _default_output(self) -> str:
        selected_model = self._selected_model_option()
        if selected_model is None:
            role = EVALUATION_TARGETS_BY_LABEL[self.target_type.get()]
            player_type, enemy_type = default_evaluation_pair_for_empty_selection(role, self._opponent_type(role))
            return str(default_single_evaluation_output_directory(player_type, enemy_type))
        request = resolve_single_evaluation_request(selected_model, self._opponent_type(selected_model.role))
        player_type = request.player_type
        enemy_type = request.enemy_type
        return str(default_single_evaluation_output_directory(player_type, enemy_type))

    def _selected_model_option(self) -> EvaluationModelOption | None:
        return self.model_options.get(self.model_artifact.get())

    def _opponent_type(self, role: str) -> str:
        if role in ("player", "auto_player"):
            return ENEMY_TYPES_BY_LABEL[self.opponent_type.get()]
        return PLAYER_TYPES_BY_LABEL[self.opponent_type.get()]

    def _set_output(self, value: str) -> None:
        self._updating_output = True
        try:
            self.output.set(value)
        finally:
            self._updating_output = False

    def _mark_custom_output(self, *_args: object) -> None:
        if not self._updating_output:
            self._auto_output = False


def build_evaluation_panel(app: Any, parent: ttk.Frame) -> EvaluationPanel:
    """创建单项评估面板。 / Build the single-evaluation panel."""
    return EvaluationPanel(app, parent, app.ui_defaults["experiment"], app.initial_enemy_type)


def _model_option_map(options: list[EvaluationModelOption]) -> dict[str, EvaluationModelOption]:
    labels: dict[str, EvaluationModelOption] = {}
    for option in options:
        labels[_unique_model_label(labels, _model_option_label(option))] = option
    return labels


def _model_option_label(option: EvaluationModelOption) -> str:
    if option.role == "auto_player":
        return PLAYER_LABELS[option.evaluation_type]
    if option.role == "auto_enemy":
        return ENEMY_LABELS[option.evaluation_type]
    return f"{TRAINING_TYPE_LABELS[option.training_type]} | {_display_timestamp(option.created_at)}"


def _unique_model_label(options: dict[str, EvaluationModelOption], base_label: str) -> str:
    if base_label not in options:
        return base_label
    index = 2
    while f"{base_label} #{index}" in options:
        index += 1
    return f"{base_label} #{index}"


def _display_timestamp(value: str) -> str:
    return value.replace("T", " ") if value else "unknown"


__all__ = [
    "EvaluationPanel",
    "build_evaluation_panel",
]
