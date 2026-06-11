"""模型训练面板和后台训练任务。 / Model training panel and background training tasks."""

from __future__ import annotations

import threading
import tkinter as tk
from queue import Empty, Queue
from pathlib import Path
from typing import Any
from tkinter import ttk

from train import train_dqn_enemy, train_dqn_player, train_q_enemy, train_q_player
from train.artifacts import (
    TRAINING_STATUS_INCOMPLETE,
    resolve_info_path,
)
from ui.components.controls import create_action_button, create_message_area, create_select, create_stepper, set_button_visual
from ui.settings.layout import (
    BUTTON_BAR_HEIGHT,
    FIELD_ROW_HEIGHT,
    FORM_FIELD_WIDTH,
    FORM_HEIGHT,
    FORM_LABEL_WIDTH,
    FORM_SECOND_FIELD_WIDTH,
    FORM_SECOND_LABEL_WIDTH,
    FORM_WIDTH,
    RUN_ACTION_BUTTON_WIDTH,
    RUN_ACTION_GAP,
    RUN_ACTION_PROGRESS_WIDTH,
    RUN_ACTION_ROW_WIDTH,
    lock_widget_size,
)
from ui.settings.options import (
    ENEMY_LABELS,
    ENEMY_TYPES_BY_LABEL,
    PLAYER_LABELS,
    PLAYER_TYPES_BY_LABEL,
    TRAINING_ALGORITHM_LABELS,
    TRAINING_TARGET_LABELS,
    TRAINING_TYPE_LABELS,
)
from ui.settings.theme import BUTTON_BUSY, BUTTON_NORMAL
from ui.panels.training_options import (
    NO_REFERENCE_LABEL,
    NO_RESUME_LABEL,
    build_training_reference_options,
    build_training_resume_options,
    default_enemy_type_for_algorithm,
    default_player_type_for_algorithm,
    default_training_output_directory,
    training_output_model_path,
)
from utils.training_log import log_error

PREVIEW_REFRESH_MS = 5000


class TrainingPanel:
    """模型训练界面的状态和后台训练控制器。 / State and background-training controller for the training UI."""
    def __init__(self, app: Any, parent: ttk.Frame, defaults: dict[str, Any]):
        self.app = app
        self.target = tk.StringVar(value=TRAINING_TARGET_LABELS[defaults["target"]])
        self.algorithm = tk.StringVar(value=TRAINING_ALGORITHM_LABELS[defaults["algorithm"]])
        self.enemy_type = tk.StringVar(value=ENEMY_LABELS[self._default_enemy_type()])
        self.player_type = tk.StringVar(value=PLAYER_LABELS[self._default_player_type()])
        self.episodes = tk.IntVar(value=defaults["episodes"])
        self.seed = tk.StringVar(value=defaults["seed"] or "")
        self._auto_output = True
        self._updating_output = False
        self.output = tk.StringVar(value="")
        self._set_output(self._default_output())
        self.output.trace_add("write", self._mark_custom_output)
        self.reference_model = tk.StringVar(value=NO_REFERENCE_LABEL)
        self.resume_run = tk.StringVar(value=NO_RESUME_LABEL)
        self.reference_options: dict[str, Path | None] = {}
        self.resume_options: dict[str, dict[str, Any] | None] = {}
        self.reference_select: tk.Frame | None = None
        self.resume_select: tk.Frame | None = None
        self.status = tk.StringVar(value="已准备好训练 AI 模型。")
        self.progress = tk.IntVar(value=0)
        self.running = False
        self.stop_event: threading.Event | None = None
        self.queue: Queue[tuple[str, object]] = Queue()
        self.latest_preview: tuple[list[list[int]], int, int, int] | None = None

        self._build(parent)
        self.refresh_target()

    def _build(self, parent: ttk.Frame) -> None:
        training = ttk.LabelFrame(parent, text="模型训练设置", width=FORM_WIDTH, height=FORM_HEIGHT, padding=16)
        lock_widget_size(training, width=FORM_WIDTH, height=FORM_HEIGHT)
        training.grid(row=0, column=0, sticky="nsew")
        training.columnconfigure(0, weight=0, minsize=FORM_LABEL_WIDTH)
        training.columnconfigure(1, weight=0, minsize=FORM_FIELD_WIDTH)
        training.columnconfigure(2, weight=0, minsize=FORM_SECOND_LABEL_WIDTH)
        training.columnconfigure(3, weight=0, minsize=FORM_SECOND_FIELD_WIDTH)
        for row in range(6):
            training.rowconfigure(row, weight=0, minsize=FIELD_ROW_HEIGHT)
        training.rowconfigure(6, weight=0, minsize=BUTTON_BAR_HEIGHT + 12)
        training.rowconfigure(7, weight=0, minsize=72)

        ttk.Label(training, text="训练对象").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=7)
        create_select(
            training,
            self.target,
            ("玩家 AI", "敌对 AI"),
            command=lambda _: self.refresh_target(),
        ).grid(row=0, column=1, sticky="ew", pady=7)
        ttk.Label(training, text="算法").grid(row=0, column=2, sticky="w", padx=(16, 10), pady=7)
        create_select(
            training,
            self.algorithm,
            tuple(TRAINING_ALGORITHM_LABELS.values()),
            command=lambda _: self.refresh_target(),
        ).grid(row=0, column=3, sticky="ew", pady=7)

        self.enemy_label = ttk.Label(training, text="对手敌人")
        self.enemy_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=7)
        self.enemy_menu = create_select(
            training,
            self.enemy_type,
            tuple(ENEMY_LABELS.values()),
        )
        self.enemy_menu.grid(row=1, column=1, columnspan=3, sticky="ew", pady=7)

        self.player_label = ttk.Label(training, text="固定玩家")
        self.player_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=7)
        self.player_menu = create_select(
            training,
            self.player_type,
            tuple(PLAYER_LABELS.values()),
        )
        self.player_menu.grid(row=1, column=1, columnspan=3, sticky="ew", pady=7)

        ttk.Label(training, text="训练局数").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=7)
        create_stepper(
            training,
            self.episodes,
            from_=1,
            to=100000,
            width=10,
        ).grid(
            row=2,
            column=1,
            sticky="ew",
            pady=7,
        )

        ttk.Label(training, text="随机种子").grid(row=2, column=2, sticky="w", padx=(16, 10), pady=7)
        ttk.Entry(training, textvariable=self.seed).grid(row=2, column=3, sticky="ew", pady=7, ipady=2)

        ttk.Label(training, text="参考模型").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=7)
        self.reference_select_host = ttk.Frame(training, style="Panel.TFrame", height=FIELD_ROW_HEIGHT - 14)
        lock_widget_size(self.reference_select_host, height=FIELD_ROW_HEIGHT - 14)
        self.reference_select_host.grid(row=3, column=1, columnspan=3, sticky="ew", pady=7)
        self.reference_select_host.columnconfigure(0, weight=1)

        ttk.Label(training, text="继续训练").grid(row=4, column=0, sticky="w", padx=(0, 10), pady=7)
        self.resume_select_host = ttk.Frame(training, style="Panel.TFrame", height=FIELD_ROW_HEIGHT - 14)
        lock_widget_size(self.resume_select_host, height=FIELD_ROW_HEIGHT - 14)
        self.resume_select_host.grid(row=4, column=1, columnspan=3, sticky="ew", pady=7)
        self.resume_select_host.columnconfigure(0, weight=1)

        ttk.Label(training, text="输出目录").grid(row=5, column=0, sticky="w", padx=(0, 10), pady=7)
        ttk.Entry(training, textvariable=self.output).grid(
            row=5,
            column=1,
            columnspan=3,
            sticky="ew",
            pady=7,
            ipady=2,
        )

        action_row = ttk.Frame(training, style="Panel.TFrame", width=RUN_ACTION_ROW_WIDTH, height=BUTTON_BAR_HEIGHT)
        lock_widget_size(action_row, width=RUN_ACTION_ROW_WIDTH, height=BUTTON_BAR_HEIGHT)
        action_row.grid(row=6, column=0, columnspan=4, sticky="w", pady=(10, 0))
        action_row.columnconfigure(0, weight=0, minsize=RUN_ACTION_BUTTON_WIDTH)
        action_row.columnconfigure(1, weight=0, minsize=RUN_ACTION_GAP)
        action_row.columnconfigure(2, weight=0, minsize=RUN_ACTION_PROGRESS_WIDTH)
        action_row.rowconfigure(0, weight=0, minsize=BUTTON_BAR_HEIGHT)

        self.button = create_action_button(action_row, text="训练 AI 模型", command=self.start)
        self.button.grid(row=0, column=0, sticky="nsew")
        self.progress_bar = ttk.Progressbar(
            action_row,
            variable=self.progress,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.grid(row=0, column=2, sticky="ew")
        create_message_area(training, self.status, height=72).grid(
            row=7,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(8, 0),
        )

    def refresh_target(self) -> None:
        if self.target.get() == "玩家 AI":
            self.player_label.grid_remove()
            self.player_menu.grid_remove()
            self.enemy_label.grid()
            self.enemy_menu.grid()
            self.enemy_type.set(ENEMY_LABELS[self._default_enemy_type()])
            self._reload_artifact_selects()
            self.refresh_output()
            self._refresh_button()
            self.status.set(f"将训练玩家 AI：算法为{self.algorithm.get()}，只需选择对手敌人。")
        else:
            self.enemy_label.grid_remove()
            self.enemy_menu.grid_remove()
            self.player_label.grid()
            self.player_menu.grid()
            self.player_type.set(PLAYER_LABELS[self._default_player_type()])
            self._reload_artifact_selects()
            self.refresh_output()
            self._refresh_button()
            self.status.set(f"将训练敌对 AI：算法为{self.algorithm.get()}，只需选择固定玩家。")

    def start(self) -> None:
        if self.running:
            self.stop()
            return

        try:
            episodes = int(self.episodes.get())
        except (TypeError, ValueError, tk.TclError):
            self.status.set("训练局数必须是正整数。")
            return
        if episodes < 1:
            self.status.set("训练局数至少为 1。")
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

        try:
            output_path = training_output_model_path(self._config_key(), self.output.get())
        except ValueError as exc:
            self.status.set(str(exc))
            return
        output = str(output_path)
        reference_model_path = self._selected_reference_path()
        resume_artifact = self._selected_resume_artifact()
        resume_run_path = self._selected_resume_run_path()
        if resume_artifact is not None:
            completed = int(resume_artifact.get("completed_episodes") or 0)
            if episodes < completed:
                self.status.set(f"继续训练总局数不能低于当前已训练局数 {completed}。")
                return
        enemy_type = ENEMY_TYPES_BY_LABEL[self.enemy_type.get()]
        player_type = PLAYER_TYPES_BY_LABEL[self.player_type.get()]

        self.running = True
        self.stop_event = threading.Event()
        self.progress.set(0)
        self._refresh_button()
        self.status.set(f"正在训练 {episodes} 局...")
        self.latest_preview = None

        # 训练可能很慢，放入 daemon 线程；界面通过队列接收进度和最终摘要。
        # Training can be slow, so it runs in a daemon thread while the UI receives queued progress.
        worker = threading.Thread(
            target=self._worker,
            args=(
                self.target.get(),
                self.algorithm.get(),
                enemy_type,
                player_type,
                episodes,
                seed,
                output,
                reference_model_path,
                resume_run_path,
                self.stop_event,
            ),
            daemon=True,
        )
        worker.start()
        self.app.root.after(100, self._poll_queue)
        self.app.root.after(PREVIEW_REFRESH_MS, self._refresh_preview)

    def stop(self) -> None:
        if not self.running or self.stop_event is None:
            return
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.button.configure(text="停止中...", state=tk.DISABLED)
        self.status.set("已请求停止训练，当前局结束后会保存未完成进度。")

    def _worker(
        self,
        target: str,
        algorithm: str,
        enemy_type: str,
        player_type: str,
        episodes: int,
        seed: int | None,
        output: str | None,
        reference_model_path: Path | None,
        resume_run_path: Path | None,
        stop_event: threading.Event | None,
    ) -> None:
        try:
            if target == "敌对 AI" and algorithm == "深度 DQN":
                summary = train_dqn_enemy(
                    episodes=episodes,
                    player_type=player_type,
                    seed=seed,
                    output=output,
                    reference_model_path=reference_model_path,
                    resume_run_path=resume_run_path,
                    stop_event=stop_event,
                    progress_callback=lambda current, total, state, epsilon, device: self.queue.put(
                        (
                            "progress",
                            (
                                target,
                                current,
                                total,
                                state.max_tile,
                                state.score,
                                epsilon,
                                device,
                                [row[:] for row in state.board],
                                state.steps,
                            ),
                        )
                    ),
                )
            elif target == "敌对 AI":
                summary = train_q_enemy(
                    episodes=episodes,
                    player_type=player_type,
                    seed=seed,
                    output=output,
                    reference_model_path=reference_model_path,
                    resume_run_path=resume_run_path,
                    stop_event=stop_event,
                    progress_callback=lambda current, total, state, epsilon: self.queue.put(
                        (
                            "progress",
                            (
                                target,
                                current,
                                total,
                                state.max_tile,
                                state.score,
                                epsilon,
                                "cpu",
                                [row[:] for row in state.board],
                                state.steps,
                            ),
                        )
                    ),
                )
            elif algorithm == "深度 DQN":
                summary = train_dqn_player(
                    episodes=episodes,
                    enemy_type=enemy_type,
                    seed=seed,
                    output=output,
                    reference_model_path=reference_model_path,
                    resume_run_path=resume_run_path,
                    stop_event=stop_event,
                    progress_callback=lambda current, total, state, epsilon, device: self.queue.put(
                        (
                            "progress",
                            (
                                target,
                                current,
                                total,
                                state.max_tile,
                                state.score,
                                epsilon,
                                device,
                                [row[:] for row in state.board],
                                state.steps,
                            ),
                        )
                    ),
                )
            else:
                summary = train_q_player(
                    episodes=episodes,
                    enemy_type=enemy_type,
                    seed=seed,
                    output=output,
                    reference_model_path=reference_model_path,
                    resume_run_path=resume_run_path,
                    stop_event=stop_event,
                    progress_callback=lambda current, total, state, epsilon: self.queue.put(
                        (
                            "progress",
                            (
                                target,
                                current,
                                total,
                                state.max_tile,
                                state.score,
                                epsilon,
                                "cpu",
                                [row[:] for row in state.board],
                                state.steps,
                            ),
                        )
                    ),
                )
        except Exception as exc:  # pragma: no cover - surfaced through the GUI.
            log_error(
                "gui_training_worker",
                exc,
                {
                    "target": target,
                    "algorithm": algorithm,
                    "enemy_type": enemy_type,
                    "player_type": player_type,
                    "episodes": episodes,
                    "seed": seed,
                    "output": output,
                    "reference_model_path": reference_model_path,
                    "resume_run_path": resume_run_path,
                },
            )
            self.queue.put(("error", str(exc)))
            return
        self.queue.put(("done", (target, summary)))

    def _poll_queue(self) -> None:
        while True:
            try:
                event, payload = self.queue.get_nowait()
            except Empty:
                break

            if event == "progress":
                target, current, total, max_tile, score, epsilon, device, board, steps = payload
                self.latest_preview = (board, score, steps, max_tile)
                self.progress.set(int(current * 100 / total))
                self.status.set(
                    f"{target} {current}/{total} | 玩家最大块 {max_tile} | 玩家分数 {score} | 探索率 {epsilon:.2f} | 设备 {device}"
                )
            elif event == "done":
                target, summary = payload
                self.running = False
                self.stop_event = None
                self._refresh_button()
                self.progress.set(100 if summary.status != TRAINING_STATUS_INCOMPLETE else self.progress.get())
                self._render_latest_preview()
                self.app.game_panel.reload_ai_player()
                if hasattr(self.app, "training_platform_panel"):
                    self.app.training_platform_panel._reload_selects()
                result_window = getattr(self.app, "result_manager_window", None)
                if result_window is not None and result_window.window.winfo_exists():
                    result_window.refresh()
                self._reload_artifact_selects()
                if self._auto_output:
                    self.refresh_output()
                if summary.status == TRAINING_STATUS_INCOMPLETE:
                    self.status.set(
                        f"训练已停止，进度 {summary.completed_episodes}/{summary.target_episodes} 已保存到 {summary.output_path}"
                    )
                elif target == "敌对 AI":
                    self.status.set(
                        f"敌对模型已保存到 {summary.output_path} | 玩家平均最大块 {summary.average_player_max_tile:.1f} | 最好压制 {summary.best_suppressed_max_tile}"
                    )
                else:
                    self.status.set(
                        f"玩家模型已保存到 {summary.output_path} | 平均最大块 {summary.average_max_tile:.1f} | 最好 {summary.best_max_tile}"
                    )
            elif event == "error":
                self.running = False
                self.stop_event = None
                self._refresh_button()
                self.status.set(f"训练失败：{payload}")

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
        self.app.render_preview(board, score, steps, max_tile, "训练")

    def _refresh_button(self) -> None:
        idle_text = "训练敌对 AI" if self.target.get() == "敌对 AI" else "训练玩家 AI"
        if self.running:
            if self.stop_event is not None and self.stop_event.is_set():
                self.button.configure(text="停止中...", state=tk.DISABLED)
            else:
                self.button.configure(text="停止训练", state=tk.NORMAL)
            set_button_visual(self.button, BUTTON_BUSY)
        else:
            self.button.configure(text=idle_text, state=tk.NORMAL)
            set_button_visual(self.button, BUTTON_NORMAL)

    def _reload_artifact_selects(self) -> None:
        training_type = self._config_key()
        previous_reference = self._selected_reference_path()
        previous_resume = self.resume_run.get()

        self.reference_options = build_training_reference_options(training_type)
        if previous_reference is not None and not self._select_reference_path(previous_reference):
            self.reference_model.set(NO_REFERENCE_LABEL)
        elif previous_reference is None and self.reference_model.get() not in self.reference_options:
            self.reference_model.set(NO_REFERENCE_LABEL)

        self.resume_options = build_training_resume_options(training_type)
        if previous_resume not in self.resume_options:
            self.resume_run.set(NO_RESUME_LABEL)

        if self.reference_select is not None:
            self.reference_select.destroy()
        self.reference_select = create_select(
            self.reference_select_host,
            self.reference_model,
            tuple(self.reference_options),
        )
        self.reference_select.grid(row=0, column=0, sticky="ew")

        if self.resume_select is not None:
            self.resume_select.destroy()
        self.resume_select = create_select(
            self.resume_select_host,
            self.resume_run,
            tuple(self.resume_options),
            command=lambda _: self._on_resume_selected(),
        )
        self.resume_select.grid(row=0, column=0, sticky="ew")

    def _on_resume_selected(self) -> None:
        artifact = self._selected_resume_artifact()
        if artifact is None:
            self.refresh_output()
            return

        completed = int(artifact.get("completed_episodes") or 0)
        target = int(artifact.get("target_episodes") or completed)
        self.episodes.set(max(target, completed))

        self.reference_model.set(NO_REFERENCE_LABEL)
        previous_reference = resolve_info_path(artifact.get("reference_model_path"))
        if previous_reference is not None and previous_reference.exists():
            if not self._select_reference_path(previous_reference):
                self.reference_model.set(NO_REFERENCE_LABEL)
        self._auto_output = True
        self.refresh_output()

    def _selected_reference_path(self) -> Path | None:
        return self.reference_options.get(self.reference_model.get())

    def _selected_resume_artifact(self) -> dict[str, Any] | None:
        return self.resume_options.get(self.resume_run.get())

    def _selected_resume_run_path(self) -> Path | None:
        artifact = self._selected_resume_artifact()
        if artifact is None:
            return None
        info_path = resolve_info_path(artifact.get("info_path"))
        return info_path.parent if info_path is not None else None

    def _select_reference_path(self, path: Path) -> bool:
        try:
            target = path.resolve()
        except OSError:
            target = path
        for label, candidate in self.reference_options.items():
            if candidate is None:
                continue
            try:
                matched = candidate.resolve() == target
            except OSError:
                matched = candidate == path
            if matched:
                self.reference_model.set(label)
                return True
        return False

    def _default_output(self) -> str:
        return str(default_training_output_directory(self._config_key()))

    def refresh_output(self) -> None:
        if self._auto_output:
            self._set_output(self._default_output())

    def _set_output(self, value: str) -> None:
        self._updating_output = True
        try:
            self.output.set(value)
        finally:
            self._updating_output = False

    def _mark_custom_output(self, *_args: object) -> None:
        if not self._updating_output:
            self._auto_output = False

    def _default_enemy_type(self) -> str:
        return default_enemy_type_for_algorithm(self._algorithm_key())

    def _default_player_type(self) -> str:
        return default_player_type_for_algorithm(self._algorithm_key())

    def _config_key(self) -> str:
        target = "enemy" if self.target.get() == "敌对 AI" else "player"
        return f"{target}_{self._algorithm_key()}"

    def _algorithm_key(self) -> str:
        return "dqn" if self.algorithm.get() == "深度 DQN" else "q"


def build_training_panel(app: Any, parent: ttk.Frame) -> TrainingPanel:
    """创建模型训练面板。 / Build the model training panel."""
    return TrainingPanel(app, parent, app.ui_defaults["training"])


__all__ = ["TrainingPanel", "build_training_panel"]
