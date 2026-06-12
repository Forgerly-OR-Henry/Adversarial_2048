"""模型训练面板和后台训练任务。 / Model training panel and background training tasks."""

from __future__ import annotations

import threading
import tkinter as tk
from queue import Empty, Queue
from pathlib import Path
from typing import Any
from tkinter import ttk

from domain.train import train_dqn_enemy, train_dqn_player, train_q_enemy, train_q_player
from domain.train.artifacts import (
    TRAINING_STATUS_INCOMPLETE,
    resolve_info_path,
)
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
from ui.settings.options import (
    ENEMY_LABELS,
    ENEMY_TYPES_BY_LABEL,
    NO_REFERENCE_LABEL,
    NO_RESUME_LABEL,
    PLAYER_LABELS,
    PLAYER_TYPES_BY_LABEL,
    REFERENCE_TYPE_OPTIONS,
    REFERENCE_TYPES_BY_LABEL,
    TRAINING_ALGORITHM_LABELS,
    TRAINING_TARGET_LABELS,
    TRAINING_TYPE_LABELS,
)
from ui.settings.theme import BUTTON_BUSY, BUTTON_NORMAL
from workflows.training import (
    build_training_reference_options,
    build_training_resume_options,
    default_enemy_type_for_algorithm,
    default_player_type_for_algorithm,
    default_training_output_directory,
    REFERENCE_TYPE_INITIAL_WEIGHTS,
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
        self.episodes = tk.StringVar(value=str(defaults["episodes"]))
        self.seed = tk.StringVar(value=defaults["seed"] or "")
        self._auto_output = True
        self._updating_output = False
        self.output = tk.StringVar(value="")
        self._set_output(self._default_output())
        self.output.trace_add("write", self._mark_custom_output)
        self.reference_model = tk.StringVar(value=NO_REFERENCE_LABEL)
        self.reference_type = tk.StringVar(value=REFERENCE_TYPE_OPTIONS[0])
        self.resume_run = tk.StringVar(value=NO_RESUME_LABEL)
        self.reference_options: dict[str, Path | None] = {}
        self.resume_options: dict[str, dict[str, Any] | None] = {}
        self.reference_select: tk.Frame | None = None
        self.resume_select: tk.Frame | None = None
        self.status = tk.StringVar(value="已准备好训练 AI 模型。")
        self.progress = tk.IntVar(value=0)
        self.running = False
        self.stop_event: threading.Event | None = None
        self.unlimited_training = False
        self.queue: Queue[tuple[str, object]] = Queue()
        self.latest_preview: tuple[list[list[int]], int, int, int] | None = None

        self._build(parent)
        self.refresh_target()

    def _build(self, parent: ttk.Frame) -> None:
        training, area = create_area_panel(parent, "模型训练设置")
        place = make_grid_placer(area)

        # 本行使用 20 列网格比例 3/6/3/8：
        # 左标签 0-3，左输入 3-9，右标签 9-12，右输入 12-20。
        # 若要调整宽度，保持四段相加为 20，并同步修改后续 col/colspan。
        # This row uses a 20-column 3/6/3/8 ratio; keep the spans summing to 20.
        place(create_field_label(training, text="训练对象"), 0, 0, colspan=3)
        place(
            create_select(
                training,
                self.target,
                ("玩家 AI", "敌对 AI"),
                command=lambda _: self.refresh_target(),
                **GRID_CONTROL_OPTIONS,
            ),
            0,
            3,
            colspan=6,
        )
        place(create_field_label(training, text="算法"), 0, 9, colspan=3)
        place(
            create_select(
                training,
                self.algorithm,
                tuple(TRAINING_ALGORITHM_LABELS.values()),
                command=lambda _: self.refresh_target(),
                **GRID_CONTROL_OPTIONS,
            ),
            0,
            12,
            colspan=8,
        )

        self.enemy_label = create_field_label(training, text="对手敌人")
        place(self.enemy_label, 1, 0, colspan=3)
        self.enemy_menu = create_select(
            training,
            self.enemy_type,
            tuple(ENEMY_LABELS.values()),
            **GRID_CONTROL_OPTIONS,
        )
        place(self.enemy_menu, 1, 3, colspan=17)

        self.player_label = create_field_label(training, text="固定玩家")
        place(self.player_label, 1, 0, colspan=3)
        self.player_menu = create_select(
            training,
            self.player_type,
            tuple(PLAYER_LABELS.values()),
            **GRID_CONTROL_OPTIONS,
        )
        place(self.player_menu, 1, 3, colspan=17)

        # 同样使用 3/6/3/8，方便“训练局数/随机种子”与上方选择行对齐。
        # Same 3/6/3/8 ratio to align episode and seed controls with the selectors above.
        place(create_field_label(training, text="训练局数"), 2, 0, colspan=3)
        place(
            create_stepper(
                training,
                self.episodes,
                from_=1,
                to=100000,
                width=10,
                **GRID_CONTROL_OPTIONS,
            ),
            2,
            3,
            colspan=6,
        )
        place(create_field_label(training, text="随机种子"), 2, 9, colspan=3)
        place(create_text_entry(training, self.seed, **GRID_CONTROL_OPTIONS), 2, 12, colspan=8)

        place(create_field_label(training, text="参考模型"), 3, 0, colspan=3)
        self.reference_select_host = ttk.Frame(training, style="Panel.TFrame")
        self.reference_select_host.columnconfigure(0, weight=1)
        self.reference_select_host.rowconfigure(0, weight=1)
        place(self.reference_select_host, 3, 3, colspan=10)
        place(create_field_label(training, text="参考类型"), 3, 13, colspan=3)
        place(
            create_select(
                training,
                self.reference_type,
                REFERENCE_TYPE_OPTIONS,
                **GRID_CONTROL_OPTIONS,
            ),
            3,
            16,
            colspan=4,
        )

        place(create_field_label(training, text="继续训练"), 4, 0, colspan=3)
        self.resume_select_host = ttk.Frame(training, style="Panel.TFrame")
        self.resume_select_host.columnconfigure(0, weight=1)
        self.resume_select_host.rowconfigure(0, weight=1)
        place(self.resume_select_host, 4, 3, colspan=17)

        place(create_field_label(training, text="输出目录"), 5, 0, colspan=3)
        place(create_text_entry(training, self.output, **GRID_CONTROL_OPTIONS), 5, 3, colspan=17)

        self.button = create_action_button(training, text="训练 AI 模型", command=self.start)
        place(self.button, 6, 0, colspan=5)
        self.progress_bar = ttk.Progressbar(
            training,
            variable=self.progress,
            maximum=100,
            mode="determinate",
        )
        place(self.progress_bar, 6, 5, colspan=15, sticky="ew")

        place(create_message_area(training, self.status, **GRID_CONTROL_OPTIONS), 7, 0, rowspan=2, colspan=20)

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

        episodes_text = self.episodes.get().strip()
        if episodes_text:
            try:
                episodes: int | None = int(episodes_text)
            except (TypeError, ValueError, tk.TclError):
                self.status.set("训练局数必须是正整数，或留空表示无限训练。")
                return
            if episodes < 1:
                self.status.set("训练局数至少为 1，或留空表示无限训练。")
                return
        else:
            episodes = None

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
        reference_type = self._selected_reference_type()
        if reference_type != REFERENCE_TYPE_INITIAL_WEIGHTS:
            self.status.set("当前仅支持使用参考模型作为起始权重。")
            return
        reference_model_path = self._selected_reference_path()
        resume_artifact = self._selected_resume_artifact()
        resume_run_path = self._selected_resume_run_path()
        if resume_artifact is not None:
            completed = int(resume_artifact.get("completed_episodes") or 0)
            if episodes is not None and episodes < completed:
                self.status.set(f"继续训练总局数不能低于当前已训练局数 {completed}。")
                return
        enemy_type = ENEMY_TYPES_BY_LABEL[self.enemy_type.get()]
        player_type = PLAYER_TYPES_BY_LABEL[self.player_type.get()]

        self.running = True
        self.stop_event = threading.Event()
        self.unlimited_training = episodes is None
        self.progress.set(0)
        if self.unlimited_training:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start(12)
        else:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
        self._refresh_button()
        self.status.set("正在无限训练，手动停止后会保存为已完成结果。" if episodes is None else f"正在训练 {episodes} 局...")
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
        if self.unlimited_training:
            self.status.set("已请求停止训练，当前局结束后会保存为已完成训练结果。")
        else:
            self.status.set("已请求停止训练，当前局结束后会保存未完成进度。")

    def _worker(
        self,
        target: str,
        algorithm: str,
        enemy_type: str,
        player_type: str,
        episodes: int | None,
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
                if total is None:
                    self.status.set(
                        f"{target} 第 {current} 局 | 无限训练 | 玩家最大块 {max_tile} | 玩家分数 {score} | 探索率 {epsilon:.2f} | 设备 {device}"
                    )
                else:
                    self.progress.set(int(current * 100 / total))
                    self.status.set(
                        f"{target} {current}/{total} | 玩家最大块 {max_tile} | 玩家分数 {score} | 探索率 {epsilon:.2f} | 设备 {device}"
                    )
            elif event == "done":
                target, summary = payload
                self.running = False
                self.stop_event = None
                self.unlimited_training = False
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
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
                self.unlimited_training = False
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
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

        self.reference_options = _reference_option_map(training_type)
        if previous_reference is not None and not self._select_reference_path(previous_reference):
            self.reference_model.set(NO_REFERENCE_LABEL)
        elif previous_reference is None and self.reference_model.get() not in self.reference_options:
            self.reference_model.set(NO_REFERENCE_LABEL)

        self.resume_options = _resume_option_map(training_type)
        if previous_resume not in self.resume_options:
            self.resume_run.set(NO_RESUME_LABEL)

        if self.reference_select is not None:
            self.reference_select.destroy()
        self.reference_select = create_select(
            self.reference_select_host,
            self.reference_model,
            tuple(self.reference_options),
            **GRID_CONTROL_OPTIONS,
        )
        self.reference_select.grid(row=0, column=0, sticky="nsew")

        if self.resume_select is not None:
            self.resume_select.destroy()
        self.resume_select = create_select(
            self.resume_select_host,
            self.resume_run,
            tuple(self.resume_options),
            command=lambda _: self._on_resume_selected(),
            **GRID_CONTROL_OPTIONS,
        )
        self.resume_select.grid(row=0, column=0, sticky="nsew")

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

    def _selected_reference_type(self) -> str:
        return REFERENCE_TYPES_BY_LABEL.get(self.reference_type.get(), "")

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


def _reference_option_map(training_type: str) -> dict[str, Path | None]:
    options: dict[str, Path | None] = {NO_REFERENCE_LABEL: None}
    for option in build_training_reference_options(training_type):
        suffix = "latest" if option.status == "latest" else display_timestamp(option.created_at)
        label = unique_label(
            options,
            f"{TRAINING_TYPE_LABELS.get(option.training_type, option.training_type)} | {suffix}",
        )
        options[label] = option.path
    return options


def _resume_option_map(training_type: str) -> dict[str, dict[str, Any] | None]:
    options: dict[str, dict[str, Any] | None] = {NO_RESUME_LABEL: None}
    for option in build_training_resume_options(training_type):
        label = unique_label(
            options,
            (
                f"{TRAINING_TYPE_LABELS.get(option.training_type, option.training_type)} | "
                f"未完成 {option.completed_episodes}/{option.target_episodes} | "
                f"{display_timestamp(option.created_at)}"
            ),
        )
        options[label] = option.artifact
    return options


__all__ = ["TrainingPanel", "build_training_panel"]
