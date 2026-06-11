from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import tests._path  # noqa: F401

from config import (
    get_config_value,
    get_evaluation_results_log_path,
    get_error_log_path,
    get_model_path,
    get_train_defaults,
    get_training_results_log_path,
)
from evaluation import compare_training_artifacts, run_episode
from models import LinearQModel
from models.torch_utils import checkpoint_metadata, load_torch_checkpoint
from train.artifacts import (
    TRAINING_STATUS_INCOMPLETE,
    list_incomplete_training_artifacts,
    load_training_info,
    model_path_from_info,
    training_info_status,
    training_timestamp,
)
from train.merge import merge_training_artifacts
from train.tuning import generate_tuning_candidates
from train.train_player_q import train_q_player
from ui.panels.evaluation_options import (
    EVALUATION_TARGET_LABELS,
    EvaluationModelOption,
    build_automatic_enemy_options,
    build_automatic_player_options,
    default_evaluation_pair_for_empty_selection,
    default_single_evaluation_output_directory,
    resolve_single_evaluation_request,
    single_evaluation_output_csv_path,
    training_type_evaluation_type,
    training_type_role,
)
from ui.panels.training_platform import build_training_artifact_labels
from ui.panels.training_options import (
    REFERENCE_TYPE_INITIAL_WEIGHTS,
    REFERENCE_TYPE_OPTIONS,
    build_training_reference_options,
    build_training_resume_options,
    default_training_output_directory,
    training_output_model_path,
)


def _contains_windows_absolute_path(value) -> bool:
    return bool(re.search(r"[A-Za-z]:[\\/]", json.dumps(value, ensure_ascii=False)))


class TrainingPlatformTest(unittest.TestCase):
    def test_model_paths_are_derived_from_configured_directories(self):
        configured_directory = get_config_value("paths.yaml", "models", "q_learning_player")

        self.assertFalse(Path(configured_directory).suffix)
        self.assertEqual(get_model_path("q_learning_player"), Path(configured_directory) / "player_q_model.json")

    def test_logs_configuration_is_error_only(self):
        error_log_directory = get_config_value("paths.yaml", "logs", "errors_directory")

        self.assertFalse(Path(error_log_directory).suffix)
        self.assertEqual(get_error_log_path(), Path(error_log_directory) / "log.jsonl")
        self.assertEqual(get_training_results_log_path(), Path("logs") / "system" / "training_log.jsonl")
        self.assertEqual(get_evaluation_results_log_path(), Path("logs") / "system" / "evaluation_log.jsonl")
        self.assertIsNone(get_config_value("paths.yaml", "logs", "training_results_directory"))
        self.assertIsNone(get_config_value("paths.yaml", "logs", "evaluation_results_directory"))

    def test_legacy_log_config_fields_are_ignored(self):
        def fake_get_config_value(_relative_path: str, *keys: str, default=None):
            if keys == ("logs", "directory"):
                return "logs"
            if keys in (("logs", "training_results"), ("logs", "evaluation_results"), ("logs", "errors")):
                return "legacy/file.jsonl"
            if keys == ("logs", "errors_directory"):
                return "logs/errors"
            return default

        with patch("config.get_config_value", side_effect=fake_get_config_value):
            self.assertEqual(get_training_results_log_path(), Path("logs") / "system" / "training_log.jsonl")
            self.assertEqual(get_evaluation_results_log_path(), Path("logs") / "system" / "evaluation_log.jsonl")
            self.assertEqual(get_error_log_path(), Path("logs") / "errors" / "log.jsonl")

    def test_train_latest_output_is_configured_as_directory(self):
        defaults = get_train_defaults("player_q")

        self.assertIn("latest_output_directory", defaults)
        self.assertFalse(Path(defaults["latest_output_directory"]).suffix)
        self.assertNotIn("latest_output", defaults)
        self.assertEqual(Path(defaults["runs_directory"]).parts[:3], ("models", "q_learning", "player"))
        self.assertEqual(Path(defaults["latest_output_directory"]).parts[:3], ("models", "q_learning", "player"))
        self.assertEqual(Path(defaults["latest_output_directory"]).name, "latest")

    def test_legacy_latest_output_field_is_rejected(self):
        with patch(
            "train.artifacts.get_train_defaults",
            return_value={"runs_directory": "models/q_learning/player", "latest_output": "legacy/player_q_model.json"},
        ):
            from train.artifacts import latest_training_output_path

            with self.assertRaises(KeyError):
                latest_training_output_path("player_q")

    def test_training_timestamp_uses_seconds_only(self):
        self.assertRegex(training_timestamp(), r"^\d{8}_\d{6}$")

    def test_gui_training_output_field_is_directory_only(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                output_directory = default_training_output_directory("player_q")
                output_path = training_output_model_path("player_q", output_directory)

                self.assertFalse(output_directory.suffix)
                self.assertEqual(output_path, output_directory / "player_q_model.json")
                with self.assertRaises(ValueError):
                    training_output_model_path("player_q", output_path)
                with self.assertRaises(ValueError):
                    training_output_model_path("player_dqn", Path(temp_dir) / "player_dqn_model.pt")
                with self.assertRaises(ValueError):
                    training_output_model_path("player_q", runs_root / "latest")

    def test_reference_type_extension_only_enables_initial_weights(self):
        self.assertEqual(REFERENCE_TYPE_OPTIONS, (REFERENCE_TYPE_INITIAL_WEIGHTS,))

    def test_gui_single_evaluation_output_field_is_directory_only(self):
        output_directory = default_single_evaluation_output_directory("q_ai", "random")
        output_path = single_evaluation_output_csv_path("q_ai", "random", output_directory)

        self.assertFalse(output_directory.suffix)
        self.assertRegex(output_directory.name, r"^\d{8}_\d{6}$")
        self.assertEqual(output_path, output_directory / "q_ai_vs_random.csv")
        with self.assertRaises(ValueError):
            single_evaluation_output_csv_path("q_ai", "random", output_path)
        with self.assertRaises(ValueError):
            single_evaluation_output_csv_path("q_ai", "random", output_directory / "log.jsonl")

    def test_default_training_creates_timestamped_artifact_and_latest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                summary_a = train_q_player(episodes=1, enemy_type="random", seed=101, max_steps=20)
                summary_b = train_q_player(episodes=1, enemy_type="random", seed=102, max_steps=20)

                self.assertNotEqual(summary_a.output_path, summary_b.output_path)
                self.assertTrue(summary_a.output_path.exists())
                self.assertTrue(summary_b.output_path.exists())
                self.assertTrue(summary_b.latest_output_path.exists())
                info = load_training_info(summary_a.output_path)
                self.assertIsNotNone(info)
                self.assertFalse(_contains_windows_absolute_path(info))
                artifact_labels = build_training_artifact_labels()
                self.assertEqual(artifact_labels["玩家 Q-learning | latest"], str(summary_b.latest_output_path))

                with summary_b.run_log_path.open("r", encoding="utf-8-sig") as handle:
                    last_record = json.loads([line for line in handle if line.strip()][-1])
                self.assertEqual(summary_b.run_log_path.name, "log.jsonl")
                self.assertEqual(last_record["event"], "training_completed")
                self.assertFalse(_contains_windows_absolute_path(last_record))

    def test_checkpoint_metadata_removes_path_objects(self):
        metadata = checkpoint_metadata(
            {
                "reference_model_path": Path("models/dqn/player/latest/player_dqn_model.pt"),
                "resume_run_path": Path("models/dqn/player/20260611_120000"),
                "nested": [Path("logs/errors/log.jsonl")],
            }
        )

        self.assertEqual(metadata["reference_model_path"], "models/dqn/player/latest/player_dqn_model.pt")
        self.assertEqual(metadata["resume_run_path"], "models/dqn/player/20260611_120000")
        self.assertEqual(metadata["nested"], ["logs/errors/log.jsonl"])
        self.assertFalse(any(isinstance(value, Path) for value in metadata.values()))

    def test_reference_model_initializes_q_training(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            reference = Path(temp_dir) / "reference.json"
            output = Path(temp_dir) / "output.json"
            model = LinearQModel.create()
            model.weights = [[1.0 for _ in row] for row in model.weights]
            model.save(reference)

            summary = train_q_player(
                episodes=1,
                enemy_type="random",
                seed=5,
                output=output,
                max_steps=0,
                reference_model_path=reference,
            )
            loaded = LinearQModel.load(summary.output_path)
            self.assertEqual(loaded.weights, model.weights)
            self.assertEqual(summary.reference_model_path, reference)

    def test_resume_completed_run_replaces_old_incomplete_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                stop_event = threading.Event()

                def stop_after_first(_current, _total, _state, _epsilon):
                    stop_event.set()

                stopped = train_q_player(
                    episodes=3,
                    enemy_type="random",
                    seed=21,
                    max_steps=5,
                    stop_event=stop_event,
                    progress_callback=stop_after_first,
                )
                old_run = stopped.output_path.parent
                self.assertEqual(stopped.status, TRAINING_STATUS_INCOMPLETE)
                self.assertTrue(old_run.exists())
                resumed_output = runs_root / "20260610_130000" / "player_q_model.json"

                resumed = train_q_player(
                    episodes=3,
                    enemy_type="random",
                    seed=21,
                    max_steps=5,
                    output=resumed_output,
                    resume_run_path=old_run,
                )

                self.assertEqual(resumed.status, "completed")
                self.assertEqual(resumed.completed_episodes, 3)
                self.assertIsNone(resumed.reference_model_path)
                self.assertEqual(resumed.output_path, resumed_output)
                self.assertNotEqual(resumed.output_path.parent, old_run)
                self.assertFalse(old_run.exists())
                self.assertTrue(resumed.output_path.parent.exists())
                info = load_training_info(resumed.output_path)
                self.assertIsNotNone(info)
                self.assertIsNone(info.get("reference_model_path"))

    def test_resume_stopped_run_replaces_old_incomplete_with_new_incomplete(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                first_stop = threading.Event()

                def stop_after_first(_current, _total, _state, _epsilon):
                    first_stop.set()

                first = train_q_player(
                    episodes=4,
                    enemy_type="random",
                    seed=31,
                    max_steps=5,
                    stop_event=first_stop,
                    progress_callback=stop_after_first,
                )
                old_run = first.output_path.parent

                second_stop = threading.Event()

                def stop_after_resume_episode(_current, _total, _state, _epsilon):
                    second_stop.set()

                second = train_q_player(
                    episodes=4,
                    enemy_type="random",
                    seed=31,
                    max_steps=5,
                    resume_run_path=old_run,
                    stop_event=second_stop,
                    progress_callback=stop_after_resume_episode,
                )

                self.assertEqual(second.status, TRAINING_STATUS_INCOMPLETE)
                self.assertEqual(second.completed_episodes, 2)
                self.assertFalse(old_run.exists())
                self.assertTrue(second.run_log_path.exists())
                with second.run_log_path.open("r", encoding="utf-8-sig") as handle:
                    last_record = json.loads([line for line in handle if line.strip()][-1])
                self.assertEqual(last_record["event"], "training_stopped")

    def test_resume_rejects_target_below_completed_episodes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                stop_event = threading.Event()

                def stop_after_second(current, _total, _state, _epsilon):
                    if current >= 2:
                        stop_event.set()

                stopped = train_q_player(
                    episodes=4,
                    enemy_type="random",
                    seed=41,
                    max_steps=5,
                    stop_event=stop_event,
                    progress_callback=stop_after_second,
                )
                with self.assertRaises(ValueError):
                    train_q_player(
                        episodes=1,
                        enemy_type="random",
                        seed=41,
                        max_steps=5,
                        resume_run_path=stopped.output_path.parent,
                    )

    def test_training_reference_and_resume_options_are_separate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                reference = runs_root / "20260610_120000" / "player_q_model.json"
                reference.parent.mkdir(parents=True, exist_ok=True)
                LinearQModel.create().save(reference)
                (reference.parent / "info.json").write_text(
                    json.dumps(
                        {
                            "created_at": "2026-06-10T12:00:00",
                            "training_type": "player_q",
                            "status": "completed",
                            "model_path": reference.as_posix(),
                        }
                    ),
                    encoding="utf-8",
                )
                incomplete = runs_root / "20260610_121000" / "player_q_model.json"
                incomplete.parent.mkdir(parents=True, exist_ok=True)
                LinearQModel.create().save(incomplete)
                (incomplete.parent / "info.json").write_text(
                    json.dumps(
                        {
                            "created_at": "2026-06-10T12:10:00",
                            "training_type": "player_q",
                            "status": "incomplete",
                            "model_path": incomplete.as_posix(),
                            "reference_model_path": reference.as_posix(),
                            "target_episodes": 5,
                            "completed_episodes": 2,
                        }
                    ),
                    encoding="utf-8",
                )

                reference_options = build_training_reference_options("player_q")
                resume_options = build_training_resume_options("player_q")

                self.assertTrue(any(value == reference for value in reference_options.values()))
                self.assertFalse(any(value == incomplete for value in reference_options.values()))
                self.assertEqual(len([value for value in resume_options.values() if value is not None]), 1)
                resume_artifact = next(value for value in resume_options.values() if value is not None)
                self.assertEqual(Path(resume_artifact["reference_model_path"]), reference)

    def test_resume_with_deleted_previous_reference_keeps_reference_empty(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                stop_event = threading.Event()

                def stop_after_first(_current, _total, _state, _epsilon):
                    stop_event.set()

                stopped = train_q_player(
                    episodes=3,
                    enemy_type="random",
                    seed=61,
                    max_steps=5,
                    stop_event=stop_event,
                    progress_callback=stop_after_first,
                )
                missing_reference = runs_root / "deleted_reference" / "player_q_model.json"
                info = json.loads(stopped.info_path.read_text(encoding="utf-8"))
                info["reference_model_path"] = missing_reference.as_posix()
                stopped.info_path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

                reference_options = build_training_reference_options("player_q")
                resumed = train_q_player(
                    episodes=3,
                    enemy_type="random",
                    seed=61,
                    max_steps=5,
                    resume_run_path=stopped.output_path.parent,
                )

                self.assertFalse(any(value == stopped.output_path for value in reference_options.values()))
                self.assertFalse(any(value == missing_reference for value in reference_options.values()))
                self.assertIsNone(resumed.reference_model_path)
                resumed_info = load_training_info(resumed.output_path)
                self.assertIsNotNone(resumed_info)
                self.assertIsNone(resumed_info.get("reference_model_path"))

    def test_missing_status_with_unfinished_progress_counts_as_incomplete(self):
        info = {
            "training_type": "player_q",
            "target_episodes": 10,
            "completed_episodes": 4,
        }

        self.assertEqual(training_info_status(info), TRAINING_STATUS_INCOMPLETE)

    def test_stopped_run_without_info_is_not_available_for_resume(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "runs"
            latest_root = Path(temp_dir) / "latest"
            with self._patched_player_q_roots(runs_root, latest_root):
                run_dir = runs_root / "20260610_121000"
                model_path = run_dir / "player_q_model.json"
                run_dir.mkdir(parents=True, exist_ok=True)
                LinearQModel.create().save(model_path)
                self._write_jsonl(
                    run_dir / "log.jsonl",
                    [
                        {
                            "event": "training_stopped",
                            "training_type": "player_q",
                            "timestamp": "2026-06-10T12:10:00",
                            "parameters": {
                                "output": model_path.as_posix(),
                                "target_episodes": 10,
                                "completed_episodes": 4,
                            },
                            "summary": {
                                "output_path": model_path.as_posix(),
                                "status": "incomplete",
                                "target_episodes": 10,
                                "completed_episodes": 4,
                            },
                        }
                    ],
                )

                resume_options = build_training_resume_options("player_q")
                incomplete = list_incomplete_training_artifacts("player_q")

                self.assertIsNone(load_training_info(run_dir))
                self.assertEqual(incomplete, [])
                self.assertEqual(len([value for value in resume_options.values() if value is not None]), 0)

    def test_dqn_stop_writes_incomplete_artifact_when_torch_is_available(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed.")

        from train import train_dqn_player

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "dqn_runs"
            latest_root = Path(temp_dir) / "dqn_latest"
            with self._patched_training_roots("player_dqn", runs_root, latest_root):
                stop_event = threading.Event()

                def stop_after_first(_current, _total, _state, _epsilon, _device):
                    stop_event.set()

                summary = train_dqn_player(
                    episodes=2,
                    enemy_type="random",
                    seed=51,
                    max_steps=0,
                    device="cpu",
                    batch_size=1,
                    min_replay_size=1,
                    replay_capacity=10,
                    target_update_interval=1,
                    stability={"enabled": False},
                    stop_event=stop_event,
                    progress_callback=stop_after_first,
                )

                self.assertEqual(summary.status, TRAINING_STATUS_INCOMPLETE)
                self.assertTrue(summary.info_path.exists())
                self.assertEqual(summary.info_path.name, "info.json")
                self.assertTrue(summary.run_log_path.exists())
                checkpoint = load_torch_checkpoint(torch, summary.output_path, map_location="cpu")
                self.assertEqual(checkpoint["stability"]["status"], TRAINING_STATUS_INCOMPLETE)

    def test_dqn_resume_after_second_stop_keeps_checkpoint_weights_only_loadable(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed.")

        from train import train_dqn_player

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            runs_root = Path(temp_dir) / "dqn_runs"
            latest_root = Path(temp_dir) / "dqn_latest"
            with self._patched_training_roots("player_dqn", runs_root, latest_root):
                first_stop = threading.Event()

                def stop_first(_current, _total, _state, _epsilon, _device):
                    first_stop.set()

                first = train_dqn_player(
                    episodes=3,
                    enemy_type="random",
                    seed=71,
                    max_steps=0,
                    device="cpu",
                    batch_size=1,
                    min_replay_size=1,
                    replay_capacity=10,
                    target_update_interval=1,
                    stability={"enabled": False},
                    stop_event=first_stop,
                    progress_callback=stop_first,
                )

                second_stop = threading.Event()

                def stop_second(_current, _total, _state, _epsilon, _device):
                    second_stop.set()

                second = train_dqn_player(
                    episodes=3,
                    enemy_type="random",
                    seed=71,
                    max_steps=0,
                    device="cpu",
                    batch_size=1,
                    min_replay_size=1,
                    replay_capacity=10,
                    target_update_interval=1,
                    stability={"enabled": False},
                    resume_run_path=first.output_path.parent,
                    stop_event=second_stop,
                    progress_callback=stop_second,
                )

                checkpoint = load_torch_checkpoint(torch, second.output_path, map_location="cpu")
                self.assertEqual(checkpoint["stability"]["status"], TRAINING_STATUS_INCOMPLETE)
                self.assertIsInstance(checkpoint["stability"]["resume_run_path"], str)

                resumed = train_dqn_player(
                    episodes=3,
                    enemy_type="random",
                    seed=71,
                    max_steps=0,
                    device="cpu",
                    batch_size=1,
                    min_replay_size=1,
                    replay_capacity=10,
                    target_update_interval=1,
                    stability={"enabled": False},
                    resume_run_path=second.output_path.parent,
                )
                self.assertEqual(resumed.status, "completed")

    @contextmanager
    def _patched_player_q_roots(self, runs_root: Path, latest_root: Path):
        with self._patched_training_roots("player_q", runs_root, latest_root) as defaults:
            yield defaults

    @contextmanager
    def _patched_training_roots(self, target_training_type: str, runs_root: Path, latest_root: Path):
        original_defaults = get_train_defaults(target_training_type)

        def fake_get_train_defaults(training_type: str):
            defaults = get_train_defaults(training_type)
            if training_type == target_training_type:
                defaults["runs_directory"] = str(runs_root)
                defaults["latest_output_directory"] = str(latest_root)
            return defaults

        with patch("train.artifacts.get_train_defaults", side_effect=fake_get_train_defaults):
            yield original_defaults

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    def test_q_merge_averages_weights(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path_a = Path(temp_dir) / "a.json"
            path_b = Path(temp_dir) / "b.json"
            output = Path(temp_dir) / "merged.json"
            model_a = LinearQModel.create()
            model_b = LinearQModel.create()
            model_a.weights = [[1.0 for _ in row] for row in model_a.weights]
            model_b.weights = [[3.0 for _ in row] for row in model_b.weights]
            model_a.save(path_a)
            model_b.save(path_b)

            summary = merge_training_artifacts(path_a, path_b, output=output, weight_a=0.25)
            merged = LinearQModel.load(summary.output_path)

            self.assertEqual(merged.weights[0][0], 2.5)
            self.assertTrue(summary.info_path.exists())

    def test_q_merge_rejects_incompatible_models(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path_a = Path(temp_dir) / "a.json"
            path_b = Path(temp_dir) / "b.json"
            LinearQModel.create().save(path_a)
            path_b.write_text('{"model_type":"other","actions":[],"weights":[]}', encoding="utf-8")

            with self.assertRaises(ValueError):
                merge_training_artifacts(path_a, path_b)

    def test_dqn_merge_averages_state_dict_when_torch_is_available(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("PyTorch is not installed.")

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path_a = Path(temp_dir) / "player_a.pt"
            path_b = Path(temp_dir) / "player_b.pt"
            output = Path(temp_dir) / "player_merged.pt"
            torch.save({"model_state_dict": {"weight": torch.tensor([1.0, 1.0])}}, path_a)
            torch.save({"model_state_dict": {"weight": torch.tensor([3.0, 3.0])}}, path_b)

            summary = merge_training_artifacts(path_a, path_b, output=output, weight_a=0.25)
            checkpoint = load_torch_checkpoint(torch, summary.output_path, map_location="cpu")

            self.assertTrue(torch.equal(checkpoint["model_state_dict"]["weight"], torch.tensor([2.5, 2.5])))

    def test_run_episode_uses_specific_model_path(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "specific.json"
            LinearQModel.create().save(path)
            state = run_episode(
                player_type="q_ai",
                enemy_type="random",
                seed=5,
                max_steps=5,
                player_model_path=path,
            )
            self.assertGreaterEqual(state.max_tile, 2)

    def test_compare_training_artifacts_uses_same_role(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path_a = Path(temp_dir) / "a.json"
            path_b = Path(temp_dir) / "b.json"
            LinearQModel.create().save(path_a)
            LinearQModel.create().save(path_b)

            comparison = compare_training_artifacts(path_a, path_b, episodes=1, seed=1)

            self.assertIn(comparison.winner, ("A", "B"))
            self.assertEqual(comparison.stats_a.episodes, 1)

    def test_tuning_candidates_are_stable_for_player_and_enemy_rank_direction(self):
        player_candidates = generate_tuning_candidates("player", "q_learning", count=3)
        enemy_candidates = generate_tuning_candidates("enemy", "q_learning", count=3)

        self.assertEqual([item.name for item in player_candidates], ["baseline", "faster", "explore"])
        self.assertEqual([item.name for item in enemy_candidates], ["baseline", "faster", "explore"])

    def test_single_evaluation_maps_training_type_to_complete_side(self):
        self.assertEqual(training_type_role("player_q"), "player")
        self.assertEqual(training_type_evaluation_type("player_q"), "q_ai")
        self.assertEqual(training_type_role("enemy_dqn"), "enemy")
        self.assertEqual(training_type_evaluation_type("enemy_dqn"), "dqn_enemy")

    def test_single_evaluation_target_order_starts_with_automatic_options(self):
        self.assertEqual(
            tuple(EVALUATION_TARGET_LABELS.values()),
            ("自动玩家", "玩家模型", "自动敌人", "敌对模型"),
        )

    def test_single_evaluation_default_pair_supports_automatic_targets(self):
        self.assertEqual(
            default_evaluation_pair_for_empty_selection("auto_player", "随机敌人"),
            ("heuristic", "random"),
        )
        self.assertEqual(
            default_evaluation_pair_for_empty_selection("auto_enemy", "启发式玩家"),
            ("heuristic", "random"),
        )

    def test_single_evaluation_request_uses_only_selected_model_path(self):
        player_option = EvaluationModelOption(
            label="player",
            path=Path("models/q_learning/player/20260610/player_q_model.json"),
            training_type="player_q",
            role="player",
            evaluation_type="q_ai",
            created_at="20260610",
        )
        player_request = resolve_single_evaluation_request(player_option, "随机敌人")
        self.assertEqual(player_request.player_type, "q_ai")
        self.assertEqual(player_request.enemy_type, "random")
        self.assertEqual(player_request.player_model_path, player_option.path)
        self.assertIsNone(player_request.enemy_model_path)

        enemy_option = EvaluationModelOption(
            label="enemy",
            path=Path("models/dqn/enemy/20260610/enemy_dqn_model.pt"),
            training_type="enemy_dqn",
            role="enemy",
            evaluation_type="dqn_enemy",
            created_at="20260610",
        )
        enemy_request = resolve_single_evaluation_request(enemy_option, "启发式玩家")
        self.assertEqual(enemy_request.player_type, "heuristic")
        self.assertEqual(enemy_request.enemy_type, "dqn_enemy")
        self.assertIsNone(enemy_request.player_model_path)
        self.assertEqual(enemy_request.enemy_model_path, enemy_option.path)

    def test_single_evaluation_can_run_automatic_player_options(self):
        options = build_automatic_player_options()
        labels = [option.label for option in options]

        self.assertEqual(labels, ["启发式玩家", "轻量 Q 玩家", "深度 DQN 玩家"])
        heuristic = options[0]
        q_latest = options[1]
        heuristic_request = resolve_single_evaluation_request(heuristic, "贪心敌人")
        q_latest_request = resolve_single_evaluation_request(q_latest, "随机敌人")

        self.assertEqual(heuristic_request.player_type, "heuristic")
        self.assertEqual(heuristic_request.enemy_type, "greedy")
        self.assertIsNone(heuristic_request.player_model_path)
        self.assertIsNone(heuristic_request.enemy_model_path)
        self.assertEqual(q_latest_request.player_type, "q_ai")
        self.assertIsNone(q_latest_request.player_model_path)

    def test_single_evaluation_can_run_automatic_enemy_options(self):
        options = build_automatic_enemy_options()
        labels = [option.label for option in options]

        self.assertEqual(labels, ["随机敌人", "贪心敌人", "轻量 Q 敌人", "深度 DQN 敌人"])
        greedy = options[1]
        q_latest = options[2]
        greedy_request = resolve_single_evaluation_request(greedy, "启发式玩家")
        q_latest_request = resolve_single_evaluation_request(q_latest, "随机玩家")

        self.assertEqual(greedy_request.player_type, "heuristic")
        self.assertEqual(greedy_request.enemy_type, "greedy")
        self.assertIsNone(greedy_request.player_model_path)
        self.assertIsNone(greedy_request.enemy_model_path)
        self.assertEqual(q_latest_request.player_type, "random")
        self.assertEqual(q_latest_request.enemy_type, "q_enemy")
        self.assertIsNone(q_latest_request.enemy_model_path)


if __name__ == "__main__":
    unittest.main()
