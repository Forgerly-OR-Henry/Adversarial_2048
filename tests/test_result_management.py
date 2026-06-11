from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import tests._path  # noqa: F401

from results import (
    ManagedResult,
    compact_managed_result_logs,
    compact_system_logs,
    delete_managed_results,
    list_managed_results,
    promote_training_result_to_latest,
)
from train.artifacts import TRAINING_STATUS_INCOMPLETE


class ResultManagementTest(unittest.TestCase):
    def setUp(self):
        self.root = Path("tmp_result_management_tests")
        if self.root.exists():
            shutil.rmtree(self.root)
        self.runs_root = self.root / "models" / "q_learning" / "player"
        self.latest_root = self.runs_root / "latest"
        self.experiment_root = self.root / "outputs" / "experiments"
        self.log_root = self.root / "logs" / "system"
        self.error_root = self.root / "logs" / "errors"
        self.training_log = self.log_root / "training_log.jsonl"
        self.evaluation_log = self.log_root / "evaluation_log.jsonl"
        self.error_log = self.error_root / "log.jsonl"
        self.training_roots = {"player_q": self.runs_root}
        self.latest_roots = {"player_q": self.latest_root}

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_lists_training_latest_and_evaluation_results(self):
        run_dir = self._create_training_run("20260610_120000")
        latest_model = self.latest_root / "player_q_model.json"
        latest_model.parent.mkdir(parents=True, exist_ok=True)
        latest_model.write_text("{}", encoding="utf-8")
        eval_dir, csv_path = self._create_evaluation_run("20260610_130000", "heuristic_vs_random.csv")

        results = list_managed_results(
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        )
        paths = {item.path for item in results}

        self.assertIn(run_dir, paths)
        self.assertIn(self.latest_root, paths)
        self.assertIn(eval_dir, paths)
        self.assertTrue(any(item.is_latest and item.path == self.latest_root for item in results))

    def test_deletes_training_run_and_related_logs(self):
        run_dir = self._create_training_run("20260610_120001")
        other_dir = self._create_training_run("20260610_120002")
        self._write_jsonl(
            self.training_log,
            [
                {"event": "training_completed", "summary": {"output_path": f"{run_dir.as_posix()}/player_q_model.json"}},
                {"event": "training_completed", "summary": {"output_path": f"{other_dir.as_posix()}/player_q_model.json"}},
            ],
        )

        result = self._result_for(run_dir)
        summary = delete_managed_results(
            [result],
            training_log_path=self.training_log,
            evaluation_log_path=self.evaluation_log,
            error_log_path=self.error_log,
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        )

        self.assertFalse(run_dir.exists())
        self.assertEqual(summary.removed_training_log_rows, 1)
        remaining = self._read_jsonl(self.training_log)
        self.assertEqual(len(remaining), 1)
        self.assertIn(other_dir.as_posix(), remaining[0]["summary"]["output_path"])

    def test_compacts_selected_training_logs_to_latest_row(self):
        run_dir = self._create_training_run("20260610_120007")
        local_log = run_dir / "log.jsonl"
        self._write_jsonl(
            local_log,
            [
                {"event": "training_progress", "episode": 1},
                {"event": "training_progress", "episode": 2},
                {"event": "training_completed", "episode": 3},
            ],
        )

        summary = compact_managed_result_logs(
            [self._result_for(run_dir)],
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        )

        self.assertTrue(run_dir.exists())
        self.assertEqual(summary.compacted_paths, (local_log,))
        self.assertEqual(summary.removed_log_rows, 2)
        remaining = self._read_jsonl(local_log)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["event"], "training_completed")

    def test_compacts_selected_evaluation_logs_and_prefixed_logs(self):
        eval_dir, csv_path = self._create_evaluation_run("20260610_131500", "q_ai_vs_random.csv")
        standard_log = eval_dir / "log.jsonl"
        prefixed_log = eval_dir / "evaluation_log.jsonl"
        self._write_jsonl(
            standard_log,
            [
                {"event": "evaluation_progress", "episode": 1},
                {"event": "evaluation_completed", "episode": 2},
            ],
        )
        self._write_jsonl(
            prefixed_log,
            [
                {"event": "preview", "episode": 1},
                {"event": "preview", "episode": 2},
                {"event": "preview", "episode": 3},
            ],
        )

        result = ManagedResult(
            result_type="evaluation",
            path=eval_dir,
            display_path=csv_path,
            created_at="2026-06-10T13:15:00",
            size_bytes=10,
        )
        summary = compact_managed_result_logs(
            [result],
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        )

        self.assertEqual(set(summary.compacted_paths), {standard_log, prefixed_log})
        self.assertEqual(summary.removed_log_rows, 3)
        self.assertEqual(self._read_jsonl(standard_log)[0]["event"], "evaluation_completed")
        self.assertEqual(self._read_jsonl(prefixed_log)[0]["episode"], 3)

    def test_compacts_system_and_error_logs_to_recent_ten_rows(self):
        extra_system_log = self.log_root / "worker_log.jsonl"
        self._write_jsonl(self.training_log, [{"event": "training", "index": index} for index in range(12)])
        self._write_jsonl(self.evaluation_log, [{"event": "evaluation", "index": index} for index in range(8)])
        self._write_jsonl(extra_system_log, [{"event": "worker", "index": index} for index in range(11)])
        self._write_jsonl(self.error_log, [{"event": "error", "index": index} for index in range(13)])

        summary = compact_system_logs(
            keep=10,
            log_directory=self.root / "logs",
            training_log_path=self.training_log,
            evaluation_log_path=self.evaluation_log,
            error_log_path=self.error_log,
        )

        self.assertEqual(set(summary.compacted_paths), {self.training_log, extra_system_log, self.error_log})
        self.assertEqual(summary.removed_log_rows, 6)
        self.assertEqual([row["index"] for row in self._read_jsonl(self.training_log)], list(range(2, 12)))
        self.assertEqual([row["index"] for row in self._read_jsonl(extra_system_log)], list(range(1, 11)))
        self.assertEqual([row["index"] for row in self._read_jsonl(self.error_log)], list(range(3, 13)))
        self.assertEqual(len(self._read_jsonl(self.evaluation_log)), 8)

    def test_latest_requires_explicit_permission(self):
        latest_model = self.latest_root / "player_q_model.json"
        latest_model.parent.mkdir(parents=True, exist_ok=True)
        latest_model.write_text("{}", encoding="utf-8")
        result = ManagedResult(
            result_type="training",
            path=self.latest_root,
            display_path=latest_model,
            created_at="2026-06-10T12:00:00",
            size_bytes=2,
            is_latest=True,
            training_type="player_q",
        )

        with self.assertRaises(PermissionError):
            delete_managed_results(
                [result],
                training_log_path=self.training_log,
                evaluation_log_path=self.evaluation_log,
                error_log_path=self.error_log,
                training_roots=self.training_roots,
                latest_roots=self.latest_roots,
                experiment_directory=self.experiment_root,
            )

        delete_managed_results(
            [result],
            allow_latest=True,
            training_log_path=self.training_log,
            evaluation_log_path=self.evaluation_log,
            error_log_path=self.error_log,
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        )
        self.assertFalse(self.latest_root.exists())

    def test_promotes_training_run_to_latest_and_replaces_old_latest(self):
        run_dir = self._create_training_run("20260610_120003")
        source_model = run_dir / "player_q_model.json"
        source_model.write_text('{"source":"new"}', encoding="utf-8")
        old_latest = self.latest_root / "player_q_model.json"
        old_extra = self.latest_root / "old_checkpoint.tmp"
        old_latest.parent.mkdir(parents=True, exist_ok=True)
        old_latest.write_text('{"source":"old"}', encoding="utf-8")
        old_extra.write_text("old", encoding="utf-8")

        summary = promote_training_result_to_latest(
            self._result_for(run_dir),
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
        )

        self.assertTrue(run_dir.exists())
        self.assertTrue(summary.removed_previous_latest)
        self.assertEqual(summary.training_type, "player_q")
        self.assertEqual(summary.latest_model_path, old_latest)
        self.assertEqual(old_latest.read_text(encoding="utf-8"), '{"source":"new"}')
        self.assertFalse(old_extra.exists())

    def test_rejects_promoting_evaluation_result_to_latest(self):
        csv_path = self._create_evaluation_csv("heuristic_vs_random.csv")
        result = ManagedResult(
            result_type="evaluation",
            path=csv_path,
            display_path=csv_path,
            created_at="2026-06-10T12:00:00",
            size_bytes=10,
        )

        with self.assertRaises(ValueError):
            promote_training_result_to_latest(
                result,
                training_roots=self.training_roots,
                latest_roots=self.latest_roots,
            )

    def test_rejects_promoting_incomplete_training_result_to_latest(self):
        run_dir = self._create_training_run("20260610_120004", status=TRAINING_STATUS_INCOMPLETE)

        with self.assertRaises(ValueError):
            promote_training_result_to_latest(
                self._result_for(run_dir),
                training_roots=self.training_roots,
                latest_roots=self.latest_roots,
            )
        with self.assertRaises(ValueError):
            promote_training_result_to_latest(
                run_dir,
                training_roots=self.training_roots,
                latest_roots=self.latest_roots,
            )

    def test_missing_status_with_unfinished_progress_is_listed_incomplete(self):
        run_dir = self._create_training_run("20260610_120005", status=None, completed_episodes=3, target_episodes=8)

        result = self._result_for(run_dir)

        self.assertEqual(result.status, TRAINING_STATUS_INCOMPLETE)
        with self.assertRaises(ValueError):
            promote_training_result_to_latest(
                run_dir,
                training_roots=self.training_roots,
                latest_roots=self.latest_roots,
            )

    def test_run_log_without_info_is_not_listed_as_training_result(self):
        run_dir = self._create_run_log_without_info("20260610_120006", completed_episodes=3, target_episodes=8)

        results = list_managed_results(
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        )

        self.assertNotIn(run_dir, {result.path for result in results})
        with self.assertRaises(ValueError):
            promote_training_result_to_latest(
                run_dir,
                training_roots=self.training_roots,
                latest_roots=self.latest_roots,
            )

    def test_deletes_evaluation_csv_and_related_logs(self):
        eval_dir, csv_path = self._create_evaluation_run("20260610_131000", "q_ai_vs_random.csv")
        local_log = eval_dir / "log.jsonl"
        self._write_jsonl(
            local_log,
            [{"event": "evaluation_completed", "summary": {"output_path": csv_path.as_posix()}}],
        )
        self._write_jsonl(
            self.error_log,
            [
                {"event": "error", "parameters": {"output": csv_path.as_posix()}},
                {"event": "error", "parameters": {"output": "unrelated.csv"}},
            ],
        )
        result = ManagedResult(
            result_type="evaluation",
            path=eval_dir,
            display_path=csv_path,
            created_at="2026-06-10T12:00:00",
            size_bytes=10,
            log_hints=(eval_dir.as_posix(), csv_path.as_posix()),
        )

        summary = delete_managed_results(
            [result],
            training_log_path=self.training_log,
            evaluation_log_path=self.evaluation_log,
            error_log_path=self.error_log,
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        )

        self.assertFalse(eval_dir.exists())
        self.assertEqual(summary.removed_evaluation_log_rows, 0)
        self.assertEqual(summary.removed_error_log_rows, 1)
        self.assertEqual(len(self._read_jsonl(self.error_log)), 1)

    def test_rejects_paths_outside_managed_directories(self):
        outside = self.root / "not-managed" / "file.txt"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("x", encoding="utf-8")

        with self.assertRaises(ValueError):
            delete_managed_results(
                [outside],
                training_log_path=self.training_log,
                evaluation_log_path=self.evaluation_log,
                error_log_path=self.error_log,
                training_roots=self.training_roots,
                latest_roots=self.latest_roots,
                experiment_directory=self.experiment_root,
            )
        self.assertTrue(outside.exists())

    def _create_training_run(
        self,
        name: str,
        *,
        status: str | None = "completed",
        completed_episodes: int | None = None,
        target_episodes: int = 10,
    ) -> Path:
        run_dir = self.runs_root / name
        model_path = run_dir / "player_q_model.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{}", encoding="utf-8")
        completed = (
            completed_episodes
            if completed_episodes is not None
            else 4 if status == TRAINING_STATUS_INCOMPLETE else target_episodes
        )
        payload = {
            "created_at": "2026-06-10T12:00:00",
            "training_type": "player_q",
            "target_episodes": target_episodes,
            "completed_episodes": completed,
            "model_path": model_path.as_posix(),
            "summary": {
                "output_path": model_path.as_posix(),
                "info_path": (run_dir / "info.json").as_posix(),
            },
        }
        if status is not None:
            payload["status"] = status
        (run_dir / "info.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return run_dir

    def _create_run_log_without_info(
        self,
        name: str,
        *,
        completed_episodes: int,
        target_episodes: int,
    ) -> Path:
        run_dir = self.runs_root / name
        model_path = run_dir / "player_q_model.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path.write_text("{}", encoding="utf-8")
        self._write_jsonl(
            run_dir / "log.jsonl",
            [
                {
                    "event": "training_stopped",
                    "training_type": "player_q",
                    "timestamp": "2026-06-10T12:00:00",
                    "parameters": {
                        "output": model_path.as_posix(),
                        "target_episodes": target_episodes,
                        "completed_episodes": completed_episodes,
                    },
                    "summary": {
                        "output_path": model_path.as_posix(),
                        "status": "incomplete",
                        "target_episodes": target_episodes,
                        "completed_episodes": completed_episodes,
                    },
                }
            ],
        )
        return run_dir

    def _create_evaluation_run(self, name: str, csv_name: str) -> tuple[Path, Path]:
        run_dir = self.experiment_root / name
        csv_path = run_dir / csv_name
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("episode,score\n1,2\n", encoding="utf-8")
        return run_dir, csv_path

    def _create_evaluation_csv(self, name: str) -> Path:
        csv_path = self.experiment_root / name
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("episode,score\n1,2\n", encoding="utf-8")
        return csv_path

    def _result_for(self, path: Path) -> ManagedResult:
        for result in list_managed_results(
            training_roots=self.training_roots,
            latest_roots=self.latest_roots,
            experiment_directory=self.experiment_root,
        ):
            if result.path == path:
                return result
        raise AssertionError(f"Result not listed: {path}")

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8-sig") as handle:
            return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    unittest.main()
