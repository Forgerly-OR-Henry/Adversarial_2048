from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch
from pathlib import Path

import tests._path  # noqa: F401

from cli.commands import run_auto
from cli.parser import build_parser
from domain.evaluation import default_experiment_path, run_episode, run_experiment
from ui.settings.options import PLAYER_LABELS, PLAYER_TYPES_BY_LABEL, TRAINING_ALGORITHM_LABELS
from domain.players.heuristic_player import HeuristicPlayer
from domain.players import create_player
from utils import ExperimentRecorder


class EndToEndTest(unittest.TestCase):
    def test_auto_episode_completes(self):
        state = run_episode(player_type="heuristic", enemy_type="greedy", seed=42, max_steps=500)
        self.assertGreaterEqual(state.max_tile, 2)
        self.assertGreaterEqual(state.score, 0)
        self.assertGreaterEqual(state.steps, 0)

    def test_auto_csv_has_expected_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.csv"
            saved_path = run_auto(
                player_type="heuristic",
                enemy_type="greedy",
                episodes=10,
                seed=100,
                output=str(output),
            )

            self.assertEqual(saved_path, output)
            with output.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ExperimentRecorder.fieldnames)
                rows = list(reader)
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["episode (局数)"], "1")
            self.assertEqual(rows[0]["player_type (玩家)"], "heuristic (启发式玩家)")
            self.assertEqual(rows[0]["enemy_type (敌人)"], "greedy (贪心敌人)")

    def test_run_experiment_shared_by_cli_and_gui(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "shared.csv"
            saved_path = run_experiment(
                player_type="random",
                enemy_type="random",
                episodes=2,
                seed=1,
                output=output,
            )
            self.assertEqual(saved_path, output)
            self.assertTrue(output.exists())
            with (output.parent / "log.jsonl").open("r", encoding="utf-8-sig") as handle:
                last_record = json.loads([line for line in handle if line.strip()][-1])
            self.assertEqual(last_record["event"], "evaluation_completed")
            self.assertEqual(last_record["summary"]["output_path"], output.name)

    def test_default_experiment_path_uses_outputs(self):
        output_path = default_experiment_path("heuristic", "random")
        self.assertEqual(output_path.parent.parent, Path("outputs") / "experiments")
        self.assertRegex(output_path.parent.name, r"^\d{8}_\d{6}$")
        self.assertEqual(output_path.name, "heuristic_vs_random.csv")
        self.assertEqual(output_path.suffix, ".csv")

    def test_no_arg_parser_allows_default_gui(self):
        args = build_parser().parse_args([])
        self.assertIsNone(args.command)

    def test_q_ai_is_available_to_auto_parser(self):
        args = build_parser().parse_args(["auto", "--player", "q_ai", "--episodes", "1"])
        self.assertEqual(args.player, "q_ai")

    def test_dqn_player_label_is_available(self):
        label = PLAYER_LABELS["dqn_player"]
        self.assertEqual(PLAYER_TYPES_BY_LABEL[label], "dqn_player")

    def test_heuristic_player_can_be_default_auto_player(self):
        self.assertIsInstance(create_player("heuristic"), HeuristicPlayer)

    def test_training_algorithm_order_is_light_to_deep(self):
        self.assertEqual(tuple(TRAINING_ALGORITHM_LABELS.values()), ("轻量 Q-learning", "深度 DQN"))

    def test_train_player_parser(self):
        args = build_parser().parse_args(["train-player", "--episodes", "2", "--enemy", "random"])
        self.assertEqual(args.command, "train-player")
        self.assertEqual(args.episodes, 2)
        self.assertEqual(args.enemy, "random")

    def test_train_enemy_parser(self):
        args = build_parser().parse_args(["train-enemy", "--episodes", "2", "--player", "random"])
        self.assertEqual(args.command, "train-enemy")
        self.assertEqual(args.episodes, 2)
        self.assertEqual(args.player, "random")

    def test_train_player_dqn_parser(self):
        args = build_parser().parse_args(["train-player-dqn", "--episodes", "2", "--enemy", "random"])
        self.assertEqual(args.command, "train-player-dqn")
        self.assertEqual(args.episodes, 2)
        self.assertEqual(args.enemy, "random")

    def test_train_enemy_dqn_parser(self):
        args = build_parser().parse_args(["train-enemy-dqn", "--episodes", "2", "--player", "heuristic"])
        self.assertEqual(args.command, "train-enemy-dqn")
        self.assertEqual(args.episodes, 2)
        self.assertEqual(args.player, "heuristic")

    def test_q_enemy_is_available_to_auto_parser(self):
        args = build_parser().parse_args(["auto", "--enemy", "q_enemy", "--episodes", "1"])
        self.assertEqual(args.enemy, "q_enemy")

    def test_default_enemies_are_random(self):
        auto_args = build_parser().parse_args(["auto"])
        gui_args = build_parser().parse_args(["gui"])
        train_args = build_parser().parse_args(["train-player"])
        self.assertEqual(auto_args.enemy, "random")
        self.assertEqual(gui_args.enemy, "random")
        self.assertEqual(train_args.enemy, "random")
        self.assertIsNone(train_args.episodes)

    def test_gui_player_selects_initial_automatic_player(self):
        args = build_parser().parse_args(["gui", "--player", "q_ai", "--enemy", "greedy"])
        self.assertEqual(args.player, "q_ai")
        self.assertEqual(args.enemy, "greedy")
        with patch("cli.commands.run_gui") as mocked_run_gui:
            from cli.commands import dispatch

            dispatch(args)
        mocked_run_gui.assert_called_once_with(player_type="q_ai", enemy_type="greedy")

    def test_gui_player_defaults_to_heuristic_and_rejects_human(self):
        parser = build_parser()
        args = parser.parse_args(["gui"])
        self.assertEqual(args.player, "heuristic")
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args(["gui", "--player", "human"])

    def test_training_merge_parser_has_no_latest_publish_flag(self):
        parser = build_parser()
        args = parser.parse_args(["training-merge", "--a", "a.json", "--b", "b.json"])
        self.assertFalse(hasattr(args, "publish_latest"))
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args(["training-merge", "--a", "a.json", "--b", "b.json", "--publish-latest"])

    def test_import_main_does_not_eagerly_load_torch(self):
        command = [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); import main; print('torch' in sys.modules)",
        ]
        result = subprocess.run(command, cwd=Path.cwd(), check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
