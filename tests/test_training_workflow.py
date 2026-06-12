from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tests._path  # noqa: F401

from workflows.training import (
    REFERENCE_TYPE_INITIAL_WEIGHTS,
    TrainingProgress,
    TrainingRunRequest,
    parse_training_episodes,
    parse_training_seed,
    run_training_request,
    training_request_log_parameters,
    validate_reference_type,
    validate_resume_episodes,
)


class TrainingWorkflowTest(unittest.TestCase):
    def test_parse_training_episodes_accepts_blank_as_unlimited(self):
        self.assertIsNone(parse_training_episodes(""))
        self.assertIsNone(parse_training_episodes("   "))
        self.assertEqual(parse_training_episodes("12"), 12)
        with self.assertRaisesRegex(ValueError, "训练局数"):
            parse_training_episodes("abc")
        with self.assertRaisesRegex(ValueError, "至少"):
            parse_training_episodes("0")

    def test_parse_training_seed_accepts_blank_or_integer(self):
        self.assertIsNone(parse_training_seed(""))
        self.assertEqual(parse_training_seed("42"), 42)
        with self.assertRaisesRegex(ValueError, "随机种子"):
            parse_training_seed("seed")

    def test_reference_and_resume_validation(self):
        validate_reference_type(REFERENCE_TYPE_INITIAL_WEIGHTS)
        with self.assertRaisesRegex(ValueError, "起始权重"):
            validate_reference_type("distillation")
        validate_resume_episodes(10, {"completed_episodes": 7})
        with self.assertRaisesRegex(ValueError, "不能低于"):
            validate_resume_episodes(6, {"completed_episodes": 7})

    def test_run_training_request_normalizes_q_progress(self):
        events: list[TrainingProgress] = []
        summary = object()
        state = SimpleNamespace(max_tile=128, score=2048, board=[[2, 4], [8, 16]], steps=31)

        def fake_train_q_player(**kwargs):
            self.assertEqual(kwargs["enemy_type"], "random")
            self.assertIsNone(kwargs["stop_event"])
            kwargs["progress_callback"](2, 5, state, 0.25)
            return summary

        request = TrainingRunRequest(
            target="player",
            algorithm="q",
            enemy_type="random",
            player_type="heuristic",
            episodes=5,
            seed=9,
            output=Path("models/q_learning/player/run/player_q_model.json"),
            reference_model_path=None,
            resume_run_path=None,
        )
        with patch("domain.train.train_q_player", side_effect=fake_train_q_player):
            result = run_training_request(request, progress_callback=events.append)

        self.assertIs(result, summary)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].target, "player")
        self.assertEqual(events[0].device, "cpu")
        self.assertEqual(events[0].board, state.board)
        self.assertIsNot(events[0].board, state.board)

    def test_run_training_request_normalizes_dqn_enemy_progress(self):
        events: list[TrainingProgress] = []
        summary = object()
        state = SimpleNamespace(max_tile=64, score=512, board=[[0, 2], [4, 8]], steps=17)

        def fake_train_dqn_enemy(**kwargs):
            self.assertEqual(kwargs["player_type"], "heuristic")
            kwargs["progress_callback"](3, None, state, 0.5, "cuda")
            return summary

        request = TrainingRunRequest(
            target="enemy",
            algorithm="dqn",
            enemy_type="random",
            player_type="heuristic",
            episodes=None,
            seed=None,
            output=Path("models/dqn/enemy/run/enemy_dqn_model.pt"),
            reference_model_path=Path("models/dqn/enemy/latest/enemy_dqn_model.pt"),
            resume_run_path=Path("models/dqn/enemy/old"),
        )
        with patch("domain.train.train_dqn_enemy", side_effect=fake_train_dqn_enemy):
            result = run_training_request(request, progress_callback=events.append)

        self.assertIs(result, summary)
        self.assertEqual(events[0].target, "enemy")
        self.assertIsNone(events[0].total)
        self.assertEqual(events[0].device, "cuda")

    def test_training_request_log_parameters_omits_stop_event(self):
        request = TrainingRunRequest(
            target="player",
            algorithm="q",
            enemy_type="random",
            player_type="heuristic",
            episodes=1,
            seed=None,
            output=None,
            reference_model_path=None,
            resume_run_path=None,
            stop_event=object(),
        )
        parameters = training_request_log_parameters(request)

        self.assertEqual(parameters["training_type"], "player_q")
        self.assertNotIn("stop_event", parameters)


if __name__ == "__main__":
    unittest.main()
