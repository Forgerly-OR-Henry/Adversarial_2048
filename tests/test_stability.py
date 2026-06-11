from __future__ import annotations

import unittest

import tests._path  # noqa: F401

from config import get_train_defaults
from train.stability import StabilityConfig, StabilityController, stability_config_from_mapping


class StabilityControllerTest(unittest.TestCase):
    def test_config_loads_from_train_yaml(self):
        defaults = get_train_defaults("player_dqn")
        config = stability_config_from_mapping(defaults["stability"])
        self.assertTrue(config.enabled)
        self.assertEqual(config.eval_interval, 20)
        self.assertEqual(config.max_grad_norm, 1.0)

    def test_learning_rate_and_epsilon_are_clamped(self):
        config = StabilityConfig(
            min_learning_rate=1e-5,
            max_learning_rate=3e-4,
            min_epsilon=0.05,
            max_epsilon=1.0,
        )
        controller = StabilityController(config=config, learning_rate=0.01, epsilon=2.0)
        self.assertEqual(controller.learning_rate, 3e-4)
        self.assertEqual(controller.epsilon, 1.0)

    def test_improvement_records_best_state_and_reduces_exploration(self):
        config = StabilityConfig(eval_interval=1, eval_window=1, epsilon_decay_on_improve=0.5)
        controller = StabilityController(config=config, learning_rate=1e-4, epsilon=0.8)

        decision = controller.observe(episode=1, metric=100.0, model_state={"weights": [1]})

        self.assertTrue(decision.improved)
        self.assertEqual(controller.best_metric, 100.0)
        self.assertEqual(controller.best_state, {"weights": [1]})
        self.assertEqual(controller.epsilon, 0.4)

    def test_repeated_degradation_triggers_rollback(self):
        config = StabilityConfig(
            eval_interval=1,
            eval_window=1,
            drop_tolerance=0.2,
            bad_eval_patience=2,
            lr_decay=0.5,
            epsilon_decay_on_improve=1.0,
            epsilon_boost_on_drop=2.0,
            max_epsilon=1.0,
        )
        controller = StabilityController(config=config, learning_rate=1e-4, epsilon=0.2)
        controller.observe(episode=1, metric=100.0, model_state={"weights": [1]})

        first_drop = controller.observe(episode=2, metric=70.0, model_state={"weights": [2]})
        second_drop = controller.observe(episode=3, metric=70.0, model_state={"weights": [3]})

        self.assertTrue(first_drop.degraded)
        self.assertFalse(first_drop.rollback)
        self.assertTrue(second_drop.rollback)
        self.assertEqual(controller.learning_rate, 2.5e-5)
        self.assertEqual(controller.epsilon, 0.8)

    def test_minimize_mode_treats_lower_metric_as_better(self):
        config = StabilityConfig(eval_interval=1, eval_window=1)
        controller = StabilityController(config=config, learning_rate=1e-4, epsilon=0.8, maximize=False)

        controller.observe(episode=1, metric=100.0, model_state={"weights": [1]})
        decision = controller.observe(episode=2, metric=80.0, model_state={"weights": [2]})

        self.assertTrue(decision.improved)
        self.assertEqual(controller.best_metric, 80.0)


if __name__ == "__main__":
    unittest.main()
