"""命令行参数和子命令定义。 / Command-line arguments and subcommand definitions."""

from __future__ import annotations

import argparse

from config import get_evaluate_defaults, get_train_defaults, get_ui_defaults


def build_parser() -> argparse.ArgumentParser:
    """构建 adversarial_2048 的 argparse 解析器。 / Build the argparse parser for adversarial_2048."""
    evaluate_defaults = get_evaluate_defaults()
    ui_defaults = get_ui_defaults()
    player_q_defaults = get_train_defaults("player_q")
    enemy_q_defaults = get_train_defaults("enemy_q")
    player_dqn_defaults = get_train_defaults("player_dqn")
    enemy_dqn_defaults = get_train_defaults("enemy_dqn")

    parser = argparse.ArgumentParser(description="Adversarial 2048 v1")
    subparsers = parser.add_subparsers(dest="command")

    play = subparsers.add_parser("play", help="Play manually in the terminal.")
    enemy_choices = ["random", "greedy", "q_enemy", "dqn_enemy"]
    player_choices = ["random", "heuristic", "q_ai", "dqn_player"]

    play.add_argument("--enemy", choices=enemy_choices, default=ui_defaults["enemy"])
    play.add_argument("--seed", type=int, default=None)

    auto = subparsers.add_parser("auto", help="Run automated experiments.")
    auto.add_argument("--player", choices=player_choices, default=evaluate_defaults["player"])
    auto.add_argument("--enemy", choices=enemy_choices, default=evaluate_defaults["enemy"])
    auto.add_argument("--episodes", type=int, default=evaluate_defaults["episodes"])
    auto.add_argument("--seed", type=int, default=evaluate_defaults.get("seed"))
    auto.add_argument("--output", default=evaluate_defaults.get("output"))

    gui = subparsers.add_parser("gui", help="Start the Tkinter GUI.")
    gui.add_argument("--player", choices=["human", "heuristic", "q_ai", "dqn_player"], default=ui_defaults["player"])
    gui.add_argument("--enemy", choices=enemy_choices, default=ui_defaults["enemy"])

    train_player = subparsers.add_parser("train-player", help="Train the Q-model player.")
    train_player.add_argument("--episodes", type=int, default=player_q_defaults["episodes"])
    train_player.add_argument("--enemy", choices=enemy_choices, default=player_q_defaults["enemy"])
    train_player.add_argument("--seed", type=int, default=player_q_defaults.get("seed"))
    train_player.add_argument("--output", default=None)
    train_player.add_argument("--learning-rate", type=float, default=player_q_defaults["learning_rate"])
    train_player.add_argument("--gamma", type=float, default=player_q_defaults["gamma"])

    train_player_dqn = subparsers.add_parser("train-player-dqn", help="Train the PyTorch DQN player.")
    train_player_dqn.add_argument("--episodes", type=int, default=player_dqn_defaults["episodes"])
    train_player_dqn.add_argument("--enemy", choices=enemy_choices, default=player_dqn_defaults["enemy"])
    train_player_dqn.add_argument("--seed", type=int, default=player_dqn_defaults.get("seed"))
    train_player_dqn.add_argument("--output", default=None)
    train_player_dqn.add_argument("--learning-rate", type=float, default=player_dqn_defaults["learning_rate"])
    train_player_dqn.add_argument("--gamma", type=float, default=player_dqn_defaults["gamma"])

    train_enemy = subparsers.add_parser("train-enemy", help="Train the Q-model enemy.")
    train_enemy.add_argument("--episodes", type=int, default=enemy_q_defaults["episodes"])
    train_enemy.add_argument("--player", choices=player_choices, default=enemy_q_defaults["player"])
    train_enemy.add_argument("--seed", type=int, default=enemy_q_defaults.get("seed"))
    train_enemy.add_argument("--output", default=None)
    train_enemy.add_argument("--learning-rate", type=float, default=enemy_q_defaults["learning_rate"])
    train_enemy.add_argument("--gamma", type=float, default=enemy_q_defaults["gamma"])

    train_enemy_dqn = subparsers.add_parser("train-enemy-dqn", help="Train the PyTorch DQN enemy.")
    train_enemy_dqn.add_argument("--episodes", type=int, default=enemy_dqn_defaults["episodes"])
    train_enemy_dqn.add_argument("--player", choices=player_choices, default=enemy_dqn_defaults["player"])
    train_enemy_dqn.add_argument("--seed", type=int, default=enemy_dqn_defaults.get("seed"))
    train_enemy_dqn.add_argument("--output", default=None)
    train_enemy_dqn.add_argument("--learning-rate", type=float, default=enemy_dqn_defaults["learning_rate"])
    train_enemy_dqn.add_argument("--gamma", type=float, default=enemy_dqn_defaults["gamma"])

    subparsers.add_parser("training-list", help="List timestamped training artifacts.")

    training_compare = subparsers.add_parser("training-compare", help="Compare two training artifacts.")
    training_compare.add_argument("--a", required=True)
    training_compare.add_argument("--b", required=True)
    training_compare.add_argument("--episodes", type=int, required=True)
    training_compare.add_argument("--seed", type=int, default=None)
    training_compare.add_argument("--player", default="heuristic")
    training_compare.add_argument("--enemy", default="random")

    training_merge = subparsers.add_parser("training-merge", help="Merge two training artifacts.")
    training_merge.add_argument("--a", required=True)
    training_merge.add_argument("--b", required=True)
    training_merge.add_argument("--output", default=None)
    training_merge.add_argument("--weight-a", type=float, default=0.5)
    training_merge.add_argument("--publish-latest", action="store_true")

    training_tune = subparsers.add_parser("training-tune", help="Run automatic short tuning trials.")
    training_tune.add_argument("--target", choices=["player", "enemy"], required=True)
    training_tune.add_argument("--algorithm", choices=["q_learning", "dqn"], required=True)
    training_tune.add_argument("--candidates", type=int, default=None)
    training_tune.add_argument("--training-episodes", type=int, default=None)
    training_tune.add_argument("--evaluation-episodes", type=int, default=None)
    training_tune.add_argument("--seed", type=int, default=None)

    return parser
