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
    gui.add_argument("--player", choices=["heuristic", "q_ai", "dqn_player"], default=ui_defaults["player"])
    gui.add_argument("--enemy", choices=enemy_choices, default=ui_defaults["enemy"])

    _add_training_parser(
        subparsers,
        "train-player",
        "Train the Q-model player.",
        player_q_defaults,
        opponent_argument="enemy",
        opponent_choices=enemy_choices,
    )
    _add_training_parser(
        subparsers,
        "train-player-dqn",
        "Train the PyTorch DQN player.",
        player_dqn_defaults,
        opponent_argument="enemy",
        opponent_choices=enemy_choices,
    )
    _add_training_parser(
        subparsers,
        "train-enemy",
        "Train the Q-model enemy.",
        enemy_q_defaults,
        opponent_argument="player",
        opponent_choices=player_choices,
    )
    _add_training_parser(
        subparsers,
        "train-enemy-dqn",
        "Train the PyTorch DQN enemy.",
        enemy_dqn_defaults,
        opponent_argument="player",
        opponent_choices=player_choices,
    )

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

    training_tune = subparsers.add_parser("training-tune", help="Run automatic short tuning trials.")
    training_tune.add_argument("--target", choices=["player", "enemy"], required=True)
    training_tune.add_argument("--algorithm", choices=["q_learning", "dqn"], required=True)
    training_tune.add_argument("--candidates", type=int, default=None)
    training_tune.add_argument("--training-episodes", type=int, default=None)
    training_tune.add_argument("--evaluation-episodes", type=int, default=None)
    training_tune.add_argument("--seed", type=int, default=None)

    return parser


def _add_training_parser(
    subparsers: argparse._SubParsersAction,
    command: str,
    help_text: str,
    defaults: dict,
    *,
    opponent_argument: str,
    opponent_choices: list[str],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(command, help=help_text)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(f"--{opponent_argument}", choices=opponent_choices, default=defaults[opponent_argument])
    parser.add_argument("--seed", type=int, default=defaults.get("seed"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--gamma", type=float, default=defaults["gamma"])
    return parser
