"""命令行模式的执行层，连接对局、评估和训练入口。 / Execution layer for CLI modes, wiring gameplay, evaluation, and training entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from domain.enemies import create_enemy
from domain.evaluation import compare_training_artifacts, run_experiment
from domain.game.board import format_board
from domain.game.env import GameEnv
from domain.players import HumanPlayer
from domain.train import (
    list_training_artifacts,
    merge_training_artifacts,
    run_auto_tuning,
    train_dqn_enemy,
    train_dqn_player,
    train_q_enemy,
    train_q_player,
)
from ui import run_gui
from utils.serialization import json_ready
from utils.training_log import log_error


def play_terminal(enemy_type: str, seed: int | None) -> None:
    """运行命令行人工对局。 / Run an interactive terminal game."""
    env = GameEnv(enemy=create_enemy(enemy_type), seed=seed)
    player = HumanPlayer()
    state = env.reset()

    while not state.done:
        print()
        print(format_board(state.board))
        print(f"Score: {state.score} | Steps: {state.steps} | Max: {state.max_tile} | Enemy: {enemy_type}")
        action = player.select_action(state, env.get_legal_actions())
        if action is None:
            print("Quit.")
            return
        state = env.step(action)

    print()
    print(format_board(state.board))
    print(f"Game over. Score: {state.score} | Steps: {state.steps} | Max: {state.max_tile}")


def run_auto(player_type: str, enemy_type: str, episodes: int, seed: int | None, output: str | None) -> Path:
    """运行多局自动实验并返回汇总统计。 / Run automated episodes and return aggregate statistics."""
    return run_experiment(
        player_type=player_type,
        enemy_type=enemy_type,
        episodes=episodes,
        seed=seed,
        output=output,
    )


def _print_json(value) -> None:
    print(json.dumps(json_ready(value), ensure_ascii=False, indent=2))


def dispatch(args: argparse.Namespace) -> None:
    """根据解析后的子命令调用对应功能。 / Dispatch parsed subcommands to the matching feature."""
    try:
        if args.command is None:
            run_gui()
        elif args.command == "play":
            play_terminal(enemy_type=args.enemy, seed=args.seed)
        elif args.command == "auto":
            output_path = run_auto(
                player_type=args.player,
                enemy_type=args.enemy,
                episodes=args.episodes,
                seed=args.seed,
                output=args.output,
            )
            print(f"Saved {args.episodes} episodes to {output_path}")
        elif args.command == "gui":
            run_gui(player_type=args.player, enemy_type=args.enemy)
        elif args.command == "train-player":
            summary = train_q_player(
                episodes=args.episodes,
                enemy_type=args.enemy,
                seed=args.seed,
                output=args.output,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
            )
            _print_player_training_summary("Q player", summary)
        elif args.command == "train-enemy":
            summary = train_q_enemy(
                episodes=args.episodes,
                player_type=args.player,
                seed=args.seed,
                output=args.output,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
            )
            _print_enemy_training_summary("Q enemy", summary)
        elif args.command == "train-player-dqn":
            summary = train_dqn_player(
                episodes=args.episodes,
                enemy_type=args.enemy,
                seed=args.seed,
                output=args.output,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
            )
            _print_player_training_summary("DQN player", summary)
        elif args.command == "train-enemy-dqn":
            summary = train_dqn_enemy(
                episodes=args.episodes,
                player_type=args.player,
                seed=args.seed,
                output=args.output,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
            )
            _print_enemy_training_summary("DQN enemy", summary)
        elif args.command == "training-list":
            _print_json(list_training_artifacts())
        elif args.command == "training-compare":
            comparison = compare_training_artifacts(
                artifact_a=args.a,
                artifact_b=args.b,
                episodes=args.episodes,
                seed=args.seed,
                player_type=args.player,
                enemy_type=args.enemy,
            )
            _print_json(comparison)
        elif args.command == "training-merge":
            summary = merge_training_artifacts(
                artifact_a=args.a,
                artifact_b=args.b,
                output=args.output,
                weight_a=args.weight_a,
            )
            _print_json(summary)
        elif args.command == "training-tune":
            results = run_auto_tuning(
                target=args.target,
                algorithm=args.algorithm,
                candidates=args.candidates,
                training_episodes=args.training_episodes,
                evaluation_episodes=args.evaluation_episodes,
                seed=args.seed,
            )
            _print_json(results)
    except ModuleNotFoundError as exc:
        log_error("cli", exc, vars(args))
        print(f"Missing dependency: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        log_error("cli", exc, vars(args))
        raise


def _print_player_training_summary(label: str, summary) -> None:
    print(f"Saved {label} model to {summary.output_path}")
    device = f"device={summary.device}, " if hasattr(summary, "device") else ""
    print(
        "Training summary: "
        f"episodes={summary.episodes}, "
        f"{device}"
        f"average_score={summary.average_score:.1f}, "
        f"average_max_tile={summary.average_max_tile:.1f}, "
        f"best_max_tile={summary.best_max_tile}"
    )


def _print_enemy_training_summary(label: str, summary) -> None:
    print(f"Saved {label} model to {summary.output_path}")
    device = f"device={summary.device}, " if hasattr(summary, "device") else ""
    print(
        "Training summary: "
        f"episodes={summary.episodes}, "
        f"{device}"
        f"average_player_score={summary.average_player_score:.1f}, "
        f"average_player_max_tile={summary.average_player_max_tile:.1f}, "
        f"best_suppressed_max_tile={summary.best_suppressed_max_tile}"
    )
