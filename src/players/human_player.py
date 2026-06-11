"""命令行人工玩家输入适配。 / Command-line human player input adapter."""

from __future__ import annotations

from game.constants import ACTION_ALIASES
from game.state import GameState
from players.base_player import BasePlayer


class HumanPlayer(BasePlayer):
    """从命令行读取人工玩家动作。 / Reads human player moves from the terminal."""
    name = "human"

    def parse_action(self, raw_action: str) -> str | None:
        return ACTION_ALIASES.get(raw_action.strip().lower())

    def select_action(self, state: GameState, legal_actions: list[str]) -> str | None:
        while True:
            raw_action = input("Move (W/A/S/D or q): ").strip().lower()
            if raw_action == "q":
                return None
            action = self.parse_action(raw_action)
            if action in legal_actions:
                return action
            print("Invalid move.")
