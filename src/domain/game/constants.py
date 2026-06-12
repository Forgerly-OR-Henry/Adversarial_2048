"""2048 游戏常量、动作和默认出块概率。 / 2048 constants, actions, and default tile spawn probabilities."""

BOARD_SIZE = 4
EMPTY = 0

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"

ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_ALIASES = {
    "w": UP,
    "a": LEFT,
    "s": DOWN,
    "d": RIGHT,
    "up": UP,
    "down": DOWN,
    "left": LEFT,
    "right": RIGHT,
}

DEFAULT_INITIAL_TILES = 2
STANDARD_FOUR_PROBABILITY = 0.1
