"""可复用 Tkinter UI 组件。 / Reusable Tkinter UI components."""

from ui.components.board_view import BoardView
from ui.components.buttons import create_action_button, set_button_visual
from ui.components.inputs import (
    CONTROL_HEIGHT,
    DROPDOWN_MAX_VISIBLE_ROWS,
    DROPDOWN_ROW_HEIGHT,
    GRID_CONTROL_OPTIONS,
    SELECT_BUTTON_SIZE,
    STEPPER_BUTTON_HEIGHT,
    STEPPER_BUTTON_WIDTH,
    create_select,
    create_stepper,
    create_text_entry,
)
from ui.components.messages import (
    MESSAGE_WRAP_LENGTH,
    RESULT_AREA_HEIGHT,
    STATUS_AREA_HEIGHT,
    create_message_area,
)

__all__ = [
    "BoardView",
    "CONTROL_HEIGHT",
    "DROPDOWN_MAX_VISIBLE_ROWS",
    "DROPDOWN_ROW_HEIGHT",
    "GRID_CONTROL_OPTIONS",
    "MESSAGE_WRAP_LENGTH",
    "RESULT_AREA_HEIGHT",
    "SELECT_BUTTON_SIZE",
    "STATUS_AREA_HEIGHT",
    "STEPPER_BUTTON_HEIGHT",
    "STEPPER_BUTTON_WIDTH",
    "create_action_button",
    "create_message_area",
    "create_select",
    "create_stepper",
    "create_text_entry",
    "set_button_visual",
]
