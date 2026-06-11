"""Tkinter 通用控件构造和按钮状态样式。 / Shared Tkinter control builders and button-state styling."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence

from ui.settings.theme import (
    BASE_FONT,
    BASE_FONT_BOLD,
    BORDER,
    BUTTON_FONT,
    BUTTON_NORMAL,
    CONTROL_ACTIVE,
    CONTROL_BG,
    MUTED,
    PANEL_BG,
    TEXT,
)
CONTROL_HEIGHT = 56
MESSAGE_WRAP_LENGTH = 820
STATUS_AREA_HEIGHT = 88
RESULT_AREA_HEIGHT = 126
DROPDOWN_ROW_HEIGHT = 44
DROPDOWN_MAX_VISIBLE_ROWS = 10
SELECT_BUTTON_SIZE = CONTROL_HEIGHT - 2
STEPPER_BUTTON_WIDTH = SELECT_BUTTON_SIZE // 2
STEPPER_BUTTON_HEIGHT = SELECT_BUTTON_SIZE // 2


def create_select(
    parent: tk.Misc,
    variable: tk.StringVar,
    values: Sequence[str],
    command: Callable[[str], None] | None = None,
    width: int | None = None,
) -> tk.Frame:
    """创建统一样式的下拉选择控件。 / Create a consistently styled combobox."""
    select = tk.Frame(parent, bg=BORDER, height=CONTROL_HEIGHT)
    if width is not None:
        select.configure(width=width * 18)
    select.grid_propagate(False)
    select.columnconfigure(0, weight=1)
    select.rowconfigure(0, weight=1)
    select.rowconfigure(0, minsize=CONTROL_HEIGHT)

    content = tk.Frame(select, bg=CONTROL_BG)
    content.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, minsize=SELECT_BUTTON_SIZE)
    content.rowconfigure(0, weight=1)
    content.rowconfigure(0, minsize=SELECT_BUTTON_SIZE)

    label = tk.Label(
        content,
        textvariable=variable,
        bg=CONTROL_BG,
        fg=TEXT,
        font=BASE_FONT,
        anchor="w",
        padx=14,
    )
    label.grid(row=0, column=0, sticky="nsew")

    button = tk.Button(
        content,
        text="▼",
        bg=CONTROL_BG,
        fg=TEXT,
        activebackground=CONTROL_ACTIVE,
        activeforeground=TEXT,
        borderwidth=0,
        highlightthickness=0,
        font=(BASE_FONT[0], 10, "bold"),
        cursor="hand2",
    )
    button.grid(row=0, column=1, sticky="nsew")

    popup: tk.Toplevel | None = None

    def choose(value: str) -> None:
        nonlocal popup
        variable.set(value)
        if popup is not None and popup.winfo_exists():
            popup.destroy()
        popup = None
        if command is not None:
            command(value)
        select.focus_set()

    def close_popup(_event: tk.Event | None = None) -> None:
        nonlocal popup
        if popup is not None and popup.winfo_exists():
            popup.destroy()
        popup = None

    def is_descendant(widget: tk.Misc, ancestor: tk.Misc) -> bool:
        current: tk.Misc | None = widget
        while current is not None:
            if current == ancestor:
                return True
            try:
                current = current.master
            except AttributeError:
                return False
        return False

    def close_on_external_click(event: tk.Event) -> None:
        if popup is None or not popup.winfo_exists():
            return
        if is_descendant(event.widget, select) or is_descendant(event.widget, popup):
            return
        close_popup()

    def open_menu(_event: tk.Event | None = None) -> str:
        nonlocal popup
        if popup is not None and popup.winfo_exists():
            close_popup()
            return "break"

        select.update_idletasks()
        x = select.winfo_rootx()
        visible_rows = max(1, min(len(values), DROPDOWN_MAX_VISIBLE_ROWS))
        dropdown_height = DROPDOWN_ROW_HEIGHT * visible_rows + 2
        y = select.winfo_rooty() + select.winfo_height()
        if y + dropdown_height > select.winfo_screenheight():
            y = max(0, select.winfo_rooty() - dropdown_height)

        popup = tk.Toplevel(select)
        popup.overrideredirect(True)
        popup.configure(bg=BORDER)
        popup.geometry(f"{select.winfo_width()}x{dropdown_height}+{x}+{y}")
        popup.transient(select.winfo_toplevel())

        outer = tk.Frame(popup, bg=CONTROL_BG, highlightthickness=1, highlightbackground=BORDER)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        if len(values) > DROPDOWN_MAX_VISIBLE_ROWS:
            canvas = tk.Canvas(
                outer,
                bg=CONTROL_BG,
                borderwidth=0,
                highlightthickness=0,
                yscrollincrement=DROPDOWN_ROW_HEIGHT,
            )
            scrollbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.grid(row=0, column=0, sticky="nsew")
            scrollbar.grid(row=0, column=1, sticky="ns")
            body = tk.Frame(canvas, bg=CONTROL_BG)
            body_window = canvas.create_window((0, 0), window=body, anchor="nw")

            def sync_scroll_region(_event: tk.Event | None = None) -> None:
                canvas.configure(scrollregion=canvas.bbox("all"))

            def sync_body_width(event: tk.Event) -> None:
                canvas.itemconfigure(body_window, width=event.width)

            def scroll_with_wheel(event: tk.Event) -> str:
                direction = -1 if event.delta > 0 else 1
                canvas.yview_scroll(direction, "units")
                return "break"

            body.bind("<Configure>", sync_scroll_region)
            canvas.bind("<Configure>", sync_body_width)
            canvas.bind("<MouseWheel>", scroll_with_wheel)
            body.bind("<MouseWheel>", scroll_with_wheel)
        else:
            body = outer

        for index, value in enumerate(values):
            item = tk.Label(
                body,
                text=value,
                bg=CONTROL_BG,
                fg=TEXT,
                font=BASE_FONT,
                anchor="w",
                padx=30,
                height=1,
            )
            item.grid(row=index, column=0, sticky="nsew")
            body.rowconfigure(index, minsize=DROPDOWN_ROW_HEIGHT)
            item.bind("<Enter>", lambda event: event.widget.configure(bg=CONTROL_ACTIVE))
            item.bind("<Leave>", lambda event: event.widget.configure(bg=CONTROL_BG))
            item.bind("<ButtonRelease-1>", lambda _event, selected=value: choose(selected))
            if len(values) > DROPDOWN_MAX_VISIBLE_ROWS:
                item.bind("<MouseWheel>", scroll_with_wheel)
        body.columnconfigure(0, weight=1)

        popup.bind("<Escape>", close_popup)
        popup.lift()
        return "break"

    def set_hover(active: bool) -> None:
        color = CONTROL_ACTIVE if active else CONTROL_BG
        button.configure(bg=color)

    for widget in (select, content, label, button):
        widget.bind("<Button-1>", open_menu)
        widget.bind("<Return>", open_menu)
        widget.bind("<space>", open_menu)
    select.bind_all("<ButtonPress-1>", close_on_external_click, add="+")
    select.bind_all("<Escape>", close_popup, add="+")
    button.bind("<Enter>", lambda _: set_hover(True))
    button.bind("<Leave>", lambda _: set_hover(False))
    return select


def create_stepper(
    parent: tk.Misc,
    variable: tk.IntVar,
    from_: int,
    to: int,
    width: int = 10,
) -> tk.Frame:
    """创建带加减按钮的数值输入控件。 / Create a numeric input with minus and plus buttons."""
    stepper = tk.Frame(parent, bg=BORDER, height=CONTROL_HEIGHT)
    stepper.grid_propagate(False)
    stepper.columnconfigure(0, weight=1)
    stepper.rowconfigure(0, weight=1)

    content = tk.Frame(stepper, bg=CONTROL_BG)
    content.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
    content.columnconfigure(0, weight=1)
    content.rowconfigure(0, weight=1)

    entry = tk.Entry(
        content,
        textvariable=variable,
        bg=CONTROL_BG,
        fg=TEXT,
        insertbackground=TEXT,
        disabledforeground=MUTED,
        borderwidth=0,
        highlightthickness=0,
        font=BASE_FONT,
        width=width,
    )
    entry.grid(row=0, column=0, sticky="nsew", padx=(14, 6))

    arrows = tk.Frame(content, bg=PANEL_BG, width=STEPPER_BUTTON_WIDTH, height=CONTROL_HEIGHT - 2)
    arrows.grid(row=0, column=1, sticky="ns")
    arrows.grid_propagate(False)

    def set_value(delta: int) -> None:
        try:
            current = int(variable.get())
        except (TypeError, ValueError, tk.TclError):
            current = from_
        variable.set(max(from_, min(to, current + delta)))

    def make_arrow(text: str, delta: int, row: int) -> tk.Button:
        button = tk.Button(
            arrows,
            text=text,
            command=lambda: set_value(delta),
            bg=CONTROL_BG,
            fg=TEXT,
            activebackground=CONTROL_ACTIVE,
            activeforeground=TEXT,
            borderwidth=0,
            highlightthickness=0,
            font=(BASE_FONT[0], 8, "bold"),
            cursor="hand2",
        )
        button.grid(row=row, column=0, sticky="nsew")
        return button

    arrows.rowconfigure(0, minsize=STEPPER_BUTTON_HEIGHT, weight=1)
    arrows.rowconfigure(1, minsize=STEPPER_BUTTON_HEIGHT, weight=1)
    arrows.columnconfigure(0, minsize=STEPPER_BUTTON_WIDTH, weight=1)
    make_arrow("▲", 1, 0)
    make_arrow("▼", -1, 1)
    return stepper


def create_message_area(
    parent: tk.Misc,
    variable: tk.StringVar,
    *,
    height: int = STATUS_AREA_HEIGHT,
    wraplength: int = MESSAGE_WRAP_LENGTH,
) -> tk.Frame:
    """创建固定尺寸的提示文本区域，避免长文本撑动表单布局。 / Create a fixed message area."""
    area = tk.Frame(parent, bg=PANEL_BG, height=height)
    area.grid_propagate(False)
    area.columnconfigure(0, weight=1)
    area.rowconfigure(0, weight=1)

    label = tk.Label(
        area,
        textvariable=variable,
        bg=PANEL_BG,
        fg=MUTED,
        font=BASE_FONT,
        anchor="nw",
        justify="left",
        wraplength=wraplength,
    )
    label.grid(row=0, column=0, sticky="nsew")
    return area


def create_action_button(
    parent: tk.Misc,
    text: str,
    command: Callable[[], None],
    *,
    compact: bool = False,
) -> tk.Button:
    """创建统一样式的操作按钮。 / Create a consistently styled action button."""
    button = tk.Button(
        parent,
        text=text,
        command=command,
        font=BASE_FONT_BOLD if compact else BUTTON_FONT,
        borderwidth=0,
        highlightthickness=0,
        padx=12 if compact else 18,
        pady=2 if compact else 6,
        anchor=tk.CENTER,
        cursor="hand2",
    )
    set_button_visual(button, BUTTON_NORMAL)
    return button


def set_button_visual(button: tk.Button, visual: dict[str, object]) -> None:
    """根据按钮状态切换强调样式。 / Switch emphasized styling based on button state."""
    button.configure(**visual)


__all__ = [
    "RESULT_AREA_HEIGHT",
    "STATUS_AREA_HEIGHT",
    "create_action_button",
    "create_message_area",
    "create_select",
    "create_stepper",
    "set_button_visual",
]
