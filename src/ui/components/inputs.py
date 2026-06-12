"""输入类控件。 / Input widgets."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Sequence

from ui.settings.theme import (
    BASE_FONT,
    BORDER,
    CONTROL_ACTIVE,
    CONTROL_BG,
    MUTED,
    PANEL_BG,
    TEXT,
)

CONTROL_HEIGHT = 56
DROPDOWN_ROW_HEIGHT = 44
DROPDOWN_MAX_VISIBLE_ROWS = 10
SELECT_BUTTON_SIZE = CONTROL_HEIGHT - 2
STEPPER_BUTTON_WIDTH = SELECT_BUTTON_SIZE // 2
STEPPER_BUTTON_HEIGHT = SELECT_BUTTON_SIZE // 2
GRID_CONTROL_OPTIONS = {"fixed_height": False}


def _is_descendant(widget: tk.Misc, ancestor: tk.Misc) -> bool:
    """判断 widget 是否属于某个控件。 / Return whether widget belongs to an ancestor widget."""
    current: tk.Misc | None = widget
    while current is not None:
        if current == ancestor:
            return True
        try:
            current = current.master
        except AttributeError:
            return False
    return False


def _bind_entry_blur_on_external_click(container: tk.Misc, entry: tk.Entry) -> None:
    """点击输入框外部时移走焦点，隐藏插入光标。 / Move focus away on external clicks to hide caret."""

    def blur_on_external_click(event: tk.Event) -> None:
        if _is_descendant(event.widget, container):
            return
        try:
            if entry.focus_get() == entry:
                container.focus_set()
        except tk.TclError:
            return

    container.bind_all("<ButtonPress-1>", blur_on_external_click, add="+")


def create_select(
    parent: tk.Misc,
    variable: tk.StringVar,
    values: Sequence[str],
    command: Callable[[str], None] | None = None,
    width: int | None = None,
    *,
    fixed_height: bool = True,
) -> tk.Frame:
    """创建统一样式的下拉选择控件。 / Create a consistently styled combobox."""
    select = tk.Frame(parent, bg=BORDER)
    if fixed_height:
        select.configure(height=CONTROL_HEIGHT)
    if width is not None:
        select.configure(width=width * 18)
    select.grid_propagate(False)
    select.columnconfigure(0, weight=1)
    select.rowconfigure(0, weight=1)
    if fixed_height:
        select.rowconfigure(0, minsize=CONTROL_HEIGHT)

    content = tk.Frame(select, bg=CONTROL_BG)
    content.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
    content.columnconfigure(0, weight=1)
    content.columnconfigure(1, minsize=SELECT_BUTTON_SIZE)
    content.rowconfigure(0, weight=1)
    if fixed_height:
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

    def close_on_external_click(event: tk.Event) -> None:
        if popup is None or not popup.winfo_exists():
            return
        if _is_descendant(event.widget, select) or _is_descendant(event.widget, popup):
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
    variable: tk.Variable,
    from_: int,
    to: int,
    width: int = 10,
    *,
    fixed_height: bool = True,
) -> tk.Frame:
    """创建带加减按钮的数值输入控件。 / Create a numeric input with minus and plus buttons."""
    stepper = tk.Frame(parent, bg=BORDER)
    if fixed_height:
        stepper.configure(height=CONTROL_HEIGHT)
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

    arrows = tk.Frame(content, bg=PANEL_BG, width=STEPPER_BUTTON_WIDTH)
    if fixed_height:
        arrows.configure(height=CONTROL_HEIGHT - 2)
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

    arrows.rowconfigure(0, minsize=STEPPER_BUTTON_HEIGHT if fixed_height else 0, weight=1)
    arrows.rowconfigure(1, minsize=STEPPER_BUTTON_HEIGHT if fixed_height else 0, weight=1)
    arrows.columnconfigure(0, minsize=STEPPER_BUTTON_WIDTH, weight=1)
    make_arrow("▲", 1, 0)
    make_arrow("▼", -1, 1)
    _bind_entry_blur_on_external_click(stepper, entry)
    return stepper


def create_text_entry(
    parent: tk.Misc,
    variable: tk.StringVar,
    width: int | None = None,
    *,
    fixed_height: bool = True,
) -> tk.Frame:
    """创建与下拉框同高的文本输入控件。 / Create a text input matching select height."""
    input_box = tk.Frame(parent, bg=BORDER)
    if fixed_height:
        input_box.configure(height=CONTROL_HEIGHT)
    if width is not None:
        input_box.configure(width=width * 18)
    input_box.grid_propagate(False)
    input_box.columnconfigure(0, weight=1)
    input_box.rowconfigure(0, weight=1)
    if fixed_height:
        input_box.rowconfigure(0, minsize=CONTROL_HEIGHT)

    content = tk.Frame(input_box, bg=CONTROL_BG)
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
        relief=tk.FLAT,
        font=BASE_FONT,
        width=width or 0,
    )
    entry.grid(row=0, column=0, sticky="nsew", padx=14)

    def focus_entry(_event: tk.Event) -> None:
        entry.focus_set()

    input_box.bind("<Button-1>", focus_entry)
    content.bind("<Button-1>", focus_entry)
    _bind_entry_blur_on_external_click(input_box, entry)
    return input_box


__all__ = [
    "CONTROL_HEIGHT",
    "DROPDOWN_MAX_VISIBLE_ROWS",
    "DROPDOWN_ROW_HEIGHT",
    "GRID_CONTROL_OPTIONS",
    "SELECT_BUTTON_SIZE",
    "STEPPER_BUTTON_HEIGHT",
    "STEPPER_BUTTON_WIDTH",
    "create_select",
    "create_stepper",
    "create_text_entry",
]
