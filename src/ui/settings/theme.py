"""Tkinter 主题、颜色、字体和 DPI 设置。 / Tkinter theme, colors, fonts, and DPI setup."""

from __future__ import annotations

import ctypes
import tkinter as tk
from tkinter import ttk

APP_BG = "#f5f3ee"
PANEL_BG = "#fbfaf7"
TEXT = "#26231f"
MUTED = "#6c645b"
ACCENT = "#2f8f5b"
ACCENT_DARK = "#26784c"
BUSY = "#3f628b"
BORDER = "#ddd6ca"
BOARD_BG = "#a99d91"
CELL_BG = "#d7cec4"
CONTROL_BG = "#ffffff"
CONTROL_ACTIVE = "#f0ece4"

FONT_FAMILY = "Microsoft YaHei UI"
BASE_FONT = (FONT_FAMILY, 16)
BASE_FONT_BOLD = (FONT_FAMILY, 16, "bold")
SCORE_FONT = (FONT_FAMILY, 20, "bold")
BUTTON_FONT = (FONT_FAMILY, 16, "bold")
TILE_FONT_FAMILY = "Segoe UI"

BUTTON_NORMAL = {
    "bg": "#efe4d1",
    "fg": TEXT,
    "activebackground": "#e3d3bc",
    "activeforeground": TEXT,
    "disabledforeground": "#ece8df",
    "relief": tk.FLAT,
}
BUTTON_PRESSED = {
    "bg": "#6f5438",
    "fg": "#ffffff",
    "activebackground": "#5a432d",
    "activeforeground": "#ffffff",
    "disabledforeground": "#eadfce",
    "relief": tk.FLAT,
}
BUTTON_BUSY = {
    "bg": BUSY,
    "fg": "#ffffff",
    "activebackground": "#345275",
    "activeforeground": "#ffffff",
    "disabledforeground": "#dde6f0",
    "relief": tk.FLAT,
}


def enable_high_dpi() -> None:
    """在 Windows 上启用更清晰的高 DPI 显示。 / Enable sharper high-DPI rendering on Windows."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def apply_theme(root: tk.Tk) -> None:
    """把项目主题样式应用到 Tk 根窗口。 / Apply the project theme styles to the Tk root window."""
    root.configure(bg=APP_BG)
    root.tk.call("tk", "scaling", 1.25)
    root.option_add("*Font", BASE_FONT)
    root.option_add("*Menu.font", BASE_FONT)
    root.option_add("*Menu.background", "#ffffff")
    root.option_add("*Menu.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.font", BASE_FONT)
    root.option_add("*TCombobox*Listbox.background", "#ffffff")
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", "#efe8dc")
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=BASE_FONT, background=APP_BG, foreground=TEXT)
    style.configure("TFrame", background=APP_BG)
    style.configure("Panel.TFrame", background=PANEL_BG)
    style.configure("TLabel", background=PANEL_BG, foreground=TEXT, font=BASE_FONT)
    style.configure("Muted.TLabel", background=PANEL_BG, foreground=MUTED, font=BASE_FONT)
    style.configure("Score.TLabel", background=PANEL_BG, foreground=TEXT, font=SCORE_FONT)
    style.configure(
        "TLabelframe",
        background=PANEL_BG,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        relief=tk.SOLID,
    )
    style.configure("TLabelframe.Label", background=PANEL_BG, foreground=TEXT, font=BASE_FONT_BOLD)
    style.configure(
        "TMenubutton",
        background="#ffffff",
        foreground=TEXT,
        font=BASE_FONT,
        bordercolor=BORDER,
        arrowcolor=TEXT,
        padding=(16, 10),
        relief=tk.FLAT,
    )
    style.map(
        "TMenubutton",
        background=[("active", "#f0ece4"), ("disabled", "#ece8df")],
        foreground=[("disabled", MUTED)],
    )
    style.configure(
        "Clean.TCombobox",
        fieldbackground="#ffffff",
        background="#ffffff",
        foreground=TEXT,
        arrowcolor=TEXT,
        bordercolor="#e6dfd4",
        lightcolor="#ffffff",
        darkcolor="#e6dfd4",
        font=BASE_FONT,
        padding=(14, 10),
        relief=tk.FLAT,
    )
    style.map(
        "Clean.TCombobox",
        fieldbackground=[("readonly", "#ffffff"), ("disabled", "#ece8df")],
        background=[("active", "#f0ece4"), ("readonly", "#ffffff"), ("disabled", "#ece8df")],
        foreground=[("disabled", MUTED)],
        arrowcolor=[("disabled", MUTED)],
    )
    style.configure(
        "TEntry",
        fieldbackground="#ffffff",
        foreground=TEXT,
        font=BASE_FONT,
        bordercolor=BORDER,
        padding=(12, 9),
    )
    style.configure(
        "Clean.TSpinbox",
        fieldbackground="#ffffff",
        background="#ffffff",
        foreground=TEXT,
        font=BASE_FONT,
        bordercolor="#e6dfd4",
        arrowcolor=TEXT,
        arrowsize=18,
        lightcolor="#ffffff",
        darkcolor="#e6dfd4",
        padding=(12, 9),
        relief=tk.FLAT,
    )
    style.map(
        "Clean.TSpinbox",
        fieldbackground=[("readonly", "#ffffff"), ("disabled", "#ece8df")],
        background=[("active", "#f0ece4"), ("disabled", "#ece8df")],
        foreground=[("disabled", MUTED)],
        arrowcolor=[("active", ACCENT), ("disabled", MUTED)],
    )
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor="#e8e2d7",
        background=ACCENT,
        bordercolor=BORDER,
        lightcolor=ACCENT,
        darkcolor=ACCENT,
    )


__all__ = [
    "APP_BG",
    "BASE_FONT",
    "BOARD_BG",
    "BUTTON_BUSY",
    "BUTTON_FONT",
    "BUTTON_NORMAL",
    "BUTTON_PRESSED",
    "CELL_BG",
    "CONTROL_ACTIVE",
    "CONTROL_BG",
    "PANEL_BG",
    "SCORE_FONT",
    "TEXT",
    "TILE_FONT_FAMILY",
    "apply_theme",
    "enable_high_dpi",
]
