"""命令行包导出解析器和分发入口。 / CLI package exports parser and dispatch entrypoints."""

from cli.commands import dispatch
from cli.parser import build_parser

__all__ = ["build_parser", "dispatch"]
