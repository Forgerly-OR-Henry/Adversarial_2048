"""程序主入口，解析命令行后分发到 CLI。 / Program entrypoint that parses arguments and dispatches to the CLI."""

from __future__ import annotations


def main() -> None:
    """解析命令行参数并执行对应命令。 / Parse command-line arguments and execute the selected command."""
    from cli.commands import dispatch
    from cli.parser import build_parser

    args = build_parser().parse_args()
    dispatch(args)


if __name__ == "__main__":
    main()
