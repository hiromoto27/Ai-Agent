"""Интерактивный CLI для локального ИИ-агента.

Примеры:
  python -m ai_agent.cli
  python -m ai_agent.cli --once "Создай файл заметок в рабочей папке"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_agent.app import DEFAULT_STATE_DIR, DEFAULT_WORKSPACE, build_agent


def _cli_confirm(action: str, context: dict) -> bool:
    print(f"\n[Требуется разрешение] Действие: {action}")
    for key, value in context.items():
        print(f"  {key}: {value}")
    answer = input("Разрешить? [y/N]: ").strip().lower()
    return answer in ("y", "yes", "д", "да")


def _auto_approve(action: str, context: dict) -> bool:
    return True


def _print_result(result) -> None:
    print(result.final_text)
    for step in result.steps:
        status = "OK" if step.result.ok else "ОШИБКА"
        print(f"  [{status}] {step.tool_name}({step.arguments})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-agent", description="Локальный самообучаемый ИИ-агент")
    parser.add_argument("--once", metavar="TASK", help="Выполнить одну задачу и выйти (для автоматизации/CI)")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE, help="Рабочая директория агента")
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR, help="Директория конфигов/памяти")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Автоматически подтверждать все запросы на разрешение (небезопасно; для тестов/автоматизации)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    confirm_callback = _auto_approve if args.yes else _cli_confirm
    agent = build_agent(
        state_dir=args.state_dir,
        workspace_root=args.workspace,
        confirm_callback=confirm_callback,
    )

    print(f"Рабочая директория: {args.workspace}")
    print(f"Профиль производительности (автонастройка): {agent.skill_context.profile.tier}")

    if args.once:
        result = agent.run_task(args.once)
        _print_result(result)
        return 0 if result.success else 1

    print("Локальный ИИ-агент. Введите задачу ('exit' или Ctrl+C для выхода).")
    while True:
        try:
            task = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nПока!")
            return 0
        if not task:
            continue
        if task.lower() in ("exit", "quit", "выход"):
            return 0
        result = agent.run_task(task)
        _print_result(result)


if __name__ == "__main__":
    sys.exit(main())
