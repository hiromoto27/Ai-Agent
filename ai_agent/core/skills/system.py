"""Взаимодействие с системой (выполнение команд) — только с разрешения
пользователя. По умолчанию заблокировано политикой (shell.enabled=false);
даже если включено, PolicyEngine всегда требует подтверждения перед
фактическим запуском (см. confirmation_required_for в policy.yaml)."""

from __future__ import annotations

import shlex
import subprocess

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

MAX_OUTPUT_CHARS = 20_000


class RunCommandSkill(Skill):
    spec = SkillSpec(
        name="system.run_command",
        description=(
            "Выполнить команду в системной оболочке рабочей директории. "
            "Требует явного разрешения пользователя (политика + подтверждение)."
        ),
        parameters=[
            SkillParam("command", "string", "Команда для выполнения"),
            SkillParam(
                "timeout_sec", "integer", "Таймаут в секундах", required=False
            ),
        ],
    )

    def _run(self, context: SkillContext, command: str, timeout_sec: int | None = None) -> SkillResult:
        context.policy.enforce("shell.execute", command=command)

        timeout = timeout_sec or context.profile.max_script_timeout_sec
        timeout = min(timeout, context.profile.max_script_timeout_sec * 4)

        try:
            # posix=True всегда, даже на Windows: это про синтаксис разбора
            # строки (корректная обработка кавычек вокруг аргументов типа
            # `-c "..."`), а не про целевую ОС. posix=False на Windows
            # сохраняет кавычки как часть токена и молча ломает такие
            # команды (см. историю коммитов) — posix=True работает
            # одинаково правильно на всех платформах.
            args = shlex.split(command, posix=True)
        except ValueError as e:
            return SkillResult(ok=False, error=f"не удалось разобрать команду: {e}")
        if not args:
            return SkillResult(ok=False, error="пустая команда")

        try:
            result = subprocess.run(
                args,
                cwd=context.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return SkillResult(ok=False, error=f"команда не найдена: {args[0]}")
        except subprocess.TimeoutExpired:
            return SkillResult(ok=False, error=f"команда превысила таймаут {timeout}с")

        stdout = result.stdout[:MAX_OUTPUT_CHARS]
        stderr = result.stderr[:MAX_OUTPUT_CHARS]
        output = f"exit_code={result.returncode}\nstdout:\n{stdout}"
        if stderr:
            output += f"\nstderr:\n{stderr}"

        return SkillResult(
            ok=result.returncode == 0,
            output=output,
            error="" if result.returncode == 0 else f"команда завершилась с кодом {result.returncode}",
            data={"exit_code": result.returncode, "stdout": stdout, "stderr": stderr},
        )


def register_system_skills(registry) -> None:
    registry.register(RunCommandSkill())
