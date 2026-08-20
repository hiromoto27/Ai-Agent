"""Навыки для написания и (с разрешения) запуска агентом вспомогательных
скриптов — способ агента создавать себе новые мини-инструменты на лету.

Запись скрипта — безопасная операция (это просто файл в
workspace/scripts), включена по умолчанию. Запуск скрипта — отдельно
гейтится политикой (scripting.execute_enabled) и требует подтверждения.
"""

from __future__ import annotations

import subprocess
import sys

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

SCRIPTS_SUBDIR = "scripts"

_INTERPRETERS = {
    ".py": [sys.executable],
    ".sh": ["bash"],
    ".ps1": ["powershell", "-NoProfile", "-File"],
    ".bat": [],  # запускается напрямую через cmd
}

MAX_OUTPUT_CHARS = 20_000


def _safe_script_name(filename: str) -> str:
    name = filename.strip().replace("\\", "/").split("/")[-1]
    if not name or name in (".", ".."):
        raise ValueError("некорректное имя файла скрипта")
    return name


class WriteScriptSkill(Skill):
    spec = SkillSpec(
        name="scripting.write_script",
        description=(
            "Создать вспомогательный скрипт (Python/Bash/PowerShell) в папке "
            "scripts/ рабочей директории. Это только запись файла, без запуска."
        ),
        parameters=[
            SkillParam("filename", "string", "Имя файла, например helper.py"),
            SkillParam("content", "string", "Содержимое скрипта"),
        ],
    )

    def _run(self, context: SkillContext, filename: str, content: str) -> SkillResult:
        context.policy.enforce("scripting.write")
        try:
            name = _safe_script_name(filename)
        except ValueError as e:
            return SkillResult(ok=False, error=str(e))

        target = context.resolve_path(name, subdir=SCRIPTS_SUBDIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return SkillResult(
            ok=True,
            output=f"Скрипт сохранён: {target}",
            data={"path": str(target)},
        )


class RunScriptSkill(Skill):
    spec = SkillSpec(
        name="scripting.run_script",
        description=(
            "Запустить ранее созданный скрипт из scripts/. Требует явного "
            "разрешения пользователя (scripting.execute_enabled + подтверждение)."
        ),
        parameters=[
            SkillParam("filename", "string", "Имя файла скрипта в scripts/"),
            SkillParam(
                "args",
                "array",
                "Аргументы командной строки",
                required=False,
                items={"type": "string"},
            ),
        ],
    )

    def _run(self, context: SkillContext, filename: str, args: list[str] | None = None) -> SkillResult:
        context.policy.enforce("scripting.execute")
        try:
            name = _safe_script_name(filename)
        except ValueError as e:
            return SkillResult(ok=False, error=str(e))

        scripts_dir = (context.workspace_root / SCRIPTS_SUBDIR).resolve()
        target = (scripts_dir / name).resolve()
        try:
            target.relative_to(scripts_dir)
        except ValueError:
            return SkillResult(ok=False, error="попытка выйти за пределы папки scripts/ запрещена")
        if not target.exists():
            return SkillResult(ok=False, error=f"скрипт не найден: {target}")

        suffix = target.suffix.lower()
        if suffix not in _INTERPRETERS:
            return SkillResult(ok=False, error=f"неподдерживаемый тип скрипта: {suffix}")

        cmd = [*_INTERPRETERS[suffix], str(target), *(args or [])] if suffix != ".bat" else [str(target), *(args or [])]

        timeout = context.profile.max_script_timeout_sec
        try:
            result = subprocess.run(
                cmd,
                cwd=context.workspace_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SkillResult(ok=False, error=f"скрипт превысил таймаут {timeout}с")
        except FileNotFoundError as e:
            return SkillResult(ok=False, error=f"интерпретатор не найден: {e}")

        stdout = result.stdout[:MAX_OUTPUT_CHARS]
        stderr = result.stderr[:MAX_OUTPUT_CHARS]
        output = f"exit_code={result.returncode}\nstdout:\n{stdout}"
        if stderr:
            output += f"\nstderr:\n{stderr}"

        return SkillResult(
            ok=result.returncode == 0,
            output=output,
            error="" if result.returncode == 0 else f"скрипт завершился с кодом {result.returncode}",
            data={"exit_code": result.returncode, "stdout": stdout, "stderr": stderr},
        )


def register_scripting_skills(registry) -> None:
    registry.register(WriteScriptSkill())
    registry.register(RunScriptSkill())
