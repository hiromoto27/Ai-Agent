"""Файловые навыки: чтение/запись/листинг, всегда через PolicyEngine."""

from __future__ import annotations

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

MAX_READ_CHARS = 200_000


class ReadFileSkill(Skill):
    spec = SkillSpec(
        name="files.read",
        description="Прочитать текстовый файл. Путь может быть относительным (от рабочей папки) или абсолютным.",
        parameters=[SkillParam("path", "string", "Путь к файлу")],
    )

    def _run(self, context: SkillContext, path: str) -> SkillResult:
        target = context.resolve_path(path)
        context.policy.enforce("files.read", path=target)
        if not target.exists():
            return SkillResult(ok=False, error=f"файл не найден: {target}")
        if not target.is_file():
            return SkillResult(ok=False, error=f"это не файл: {target}")
        text = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > MAX_READ_CHARS
        if truncated:
            text = text[:MAX_READ_CHARS]
        return SkillResult(
            ok=True,
            output=text,
            data={"path": str(target), "truncated": truncated},
        )


class WriteFileSkill(Skill):
    spec = SkillSpec(
        name="files.write",
        description="Записать текст в файл (создаёт директории при необходимости).",
        parameters=[
            SkillParam("path", "string", "Путь к файлу"),
            SkillParam("content", "string", "Содержимое файла"),
        ],
    )

    def _run(self, context: SkillContext, path: str, content: str) -> SkillResult:
        target = context.resolve_path(path)
        context.policy.enforce("files.write", path=target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return SkillResult(
            ok=True,
            output=f"Записано {len(content)} символов в {target}",
            data={"path": str(target)},
        )


class ListDirSkill(Skill):
    spec = SkillSpec(
        name="files.list",
        description="Показать содержимое директории (нерекурсивно).",
        parameters=[SkillParam("path", "string", "Путь к директории", required=False)],
    )

    def _run(self, context: SkillContext, path: str = ".") -> SkillResult:
        target = context.resolve_path(path)
        context.policy.enforce("files.read", path=target)
        if not target.exists():
            return SkillResult(ok=False, error=f"директория не найдена: {target}")
        if not target.is_dir():
            return SkillResult(ok=False, error=f"это не директория: {target}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return SkillResult(ok=True, output="\n".join(entries), data={"path": str(target), "count": len(entries)})


def register_file_skills(registry) -> None:
    registry.register(ReadFileSkill())
    registry.register(WriteFileSkill())
    registry.register(ListDirSkill())
