"""Базовые типы для системы навыков (инструментов) агента.

Каждый навык — это отдельный класс с декларативной спецификацией
параметров (для передачи LLM как tool-schema) и методом ``run``,
который получает ``SkillContext`` (рабочая директория, policy engine,
профиль производительности) и именованные аргументы.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ai_agent.core.autotune import Profile, build_profile, detect_hardware
from ai_agent.core.logging_setup import get_logger
from ai_agent.core.policy import PermissionDenied, PolicyEngine

logger = get_logger("skills")

_SECRET_KWARG_MARKERS = ("token", "key", "password", "secret")


def _redact_kwargs(kwargs: dict) -> dict:
    """Не пишем в лог значения аргументов, похожих на секреты (токены и
    т.п.) — лог-файл может попасть в переписку при разборе проблемы."""
    return {
        k: ("***" if any(marker in k.lower() for marker in _SECRET_KWARG_MARKERS) else v)
        for k, v in kwargs.items()
    }


@dataclass
class SkillParam:
    name: str
    type: str  # "string" | "integer" | "number" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    items: Optional[dict] = None  # для type == "array"


@dataclass
class SkillSpec:
    name: str
    description: str
    parameters: list[SkillParam] = field(default_factory=list)

    def to_tool_schema(self) -> dict:
        properties: dict[str, Any] = {}
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.items is not None:
                prop["items"] = p.items
            properties[p.name] = prop
        required = [p.name for p in self.parameters if p.required]
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class SkillResult:
    ok: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""

    def to_tool_message(self) -> str:
        if self.ok:
            return self.output
        return f"ОШИБКА: {self.error}"


@dataclass
class SkillContext:
    workspace_root: Path
    policy: PolicyEngine
    profile: Profile = field(default_factory=lambda: build_profile(detect_hardware()))

    def resolve_path(self, path_str: str, subdir: Optional[str] = None) -> Path:
        """Разрешает путь: абсолютный — как есть, относительный — от workspace
        (опционально от подпапки, например 'documents' или 'scripts')."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        base = self.workspace_root / subdir if subdir else self.workspace_root
        return (base / p).resolve()


class Skill(ABC):
    """Базовый класс навыка.

    Подклассы реализуют ``_run`` (логику), а публичный ``run`` — общая
    обёртка, которая гарантирует, что PermissionDenied и любые прочие
    исключения всегда превращаются в ``SkillResult(ok=False, ...)``,
    а не роняют агента, независимо от того, вызван навык напрямую или
    через ``SkillRegistry.invoke``.
    """

    spec: SkillSpec

    def run(self, context: SkillContext, **kwargs) -> SkillResult:
        try:
            return self._run(context, **kwargs)
        except PermissionDenied as e:
            logger.info("%s отклонён политикой: %s", self.spec.name, e)
            return SkillResult(ok=False, error=f"доступ запрещён: {e}")
        except Exception as e:  # защитный барьер — навык не должен ронять агента
            logger.exception("%s упал с исключением (аргументы: %s)", self.spec.name, _redact_kwargs(kwargs))
            return SkillResult(ok=False, error=f"{type(e).__name__}: {e}")

    @abstractmethod
    def _run(self, context: SkillContext, **kwargs) -> SkillResult:
        raise NotImplementedError


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.spec.name] = skill

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"неизвестный навык: {name}")
        return self._skills[name]

    def has(self, name: str) -> bool:
        return name in self._skills

    def list_specs(self) -> list[SkillSpec]:
        return [s.spec for s in self._skills.values()]

    def tool_schemas(self) -> list[dict]:
        return [s.spec.to_tool_schema() for s in self._skills.values()]

    def invoke(self, name: str, context: SkillContext, **kwargs) -> SkillResult:
        try:
            skill = self.get(name)
        except KeyError as e:
            return SkillResult(ok=False, error=str(e))
        return skill.run(context, **kwargs)
