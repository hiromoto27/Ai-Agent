"""Установка новых Python-пакетов — механизм приобретения агентом новых
навыков/инструментов, найденных в интернете. Всегда с разрешения
пользователя (policy: package_install.enabled + подтверждение).

MVP ограничение: пакеты ставятся в то же окружение, где работает агент,
а не в отдельное песочное venv (это заявлено как развитие в PLAN.md,
раздел 4.5). Риск компенсируется тем, что установка в принципе выключена
по умолчанию и требует явного разрешения + подтверждения на каждый пакет.
"""

from __future__ import annotations

import re
import subprocess
import sys

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,_-]+\])?(==[A-Za-z0-9.\-*+!]+)?$")


class InstallPackageSkill(Skill):
    spec = SkillSpec(
        name="skills.install_package",
        description=(
            "Установить пакет Python (pip), которого не хватает для решения задачи. "
            "Используй, когда для задачи нужна библиотека, которой нет. Требует разрешения пользователя."
        ),
        parameters=[
            SkillParam("package", "string", "Имя пакета, опционально с версией: requests==2.32.0"),
        ],
    )

    def __init__(self, pip_cmd: list[str] | None = None) -> None:
        self.pip_cmd = pip_cmd or [sys.executable, "-m", "pip"]

    def _run(self, context: SkillContext, package: str) -> SkillResult:
        if not _PACKAGE_NAME_RE.match(package.strip()):
            return SkillResult(ok=False, error=f"некорректное имя пакета: {package!r}")

        context.policy.enforce("package.install", package=package)

        try:
            result = subprocess.run(
                [*self.pip_cmd, "install", package],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return SkillResult(ok=False, error="установка пакета превысила таймаут")

        if result.returncode != 0:
            return SkillResult(ok=False, error=result.stderr[-4000:] or "неизвестная ошибка pip")

        return SkillResult(
            ok=True,
            output=f"Пакет установлен: {package}",
            data={"package": package},
        )


def register_package_skills(registry, pip_cmd: list[str] | None = None) -> None:
    registry.register(InstallPackageSkill(pip_cmd=pip_cmd))
