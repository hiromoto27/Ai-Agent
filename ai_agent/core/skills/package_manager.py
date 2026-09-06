"""Установка новых Python-пакетов — механизм приобретения агентом новых
навыков/инструментов, найденных в интернете. Всегда с разрешения
пользователя (policy: package_install.enabled + подтверждение).

Перед установкой (и по отдельному запросу через
``skills.estimate_package_size``) агент может узнать примерный размер
пакета на PyPI — чтобы предупредить пользователя, сколько места
потребуется, до того как что-то реально скачается на диск.

MVP ограничение: пакеты ставятся в то же окружение, где работает агент,
а не в отдельное песочное venv (это заявлено как развитие в PLAN.md,
раздел 4.5). Риск компенсируется тем, что установка в принципе выключена
по умолчанию и требует явного разрешения + подтверждения на каждый пакет.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any, Optional, Protocol

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,_-]+\])?(==[A-Za-z0-9.\-*+!]+)?$")

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
PYPI_JSON_URL_VERSIONED = "https://pypi.org/pypi/{name}/{version}/json"


class HttpClient(Protocol):
    def get(self, url: str) -> Any: ...


def _default_http_client() -> HttpClient:
    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx не установлен (extras: web)") from e
    return httpx.Client(timeout=15.0)


def fetch_pypi_size(
    name: str, version: str = "", http_client: Optional[HttpClient] = None
) -> tuple[int, str]:
    """Возвращает (байты, версия) для пакета с PyPI — размер самого
    большого файла релиза (обычно .whl), как консервативная оценка.

    Это оценка ТОЛЬКО самого пакета, без учёта его же зависимостей —
    честная и простая оценка, вместо того чтобы эмулировать полное
    разрешение зависимостей pip (не гарантирует итоговый объём
    установки, но даёт представление о порядке величины)."""
    owns_client = http_client is None
    client = http_client or _default_http_client()
    url = (
        PYPI_JSON_URL_VERSIONED.format(name=name, version=version)
        if version
        else PYPI_JSON_URL.format(name=name)
    )
    try:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            client.close()

    resolved_version = data["info"]["version"]
    files = data.get("urls") or data.get("releases", {}).get(resolved_version, [])
    if not files:
        raise ValueError(f"на PyPI нет файлов релиза для {name} {resolved_version}")
    size_bytes = max(f["size"] for f in files)
    return size_bytes, resolved_version


def _format_size_note(name: str, size_bytes: int, version: str) -> str:
    size_mb = round(size_bytes / (1024**2), 2)
    return f"~{size_mb} МБ ({name} {version}, без учёта зависимостей)"


class EstimatePackageSizeSkill(Skill):
    spec = SkillSpec(
        name="skills.estimate_package_size",
        description=(
            "Узнать примерный размер (в МБ) пакета Python с PyPI ДО установки через "
            "skills.install_package — используй это, чтобы сообщить пользователю, сколько места "
            "потребуется, прежде чем ставить что-то новое, особенно если пакет может быть большим "
            "(модели, ML-библиотеки и т.п.)."
        ),
        parameters=[SkillParam("package", "string", "Имя пакета, опционально с версией: requests==2.32.0")],
    )

    def __init__(self, http_client: Optional[HttpClient] = None) -> None:
        self.http_client = http_client

    def _run(self, context: SkillContext, package: str) -> SkillResult:
        if not _PACKAGE_NAME_RE.match(package.strip()):
            return SkillResult(ok=False, error=f"некорректное имя пакета: {package!r}")
        name, _, version = package.strip().partition("==")

        context.policy.enforce("web.fetch", domain="pypi.org")

        try:
            size_bytes, resolved_version = fetch_pypi_size(name, version, http_client=self.http_client)
        except ImportError as e:
            return SkillResult(ok=False, error=str(e))
        except Exception as e:
            return SkillResult(ok=False, error=f"не удалось получить размер пакета с PyPI: {e}")

        return SkillResult(
            ok=True,
            output=_format_size_note(name, size_bytes, resolved_version),
            data={"package": name, "version": resolved_version, "size_bytes": size_bytes},
        )


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

    def __init__(self, pip_cmd: list[str] | None = None, http_client: Optional[HttpClient] = None) -> None:
        self.pip_cmd = pip_cmd or [sys.executable, "-m", "pip"]
        self.http_client = http_client

    def _run(self, context: SkillContext, package: str) -> SkillResult:
        if not _PACKAGE_NAME_RE.match(package.strip()):
            return SkillResult(ok=False, error=f"некорректное имя пакета: {package!r}")

        # Лучшая попытка: неудача (сеть/PyPI недоступны, политика запрещает
        # web.fetch) не должна мешать самой установке — это просто подсказка
        # пользователю в диалоге подтверждения, а не обязательное условие.
        name, _, version = package.strip().partition("==")
        size_note = "неизвестен (не удалось узнать заранее)"
        try:
            context.policy.enforce("web.fetch", domain="pypi.org")
            size_bytes, resolved_version = fetch_pypi_size(name, version, http_client=self.http_client)
            size_note = _format_size_note(name, size_bytes, resolved_version)
        except Exception:
            pass

        context.policy.enforce("package.install", package=package, estimated_size=size_note)

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
    registry.register(EstimatePackageSizeSkill())
    registry.register(InstallPackageSkill(pip_cmd=pip_cmd))
