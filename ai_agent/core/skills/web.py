"""Доступ в интернет: получение страниц и поиск (для поиска информации и
новых инструментов/навыков, недостающих агенту). Каждый запрос проходит
проверку домена через PolicyEngine."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

MAX_BODY_CHARS = 50_000
DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"

_RESULT_LINK_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _TAG_RE.sub("", html).strip()


class FetchUrlSkill(Skill):
    spec = SkillSpec(
        name="web.fetch_url",
        description="Скачать содержимое веб-страницы по URL (GET-запрос).",
        parameters=[SkillParam("url", "string", "Полный URL, включая http(s)://")],
    )

    def _run(self, context: SkillContext, url: str) -> SkillResult:
        try:
            import httpx
        except ImportError:
            return SkillResult(ok=False, error="httpx не установлен (extras: web)")

        domain = urlparse(url).hostname or ""
        context.policy.enforce("web.fetch", domain=domain)

        try:
            response = httpx.get(url, timeout=15, follow_redirects=True)
        except httpx.HTTPError as e:
            return SkillResult(ok=False, error=f"ошибка запроса: {e}")

        text = response.text[:MAX_BODY_CHARS]
        return SkillResult(
            ok=response.is_success,
            output=text,
            error="" if response.is_success else f"HTTP {response.status_code}",
            data={"status_code": response.status_code, "url": str(response.url)},
        )


class SearchWebSkill(Skill):
    spec = SkillSpec(
        name="web.search",
        description=(
            "Найти информацию, библиотеки или инструменты в интернете по текстовому запросу. "
            "Используй, когда для задачи не хватает знаний или готового навыка."
        ),
        parameters=[
            SkillParam("query", "string", "Поисковый запрос"),
            SkillParam("max_results", "integer", "Максимум результатов", required=False),
        ],
    )

    def __init__(self, endpoint: str = DEFAULT_SEARCH_ENDPOINT) -> None:
        self.endpoint = endpoint

    def _run(self, context: SkillContext, query: str, max_results: int = 5) -> SkillResult:
        try:
            import httpx
        except ImportError:
            return SkillResult(ok=False, error="httpx не установлен (extras: web)")

        domain = urlparse(self.endpoint).hostname or ""
        context.policy.enforce("web.fetch", domain=domain)

        try:
            response = httpx.get(self.endpoint, params={"q": query}, timeout=15)
        except httpx.HTTPError as e:
            return SkillResult(ok=False, error=f"ошибка поиска: {e}")

        if not response.is_success:
            return SkillResult(ok=False, error=f"HTTP {response.status_code}")

        matches = _RESULT_LINK_RE.findall(response.text)[:max_results]
        results = [{"url": href, "title": _strip_tags(title)} for href, title in matches]

        if not results:
            return SkillResult(ok=True, output="Ничего не найдено.", data={"results": []})

        lines = [f"- {r['title']}\n  {r['url']}" for r in results]
        return SkillResult(ok=True, output="\n".join(lines), data={"results": results})


def register_web_skills(registry, search_endpoint: str = DEFAULT_SEARCH_ENDPOINT) -> None:
    registry.register(FetchUrlSkill())
    registry.register(SearchWebSkill(endpoint=search_endpoint))
