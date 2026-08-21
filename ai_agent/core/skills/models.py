"""Навыки для подбора и получения LLM-моделей с Hugging Face Hub.

``models.recommend`` — локальная операция (без сети): подсказывает, какие
модели реально потянет текущий ПК, на основе автонастройки (см.
core.autotune). ``models.search_huggingface`` ищет модели на Hub.
``models.download_huggingface`` скачивает модель в workspace/models —
единственная из трёх, требующая отдельного разрешения политики
(model_download.enabled + подтверждение), так как это может занять
гигабайты места и трафика.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai_agent.core.autotune import detect_hardware
from ai_agent.core.hf_models import (
    DownloadFn,
    HFApiClient,
    clear_token,
    download_model,
    get_saved_token,
    is_auth_error,
    recommend_models,
    save_token,
    search_models,
)

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

MODELS_SUBDIR = "models"
_IGNORED_ENTRIES = {"README.txt"}


def _entry_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


class RecommendModelsSkill(Skill):
    spec = SkillSpec(
        name="models.recommend",
        description=(
            "Подобрать LLM-модели с Hugging Face, которые реально потянет текущий ПК, "
            "исходя из его характеристик (CPU/RAM/GPU). Не требует интернета и разрешений."
        ),
        parameters=[],
    )

    def _run(self, context: SkillContext) -> SkillResult:
        hw = detect_hardware()
        recommendations = recommend_models(hw, context.profile)
        if not recommendations:
            return SkillResult(
                ok=True,
                output=(
                    f"Для текущего железа ({hw.cpu_cores} ядер, {hw.total_ram_gb} ГБ ОЗУ, "
                    f"GPU: {'есть' if hw.has_gpu else 'нет'}) подходящих моделей из "
                    "курируемого списка не нашлось — маловато памяти даже для самой лёгкой."
                ),
                data={"hardware": hw.__dict__, "models": []},
            )
        lines = [
            f"Железо: {hw.cpu_cores} ядер, {hw.total_ram_gb} ГБ ОЗУ, GPU: {'есть' if hw.has_gpu else 'нет'}",
            "Рекомендуемые модели (по возрастанию размера):",
        ]
        for m in recommendations:
            lines.append(f"- {m.label} ({m.repo_id}), ~{m.approx_size_gb} ГБ, {m.quantization} — {m.notes}")
        return SkillResult(
            ok=True,
            output="\n".join(lines),
            data={
                "hardware": hw.__dict__,
                "models": [
                    {
                        "repo_id": m.repo_id,
                        "label": m.label,
                        "approx_size_gb": m.approx_size_gb,
                        "quantization": m.quantization,
                        "notes": m.notes,
                    }
                    for m in recommendations
                ],
            },
        )


class SearchHuggingFaceSkill(Skill):
    spec = SkillSpec(
        name="models.search_huggingface",
        description="Найти модели на Hugging Face Hub по текстовому запросу (отсортировано по популярности).",
        parameters=[
            SkillParam("query", "string", "Поисковый запрос, например 'qwen2.5 instruct gguf'"),
            SkillParam("max_results", "integer", "Максимум результатов", required=False),
        ],
    )

    def __init__(self, api: Optional[HFApiClient] = None) -> None:
        self.api = api

    def _run(self, context: SkillContext, query: str, max_results: int = 10) -> SkillResult:
        context.policy.enforce("web.fetch", domain="huggingface.co")
        try:
            results = search_models(query, limit=max_results, api=self.api)
        except ImportError:
            return SkillResult(ok=False, error="huggingface_hub не установлен (extras: hf)")
        except Exception as e:
            if is_auth_error(e):
                return SkillResult(
                    ok=False,
                    error="нужна авторизация на Hugging Face (см. вкладку «Модели» — «Авторизация»)",
                    data={"auth_required": True},
                )
            return SkillResult(ok=False, error=f"ошибка поиска на Hugging Face: {e}")

        if not results:
            return SkillResult(ok=True, output="Ничего не найдено.", data={"results": []})

        lines = [f"- {r.id} (загрузок: {r.downloads}, лайков: {r.likes})" for r in results]
        return SkillResult(
            ok=True,
            output="\n".join(lines),
            data={"results": [{"id": r.id, "downloads": r.downloads, "likes": r.likes, "tags": r.tags} for r in results]},
        )


class DownloadHuggingFaceModelSkill(Skill):
    spec = SkillSpec(
        name="models.download_huggingface",
        description=(
            "Скачать модель с Hugging Face Hub в workspace/models. Может занять много места и трафика — "
            "требует явного разрешения пользователя (policy: model_download.enabled + подтверждение)."
        ),
        parameters=[
            SkillParam("repo_id", "string", "Идентификатор репозитория, например Qwen/Qwen2.5-1.5B-Instruct-GGUF"),
            SkillParam(
                "allow_patterns",
                "array",
                "Скачать только файлы по этим маскам (например конкретный .gguf-файл), иначе весь репозиторий",
                required=False,
                items={"type": "string"},
            ),
        ],
    )

    def __init__(self, download_fn: Optional[DownloadFn] = None) -> None:
        self.download_fn = download_fn

    def _run(self, context: SkillContext, repo_id: str, allow_patterns: Optional[list[str]] = None) -> SkillResult:
        context.policy.enforce("model.download", repo_id=repo_id)
        target_dir = context.resolve_path(".", subdir=MODELS_SUBDIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            local_path = download_model(repo_id, target_dir, allow_patterns=allow_patterns, download_fn=self.download_fn)
        except ImportError:
            return SkillResult(ok=False, error="huggingface_hub не установлен (extras: hf)")
        except Exception as e:
            if is_auth_error(e):
                return SkillResult(
                    ok=False,
                    error=(
                        f"модель {repo_id} требует авторизации на Hugging Face "
                        "(закрытый доступ или нужен токен) — см. вкладку «Модели» — «Авторизация»"
                    ),
                    data={"auth_required": True, "repo_id": repo_id},
                )
            return SkillResult(ok=False, error=f"ошибка скачивания модели: {e}")

        return SkillResult(
            ok=True,
            output=f"Модель скачана: {local_path}",
            data={"repo_id": repo_id, "path": str(local_path)},
        )


class ListLocalModelsSkill(Skill):
    spec = SkillSpec(
        name="models.list_local",
        description=(
            "Показать модели, уже лежащие локально в workspace/models — скачанные через "
            "models.download_huggingface или добавленные вручную (пользователь может просто "
            "скопировать туда файлы моделей из другого источника)."
        ),
        parameters=[],
    )

    def _run(self, context: SkillContext) -> SkillResult:
        target_dir = context.resolve_path(".", subdir=MODELS_SUBDIR)
        if not target_dir.exists():
            return SkillResult(ok=True, output="Папка models пуста.", data={"models": []})

        entries = []
        for p in sorted(target_dir.iterdir()):
            if p.name in _IGNORED_ENTRIES:
                continue
            size_gb = round(_entry_size_bytes(p) / (1024**3), 3)
            entries.append({"name": p.name, "is_dir": p.is_dir(), "size_gb": size_gb})

        if not entries:
            return SkillResult(ok=True, output="Папка models пуста.", data={"models": [], "path": str(target_dir)})

        lines = [
            f"- {e['name']} ({'папка' if e['is_dir'] else 'файл'}, ~{e['size_gb']} ГБ)" for e in entries
        ]
        return SkillResult(
            ok=True,
            output="\n".join(lines),
            data={"models": entries, "path": str(target_dir)},
        )


class SetHuggingFaceTokenSkill(Skill):
    spec = SkillSpec(
        name="models.set_hf_token",
        description=(
            "Сохранить access-токен Hugging Face (для скачивания закрытых/gated моделей). "
            "Токен получают на странице https://huggingface.co/settings/tokens."
        ),
        parameters=[SkillParam("token", "string", "Access-токен Hugging Face")],
    )

    def _run(self, context: SkillContext, token: str) -> SkillResult:
        context.policy.enforce("web.fetch", domain="huggingface.co")
        try:
            info = save_token(token)
        except ImportError:
            return SkillResult(ok=False, error="huggingface_hub не установлен (extras: hf)")
        except ValueError as e:
            return SkillResult(ok=False, error=str(e))
        except Exception as e:
            return SkillResult(ok=False, error=f"токен не принят Hugging Face: {e}")

        username = info.get("name") or info.get("fullname") or "неизвестно"
        return SkillResult(ok=True, output=f"Вход выполнен: {username}", data={"username": username})


class ClearHuggingFaceTokenSkill(Skill):
    spec = SkillSpec(
        name="models.clear_hf_token",
        description="Удалить локально сохранённый токен Hugging Face (выйти из аккаунта).",
        parameters=[],
    )

    def _run(self, context: SkillContext) -> SkillResult:
        try:
            clear_token()
        except ImportError:
            return SkillResult(ok=False, error="huggingface_hub не установлен (extras: hf)")
        return SkillResult(ok=True, output="Токен удалён.")


def hf_login_status() -> Optional[str]:
    """Вспомогательная функция для UI: есть ли сохранённый токен (без сетевого запроса)."""
    return get_saved_token()


def register_model_skills(registry) -> None:
    registry.register(RecommendModelsSkill())
    registry.register(SearchHuggingFaceSkill())
    registry.register(DownloadHuggingFaceModelSkill())
    registry.register(ListLocalModelsSkill())
    registry.register(SetHuggingFaceTokenSkill())
    registry.register(ClearHuggingFaceTokenSkill())
