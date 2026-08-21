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

from typing import Optional

from ai_agent.core.autotune import detect_hardware
from ai_agent.core.hf_models import DownloadFn, HFApiClient, download_model, recommend_models, search_models

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec

MODELS_SUBDIR = "models"


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
            return SkillResult(ok=False, error=f"ошибка скачивания модели: {e}")

        return SkillResult(
            ok=True,
            output=f"Модель скачана: {local_path}",
            data={"repo_id": repo_id, "path": str(local_path)},
        )


def register_model_skills(registry) -> None:
    registry.register(RecommendModelsSkill())
    registry.register(SearchHuggingFaceSkill())
    registry.register(DownloadHuggingFaceModelSkill())
