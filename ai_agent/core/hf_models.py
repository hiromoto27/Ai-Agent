"""Интеграция с Hugging Face Hub: подбор моделей под характеристики текущего
ПК, поиск и (с разрешения) скачивание.

Рекомендации строятся на курируемом списке моделей, а не на "сыром" API
Hugging Face: API не даёт надёжно вычислить объём/качество модели по одним
только тегам, поэтому используется тот же принцип, что у Ollama/LM Studio —
проверенный список моделей с известным размером и требованиями к памяти.
Поиск (``search_models``) и скачивание (``download_model``) отдельно
позволяют подтягивать любые модели с Hub напрямую.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from ai_agent.core.autotune import HardwareInfo, Profile

# huggingface_hub по умолчанию качает большие файлы через ускоренный бэкенд
# "Xet" (отдельный протокол/CDN, отличный от обычного HTTPS). На части
# сетей (корпоративные фаервол/прокси, некоторые антивирусы) он не проходит,
# при этом мелкие служебные файлы репозитория (README/config.json), которые
# идут обычным HTTP, скачиваются нормально — со стороны выглядит как "папка
# модели создалась, но веса не скачались" (0 байт вместо гигабайтов).
# Отключаем Xet и используем обычный HTTPS — медленнее, зато надёжнее на
# любой сети, что важнее для десктоп-приложения массового пользователя.
# Переменную нужно выставить ДО первого импорта huggingface_hub (эта строка
# исполняется при импорте ai_agent.core.hf_models, что происходит раньше).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass(frozen=True)
class CuratedModel:
    repo_id: str
    label: str
    approx_size_gb: float
    quantization: str
    min_ram_gb: float
    requires_gpu: bool
    notes: str


# Проверенные локальные модели (GGUF-квантование, кроме отмеченных fp16),
# отсортированные примерно по возрастанию требований к железу.
CURATED_MODELS: list[CuratedModel] = [
    CuratedModel(
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "Qwen2.5 0.5B Instruct",
        0.4, "Q4_K_M", 2, False,
        "Самая лёгкая модель — работает почти на любом ПК",
    ),
    CuratedModel(
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "Qwen2.5 1.5B Instruct",
        1.0, "Q4_K_M", 4, False,
        "Хороший баланс скорости и качества для слабых ПК",
    ),
    CuratedModel(
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "Phi-3.5 mini Instruct",
        2.4, "Q4_K_M", 6, False,
        "Компактная, неплохо рассуждает для своего размера",
    ),
    CuratedModel(
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5 7B Instruct",
        4.7, "Q4_K_M", 8, False,
        "Универсальный выбор для среднего ПК (8+ ГБ ОЗУ)",
    ),
    CuratedModel(
        "meta-llama/Llama-3.1-8B-Instruct",
        "Llama 3.1 8B Instruct",
        16.0, "fp16", 16, True,
        "Полная точность — нужна видеокарта с 16+ ГБ VRAM",
    ),
    CuratedModel(
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "Qwen2.5 14B Instruct",
        9.0, "Q4_K_M", 16, False,
        "Мощная модель для производительного ПК (CPU или GPU)",
    ),
    CuratedModel(
        "mistralai/Mistral-Small-Instruct-2409",
        "Mistral Small (22B)",
        44.0, "fp16", 24, True,
        "Топ-уровень — нужна мощная видеокарта (24+ ГБ VRAM)",
    ),
]

# Запас памяти: не рекомендуем модель, которая займёт больше этой доли RAM,
# чтобы оставить место ОС и самому приложению.
_RAM_HEADROOM = 0.7


def recommend_models(hw: HardwareInfo, profile: Optional[Profile] = None) -> list[CuratedModel]:
    """Отбирает из CURATED_MODELS модели, реально подходящие под железо.

    ``profile`` пока не используется напрямую (вся логика уже в hw), но
    принимается для единообразия с остальным кодом и на случай будущей
    тонкой настройки (например лимита числа рекомендаций по tier).
    """
    suitable = [
        m
        for m in CURATED_MODELS
        if hw.total_ram_gb >= m.min_ram_gb
        and m.approx_size_gb <= hw.total_ram_gb * _RAM_HEADROOM
        and (not m.requires_gpu or hw.has_gpu)
    ]
    suitable.sort(key=lambda m: m.approx_size_gb)
    return suitable


@dataclass
class HFSearchResult:
    id: str
    downloads: Optional[int]
    likes: Optional[int]
    tags: list[str]


class HFApiClient(Protocol):
    def list_models(self, *, search: str, limit: int, sort: str) -> Any: ...


def _default_api() -> HFApiClient:
    from huggingface_hub import HfApi  # импорт по требованию: huggingface_hub опционален

    return HfApi()


def search_models(query: str, limit: int = 10, api: Optional[HFApiClient] = None) -> list[HFSearchResult]:
    """Ищет модели на Hugging Face Hub по текстовому запросу."""
    client = api or _default_api()
    raw_results = client.list_models(search=query, limit=limit, sort="downloads")
    return [
        HFSearchResult(
            id=r.id,
            downloads=getattr(r, "downloads", None),
            likes=getattr(r, "likes", None),
            tags=list(getattr(r, "tags", None) or []),
        )
        for r in raw_results
    ]


DownloadFn = Callable[..., str]


def _default_download(**kwargs) -> str:
    from huggingface_hub import snapshot_download  # импорт по требованию

    return snapshot_download(**kwargs)


def download_model(
    repo_id: str,
    target_dir: Path,
    allow_patterns: Optional[list[str]] = None,
    download_fn: Optional[DownloadFn] = None,
) -> Path:
    """Скачивает модель с Hugging Face Hub в локальную папку.

    ``allow_patterns`` позволяет скачать только нужные файлы (например
    один конкретный GGUF-квант, а не все веса репозитория).
    """
    fn = download_fn or _default_download
    local_dir = target_dir / repo_id.replace("/", "__")
    result_path = fn(repo_id=repo_id, local_dir=str(local_dir), allow_patterns=allow_patterns)
    return Path(result_path)


# ---- авторизация -----------------------------------------------------------------
#
# Часть моделей на Hugging Face (например meta-llama/Llama-3.1-8B-Instruct)
# закрыта политикой доступа ("gated") — для скачивания нужен аккаунт,
# принятая лицензия и access-токен. huggingface_hub.login() сохраняет токен
# в свой стандартный локальный кеш (~/.cache/huggingface/token и на Windows
# аналогично), после чего HfApi()/snapshot_download() подхватывают его сами,
# без явной передачи token= в каждый вызов.


def is_auth_error(exc: Exception) -> bool:
    """True, если исключение похоже на "нужна авторизация/токен/лицензия"."""
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:
        return False

    if isinstance(exc, GatedRepoError):
        return True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in (401, 403)
    return False


def get_saved_token() -> Optional[str]:
    """Токен, уже сохранённый локально (через save_token), если есть."""
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def save_token(token: str) -> dict:
    """Сохраняет токен в локальный кеш huggingface_hub и проверяет его
    сразу (запрос whoami) — чтобы не сохранять явно нерабочий токен.
    Возвращает информацию о пользователе (как минимум ``name``)."""
    from huggingface_hub import login, whoami

    token = token.strip()
    if not token:
        raise ValueError("пустой токен")
    info = whoami(token=token)
    login(token=token, add_to_git_credential=False)
    return info


def clear_token() -> None:
    from huggingface_hub import logout

    logout()
