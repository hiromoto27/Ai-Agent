"""Настройки LLM-провайдера агента: какой источник ответов использовать
(облачный Claude API, локальная GGUF-модель или офлайн-заглушка для
тестов), с какими параметрами, и пользовательский системный промпт
(инструкции, определяющие поведение агента поверх базовых).

Хранится отдельно от ``config/policy.yaml``/``settings.yaml`` — это не
про права доступа и не про автонастройку под железо, а про то, "кто
отвечает" и "как себя вести".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# "auto" — подобрать первый реально доступный провайдер (Claude API, если
# задан ключ, иначе локальная модель, если указан путь, иначе офлайн-эхо).
# Явный выбор ("anthropic"/"local"/"echo") не деградирует молча — если он
# не может стартовать, пользователь должен это увидеть, а не получить
# незаметно эхо-заглушку вместо настоящей модели.
DEFAULT_PROVIDER = "auto"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_LOCAL_N_CTX = 4096


@dataclass
class LLMSettings:
    provider: str = DEFAULT_PROVIDER
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL
    anthropic_api_key: str = ""  # непусто — переопределяет переменную окружения ANTHROPIC_API_KEY
    local_model_path: str = ""  # путь к .gguf-файлу (обычно из workspace/models)
    local_n_ctx: int = DEFAULT_LOCAL_N_CTX
    system_prompt: str = ""  # доп. инструкции пользователя поверх базового системного промпта

    @classmethod
    def load(cls, path: Path) -> "LLMSettings":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            provider=data.get("provider", DEFAULT_PROVIDER),
            anthropic_model=data.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL),
            anthropic_api_key=data.get("anthropic_api_key", ""),
            local_model_path=data.get("local_model_path", ""),
            local_n_ctx=data.get("local_n_ctx", DEFAULT_LOCAL_N_CTX),
            system_prompt=data.get("system_prompt", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "provider": self.provider,
            "anthropic_model": self.anthropic_model,
            "anthropic_api_key": self.anthropic_api_key,
            "local_model_path": self.local_model_path,
            "local_n_ctx": self.local_n_ctx,
            "system_prompt": self.system_prompt,
        }
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
