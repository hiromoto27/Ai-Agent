"""Настройки модуля «Диктофон»: выбранное устройство записи и размер
STT-модели. Хранится отдельно от ``policy.yaml`` — это не про права
доступа, а про выбор оборудования/модели (как ``llm_settings.py`` для
LLM-провайдера).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class VoiceSettings:
    input_device: int | None = None  # None — устройство по умолчанию в системе
    stt_model_size: str = ""  # пусто — автоподбор под железо при установке/первом использовании

    @classmethod
    def load(cls, path: Path) -> "VoiceSettings":
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            input_device=data.get("input_device"),
            stt_model_size=data.get("stt_model_size", ""),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"input_device": self.input_device, "stt_model_size": self.stt_model_size}
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
