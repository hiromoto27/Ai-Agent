"""Установка компонентов диктофона: pip-пакеты (sounddevice/webrtcvad/
faster-whisper) + рекомендация и предзагрузка STT-модели под характеристики
текущего ПК — тот же принцип автоподбора, что и для LLM-моделей в
``ai_agent/core/hf_models.py`` (см. ``models.recommend``)."""

from __future__ import annotations

from dataclasses import dataclass

from ai_agent.core.autotune import HardwareInfo, Profile

from . import audio, stt

VOICE_PACKAGES = ["sounddevice", "webrtcvad", "faster-whisper"]

_MODEL_BY_TIER = {"low": "tiny", "medium": "base", "high": "small"}


@dataclass
class DependencyStatus:
    audio_ok: bool
    audio_reason: str
    stt_ok: bool
    stt_reason: str

    @property
    def ready(self) -> bool:
        return self.audio_ok and self.stt_ok


def check_dependencies() -> DependencyStatus:
    audio_ok, audio_reason = audio.dependencies_available()
    stt_ok, stt_reason = stt.dependencies_available()
    return DependencyStatus(audio_ok=audio_ok, audio_reason=audio_reason, stt_ok=stt_ok, stt_reason=stt_reason)


def recommend_stt_model_size(hw: HardwareInfo, profile: Profile) -> str:
    """Размер модели faster-whisper под текущее железо — та же логика
    "не впритык, а с запасом", что и для рекомендаций LLM-моделей."""
    size = _MODEL_BY_TIER.get(profile.tier, "base")
    if profile.tier == "high" and hw.has_gpu:
        size = "medium"
    return size
