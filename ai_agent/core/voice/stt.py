"""Локальное распознавание речи (STT).

``faster-whisper`` — опциональная зависимость (extras: voice), импортируется
только при создании движка, не на уровне модуля. Выбор размера модели —
дело вызывающего кода (обычно ``autotune.Profile``, как и для LLM-моделей).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def dependencies_available() -> tuple[bool, str]:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False, "faster-whisper не установлен (extras: voice)"
    return True, ""


class SpeechToText(ABC):
    @abstractmethod
    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """Распознать сырые PCM 16-bit моно байты в текст."""

    @abstractmethod
    def transcribe_file(self, path: Path) -> str:
        """Распознать аудиофайл (WAV/MP3/... — что понимает ffmpeg/движок) в текст."""


class FasterWhisperSTT(SpeechToText):
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8") -> None:
        ok, reason = dependencies_available()
        if not ok:
            raise RuntimeError(reason)
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        import numpy as np

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language="ru")
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe_file(self, path: Path) -> str:
        segments, _ = self._model.transcribe(str(path), language="ru")
        return " ".join(segment.text.strip() for segment in segments).strip()


def create_stt_engine(model_size: str = "base") -> SpeechToText:
    return FasterWhisperSTT(model_size=model_size)
