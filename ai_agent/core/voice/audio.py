"""Захват аудио с микрофона и сегментация на реплики через VAD.

``sounddevice``/``webrtcvad`` — опциональные зависимости (extras: voice),
импортируются только внутри методов, которые реально трогают железо —
модуль можно свободно импортировать даже без них установленных (нужно
только для того, чтобы вызвать ``dependencies_available()`` и показать
пользователю понятную причину, а не ``ImportError`` где-то в глубине стека).
"""

from __future__ import annotations

import array
import queue
import threading
import time
from typing import Callable, Optional

SAMPLE_RATE = 16000
FRAME_MS = 30  # webrtcvad поддерживает только 10/20/30 мс на фрейм


def dependencies_available() -> tuple[bool, str]:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False, "sounddevice не установлен (extras: voice)"
    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        return False, "webrtcvad не установлен (extras: voice)"
    return True, ""


def list_input_devices() -> list[dict]:
    """Перечисляет доступные устройства записи (микрофоны) через PortAudio.

    Каждая запись: index (для передачи в AudioCapture(device=...)), name,
    channels, default_samplerate, is_default."""
    ok, reason = dependencies_available()
    if not ok:
        raise RuntimeError(reason)
    import sounddevice as sd

    default = sd.default.device
    default_input = default[0] if isinstance(default, (list, tuple)) else default

    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": info.get("name", f"Устройство {index}"),
                "channels": info.get("max_input_channels", 1),
                "default_samplerate": info.get("default_samplerate", SAMPLE_RATE),
                "is_default": index == default_input,
            }
        )
    return devices


def frame_level(frame: bytes) -> float:
    """RMS-уровень фрейма PCM 16-bit моно, нормализованный в [0, 1] —
    используется для живого индикатора шума (не зависит от VAD/речи)."""
    if not frame:
        return 0.0
    samples = array.array("h")
    try:
        samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    except ValueError:
        return 0.0
    if not samples:
        return 0.0
    mean_square = sum(s * s for s in samples) / len(samples)
    rms = mean_square**0.5
    return min(rms / 32768.0, 1.0)


class AudioCapture:
    """Слушает микрофон и режет поток на реплики по паузам речи (VAD).

    ``record_utterances`` — генератор, отдающий PCM 16-bit моно байты
    каждой обнаруженной реплики. Останавливается по ``stop()`` (для
    постоянной записи) или по истечении ``max_seconds`` (для разовой).
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        aggressiveness: int = 2,
        silence_ms: int = 600,
        device: int | str | None = None,
        level_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        ok, reason = dependencies_available()
        if not ok:
            raise RuntimeError(reason)
        import webrtcvad

        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
        self.device = device
        # Публичный изменяемый атрибут: вызывающий код (VoiceService) может
        # выставить/поменять его в любой момент — используется для живого
        # индикатора уровня сигнала независимо от распознавания речи.
        self.level_callback = level_callback
        self._vad = webrtcvad.Vad(aggressiveness)
        self._stop_event = threading.Event()

    def record_utterances(self, max_seconds: float | None = None):
        import sounddevice as sd

        frame_len = int(self.sample_rate * FRAME_MS / 1000)
        audio_q: "queue.Queue[bytes]" = queue.Queue()

        def _callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            audio_q.put(bytes(indata))

        self._stop_event.clear()
        start = time.monotonic()
        voiced_frames: list[bytes] = []
        silence_run = 0

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=frame_len,
            dtype="int16",
            channels=1,
            device=self.device,
            callback=_callback,
        ):
            while not self._stop_event.is_set():
                if max_seconds is not None and time.monotonic() - start > max_seconds:
                    break
                try:
                    frame = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if self.level_callback is not None:
                    self.level_callback(frame_level(frame))
                if self._vad.is_speech(frame, self.sample_rate):
                    voiced_frames.append(frame)
                    silence_run = 0
                elif voiced_frames:
                    silence_run += FRAME_MS
                    if silence_run >= self.silence_ms:
                        yield b"".join(voiced_frames)
                        voiced_frames = []
                        silence_run = 0

        if voiced_frames:
            yield b"".join(voiced_frames)

    def stop(self) -> None:
        self._stop_event.set()
