"""Захват аудио с микрофона и сегментация на реплики через VAD.

``sounddevice``/``webrtcvad`` — опциональные зависимости (extras: voice),
импортируются только внутри методов, которые реально трогают железо —
модуль можно свободно импортировать даже без них установленных (нужно
только для того, чтобы вызвать ``dependencies_available()`` и показать
пользователю понятную причину, а не ``ImportError`` где-то в глубине стека).
"""

from __future__ import annotations

import queue
import threading
import time

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
    ) -> None:
        ok, reason = dependencies_available()
        if not ok:
            raise RuntimeError(reason)
        import webrtcvad

        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
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
            callback=_callback,
        ):
            while not self._stop_event.is_set():
                if max_seconds is not None and time.monotonic() - start > max_seconds:
                    break
                try:
                    frame = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
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
