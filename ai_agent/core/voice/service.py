"""Связывает захват аудио, STT и классификатор задач в рабочий цикл
диктофона: разовая запись и постоянная (фоновая) запись.

Захват и STT собираются лениво через фабрики (``audio_capture_factory``,
``stt_factory``) — по умолчанию реальные ``AudioCapture``/``FasterWhisperSTT``
(требуют extras ``voice``), но их легко подменить в тестах фейками, не
трогая настоящий микрофон/модель.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Callable, Optional

from .audio import SAMPLE_RATE, AudioCapture
from .classifier import Classification, TaskClassifier, tokenize
from .protocol import Protocol, build_protocol
from .stt import SpeechToText, create_stt_engine
from .store import Task, Utterance, VoiceStore

UtteranceCallback = Callable[[Utterance, Classification], None]


class VoiceService:
    def __init__(
        self,
        store: VoiceStore,
        classifier: TaskClassifier | None = None,
        stt_factory: Callable[[], SpeechToText] | None = None,
        audio_capture_factory: Callable[[], AudioCapture] | None = None,
        audio_dir: Path | None = None,
        retain_audio: Callable[[], bool] | None = None,
        input_device: int | str | None = None,
    ) -> None:
        self.store = store
        self.classifier = classifier or TaskClassifier()
        self._stt_factory = stt_factory or (lambda: create_stt_engine())
        # Замыкание на self.input_device, а не на значение аргумента — смена
        # устройства через set_input_device() применяется к следующей же
        # записи, без пересборки сервиса (как и retain_audio выше).
        self._audio_capture_factory = audio_capture_factory or (
            lambda: AudioCapture(device=self.input_device, level_callback=self._on_level)
        )
        self._audio_dir = audio_dir
        self._retain_audio = retain_audio or (lambda: False)
        self._stt: Optional[SpeechToText] = None
        self.input_device = input_device
        # Живой уровень сигнала микрофона (0..1), обновляется на каждый
        # фрейм независимо от VAD/распознавания — для индикатора в UI.
        self.current_level: float = 0.0

        self._continuous_thread: threading.Thread | None = None
        self._continuous_capture: AudioCapture | None = None
        self._continuous_recording_id: int | None = None

        # Профили переживают перезапуск приложения — подхватываем из БД.
        for task_id, profile in store.load_all_task_profiles().items():
            self.classifier.profiles[task_id] = profile

    @property
    def is_continuous_active(self) -> bool:
        return self._continuous_thread is not None and self._continuous_thread.is_alive()

    def _on_level(self, level: float) -> None:
        self.current_level = level

    def set_input_device(self, device: int | str | None) -> None:
        self.input_device = device

    def _get_stt(self) -> SpeechToText:
        if self._stt is None:
            self._stt = self._stt_factory()
        return self._stt

    def preload_stt(self, model_size: str | None = None) -> SpeechToText:
        """Явно (пере)загружает STT-движок — используется навыком установки
        компонентов, чтобы скачивание модели с Hugging Face происходило в
        явный, подтверждённый пользователем момент, а не незаметно при
        первой же записи."""
        if model_size:
            self._stt_factory = lambda: create_stt_engine(model_size=model_size)
        self._stt = None
        return self._get_stt()

    def transcribe_file(self, path: Path) -> str:
        return self._get_stt().transcribe_file(path)

    def _maybe_save_audio(self, utterance_id: int, pcm: bytes) -> str | None:
        if not pcm or not self._retain_audio() or self._audio_dir is None:
            return None
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        path = self._audio_dir / f"utterance_{utterance_id}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)
        return str(path)

    def _process_utterance(
        self,
        recording_id: int,
        text: str,
        pcm: bytes | None,
        on_utterance: UtteranceCallback | None,
    ) -> Utterance | None:
        text = text.strip()
        if not text:
            return None

        classification = self.classifier.classify(text)
        task_id = classification.task_id
        utterance = self.store.add_utterance(recording_id, text, task_id=task_id)

        if pcm:
            audio_path = self._maybe_save_audio(utterance.id, pcm)
            if audio_path:
                utterance.audio_path = audio_path
                self.store.set_utterance_audio_path(utterance.id, audio_path)

        if task_id is not None:
            profile = self.classifier.profiles[task_id]
            profile.apply_auto(tokenize(text))
            self.store.save_task_profile(profile)

        if on_utterance:
            on_utterance(utterance, classification)
        return utterance

    # ---- разовая запись -----------------------------------------------------------

    def record_once(
        self, max_seconds: float = 30.0, on_utterance: UtteranceCallback | None = None
    ) -> list[Utterance]:
        capture = self._audio_capture_factory()
        capture.level_callback = self._on_level  # тот же приём, что и в дефолтной фабрике выше
        stt = self._get_stt()
        recording = self.store.start_recording(mode="once")
        results: list[Utterance] = []
        try:
            for pcm in capture.record_utterances(max_seconds=max_seconds):
                text = stt.transcribe_pcm(pcm)
                utterance = self._process_utterance(recording.id, text, pcm, on_utterance)
                if utterance is not None:
                    results.append(utterance)
        finally:
            self.store.stop_recording(recording.id)
            self.current_level = 0.0
        return results

    # ---- постоянная (фоновая) запись -----------------------------------------------

    def start_continuous(self, on_utterance: UtteranceCallback | None = None) -> None:
        if self.is_continuous_active:
            return
        capture = self._audio_capture_factory()
        capture.level_callback = self._on_level
        stt = self._get_stt()
        recording = self.store.start_recording(mode="continuous")
        self._continuous_capture = capture
        self._continuous_recording_id = recording.id

        def _loop() -> None:
            for pcm in capture.record_utterances(max_seconds=None):
                text = stt.transcribe_pcm(pcm)
                self._process_utterance(recording.id, text, pcm, on_utterance)

        thread = threading.Thread(target=_loop, daemon=True, name="voice-continuous")
        self._continuous_thread = thread
        thread.start()

    def stop_continuous(self) -> None:
        if self._continuous_capture is not None:
            self._continuous_capture.stop()
        if self._continuous_thread is not None:
            self._continuous_thread.join(timeout=5)
        if self._continuous_recording_id is not None:
            self.store.stop_recording(self._continuous_recording_id)
        self._continuous_thread = None
        self._continuous_capture = None
        self._continuous_recording_id = None
        self.current_level = 0.0

    # ---- задачи и обратная связь ----------------------------------------------------

    def create_task(self, title: str, description: str = "", due_at: float | None = None) -> Task:
        task = self.store.create_task(title=title, description=description, due_at=due_at)
        profile = self.classifier.register_task(task.id, seed_text=f"{title} {description}")
        self.store.save_task_profile(profile)
        return task

    def confirm_task(self, utterance_id: int, task_id: int) -> Utterance:
        utterance = self.store.get_utterance(utterance_id)
        if utterance is None:
            raise ValueError(f"неизвестная реплика: {utterance_id}")
        self.classifier.confirm(task_id, utterance.text)
        self.store.save_task_profile(self.classifier.profiles[task_id])
        self.store.set_utterance_task(utterance_id, task_id)
        utterance.task_id = task_id
        return utterance

    def reject_task(self, utterance_id: int, task_id: int) -> Utterance:
        utterance = self.store.get_utterance(utterance_id)
        if utterance is None:
            raise ValueError(f"неизвестная реплика: {utterance_id}")
        self.classifier.reject(task_id, utterance.text)
        self.store.save_task_profile(self.classifier.profiles[task_id])
        if utterance.task_id == task_id:
            self.store.set_utterance_task(utterance_id, None)
            utterance.task_id = None
        return utterance

    # ---- протокол -------------------------------------------------------------------

    def generate_protocol(self, recording_id: int | None = None) -> Protocol:
        tasks = self.store.list_tasks()
        utterances = self.store.list_utterances(recording_id=recording_id)
        return build_protocol(tasks, utterances)
