import threading
import time
from pathlib import Path

from ai_agent.core.voice.service import VoiceService
from ai_agent.core.voice.store import VoiceStore


class FakeCapture:
    """Отдаёт фиксированный список "реплик" (PCM-байтов) и завершается —
    имитирует разовую запись без реального микрофона."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def record_utterances(self, max_seconds=None):
        yield from self._chunks

    def stop(self) -> None:
        pass


class FakeContinuousCapture:
    """Как FakeCapture, но после отдачи чанков "висит", пока не вызовут
    stop() — имитирует постоянную (фоновую) запись."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self._stop_event = threading.Event()

    def record_utterances(self, max_seconds=None):
        yield from self._chunks
        while not self._stop_event.wait(timeout=0.01):
            pass

    def stop(self) -> None:
        self._stop_event.set()


class FakeSTT:
    def __init__(self, texts: list[str]):
        self._texts = list(texts)

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        return self._texts.pop(0) if self._texts else ""

    def transcribe_file(self, path) -> str:
        return "распознанный текст файла"


def test_record_once_transcribes_and_classifies(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    service = VoiceService(
        store=store,
        audio_capture_factory=lambda: FakeCapture([b"chunk1", b"chunk2"]),
        stt_factory=lambda: FakeSTT(["Обсудили бюджет квартала", "Просто болтали"]),
    )
    task = service.create_task(title="Бюджет", description="Финансовый план квартала")

    utterances = service.record_once(max_seconds=5)

    assert len(utterances) == 2
    assert utterances[0].task_id == task.id
    assert utterances[1].task_id is None


def test_record_once_skips_empty_transcriptions(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    service = VoiceService(
        store=store,
        audio_capture_factory=lambda: FakeCapture([b"silence"]),
        stt_factory=lambda: FakeSTT([""]),
    )
    utterances = service.record_once(max_seconds=5)
    assert utterances == []


def test_record_once_saves_audio_when_retain_enabled(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    audio_dir = tmp_path / "recordings"
    pcm_chunk = b"\x00\x01" * 800
    service = VoiceService(
        store=store,
        audio_capture_factory=lambda: FakeCapture([pcm_chunk]),
        stt_factory=lambda: FakeSTT(["Тестовая фраза"]),
        audio_dir=audio_dir,
        retain_audio=lambda: True,
    )
    utterances = service.record_once(max_seconds=5)
    assert len(utterances) == 1
    saved_path = Path(utterances[0].audio_path)
    assert saved_path.exists()
    assert saved_path.suffix == ".wav"


def test_record_once_does_not_save_audio_by_default(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    audio_dir = tmp_path / "recordings"
    service = VoiceService(
        store=store,
        audio_capture_factory=lambda: FakeCapture([b"chunk"]),
        stt_factory=lambda: FakeSTT(["Фраза"]),
        audio_dir=audio_dir,
    )
    utterances = service.record_once(max_seconds=5)
    assert utterances[0].audio_path is None
    assert not audio_dir.exists()


def test_confirm_and_reject_task_update_store_and_classifier(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    service = VoiceService(store=store)
    task_a = service.create_task(title="Проект Альфа")
    recording = store.start_recording(mode="once")
    utterance = store.add_utterance(recording.id, "Обсудили статус проекта")

    service.confirm_task(utterance.id, task_a.id)
    assert store.get_utterance(utterance.id).task_id == task_a.id
    assert service.classifier.profiles[task_a.id].confirmations == 1

    service.reject_task(utterance.id, task_a.id)
    assert store.get_utterance(utterance.id).task_id is None
    assert service.classifier.profiles[task_a.id].rejections == 1


def test_task_profiles_survive_service_restart(tmp_path: Path):
    db_path = tmp_path / "voice.sqlite3"
    store1 = VoiceStore(db_path)
    service1 = VoiceService(store=store1)
    task = service1.create_task(title="Бюджет", description="Финансовый план")
    recording = store1.start_recording(mode="once")
    utterance = store1.add_utterance(recording.id, "Реплика")
    service1.confirm_task(utterance.id, task.id)
    store1.close()

    store2 = VoiceStore(db_path)
    service2 = VoiceService(store=store2)
    assert task.id in service2.classifier.profiles
    assert service2.classifier.profiles[task.id].confirmations == 1


def test_generate_protocol_via_service(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    service = VoiceService(store=store)
    task = service.create_task(title="Бюджет")
    recording = store.start_recording(mode="once")
    store.add_utterance(recording.id, "Нужно согласовать бюджет", task_id=task.id)

    protocol = service.generate_protocol()
    assert len(protocol.sections) == 1
    assert protocol.sections[0].task_title == "Бюджет"


def test_start_stop_continuous_processes_utterances_in_background(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    service = VoiceService(
        store=store,
        audio_capture_factory=lambda: FakeContinuousCapture([b"chunk1", b"chunk2"]),
        stt_factory=lambda: FakeSTT(["Первая реплика", "Вторая реплика"]),
    )
    assert service.is_continuous_active is False

    events: list = []
    service.start_continuous(on_utterance=lambda utterance, classification: events.append(utterance))
    assert service.is_continuous_active is True

    deadline = time.time() + 5
    while len(events) < 2 and time.time() < deadline:
        time.sleep(0.02)

    service.stop_continuous()
    assert service.is_continuous_active is False
    assert len(events) == 2


def test_transcribe_file_delegates_to_stt(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    service = VoiceService(store=store, stt_factory=lambda: FakeSTT([]))
    text = service.transcribe_file(tmp_path / "sample.wav")
    assert text == "распознанный текст файла"
