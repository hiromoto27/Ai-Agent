from pathlib import Path

from ai_agent.core.voice.classifier import TaskProfile, tokenize
from ai_agent.core.voice.store import VoiceStore


def test_create_and_get_task(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    task = store.create_task(title="Согласовать бюджет", description="Финансовый план на квартал")
    fetched = store.get_task(task.id)
    assert fetched is not None
    assert fetched.title == "Согласовать бюджет"
    assert fetched.active is True


def test_list_tasks_active_only(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    a = store.create_task(title="A")
    store.create_task(title="B")
    store.set_task_active(a.id, False)
    active = store.list_tasks(active_only=True)
    assert [t.title for t in active] == ["B"]
    all_tasks = store.list_tasks(active_only=False)
    assert len(all_tasks) == 2


def test_recording_and_utterance_lifecycle(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    recording = store.start_recording(mode="once")
    assert recording.ended_at is None

    task = store.create_task(title="Бюджет")
    utterance = store.add_utterance(recording.id, "Обсудили бюджет", task_id=task.id)
    assert utterance.id is not None

    store.stop_recording(recording.id)

    fetched = store.get_utterance(utterance.id)
    assert fetched is not None
    assert fetched.text == "Обсудили бюджет"
    assert fetched.task_id == task.id

    listed = store.list_utterances(recording_id=recording.id)
    assert len(listed) == 1
    by_task = store.list_utterances(task_id=task.id)
    assert len(by_task) == 1


def test_set_utterance_task_and_audio_path(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    recording = store.start_recording(mode="once")
    utterance = store.add_utterance(recording.id, "Реплика без задачи")
    assert utterance.task_id is None

    task = store.create_task(title="Задача")
    store.set_utterance_task(utterance.id, task.id)
    store.set_utterance_audio_path(utterance.id, "recordings/utterance_1.wav")

    fetched = store.get_utterance(utterance.id)
    assert fetched.task_id == task.id
    assert fetched.audio_path == "recordings/utterance_1.wav"


def test_task_profile_persistence_round_trip(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    task = store.create_task(title="Бюджет")
    profile = TaskProfile(task_id=task.id)
    profile.apply_feedback(tokenize("бюджет квартал отчёт"), positive=True)
    profile.apply_feedback(tokenize("маркетинг реклама"), positive=False)
    store.save_task_profile(profile)

    loaded = store.load_task_profile(task.id)
    assert loaded is not None
    assert loaded.positive_words == profile.positive_words
    assert loaded.negative_words == profile.negative_words
    assert loaded.confirmations == 1
    assert loaded.rejections == 1

    all_profiles = store.load_all_task_profiles()
    assert task.id in all_profiles


def test_save_task_profile_upserts(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    task = store.create_task(title="Бюджет")
    profile = TaskProfile(task_id=task.id)
    store.save_task_profile(profile)

    profile.apply_feedback(tokenize("новое слово"), positive=True)
    store.save_task_profile(profile)

    loaded = store.load_task_profile(task.id)
    assert loaded.confirmations == 1


def test_reminders_due_and_fired(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    task = store.create_task(title="Бюджет")
    reminder = store.add_reminder(task_id=task.id, fire_at=1000.0, message="Срок сегодня")

    assert store.due_reminders(now=999.0) == []
    due = store.due_reminders(now=1001.0)
    assert len(due) == 1
    assert due[0].id == reminder.id

    store.mark_reminder_fired(reminder.id)
    assert store.due_reminders(now=1001.0) == []

    all_reminders = store.list_reminders(task_id=task.id)
    assert len(all_reminders) == 1
    assert all_reminders[0].fired is True


def test_persistence_across_reopen(tmp_path: Path):
    db_path = tmp_path / "voice.sqlite3"
    store1 = VoiceStore(db_path)
    task = store1.create_task(title="Персистентная задача")
    store1.close()

    store2 = VoiceStore(db_path)
    fetched = store2.get_task(task.id)
    assert fetched is not None
    assert fetched.title == "Персистентная задача"
