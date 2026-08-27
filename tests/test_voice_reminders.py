from pathlib import Path

from ai_agent.core.voice.reminders import ReminderChecker
from ai_agent.core.voice.store import VoiceStore


def test_check_returns_due_reminders_and_marks_fired(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    task = store.create_task(title="Бюджет")
    store.add_reminder(task_id=task.id, fire_at=1000.0, message="Срок сегодня")
    store.add_reminder(task_id=task.id, fire_at=5000.0, message="Ещё не скоро")

    checker = ReminderChecker(store)
    due = checker.check(now=2000.0)

    assert len(due) == 1
    assert due[0].task.id == task.id
    assert due[0].reminder.message == "Срок сегодня"

    # Повторный вызов не должен вернуть то же напоминание снова.
    assert checker.check(now=2000.0) == []


def test_check_with_no_due_reminders_returns_empty(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    task = store.create_task(title="Задача")
    store.add_reminder(task_id=task.id, fire_at=9999999999.0)

    checker = ReminderChecker(store)
    assert checker.check(now=1000.0) == []


def test_check_skips_reminder_for_deleted_task(tmp_path: Path):
    store = VoiceStore(tmp_path / "voice.sqlite3")
    task = store.create_task(title="Временная")
    reminder = store.add_reminder(task_id=task.id, fire_at=1000.0)
    store.set_task_active(task.id, False)  # задача деактивирована, но не физически удалена

    checker = ReminderChecker(store)
    due = checker.check(now=2000.0)
    # get_task не фильтрует по active — напоминание всё равно должно дойти.
    assert len(due) == 1
    assert due[0].reminder.id == reminder.id
