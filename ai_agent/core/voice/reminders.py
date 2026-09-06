"""Проверка напоминаний по задачам — вызывается периодически (таймер в GUI
или цикл в CLI). Доставка работает только пока приложение запущено — см.
ограничение в VOICE_RECORDER_PLAN.md, раздел 7.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .store import Reminder, Task, VoiceStore


@dataclass
class DueReminder:
    reminder: Reminder
    task: Task


class ReminderChecker:
    def __init__(self, store: VoiceStore) -> None:
        self.store = store

    def check(self, now: float | None = None) -> list[DueReminder]:
        """Возвращает наступившие, ещё не сработавшие напоминания и сразу
        помечает их сработавшими (при вызове раз в интервал таймера
        каждое напоминание должно быть доставлено ровно один раз)."""
        now = now if now is not None else time.time()
        due: list[DueReminder] = []
        for reminder in self.store.due_reminders(now):
            task = self.store.get_task(reminder.task_id)
            self.store.mark_reminder_fired(reminder.id)
            if task is not None:
                due.append(DueReminder(reminder=reminder, task=task))
        return due
