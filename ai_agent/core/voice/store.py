"""Хранилище диктофона: записи, реплики, задачи, профили классификатора,
напоминания — в том же файле SQLite, что и долговременная память агента
(``~/.ai-agent/memory.sqlite3``), отдельным соединением и отдельными
таблицами (``voice_*``), без пересечения с ``MemoryStore``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .classifier import TaskProfile


@dataclass
class Recording:
    id: int
    started_at: float
    ended_at: float | None
    mode: str  # "once" | "continuous"


@dataclass
class Utterance:
    id: int
    recording_id: int
    ts: float
    text: str
    task_id: int | None
    audio_path: str | None = None


@dataclass
class Task:
    id: int
    title: str
    description: str
    due_at: float | None
    created_at: float
    active: bool = True


@dataclass
class Reminder:
    id: int
    task_id: int
    fire_at: float
    message: str
    fired: bool = False


class VoiceStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: постоянная запись работает в фоновом потоке
        # (см. voice/service.py), а хранилище создаётся в основном потоке.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS voice_recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL NOT NULL,
                ended_at REAL,
                mode TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS voice_utterances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER NOT NULL,
                ts REAL NOT NULL,
                text TEXT NOT NULL,
                task_id INTEGER,
                audio_path TEXT
            );

            CREATE TABLE IF NOT EXISTS voice_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                due_at REAL,
                created_at REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS voice_task_profiles (
                task_id INTEGER PRIMARY KEY,
                positive_words TEXT NOT NULL DEFAULT '{}',
                negative_words TEXT NOT NULL DEFAULT '{}',
                confirmations INTEGER NOT NULL DEFAULT 0,
                rejections INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS voice_reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                fire_at REAL NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                fired INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- записи -------------------------------------------------------------

    def start_recording(self, mode: str) -> Recording:
        assert mode in ("once", "continuous")
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO voice_recordings (started_at, mode) VALUES (?, ?)", (now, mode)
        )
        self._conn.commit()
        return Recording(id=cur.lastrowid, started_at=now, ended_at=None, mode=mode)

    def stop_recording(self, recording_id: int) -> None:
        self._conn.execute(
            "UPDATE voice_recordings SET ended_at = ? WHERE id = ?", (time.time(), recording_id)
        )
        self._conn.commit()

    # ---- реплики --------------------------------------------------------------

    def add_utterance(
        self,
        recording_id: int,
        text: str,
        task_id: int | None = None,
        audio_path: str | None = None,
    ) -> Utterance:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO voice_utterances (recording_id, ts, text, task_id, audio_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (recording_id, now, text, task_id, audio_path),
        )
        self._conn.commit()
        return Utterance(
            id=cur.lastrowid,
            recording_id=recording_id,
            ts=now,
            text=text,
            task_id=task_id,
            audio_path=audio_path,
        )

    def get_utterance(self, utterance_id: int) -> Utterance | None:
        row = self._conn.execute(
            "SELECT * FROM voice_utterances WHERE id = ?", (utterance_id,)
        ).fetchone()
        return self._row_to_utterance(row) if row else None

    def set_utterance_task(self, utterance_id: int, task_id: int | None) -> None:
        self._conn.execute(
            "UPDATE voice_utterances SET task_id = ? WHERE id = ?", (task_id, utterance_id)
        )
        self._conn.commit()

    def set_utterance_audio_path(self, utterance_id: int, audio_path: str) -> None:
        self._conn.execute(
            "UPDATE voice_utterances SET audio_path = ? WHERE id = ?", (audio_path, utterance_id)
        )
        self._conn.commit()

    def list_utterances(
        self, recording_id: int | None = None, task_id: int | None = None
    ) -> list[Utterance]:
        query = "SELECT * FROM voice_utterances"
        clauses = []
        params: list = []
        if recording_id is not None:
            clauses.append("recording_id = ?")
            params.append(recording_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY ts ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_utterance(row) for row in rows]

    @staticmethod
    def _row_to_utterance(row: sqlite3.Row) -> Utterance:
        return Utterance(
            id=row["id"],
            recording_id=row["recording_id"],
            ts=row["ts"],
            text=row["text"],
            task_id=row["task_id"],
            audio_path=row["audio_path"],
        )

    # ---- задачи -----------------------------------------------------------------

    def create_task(self, title: str, description: str = "", due_at: float | None = None) -> Task:
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO voice_tasks (title, description, due_at, created_at, active) "
            "VALUES (?, ?, ?, ?, 1)",
            (title, description, due_at, now),
        )
        self._conn.commit()
        return Task(id=cur.lastrowid, title=title, description=description, due_at=due_at, created_at=now)

    def get_task(self, task_id: int) -> Task | None:
        row = self._conn.execute("SELECT * FROM voice_tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def list_tasks(self, active_only: bool = True) -> list[Task]:
        query = "SELECT * FROM voice_tasks"
        if active_only:
            query += " WHERE active = 1"
        query += " ORDER BY created_at ASC"
        rows = self._conn.execute(query).fetchall()
        return [self._row_to_task(row) for row in rows]

    def set_task_active(self, task_id: int, active: bool) -> None:
        self._conn.execute(
            "UPDATE voice_tasks SET active = ? WHERE id = ?", (1 if active else 0, task_id)
        )
        self._conn.commit()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            due_at=row["due_at"],
            created_at=row["created_at"],
            active=bool(row["active"]),
        )

    # ---- профили классификатора ----------------------------------------------------

    def save_task_profile(self, profile: TaskProfile) -> None:
        self._conn.execute(
            """
            INSERT INTO voice_task_profiles (task_id, positive_words, negative_words, confirmations, rejections)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                positive_words = excluded.positive_words,
                negative_words = excluded.negative_words,
                confirmations = excluded.confirmations,
                rejections = excluded.rejections
            """,
            (
                profile.task_id,
                json.dumps(dict(profile.positive_words)),
                json.dumps(dict(profile.negative_words)),
                profile.confirmations,
                profile.rejections,
            ),
        )
        self._conn.commit()

    def load_task_profile(self, task_id: int) -> TaskProfile | None:
        row = self._conn.execute(
            "SELECT * FROM voice_task_profiles WHERE task_id = ?", (task_id,)
        ).fetchone()
        return self._row_to_profile(row) if row else None

    def load_all_task_profiles(self) -> dict[int, TaskProfile]:
        rows = self._conn.execute("SELECT * FROM voice_task_profiles").fetchall()
        return {row["task_id"]: self._row_to_profile(row) for row in rows}

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> TaskProfile:
        return TaskProfile.from_dict(
            {
                "task_id": row["task_id"],
                "positive_words": json.loads(row["positive_words"]),
                "negative_words": json.loads(row["negative_words"]),
                "confirmations": row["confirmations"],
                "rejections": row["rejections"],
            }
        )

    # ---- напоминания --------------------------------------------------------------

    def add_reminder(self, task_id: int, fire_at: float, message: str = "") -> Reminder:
        cur = self._conn.execute(
            "INSERT INTO voice_reminders (task_id, fire_at, message, fired) VALUES (?, ?, ?, 0)",
            (task_id, fire_at, message),
        )
        self._conn.commit()
        return Reminder(id=cur.lastrowid, task_id=task_id, fire_at=fire_at, message=message)

    def due_reminders(self, now: float | None = None) -> list[Reminder]:
        now = now if now is not None else time.time()
        rows = self._conn.execute(
            "SELECT * FROM voice_reminders WHERE fired = 0 AND fire_at <= ? ORDER BY fire_at ASC",
            (now,),
        ).fetchall()
        return [self._row_to_reminder(row) for row in rows]

    def mark_reminder_fired(self, reminder_id: int) -> None:
        self._conn.execute(
            "UPDATE voice_reminders SET fired = 1 WHERE id = ?", (reminder_id,)
        )
        self._conn.commit()

    def list_reminders(self, task_id: int | None = None) -> list[Reminder]:
        if task_id is None:
            rows = self._conn.execute(
                "SELECT * FROM voice_reminders ORDER BY fire_at ASC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM voice_reminders WHERE task_id = ? ORDER BY fire_at ASC", (task_id,)
            ).fetchall()
        return [self._row_to_reminder(row) for row in rows]

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"],
            task_id=row["task_id"],
            fire_at=row["fire_at"],
            message=row["message"],
            fired=bool(row["fired"]),
        )
