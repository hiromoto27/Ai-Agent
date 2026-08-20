"""Долговременная память агента ("самообучение" на опыте).

Каждая выполненная задача записывается как эпизод (задача, план, итог,
успех/неуспех, заметка-рефлексия). Перед планированием новой задачи
агент запрашивает похожие прошлые эпизоды (простой bag-of-words поиск
по косинусному сходству — без тяжёлых ML-зависимостей вроде torch,
что важно для лёгкого локального .exe) и получает их как контекст.

Это и есть механизм "обучения на опыте": агент не дообучает веса модели,
а накапливает и переиспользует опыт через retrieval.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")


def _tokenize(text: str) -> Counter:
    return Counter(t.lower() for t in _TOKEN_RE.findall(text))


def _cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class Episode:
    id: int | None
    created_at: float
    task: str
    plan_summary: str
    outcome: str
    success: bool | None
    reflection: str = ""
    tags: list[str] = field(default_factory=list)

    def to_context_snippet(self) -> str:
        status = "успех" if self.success else ("неудача" if self.success is False else "неизвестно")
        parts = [f"Задача: {self.task}", f"Итог ({status}): {self.outcome}"]
        if self.reflection:
            parts.append(f"Вывод на будущее: {self.reflection}")
        return "\n".join(parts)


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: агент выполняет задачи в фоновом потоке
        # (см. ui/worker.py), а MemoryStore создаётся в потоке GUI.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                task TEXT NOT NULL,
                plan_summary TEXT NOT NULL DEFAULT '',
                outcome TEXT NOT NULL DEFAULT '',
                success INTEGER,
                reflection TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add_episode(
        self,
        task: str,
        plan_summary: str = "",
        outcome: str = "",
        success: bool | None = None,
        reflection: str = "",
        tags: list[str] | None = None,
    ) -> Episode:
        created_at = time.time()
        tags = tags or []
        cur = self._conn.execute(
            """
            INSERT INTO episodes (created_at, task, plan_summary, outcome, success, reflection, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                task,
                plan_summary,
                outcome,
                None if success is None else int(success),
                reflection,
                json.dumps(tags, ensure_ascii=False),
            ),
        )
        self._conn.commit()
        return Episode(
            id=cur.lastrowid,
            created_at=created_at,
            task=task,
            plan_summary=plan_summary,
            outcome=outcome,
            success=success,
            reflection=reflection,
            tags=tags,
        )

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        success = None if row["success"] is None else bool(row["success"])
        return Episode(
            id=row["id"],
            created_at=row["created_at"],
            task=row["task"],
            plan_summary=row["plan_summary"],
            outcome=row["outcome"],
            success=success,
            reflection=row["reflection"],
            tags=json.loads(row["tags"]),
        )

    def recent(self, limit: int = 10) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def retrieve_similar(self, task: str, k: int = 4) -> list[Episode]:
        if k <= 0:
            return []
        query_vec = _tokenize(task)
        rows = self._conn.execute("SELECT * FROM episodes").fetchall()
        scored: list[tuple[float, Episode]] = []
        for row in rows:
            episode = self._row_to_episode(row)
            score = _cosine_similarity(query_vec, _tokenize(episode.task))
            if score > 0:
                scored.append((score, episode))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [episode for _, episode in scored[:k]]

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successes, "
            "SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) AS failures FROM episodes"
        ).fetchone()
        total = row["total"] or 0
        successes = row["successes"] or 0
        failures = row["failures"] or 0
        return {
            "total_episodes": total,
            "successes": successes,
            "failures": failures,
            "success_rate": (successes / total) if total else None,
        }
