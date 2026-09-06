"""Самообучаемая классификация реплик по задачам через контекстные слова.

Как и долговременная память агента (``ai_agent/core/memory/store.py``),
это не дообучение весов нейросети, а онлайн-обучаемый профиль ключевых
слов на задачу (bag-of-words + косинусное сходство), который уточняется
каждым подтверждением/отклонением пользователя. Специально без тяжёлых
ML-зависимостей (torch и т.п.) — см. VOICE_RECORDER_PLAN.md, раздел 4.3.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+")

# Небольшой стоп-лист самых частых служебных слов рус./англ. — без него
# любой текст выглядел бы "похожим" на любую задачу просто за счёт общих
# предлогов и союзов.
_STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "сказал", "этот", "эта",
    "это", "эти", "мы", "тебя", "их", "чем", "была", "сам", "чтобы", "без",
    "будто", "человек", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этого", "того", "потому", "этой", "какой",
    "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем",
    "чтоб", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
    "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них",
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
    "in", "on", "for", "it", "this", "that", "with", "as", "at", "by",
}


def tokenize(text: str) -> Counter:
    """Токенизирует текст в мешок слов, отбрасывая стоп-слова и однобуквенные токены."""
    words = (w.lower() for w in _TOKEN_RE.findall(text))
    return Counter(w for w in words if len(w) > 2 and w not in _STOP_WORDS)


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
class TaskProfile:
    """Взвешенный профиль ключевых слов одной задачи."""

    task_id: int
    positive_words: Counter = field(default_factory=Counter)
    negative_words: Counter = field(default_factory=Counter)
    confirmations: int = 0
    rejections: int = 0

    def score(self, tokens: Counter) -> float:
        pos = _cosine_similarity(tokens, self.positive_words)
        neg = _cosine_similarity(tokens, self.negative_words)
        return pos - 0.5 * neg

    def seed(self, text: str) -> None:
        """Первичное наполнение профиля из названия/описания задачи —
        не считается подтверждением пользователя, не влияет на ``confidence``."""
        self.positive_words.update(tokenize(text))

    def apply_auto(self, tokens: Counter) -> None:
        """Тихое усиление при уверенной автоматической классификации
        (без вопроса пользователю) — счётчики подтверждений не растут."""
        self.positive_words.update(tokens)

    def apply_feedback(self, tokens: Counter, positive: bool) -> None:
        """Явный ответ пользователя на уточняющий вопрос — обучающий сигнал."""
        if positive:
            self.positive_words.update(tokens)
            self.confirmations += 1
        else:
            self.negative_words.update(tokens)
            self.rejections += 1

    @property
    def confidence(self) -> float:
        """Доля подтверждений среди явной обратной связи — "уровень" задачи:
        чем выше, тем реже классификатор переспрашивает про эту задачу."""
        total = self.confirmations + self.rejections
        if total == 0:
            return 0.0
        return self.confirmations / total

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "positive_words": dict(self.positive_words),
            "negative_words": dict(self.negative_words),
            "confirmations": self.confirmations,
            "rejections": self.rejections,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskProfile":
        return cls(
            task_id=data["task_id"],
            positive_words=Counter(data.get("positive_words", {})),
            negative_words=Counter(data.get("negative_words", {})),
            confirmations=data.get("confirmations", 0),
            rejections=data.get("rejections", 0),
        )


@dataclass
class Classification:
    task_id: int | None
    auto: bool
    needs_confirmation: bool
    candidates: list[tuple[int, float]] = field(default_factory=list)


class TaskClassifier:
    """Ранжирует реплику по задачам и решает, назначать ли её автоматически
    или переспросить пользователя.

    ``gap_threshold`` — насколько лучший кандидат должен опережать второго,
    чтобы решение считалось однозначным. Этот порог сам смягчается с ростом
    ``confidence`` задачи (больше подтверждений → меньше вопросов) — это и
    есть измеримое "самообучение" (см. VOICE_RECORDER_PLAN.md, п. 4.3.6).
    """

    def __init__(self, gap_threshold: float = 0.2, min_score: float = 0.05) -> None:
        self.gap_threshold = gap_threshold
        self.min_score = min_score
        self.profiles: dict[int, TaskProfile] = {}

    def register_task(self, task_id: int, seed_text: str = "") -> TaskProfile:
        profile = self.profiles.setdefault(task_id, TaskProfile(task_id=task_id))
        if seed_text:
            profile.seed(seed_text)
        return profile

    def remove_task(self, task_id: int) -> None:
        self.profiles.pop(task_id, None)

    def classify(self, text: str) -> Classification:
        if not self.profiles:
            return Classification(task_id=None, auto=False, needs_confirmation=False)

        tokens = tokenize(text)
        if not tokens:
            return Classification(task_id=None, auto=False, needs_confirmation=False)

        scored = sorted(
            ((task_id, profile.score(tokens)) for task_id, profile in self.profiles.items()),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best_id, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        gap = best_score - second_score
        required_gap = self.gap_threshold * (1.0 - self.profiles[best_id].confidence)

        auto = best_score >= self.min_score and gap >= required_gap
        return Classification(
            task_id=best_id if auto else None,
            auto=auto,
            needs_confirmation=not auto,
            candidates=scored,
        )

    def confirm(self, task_id: int, text: str) -> None:
        profile = self.register_task(task_id)
        profile.apply_feedback(tokenize(text), positive=True)

    def reject(self, task_id: int, text: str) -> None:
        profile = self.register_task(task_id)
        profile.apply_feedback(tokenize(text), positive=False)
