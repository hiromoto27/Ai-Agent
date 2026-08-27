"""Сборка протокола разговора по задачам из распознанных реплик.

Пункты-действия выделяются эвристикой по ключевым словам (без LLM) —
чтобы протокол собирался и в тестовом режиме, без настроенного
провайдера. Более связный пересказ через LLM — возможное расширение
поверх этого модуля, не его замена (см. VOICE_RECORDER_PLAN.md, п. 4.4).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from .store import Task, Utterance

_ACTION_MARKERS = (
    "нужно",
    "надо",
    "необходимо",
    "сделать",
    "договорились",
    "уточнить",
    "подготовить",
    "согласовать",
    "проверить",
    "отправить",
    "напомнить",
    "к сроку",
    "до конца",
    "дедлайн",
)


def is_action_item(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _ACTION_MARKERS)


@dataclass
class ProtocolSection:
    task_id: int
    task_title: str
    items: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)


@dataclass
class Protocol:
    generated_at: float
    sections: list[ProtocolSection] = field(default_factory=list)
    unassigned: list[str] = field(default_factory=list)


def build_protocol(tasks: list[Task], utterances: list[Utterance]) -> Protocol:
    by_task: dict[int, list[str]] = defaultdict(list)
    unassigned: list[str] = []
    for utterance in utterances:
        if utterance.task_id is not None:
            by_task[utterance.task_id].append(utterance.text)
        else:
            unassigned.append(utterance.text)

    sections = []
    for task in tasks:
        items = by_task.get(task.id)
        if not items:
            continue
        sections.append(
            ProtocolSection(
                task_id=task.id,
                task_title=task.title,
                items=items,
                action_items=[text for text in items if is_action_item(text)],
            )
        )
    return Protocol(generated_at=time.time(), sections=sections, unassigned=unassigned)


def protocol_to_text(protocol: Protocol) -> str:
    lines = ["Протокол разговора", ""]
    if not protocol.sections and not protocol.unassigned:
        lines.append("(реплик нет)")
        return "\n".join(lines)

    for section in protocol.sections:
        lines.append(f"## {section.task_title}")
        for item in section.items:
            lines.append(f"- {item}")
        if section.action_items:
            lines.append("Пункты-действия:")
            for action in section.action_items:
                lines.append(f"  * {action}")
        lines.append("")

    if protocol.unassigned:
        lines.append("## Не отнесено к задачам")
        for item in protocol.unassigned:
            lines.append(f"- {item}")

    return "\n".join(lines).strip()
