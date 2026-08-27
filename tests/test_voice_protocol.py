from ai_agent.core.voice.protocol import build_protocol, is_action_item, protocol_to_text
from ai_agent.core.voice.store import Task, Utterance


def make_task(task_id: int, title: str) -> Task:
    return Task(id=task_id, title=title, description="", due_at=None, created_at=0.0)


def make_utterance(uid: int, text: str, task_id: int | None) -> Utterance:
    return Utterance(id=uid, recording_id=1, ts=float(uid), text=text, task_id=task_id)


def test_is_action_item_detects_markers():
    assert is_action_item("Нужно подготовить отчёт к пятнице")
    assert is_action_item("Договорились согласовать бюджет")
    assert not is_action_item("Погода сегодня хорошая")


def test_build_protocol_groups_by_task_and_extracts_actions():
    tasks = [make_task(1, "Бюджет"), make_task(2, "Маркетинг")]
    utterances = [
        make_utterance(1, "Нужно согласовать бюджет до пятницы", task_id=1),
        make_utterance(2, "Обсудили детали бюджета", task_id=1),
        make_utterance(3, "Подготовить рекламную кампанию", task_id=2),
        make_utterance(4, "Просто болтали ни о чём", task_id=None),
    ]

    protocol = build_protocol(tasks, utterances)

    assert len(protocol.sections) == 2
    budget_section = next(s for s in protocol.sections if s.task_id == 1)
    assert budget_section.items == [
        "Нужно согласовать бюджет до пятницы",
        "Обсудили детали бюджета",
    ]
    assert budget_section.action_items == ["Нужно согласовать бюджет до пятницы"]

    assert protocol.unassigned == ["Просто болтали ни о чём"]


def test_build_protocol_skips_tasks_without_utterances():
    tasks = [make_task(1, "Пустая задача")]
    protocol = build_protocol(tasks, [])
    assert protocol.sections == []
    assert protocol.unassigned == []


def test_protocol_to_text_contains_task_titles_and_items():
    tasks = [make_task(1, "Бюджет")]
    utterances = [make_utterance(1, "Нужно согласовать бюджет", task_id=1)]
    protocol = build_protocol(tasks, utterances)
    text = protocol_to_text(protocol)
    assert "Бюджет" in text
    assert "Нужно согласовать бюджет" in text
    assert "Пункты-действия" in text


def test_protocol_to_text_empty():
    protocol = build_protocol([], [])
    assert "реплик нет" in protocol_to_text(protocol)
