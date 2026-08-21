"""Фоновый запуск одного навыка напрямую (в обход LLM-цикла агента).

Используется для UI-действий вроде "подобрать модели"/"скачать модель" —
это прямые команды пользователя, а не задачи на естественном языке, так
что незачем гонять их через агентный цикл. PolicyEngine (включая диалоги
подтверждения через ConfirmBridge) работает точно так же, как для обычных
задач, потому что используется тот же ``SkillContext``.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ai_agent.core.skills.base import SkillResult


class SkillWorker(QThread):
    finished_skill = Signal(object)
    failed = Signal(str)

    def __init__(self, registry, context, skill_name: str, kwargs: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.registry = registry
        self.context = context
        self.skill_name = skill_name
        self.kwargs = kwargs or {}

    def run(self) -> None:
        try:
            result: SkillResult = self.registry.invoke(self.skill_name, self.context, **self.kwargs)
        except Exception as e:  # защитный барьер — поток не должен падать молча
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished_skill.emit(result)
