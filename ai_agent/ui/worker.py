"""Фоновое выполнение задачи агента, чтобы не блокировать интерфейс."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class AgentWorker(QThread):
    finished_task = Signal(object)
    failed = Signal(str)

    def __init__(self, agent, task: str, parent=None) -> None:
        super().__init__(parent)
        self.agent = agent
        self.task = task

    def run(self) -> None:
        try:
            result = self.agent.run_task(self.task)
        except Exception as e:  # защитный барьер — поток не должен падать молча
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished_task.emit(result)
