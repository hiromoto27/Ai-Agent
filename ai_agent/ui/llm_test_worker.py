"""Фоновая проверка связи с текущим LLM-провайдером (см. app.test_llm_connection)."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ai_agent.app import test_llm_connection
from ai_agent.core.llm.base import LLMProvider


class LLMConnectionTestWorker(QThread):
    finished_test = Signal(bool, str)

    def __init__(self, llm: LLMProvider, parent=None) -> None:
        super().__init__(parent)
        self.llm = llm

    def run(self) -> None:
        ok, message = test_llm_connection(self.llm)
        self.finished_test.emit(ok, message)
