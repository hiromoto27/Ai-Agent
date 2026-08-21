"""Фоновый запрос списка моделей, которые LM Studio отдаёт прямо сейчас."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ai_agent.core.llm.lmstudio_provider import list_models


class LMStudioModelsWorker(QThread):
    finished_models = Signal(list)
    failed = Signal(str)

    def __init__(self, base_url: str, api_key: str = "", parent=None) -> None:
        super().__init__(parent)
        self.base_url = base_url
        self.api_key = api_key

    def run(self) -> None:
        try:
            models = list_models(self.base_url, api_key=self.api_key)
        except Exception as e:  # защитный барьер — поток не должен падать молча
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.finished_models.emit(models)
