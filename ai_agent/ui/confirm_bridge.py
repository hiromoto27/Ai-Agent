"""Мост подтверждения прав между фоновым потоком агента и потоком GUI.

Agent.run_task выполняется в QThread (см. worker.py). Когда навыку нужно
подтверждение пользователя (PolicyEngine.confirm_callback), диалог
обязан открываться в главном потоке Qt. ConfirmBridge посылает Qt-сигнал
в главный поток и блокирует ТОЛЬКО поток агента (через threading.Event)
до тех пор, пока пользователь не ответит в диалоге.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox


class _ResultBox:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.result = False


class ConfirmBridge(QObject):
    request = Signal(str, dict, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.request.connect(self._handle)

    def confirm_callback(self, action: str, context: dict) -> bool:
        """Вызывается из потока агента. Блокирует поток агента, не GUI."""
        box = _ResultBox()
        self.request.emit(action, context, box)
        box.event.wait()
        return box.result

    @Slot(str, dict, object)
    def _handle(self, action: str, context: dict, box: _ResultBox) -> None:
        details = "\n".join(f"{k}: {v}" for k, v in context.items())
        answer = QMessageBox.question(
            None,
            "Требуется разрешение",
            f"Агент запрашивает действие:\n{action}\n\n{details}\n\nРазрешить?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        box.result = answer == QMessageBox.Yes
        box.event.set()
