"""Точка входа десктоп-приложения (GUI)."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from . import theme
from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    # На уровне приложения — чтобы тёмная тема применялась и к диалогам
    # (например QMessageBox подтверждения прав), не только к MainWindow.
    app.setStyleSheet(theme.STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
