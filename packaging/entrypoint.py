"""Точка входа для PyInstaller.

PyInstaller запускает переданный скрипт как модуль верхнего уровня
(``__main__``), поэтому файл внутри пакета ``ai_agent.ui`` с
относительным импортом (``from .main_window import ...``) не подходит
напрямую — нужен отдельный тонкий загрузчик снаружи пакета с абсолютным
импортом.
"""

from ai_agent.ui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
