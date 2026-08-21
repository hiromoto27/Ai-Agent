"""Файловое логирование агента: все сбои навыков и обращений к LLM
пишутся в ``state_dir/agent.log`` с полным traceback'ом.

Раньше единственным источником диагностики был текст ошибки в
интерфейсе (одна строка) — этого не хватало, чтобы понять, что реально
произошло на машине пользователя (сетевая ошибка? неверный формат
ответа модели? таймаут?). Лог-файл — то, что можно открыть или
приложить при сообщении о проблеме.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "ai_agent"
LOG_FILENAME = "agent.log"


def get_logger(component: str = "") -> logging.Logger:
    name = f"{LOGGER_NAME}.{component}" if component else LOGGER_NAME
    return logging.getLogger(name)


def setup_logging(state_dir: Path, level: int = logging.DEBUG) -> Path:
    """Настраивает запись логов в ``state_dir/agent.log``. Идемпотентна —
    повторный вызов (например при пересборке агента в GUI) не плодит
    дублирующиеся обработчики и не пишет одно и то же сообщение дважды."""
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / LOG_FILENAME

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    already_configured = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in logger.handlers
    )
    if not already_configured:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    return log_path
