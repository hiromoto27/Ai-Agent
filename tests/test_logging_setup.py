import logging
from pathlib import Path

from ai_agent.core.logging_setup import LOG_FILENAME, get_logger, setup_logging


def test_setup_logging_creates_file(tmp_path: Path):
    log_path = setup_logging(tmp_path)
    assert log_path == tmp_path / LOG_FILENAME

    logger = get_logger("test")
    logger.info("привет из теста")
    for handler in logging.getLogger("ai_agent").handlers:
        handler.flush()

    assert log_path.exists()
    assert "привет из теста" in log_path.read_text(encoding="utf-8")


def test_setup_logging_idempotent_no_duplicate_handlers(tmp_path: Path):
    setup_logging(tmp_path)
    handlers_after_first = len(logging.getLogger("ai_agent").handlers)
    setup_logging(tmp_path)
    handlers_after_second = len(logging.getLogger("ai_agent").handlers)

    assert handlers_after_first == handlers_after_second


def test_setup_logging_captures_exception_traceback(tmp_path: Path):
    log_path = setup_logging(tmp_path)
    logger = get_logger("test")
    try:
        raise ValueError("что-то пошло не так")
    except ValueError:
        logger.exception("сбой при тестировании")
    for handler in logging.getLogger("ai_agent").handlers:
        handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "сбой при тестировании" in content
    assert "ValueError" in content
    assert "Traceback" in content
