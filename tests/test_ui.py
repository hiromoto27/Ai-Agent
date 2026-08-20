"""GUI-смоук-тесты через офскрин-платформу Qt (без реального дисплея)."""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from ai_agent.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _wait_for_worker(window, app, timeout_ms=5000):
    import time

    start = time.monotonic()
    while window._worker is not None and window._worker.isRunning():
        app.processEvents()
        if (time.monotonic() - start) * 1000 > timeout_ms:
            raise TimeoutError("agent worker did not finish in time")
    app.processEvents()


def test_main_window_builds_with_four_tabs(qapp, tmp_path: Path):
    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    tabs = window.centralWidget()
    assert tabs.count() == 4
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == ["Чат", "Навыки", "Память", "Права доступа"]


def test_skills_tab_lists_builtin_skills(qapp, tmp_path: Path):
    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    tabs = window.centralWidget()
    skills_widget = tabs.widget(1)
    text = skills_widget.findChild(type(window.chat_log)).toPlainText()
    assert "files.write" in text
    assert "system.run_command" in text
    assert "documents.create_docx" in text


def test_send_plain_task_updates_chat_and_memory(qapp, tmp_path: Path):
    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    window.input_line.setText("привет, агент")
    window._on_send()

    assert not window.input_line.isEnabled()  # заблокирован на время выполнения
    _wait_for_worker(window, qapp)

    log = window.chat_log.toPlainText()
    assert "Вы: привет, агент" in log
    assert "Агент:" in log
    assert window.input_line.isEnabled()

    stats = window.agent.memory.stats()
    assert stats["total_episodes"] == 1


def test_send_tool_task_creates_file_after_confirmation(qapp, tmp_path: Path, monkeypatch):
    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")

    # Подменяем диалог подтверждения на автоматическое согласие, чтобы не
    # блокироваться на реальном QMessageBox в headless-тесте. PolicyEngine
    # читает confirm_callback как атрибут при каждом enforce(), поэтому
    # патчим именно его, а не bound method моста (который уже был захвачен
    # по значению при сборке агента).
    monkeypatch.setattr(window.agent.skill_context.policy, "confirm_callback", lambda action, ctx: True)

    window.input_line.setText('TOOL: files.write {"path": "notes.txt", "content": "from gui"}')
    window._on_send()
    _wait_for_worker(window, qapp)

    target = tmp_path / "ws" / "notes.txt"
    assert target.exists()
    assert target.read_text() == "from gui"
    assert "files.write" in window.chat_log.toPlainText()


def test_confirm_bridge_dialog_unblocks_agent_thread(qapp, tmp_path: Path, monkeypatch):
    """Проверяет реальный кросс-поточный мост подтверждения: диалог должен
    открыться в GUI-потоке и разблокировать поток агента после ответа."""
    from PySide6.QtWidgets import QMessageBox

    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    window.agent.skill_context.policy.config.scripting_execute_enabled = True

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes))

    window.input_line.setText('TOOL: scripting.write_script {"filename": "hi.py", "content": "print(1)"}')
    window._on_send()
    _wait_for_worker(window, qapp)

    window.input_line.setText('TOOL: scripting.run_script {"filename": "hi.py"}')
    window._on_send()
    _wait_for_worker(window, qapp, timeout_ms=5000)

    log = window.chat_log.toPlainText()
    assert "scripting.run_script" in log
    assert "[OK]" in log


def test_permissions_tab_shows_policy_summary(qapp, tmp_path: Path):
    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    tabs = window.centralWidget()
    permissions_widget = tabs.widget(3)
    text = permissions_widget.findChild(type(window.chat_log)).toPlainText()
    assert "shell.enabled: False" in text
    assert "policy.yaml" in text
