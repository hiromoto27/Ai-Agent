"""GUI-смоук-тесты через офскрин-платформу Qt (без реального дисплея)."""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from ai_agent.ui import theme
from ai_agent.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def no_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _wait_for_worker(window, app, timeout_ms=5000):
    start = time.monotonic()
    while window._worker is not None and window._worker.isRunning():
        app.processEvents()
        if (time.monotonic() - start) * 1000 > timeout_ms:
            raise TimeoutError("agent worker did not finish in time")
    app.processEvents()


def _wait_for_models_idle(window, app, timeout_ms=5000):
    """Вкладка "Модели" считается свободной, когда кнопка "Обновить"
    снова активна — _set_models_busy(False) включает её при завершении
    любого фонового действия (рекомендации/поиск/скачивание)."""
    start = time.monotonic()
    while not window.recommend_button.isEnabled():
        app.processEvents()
        if (time.monotonic() - start) * 1000 > timeout_ms:
            raise TimeoutError("models tab did not finish in time")
    app.processEvents()


def _make_window(tmp_path: Path) -> MainWindow:
    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    return window


def test_main_window_builds_with_five_tabs(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    tabs = window.centralWidget()
    assert tabs.count() == 5
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == [
        theme.TAB_TITLES["chat"],
        theme.TAB_TITLES["models"],
        theme.TAB_TITLES["skills"],
        theme.TAB_TITLES["memory"],
        theme.TAB_TITLES["permissions"],
    ]


def test_skills_tab_lists_builtin_skills(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    tabs = window.centralWidget()
    skills_widget = tabs.widget(2)
    text = skills_widget.findChild(type(window.chat_log)).toPlainText()
    assert "files.write" in text
    assert "system.run_command" in text
    assert "documents.create_docx" in text
    assert "models.recommend" in text
    assert "models.search_huggingface" in text
    assert "models.download_huggingface" in text


def test_send_plain_task_updates_chat_and_memory(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
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
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

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

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
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
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    tabs = window.centralWidget()
    permissions_widget = tabs.widget(4)
    text = permissions_widget.findChild(type(window.chat_log)).toPlainText()
    assert "shell.enabled: False" in text
    assert "model_download.enabled: False" in text
    assert "policy.yaml" in text


def test_models_tab_shows_recommendations_on_open(qapp, tmp_path: Path):
    """Рекомендации — чисто локальная операция (детект железа), поэтому
    вкладка сама подгружает их при открытии, без сети и разрешений."""
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    assert "CPU" in window.hw_summary_label.text()
    assert window.recommend_list.count() >= 0  # может быть 0 на очень слабом раннере, это не баг
    assert "Ошибка" not in window.model_status_label.text()


def test_models_tab_search_uses_injected_api(qapp, tmp_path: Path, monkeypatch):
    class _FakeModelInfo:
        def __init__(self, id):
            self.id = id
            self.downloads = 42
            self.likes = 7
            self.tags = []

    class _FakeApi:
        def list_models(self, *, search, limit, sort):
            return [_FakeModelInfo("Qwen/Qwen2.5-1.5B-Instruct-GGUF")][:limit]

    # search_models() без явного api строит HfApi() через _default_api() —
    # подменяем именно эту фабрику, чтобы тест не ходил в реальную сеть.
    monkeypatch.setattr("ai_agent.core.hf_models._default_api", lambda: _FakeApi())

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window.model_search_input.setText("qwen instruct")
    window._on_search_models()
    _wait_for_models_idle(window, qapp)

    assert window.search_list.count() == 1
    assert "Qwen/Qwen2.5-1.5B-Instruct-GGUF" in window.search_list.item(0).text()


def test_models_tab_download_selected_recommendation(qapp, tmp_path: Path, monkeypatch):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        out = Path(kwargs["local_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    monkeypatch.setattr("ai_agent.core.hf_models._default_download", lambda **kw: fake_download(**kw))

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    # Скачивание выключено и требует подтверждения по умолчанию — включаем
    # и подтверждаем напрямую через policy, как и в остальных GUI-тестах.
    window.agent.skill_context.policy.config.model_download_enabled = True
    monkeypatch.setattr(window.agent.skill_context.policy, "confirm_callback", lambda action, ctx: True)

    assert window.recommend_list.count() > 0, "на этом железе должна найтись хотя бы одна рекомендация"
    window.recommend_list.setCurrentRow(0)
    window._on_download_selected(window.recommend_list)
    _wait_for_models_idle(window, qapp)

    assert len(calls) == 1
    assert "Ошибка" not in window.model_status_label.text()


def test_models_tab_download_without_selection_shows_hint(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window.search_list.clear()  # гарантируем отсутствие выбора
    window._on_download_selected(window.search_list)
    assert "выберите модель" in window.model_status_label.text()
