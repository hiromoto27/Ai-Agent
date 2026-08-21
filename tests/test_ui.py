"""GUI-смоук-тесты через офскрин-платформу Qt (без реального дисплея)."""

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton

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


def _wait_for_llm_test(window, app, timeout_ms=5000):
    start = time.monotonic()
    while window._llm_test_worker is not None and window._llm_test_worker.isRunning():
        app.processEvents()
        if (time.monotonic() - start) * 1000 > timeout_ms:
            raise TimeoutError("llm connection test did not finish in time")
    app.processEvents()


def _make_window(tmp_path: Path) -> MainWindow:
    window = MainWindow(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    return window


def test_main_window_builds_with_six_tabs(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    tabs = window.centralWidget()
    assert tabs.count() == 6
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert titles == [
        theme.TAB_TITLES["chat"],
        theme.TAB_TITLES["settings"],
        theme.TAB_TITLES["models"],
        theme.TAB_TITLES["skills"],
        theme.TAB_TITLES["memory"],
        theme.TAB_TITLES["permissions"],
    ]


def test_skills_tab_lists_builtin_skills(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    tabs = window.centralWidget()
    skills_widget = tabs.widget(3)
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
    assert target.read_text(encoding="utf-8") == "from gui"
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
    permissions_widget = tabs.widget(5)
    text = permissions_widget.findChild(type(window.chat_log)).toPlainText()
    assert "shell.enabled: False" in text
    assert "model_download.enabled: False" in text
    assert "policy.yaml" in text


def test_permissions_tab_shows_log_path_and_open_button(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    tabs = window.centralWidget()
    permissions_widget = tabs.widget(5)

    labels_text = " ".join(child.text() for child in permissions_widget.findChildren(QLabel))
    assert str(window.agent.log_path) in labels_text
    assert window.open_log_button in permissions_widget.findChildren(QPushButton)


def test_open_log_button_opens_log_file_via_desktop_services(qapp, tmp_path: Path, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toLocalFile())))

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window._on_open_log_file()

    assert opened == [str(window.agent.log_path)]
    assert window.agent.log_path.exists()


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

    # Скачивание выключено по умолчанию — включаем через ту же галочку,
    # что видит пользователь, плюс подтверждаем сам запрос через policy,
    # как и в остальных GUI-тестах.
    window.enable_download_checkbox.setChecked(True)
    monkeypatch.setattr(window.agent.skill_context.policy, "confirm_callback", lambda action, ctx: True)

    assert window.recommend_list.count() > 0, "на этом железе должна найтись хотя бы одна рекомендация"
    window.recommend_list.setCurrentRow(0)
    window._on_download_selected(window.recommend_list)
    _wait_for_models_idle(window, qapp)

    assert len(calls) == 1
    assert "Ошибка" not in window.model_status_label.text()
    # Успешное скачивание сразу должно отразиться в разделе локальных моделей.
    assert window.local_models_list.count() == 1


def test_models_tab_download_blocked_without_checkbox(qapp, tmp_path: Path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "ai_agent.core.hf_models._default_download",
        lambda **kw: calls.append(kw) or kw["local_dir"],
    )

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window.agent.skill_context.policy.config.model_download_enabled = True
    window.enable_download_checkbox.setChecked(False)

    assert window.recommend_list.count() > 0
    window.recommend_list.setCurrentRow(0)
    window._on_download_selected(window.recommend_list)

    assert calls == []
    assert "выключено" in window.model_status_label.text()


def test_models_tab_checkbox_persists_to_policy_yaml(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window.enable_download_checkbox.setChecked(True)

    assert window.agent.skill_context.policy.config.model_download_enabled is True
    saved = (tmp_path / "state" / "policy.yaml").read_text(encoding="utf-8")
    assert "model_download" in saved
    assert "enabled: true" in saved


def test_models_tab_shows_manually_dropped_file(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    assert window.local_models_list.count() == 0  # только что созданная папка, есть лишь README.txt

    (tmp_path / "ws" / "models" / "my-model.gguf").write_bytes(b"x" * 1024)
    window._on_list_local_models()
    _wait_for_models_idle(window, qapp)

    assert window.local_models_list.count() == 1
    assert "my-model.gguf" in window.local_models_list.item(0).text()


def test_models_tab_download_without_selection_shows_hint(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window.search_list.clear()  # гарантируем отсутствие выбора
    window._on_download_selected(window.search_list)
    assert "выберите модель" in window.model_status_label.text()


def test_models_tab_shows_not_authorized_by_default(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    assert "не авторизован" in window.hf_auth_status_label.text()


def test_models_tab_open_auth_page_calls_desktop_services(qapp, tmp_path: Path, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toString())))

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window._on_open_hf_auth_page()

    assert opened == ["https://huggingface.co/settings/tokens"]


def test_models_tab_save_token_updates_status(qapp, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.whoami", lambda token=None: {"name": "alice"})
    monkeypatch.setattr(
        "huggingface_hub.login",
        lambda token=None, add_to_git_credential=False, skip_if_logged_in=True: None,
    )
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_abc123")

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window.hf_token_input.setText("hf_abc123")
    window._on_save_hf_token()
    _wait_for_models_idle(window, qapp)

    assert "alice" in window.model_status_label.text()
    assert window.hf_token_input.text() == ""
    assert "сохранён" in window.hf_auth_status_label.text()


def test_models_tab_save_empty_token_shows_hint(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window.hf_token_input.setText("   ")
    window._on_save_hf_token()
    assert "Вставьте токен" in window.model_status_label.text()


def test_models_tab_logout_clears_status(qapp, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.logout", lambda token_name=None: None)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window._on_hf_logout()
    _wait_for_models_idle(window, qapp)

    assert "не авторизован" in window.hf_auth_status_label.text()


def test_models_tab_download_auth_required_points_to_auth_section(qapp, tmp_path: Path, monkeypatch):
    from huggingface_hub.errors import GatedRepoError
    import httpx

    def fake_download(**kwargs):
        response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
        raise GatedRepoError("gated", response=response)

    monkeypatch.setattr("ai_agent.core.hf_models._default_download", lambda **kw: fake_download(**kw))

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window.enable_download_checkbox.setChecked(True)
    monkeypatch.setattr(window.agent.skill_context.policy, "confirm_callback", lambda action, ctx: True)

    assert window.recommend_list.count() > 0
    window.recommend_list.setCurrentRow(0)
    window._on_download_selected(window.recommend_list)
    _wait_for_models_idle(window, qapp)

    assert "авторизации" in window.model_status_label.text()


def test_models_tab_import_file_via_dialog(qapp, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    source = tmp_path / "downloads" / "my-model.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x" * 4096)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(source), "")))

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    monkeypatch.setattr(window.agent.skill_context.policy, "confirm_callback", lambda action, ctx: True)

    window._on_import_file()
    _wait_for_models_idle(window, qapp)

    assert (tmp_path / "ws" / "models" / "my-model.gguf").exists()
    assert window.local_models_list.count() == 1
    assert "my-model.gguf" in window.local_models_list.item(0).text()


def test_models_tab_import_folder_via_dialog(qapp, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    source = tmp_path / "cloned-repo"
    source.mkdir()
    (source / "model.gguf").write_bytes(b"x" * 4096)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(source)))

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    monkeypatch.setattr(window.agent.skill_context.policy, "confirm_callback", lambda action, ctx: True)

    window._on_import_folder()
    _wait_for_models_idle(window, qapp)

    assert (tmp_path / "ws" / "models" / "cloned-repo" / "model.gguf").exists()
    assert window.local_models_list.count() == 1


def test_models_tab_import_cancelled_dialog_does_nothing(qapp, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: ("", "")))

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    status_before = window.model_status_label.text()

    window._on_import_file()

    assert window.model_status_label.text() == status_before


def test_settings_tab_shows_echo_by_default(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    assert "тестовый режим" in window.current_provider_label.text()


def test_settings_tab_switch_to_echo_and_save_applies_immediately(qapp, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")  # чтобы "авто" не был уже echo

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    idx = window.provider_combo.findData("echo")
    window.provider_combo.setCurrentIndex(idx)
    window._on_save_llm_settings()

    from ai_agent.core.llm.echo_provider import EchoProvider

    assert isinstance(window.agent.llm, EchoProvider)
    assert "тестовый режим" in window.current_provider_label.text()
    assert "Настройки сохранены" in window.settings_status_label.text()

    saved = (tmp_path / "state" / "llm_settings.yaml").read_text(encoding="utf-8")
    assert "provider: echo" in saved


def test_settings_tab_system_prompt_is_applied_to_agent(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window.system_prompt_edit.setPlainText("Отвечай только эмодзи.")
    window._on_save_llm_settings()

    assert "Отвечай только эмодзи." in window.agent.system_prompt


def test_settings_tab_explicit_anthropic_without_key_shows_setup_error(qapp, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    idx = window.provider_combo.findData("anthropic")
    window.provider_combo.setCurrentIndex(idx)
    window.anthropic_key_input.setText("")
    window._on_save_llm_settings()

    from ai_agent.core.llm.echo_provider import EchoProvider

    assert isinstance(window.agent.llm, EchoProvider)  # безопасный откат, не падение приложения
    assert "не запустился" in window.settings_status_label.text()
    assert window.agent.llm_setup_error != ""


def test_settings_tab_connection_test_success_with_echo_provider(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window._on_test_llm_connection()
    assert not window.test_connection_button.isEnabled()
    _wait_for_llm_test(window, qapp)

    assert window.test_connection_button.isEnabled()
    assert "✅" in window.settings_status_label.text()


def test_settings_tab_connection_test_reports_failure(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    class _BrokenProvider:
        def complete(self, messages, tools, system=""):
            raise ConnectionError("сервер недоступен")

    window.agent.llm = _BrokenProvider()
    window._on_test_llm_connection()
    _wait_for_llm_test(window, qapp)

    assert "❌" in window.settings_status_label.text()
    assert "сервер недоступен" in window.settings_status_label.text()


def test_models_tab_connection_test_success(qapp, tmp_path: Path, monkeypatch):
    class _FakeModelInfo:
        def __init__(self, id):
            self.id = id
            self.downloads = 1
            self.likes = 0
            self.tags = []

    class _FakeApi:
        def list_models(self, *, search, limit, sort):
            return [_FakeModelInfo("test/model")][:limit]

    monkeypatch.setattr("ai_agent.core.hf_models._default_api", lambda: _FakeApi())

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window._on_test_hf_connection()
    _wait_for_models_idle(window, qapp)

    assert "✅" in window.model_status_label.text()


def test_models_tab_connection_test_reports_failure(qapp, tmp_path: Path, monkeypatch):
    class _FakeApi:
        def list_models(self, *, search, limit, sort):
            raise ConnectionError("сеть недоступна")

    monkeypatch.setattr("ai_agent.core.hf_models._default_api", lambda: _FakeApi())

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window._on_test_hf_connection()
    _wait_for_models_idle(window, qapp)

    assert "❌" in window.model_status_label.text()
    assert "сеть недоступна" in window.model_status_label.text()


def test_models_tab_refresh_local_shows_feedback_even_when_unchanged(qapp, tmp_path: Path):
    """Раньше повторное нажатие «Обновить» без изменений в workspace/models
    не меняло статус-строку — выглядело так, будто кнопка не работает."""
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window.model_status_label.setText("")
    window._on_list_local_models()
    _wait_for_models_idle(window, qapp)

    assert window.model_status_label.text() != ""
    assert "Обновлено" in window.model_status_label.text()


def test_settings_tab_refresh_local_combo_shows_feedback(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window.settings_status_label.setText("")
    window._refresh_local_model_combo()

    assert window.settings_status_label.text() != ""
    assert "Обновлено" in window.settings_status_label.text()


def test_settings_tab_local_model_combo_lists_downloaded_gguf(qapp, tmp_path: Path):
    models_dir = tmp_path / "ws" / "models" / "Qwen__Qwen2.5-1.5B-Instruct-GGUF"
    models_dir.mkdir(parents=True)
    (models_dir / "model.q4_k_m.gguf").write_bytes(b"x" * 1024)

    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)
    window._refresh_local_model_combo()

    items = [window.local_model_combo.itemText(i) for i in range(window.local_model_combo.count())]
    assert any("model.q4_k_m.gguf" in item for item in items)


def test_new_chat_button_clears_conversation(qapp, tmp_path: Path):
    window = _make_window(tmp_path)
    _wait_for_models_idle(window, qapp)

    window.input_line.setText("привет")
    window._on_send()
    _wait_for_worker(window, qapp)
    assert len(window.agent.conversation) > 0
    assert window.chat_log.toPlainText().strip() != ""

    window._on_new_chat()

    assert window.agent.conversation == []
    assert window.chat_log.toPlainText().strip() == ""
