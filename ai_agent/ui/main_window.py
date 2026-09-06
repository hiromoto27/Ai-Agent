"""Главное окно десктоп-приложения (чат, навыки, память, права доступа, модели, диктофон)."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai_agent.app import build_agent, build_llm_provider_safe, combine_system_prompt
from ai_agent.core.orchestrator import AgentResult, DEFAULT_SYSTEM_PROMPT
from ai_agent.core.skills.base import SkillResult
from ai_agent.core.skills.models import hf_login_status
from ai_agent.core.voice.reminders import ReminderChecker

from . import theme
from .confirm_bridge import ConfirmBridge
from .llm_test_worker import LLMConnectionTestWorker
from .lmstudio_models_worker import LMStudioModelsWorker
from .skill_worker import SkillWorker
from .worker import AgentWorker

HF_TOKENS_URL = "https://huggingface.co/settings/tokens"

_PROVIDER_CHOICES = [
    ("auto", "Автоматически (облако, если есть ключ, иначе локальная модель, иначе тест)"),
    ("anthropic", "Claude API (облако)"),
    ("local", "Локальная модель (GGUF, встроенный движок)"),
    ("lmstudio", "LM Studio (модель, запущенная в LM Studio)"),
    ("echo", "Тестовый режим (без ИИ — эхо, для проверки)"),
]


class MainWindow(QMainWindow):
    def __init__(self, state_dir: Path | None = None, workspace_root: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Локальный ИИ-агент")
        self.resize(1000, 680)
        self.setStyleSheet(theme.STYLESHEET)

        self.confirm_bridge = ConfirmBridge(self)
        kwargs = {}
        if state_dir is not None:
            kwargs["state_dir"] = state_dir
        if workspace_root is not None:
            kwargs["workspace_root"] = workspace_root
        self.agent = build_agent(confirm_callback=self.confirm_bridge.confirm_callback, **kwargs)
        self._policy_path = self.agent.skill_context.policy.audit_log_path.parent / "policy.yaml"

        self._worker: AgentWorker | None = None
        self._model_workers: list[SkillWorker] = []
        self._models_busy_count = 0
        self._llm_test_worker: LLMConnectionTestWorker | None = None
        self._lmstudio_models_worker: LMStudioModelsWorker | None = None
        self._voice_workers: list[SkillWorker] = []
        self._voice_busy_count = 0
        self._reminder_checker = ReminderChecker(self.agent.skill_context.voice.store)
        self._build_ui()

        # Напоминания по задачам работают, только пока приложение открыто
        # (см. VOICE_RECORDER_PLAN.md, раздел 7) — таймер проверяет их
        # раз в 30 секунд.
        self.reminder_timer = QTimer(self)
        self.reminder_timer.timeout.connect(self._on_check_reminders)
        self.reminder_timer.start(30_000)

        # Индикатор уровня сигнала микрофона — просто читает текущее
        # значение VoiceService.current_level (обновляется из фонового
        # потока записи), без сложной межпоточной сигнализации Qt.
        self.level_timer = QTimer(self)
        self.level_timer.timeout.connect(self._on_level_tick)
        self.level_timer.start(100)

    # ---- построение интерфейса --------------------------------------------------

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_chat_tab(), theme.TAB_TITLES["chat"])
        tabs.addTab(self._build_settings_tab(), theme.TAB_TITLES["settings"])
        tabs.addTab(self._build_models_tab(), theme.TAB_TITLES["models"])
        tabs.addTab(self._build_skills_tab(), theme.TAB_TITLES["skills"])
        tabs.addTab(self._build_memory_tab(), theme.TAB_TITLES["memory"])
        tabs.addTab(self._build_permissions_tab(), theme.TAB_TITLES["permissions"])
        tabs.addTab(self._build_voice_tab(), theme.TAB_TITLES["voice"])
        self.setCentralWidget(tabs)

    def _build_chat_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        header = QHBoxLayout()
        header.addStretch()
        self.new_chat_button = QPushButton("🔄 Новый чат")
        self.new_chat_button.setObjectName("secondary")
        self.new_chat_button.clicked.connect(self._on_new_chat)
        header.addWidget(self.new_chat_button)
        layout.addLayout(header)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        layout.addWidget(self.chat_log)

        row = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Опишите задачу для агента…")
        self.input_line.returnPressed.connect(self._on_send)
        self.send_button = QPushButton("Отправить")
        self.send_button.clicked.connect(self._on_send)
        row.addWidget(self.input_line)
        row.addWidget(self.send_button)
        layout.addLayout(row)
        return widget

    def _build_settings_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        settings = self.agent.llm_settings

        provider_status_row = QHBoxLayout()
        self.current_provider_label = QLabel()
        self.current_provider_label.setObjectName("hwSummary")
        self.current_provider_label.setWordWrap(True)
        provider_status_row.addWidget(self.current_provider_label, 1)
        self.test_connection_button = QPushButton("🔌 Проверить подключение")
        self.test_connection_button.setObjectName("secondary")
        self.test_connection_button.clicked.connect(self._on_test_llm_connection)
        provider_status_row.addWidget(self.test_connection_button)
        layout.addLayout(provider_status_row)

        layout.addWidget(QLabel("Провайдер ответов агента:"))
        self.provider_combo = QComboBox()
        for value, label in _PROVIDER_CHOICES:
            self.provider_combo.addItem(label, value)
        idx = self.provider_combo.findData(settings.provider)
        self.provider_combo.setCurrentIndex(idx if idx >= 0 else 0)
        layout.addWidget(self.provider_combo)

        layout.addWidget(QLabel("Claude API:"))
        anthropic_row = QHBoxLayout()
        self.anthropic_key_input = QLineEdit(settings.anthropic_api_key)
        self.anthropic_key_input.setPlaceholderText(
            "ANTHROPIC_API_KEY (пусто — взять из переменной окружения)"
        )
        self.anthropic_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        anthropic_row.addWidget(self.anthropic_key_input, 1)
        self.anthropic_model_input = QLineEdit(settings.anthropic_model)
        self.anthropic_model_input.setPlaceholderText("модель, например claude-sonnet-5")
        anthropic_row.addWidget(self.anthropic_model_input)
        layout.addLayout(anthropic_row)

        local_header = QHBoxLayout()
        local_header.addWidget(QLabel("Локальная модель (файл .gguf из workspace/models):"))
        local_header.addStretch()
        self.refresh_local_combo_button = QPushButton("🔄 Обновить список")
        self.refresh_local_combo_button.setObjectName("secondary")
        self.refresh_local_combo_button.clicked.connect(self._refresh_local_model_combo)
        local_header.addWidget(self.refresh_local_combo_button)
        layout.addLayout(local_header)

        self.local_model_combo = QComboBox()
        self.local_model_combo.setEditable(True)
        layout.addWidget(self.local_model_combo)
        self._refresh_local_model_combo()

        local_hint = QLabel(
            "Требует пакет llama-cpp-python (pip install \"ai-agent[local-llm]\") — не входит в "
            "обычную поставку, так как на большинстве систем ставится сборкой из исходников "
            "(нужен компилятор). Экспериментально: то, насколько модель соблюдает формат вызова "
            "инструментов, зависит от конкретной модели — крупные следуют инструкциям надёжнее мелких."
        )
        local_hint.setObjectName("statusLabel")
        local_hint.setWordWrap(True)
        layout.addWidget(local_hint)

        lmstudio_header = QHBoxLayout()
        lmstudio_header.addWidget(QLabel("LM Studio (адрес локального сервера):"))
        lmstudio_header.addStretch()
        self.refresh_lmstudio_models_button = QPushButton("🔄 Обновить список моделей")
        self.refresh_lmstudio_models_button.setObjectName("secondary")
        self.refresh_lmstudio_models_button.clicked.connect(self._on_refresh_lmstudio_models)
        lmstudio_header.addWidget(self.refresh_lmstudio_models_button)
        layout.addLayout(lmstudio_header)

        self.lmstudio_url_input = QLineEdit(settings.lmstudio_base_url)
        self.lmstudio_url_input.setPlaceholderText("http://localhost:1234/v1")
        layout.addWidget(self.lmstudio_url_input)

        self.lmstudio_model_combo = QComboBox()
        self.lmstudio_model_combo.setEditable(True)
        if settings.lmstudio_model:
            self.lmstudio_model_combo.addItem(settings.lmstudio_model)
        layout.addWidget(self.lmstudio_model_combo)

        self.lmstudio_api_key_input = QLineEdit(settings.lmstudio_api_key)
        self.lmstudio_api_key_input.setPlaceholderText(
            "API-ключ LM Studio (только если включено Require API Key) — обычно не нужен"
        )
        self.lmstudio_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.lmstudio_api_key_input)

        lmstudio_hint = QLabel(
            "Запустите модель в приложении LM Studio и включите локальный сервер "
            "(Settings → Developer → Enable Local Server), затем нажмите «Обновить список моделей» — "
            "поле модели можно оставить пустым, если в LM Studio загружена только одна модель. "
            "Ошибка «401 Unauthorized» означает, что в LM Studio включено «Require API Key» — "
            "укажите тот же ключ в поле выше."
        )
        lmstudio_hint.setObjectName("statusLabel")
        lmstudio_hint.setWordWrap(True)
        layout.addWidget(lmstudio_hint)

        layout.addWidget(QLabel("Системный промпт (стиль ответов, роль, ограничения — поверх базовых инструкций):"))
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlainText(settings.system_prompt)
        self.system_prompt_edit.setPlaceholderText(
            "Например: «Отвечай только на русском и кратко» или «Ты — ассистент по Python, "
            "объясняй код построчно»."
        )
        layout.addWidget(self.system_prompt_edit, 1)

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.save_settings_button = QPushButton("💾 Сохранить и применить")
        self.save_settings_button.clicked.connect(self._on_save_llm_settings)
        save_row.addWidget(self.save_settings_button)
        layout.addLayout(save_row)

        self.settings_status_label = QLabel("")
        self.settings_status_label.setObjectName("statusLabel")
        self.settings_status_label.setWordWrap(True)
        layout.addWidget(self.settings_status_label)

        self._refresh_current_provider_label()
        if self.agent.llm_setup_error:
            self.settings_status_label.setText(
                "⚠ Выбранный провайдер не смог запуститься, сейчас используется тестовый режим: "
                f"{self.agent.llm_setup_error}"
            )

        return widget

    def _build_skills_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        view = QTextEdit()
        view.setReadOnly(True)
        lines = [f"{spec.name} — {spec.description}" for spec in self.agent.skills.list_specs()]
        view.setPlainText("\n\n".join(lines))
        layout.addWidget(view)
        return widget

    def _build_memory_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.memory_view = QTextEdit()
        self.memory_view.setReadOnly(True)
        layout.addWidget(self.memory_view)
        self._refresh_memory_tab()
        return widget

    def _build_permissions_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        log_row = QHBoxLayout()
        log_label = QLabel(f"Журнал ошибок и диагностики: {self.agent.log_path}")
        log_label.setObjectName("statusLabel")
        log_label.setWordWrap(True)
        log_row.addWidget(log_label, 1)
        self.open_log_button = QPushButton("📄 Открыть журнал")
        self.open_log_button.setObjectName("secondary")
        self.open_log_button.clicked.connect(self._on_open_log_file)
        log_row.addWidget(self.open_log_button)
        layout.addLayout(log_row)

        view = QTextEdit()
        view.setReadOnly(True)
        config = self.agent.skill_context.policy.config
        view.setPlainText(
            "Текущая политика прав доступа (config/policy.yaml):\n\n"
            f"shell.enabled: {config.shell_enabled}\n"
            f"scripting.execute_enabled: {config.scripting_execute_enabled}\n"
            f"package_install.enabled: {config.package_install_enabled}\n"
            f"model_download.enabled: {config.model_download_enabled}\n"
            f"agents.enabled: {config.subagents_enabled}\n"
            f"workspace_only: {config.workspace_only}\n"
            f"network.enabled: {config.network_enabled}\n\n"
            f"Файл: {self._policy_path}\n"
            "Отредактируйте его и перезапустите приложение, чтобы изменить права.\n"
            "Действия, требующие подтверждения, покажут диалог во время выполнения задачи."
        )
        layout.addWidget(view)
        return widget

    def _build_voice_tab(self) -> QWidget:
        widget = QWidget()
        outer = QVBoxLayout(widget)
        mic_config = self.agent.skill_context.policy.config

        columns = QHBoxLayout()
        columns.setSpacing(14)
        left_col = QVBoxLayout()
        left_col.setSpacing(14)
        right_col = QVBoxLayout()
        right_col.setSpacing(14)

        # -- Компоненты диктофона (статус + установка) ------------------------------
        setup_group = QGroupBox("⚙ Компоненты диктофона")
        setup_layout = QVBoxLayout(setup_group)
        self.setup_status_label = QLabel("Проверяю компоненты…")
        self.setup_status_label.setObjectName("statusLabel")
        self.setup_status_label.setWordWrap(True)
        setup_layout.addWidget(self.setup_status_label)
        install_row = QHBoxLayout()
        self.install_voice_button = QPushButton("⬇ Установить / обновить компоненты")
        self.install_voice_button.clicked.connect(self._on_install_voice_dependencies)
        install_row.addWidget(self.install_voice_button)
        install_row.addStretch()
        setup_layout.addLayout(install_row)
        install_hint = QLabel(
            "Скачивает пакеты для записи/распознавания речи и модель распознавания под ваш ПК "
            "(размер подобран автоматически). Требует разрешения на установку пакетов и на "
            "скачивание моделей — каждое действие спросит подтверждение."
        )
        install_hint.setObjectName("statusLabel")
        install_hint.setWordWrap(True)
        setup_layout.addWidget(install_hint)
        left_col.addWidget(setup_group)

        # -- Права доступа к микрофону ------------------------------------------------
        permissions_group = QGroupBox("🔒 Права доступа к микрофону")
        permissions_layout = QVBoxLayout(permissions_group)
        self.mic_enabled_checkbox = QCheckBox("Разрешить запись с микрофона")
        self.mic_enabled_checkbox.setChecked(mic_config.microphone_enabled)
        self.mic_enabled_checkbox.toggled.connect(self._on_toggle_mic_enabled)
        permissions_layout.addWidget(self.mic_enabled_checkbox)

        self.mic_continuous_checkbox = QCheckBox("Разрешить постоянную запись")
        self.mic_continuous_checkbox.setChecked(mic_config.microphone_continuous_enabled)
        self.mic_continuous_checkbox.toggled.connect(self._on_toggle_mic_continuous_enabled)
        permissions_layout.addWidget(self.mic_continuous_checkbox)

        self.mic_retain_checkbox = QCheckBox("Сохранять аудиофайлы")
        self.mic_retain_checkbox.setChecked(mic_config.microphone_retain_audio)
        self.mic_retain_checkbox.toggled.connect(self._on_toggle_mic_retain_audio)
        permissions_layout.addWidget(self.mic_retain_checkbox)

        mic_hint = QLabel(
            "Разовая запись работает без переспроса, пока включена галочка выше. Постоянная "
            "(фоновая) запись при КАЖДОМ включении отдельно спросит подтверждение — риск "
            "качественно другой: агент слышит всё, что происходит рядом. По умолчанию "
            "сохраняется только текст расшифровки, не сами аудиофайлы."
        )
        mic_hint.setObjectName("statusLabel")
        mic_hint.setWordWrap(True)
        permissions_layout.addWidget(mic_hint)
        left_col.addWidget(permissions_group)

        # -- Устройство записи и уровень сигнала --------------------------------------
        device_group = QGroupBox("🎚 Устройство записи и уровень сигнала")
        device_layout = QVBoxLayout(device_group)
        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_row.addWidget(self.device_combo, 1)
        self.refresh_devices_button = QPushButton("🔄")
        self.refresh_devices_button.setObjectName("secondary")
        self.refresh_devices_button.setToolTip("Обновить список устройств")
        self.refresh_devices_button.clicked.connect(self._on_refresh_input_devices)
        device_row.addWidget(self.refresh_devices_button)
        device_layout.addLayout(device_row)

        device_layout.addWidget(QLabel("Уровень сигнала (говорите, чтобы проверить микрофон):"))
        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setValue(0)
        self.level_meter.setTextVisible(False)
        device_layout.addWidget(self.level_meter)
        left_col.addWidget(device_group)
        left_col.addStretch()

        # -- Запись --------------------------------------------------------------------
        record_group = QGroupBox("🎙 Запись")
        record_layout = QVBoxLayout(record_group)
        record_row = QHBoxLayout()
        self.record_once_button = QPushButton("🎙 Разовая запись")
        self.record_once_button.clicked.connect(self._on_record_once)
        record_row.addWidget(self.record_once_button)

        self.continuous_button = QPushButton("⏺ Включить постоянную запись")
        self.continuous_button.clicked.connect(self._on_toggle_continuous)
        record_row.addWidget(self.continuous_button)
        record_layout.addLayout(record_row)

        self.continuous_status_label = QLabel("Постоянная запись выключена.")
        self.continuous_status_label.setObjectName("statusLabel")
        record_layout.addWidget(self.continuous_status_label)
        right_col.addWidget(record_group)

        # -- Задачи ----------------------------------------------------------------------
        tasks_group = QGroupBox("🗂 Задачи (классификация по контекстным словам)")
        tasks_layout = QVBoxLayout(tasks_group)
        task_row = QHBoxLayout()
        self.new_task_title_input = QLineEdit()
        self.new_task_title_input.setPlaceholderText("Название задачи")
        task_row.addWidget(self.new_task_title_input, 1)
        self.new_task_desc_input = QLineEdit()
        self.new_task_desc_input.setPlaceholderText("Описание/контекст (для первичной классификации)")
        task_row.addWidget(self.new_task_desc_input, 1)
        self.create_task_button = QPushButton("+ Задача")
        self.create_task_button.clicked.connect(self._on_create_task)
        task_row.addWidget(self.create_task_button)
        tasks_layout.addLayout(task_row)

        self.tasks_list = QListWidget()
        self.tasks_list.setMaximumHeight(100)
        tasks_layout.addWidget(self.tasks_list)

        reminder_row = QHBoxLayout()
        reminder_row.addWidget(QLabel("Напомнить через (минут):"))
        self.reminder_minutes_input = QLineEdit("60")
        self.reminder_minutes_input.setFixedWidth(60)
        reminder_row.addWidget(self.reminder_minutes_input)
        self.set_reminder_button = QPushButton("⏰ Поставить напоминание")
        self.set_reminder_button.clicked.connect(self._on_set_reminder)
        reminder_row.addWidget(self.set_reminder_button)
        reminder_row.addStretch()
        tasks_layout.addLayout(reminder_row)
        right_col.addWidget(tasks_group)

        # -- Реплики -----------------------------------------------------------------------
        utterances_group = QGroupBox("💬 Реплики")
        utterances_layout = QVBoxLayout(utterances_group)
        utterances_layout.addWidget(QLabel("Выберите реплику и задачу, чтобы подтвердить/отклонить принадлежность:"))
        self.utterances_list = QListWidget()
        utterances_layout.addWidget(self.utterances_list, 1)

        confirm_row = QHBoxLayout()
        self.confirm_task_combo = QComboBox()
        confirm_row.addWidget(self.confirm_task_combo, 1)
        self.confirm_yes_button = QPushButton("✔ Относится")
        self.confirm_yes_button.clicked.connect(lambda: self._on_confirm_utterance(True))
        confirm_row.addWidget(self.confirm_yes_button)
        self.confirm_no_button = QPushButton("✘ Не относится")
        self.confirm_no_button.setObjectName("secondary")
        self.confirm_no_button.clicked.connect(lambda: self._on_confirm_utterance(False))
        confirm_row.addWidget(self.confirm_no_button)
        utterances_layout.addLayout(confirm_row)
        right_col.addWidget(utterances_group, 1)

        # -- Протокол --------------------------------------------------------------------
        protocol_row = QHBoxLayout()
        self.generate_protocol_button = QPushButton("📝 Сформировать протокол")
        self.generate_protocol_button.clicked.connect(self._on_generate_protocol)
        protocol_row.addWidget(self.generate_protocol_button)
        protocol_row.addStretch()
        right_col.addLayout(protocol_row)

        columns.addLayout(left_col, 1)
        columns.addLayout(right_col, 2)
        outer.addLayout(columns, 1)

        self.voice_status_label = QLabel("")
        self.voice_status_label.setObjectName("statusLabel")
        self.voice_status_label.setWordWrap(True)
        outer.addWidget(self.voice_status_label)

        self._refresh_tasks_and_utterances()
        self._refresh_continuous_status()
        self._on_check_voice_setup()
        self._on_refresh_input_devices()
        return widget

    def _build_models_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.hw_summary_label = QLabel("Определяю характеристики ПК…")
        self.hw_summary_label.setObjectName("hwSummary")
        self.hw_summary_label.setWordWrap(True)
        layout.addWidget(self.hw_summary_label)

        self.enable_download_checkbox = QCheckBox("Разрешить скачивание моделей с Hugging Face")
        self.enable_download_checkbox.setChecked(self.agent.skill_context.policy.config.model_download_enabled)
        self.enable_download_checkbox.toggled.connect(self._on_toggle_model_download)
        layout.addWidget(self.enable_download_checkbox)

        auth_header = QHBoxLayout()
        auth_header.addWidget(QLabel("Авторизация Hugging Face (нужна для закрытых/gated моделей):"))
        auth_header.addStretch()
        self.hf_auth_status_label = QLabel()
        auth_header.addWidget(self.hf_auth_status_label)
        layout.addLayout(auth_header)

        auth_row = QHBoxLayout()
        self.open_hf_auth_button = QPushButton("🔗 Открыть страницу авторизации")
        self.open_hf_auth_button.setObjectName("secondary")
        self.open_hf_auth_button.clicked.connect(self._on_open_hf_auth_page)
        auth_row.addWidget(self.open_hf_auth_button)

        self.hf_token_input = QLineEdit()
        self.hf_token_input.setPlaceholderText("Вставьте токен (hf_...) со страницы выше")
        self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_token_input.returnPressed.connect(self._on_save_hf_token)
        auth_row.addWidget(self.hf_token_input, 1)

        self.save_hf_token_button = QPushButton("💾 Сохранить")
        self.save_hf_token_button.clicked.connect(self._on_save_hf_token)
        auth_row.addWidget(self.save_hf_token_button)

        self.logout_hf_button = QPushButton("Выйти")
        self.logout_hf_button.setObjectName("secondary")
        self.logout_hf_button.clicked.connect(self._on_hf_logout)
        auth_row.addWidget(self.logout_hf_button)
        layout.addLayout(auth_row)

        self._refresh_hf_auth_status()

        test_hf_row = QHBoxLayout()
        self.test_hf_button = QPushButton("🔌 Проверить связь с Hugging Face")
        self.test_hf_button.setObjectName("secondary")
        self.test_hf_button.clicked.connect(self._on_test_hf_connection)
        test_hf_row.addWidget(self.test_hf_button)
        test_hf_row.addStretch()
        layout.addLayout(test_hf_row)

        recommend_header = QHBoxLayout()
        recommend_header.addWidget(QLabel("Рекомендации под ваш ПК:"))
        recommend_header.addStretch()
        self.recommend_button = QPushButton("🔄 Обновить")
        self.recommend_button.setObjectName("secondary")
        self.recommend_button.clicked.connect(self._on_recommend_models)
        recommend_header.addWidget(self.recommend_button)
        layout.addLayout(recommend_header)

        self.recommend_list = QListWidget()
        layout.addWidget(self.recommend_list, 1)

        self.download_recommend_button = QPushButton("⬇ Скачать выбранную модель")
        self.download_recommend_button.clicked.connect(
            lambda: self._on_download_selected(self.recommend_list)
        )
        layout.addWidget(self.download_recommend_button)

        search_row = QHBoxLayout()
        self.model_search_input = QLineEdit()
        self.model_search_input.setPlaceholderText("Поиск моделей на Hugging Face…")
        self.model_search_input.returnPressed.connect(self._on_search_models)
        self.search_button = QPushButton("🔍 Искать")
        self.search_button.clicked.connect(self._on_search_models)
        search_row.addWidget(self.model_search_input)
        search_row.addWidget(self.search_button)
        layout.addLayout(search_row)

        self.search_list = QListWidget()
        layout.addWidget(self.search_list, 1)

        self.download_search_button = QPushButton("⬇ Скачать выбранную модель")
        self.download_search_button.setObjectName("secondary")
        self.download_search_button.clicked.connect(
            lambda: self._on_download_selected(self.search_list)
        )
        layout.addWidget(self.download_search_button)

        local_header = QHBoxLayout()
        local_header.addWidget(QLabel("Локальные модели (workspace/models):"))
        local_header.addStretch()
        self.import_file_button = QPushButton("📄 Импортировать файл…")
        self.import_file_button.setObjectName("secondary")
        self.import_file_button.clicked.connect(self._on_import_file)
        local_header.addWidget(self.import_file_button)

        self.import_folder_button = QPushButton("📁 Импортировать папку…")
        self.import_folder_button.setObjectName("secondary")
        self.import_folder_button.clicked.connect(self._on_import_folder)
        local_header.addWidget(self.import_folder_button)

        self.refresh_local_button = QPushButton("🔄 Обновить")
        self.refresh_local_button.setObjectName("secondary")
        self.refresh_local_button.clicked.connect(self._on_list_local_models)
        local_header.addWidget(self.refresh_local_button)
        layout.addLayout(local_header)

        local_hint = QLabel(
            "Если скачивание через Hugging Face не работает — скачайте модель в браузере "
            "вручную и добавьте её сюда кнопкой «Импортировать файл/папку»."
        )
        local_hint.setObjectName("statusLabel")
        local_hint.setWordWrap(True)
        layout.addWidget(local_hint)

        self.local_models_list = QListWidget()
        layout.addWidget(self.local_models_list, 1)

        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("statusLabel")
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

        self._on_recommend_models()
        self._on_list_local_models()
        return widget

    # ---- обработчики: чат ----------------------------------------------------------

    def _on_send(self) -> None:
        task = self.input_line.text().strip()
        if not task:
            return
        self.input_line.clear()
        self.chat_log.append(theme.user_message_html(task))
        self._set_busy(True)

        self._worker = AgentWorker(self.agent, task, parent=self)
        self._worker.finished_task.connect(self._on_task_finished)
        self._worker.failed.connect(self._on_task_failed)
        self._worker.start()

    def _on_task_finished(self, result: AgentResult) -> None:
        self.chat_log.append(theme.agent_message_html(result.final_text))
        for step in result.steps:
            self.chat_log.append(theme.tool_step_html(step.tool_name, step.arguments, step.result.ok))
        self._set_busy(False)
        self._refresh_memory_tab()

    def _on_task_failed(self, message: str) -> None:
        self.chat_log.append(theme.system_message_html(f"[Ошибка агента] {message}"))
        self._set_busy(False)

    def _on_new_chat(self) -> None:
        self.agent.reset_conversation()
        self.chat_log.clear()

    def _set_busy(self, busy: bool) -> None:
        self.input_line.setEnabled(not busy)
        self.send_button.setEnabled(not busy)

    def _refresh_memory_tab(self) -> None:
        stats = self.agent.memory.stats()
        lines = [
            f"Всего эпизодов: {stats['total_episodes']} | "
            f"успехов: {stats['successes']} | неудач: {stats['failures']}",
            "",
        ]
        for episode in self.agent.memory.recent(20):
            status = "успех" if episode.success else ("неудача" if episode.success is False else "?")
            lines.append(f"- [{status}] {episode.task}")
        self.memory_view.setPlainText("\n".join(lines))

    # ---- обработчики: настройки LLM -------------------------------------------------

    def _refresh_current_provider_label(self) -> None:
        provider_class = type(self.agent.llm).__name__
        names = {
            "AnthropicProvider": "Claude API",
            "LocalLlamaProvider": "локальная модель",
            "LMStudioProvider": "LM Studio",
            "EchoProvider": "тестовый режим (без ИИ)",
        }
        name = names.get(provider_class, provider_class)
        self.current_provider_label.setText(f"Сейчас отвечает: {name}")

    def _on_test_llm_connection(self) -> None:
        self.test_connection_button.setEnabled(False)
        self.settings_status_label.setText("Проверяю связь с провайдером…")
        self._llm_test_worker = LLMConnectionTestWorker(self.agent.llm, parent=self)
        self._llm_test_worker.finished_test.connect(self._on_llm_connection_test_finished)
        self._llm_test_worker.start()

    def _on_llm_connection_test_finished(self, ok: bool, message: str) -> None:
        self.test_connection_button.setEnabled(True)
        provider_class = type(self.agent.llm).__name__
        if ok:
            self.settings_status_label.setText(
                f"✅ Связь есть — {provider_class} ответил: «{message}»"
            )
        else:
            self.settings_status_label.setText(
                f"❌ Связи нет ({provider_class}): {message}"
            )
            log_hint = getattr(self.agent, "log_path", None)
            if log_hint:
                self.settings_status_label.setText(
                    self.settings_status_label.text() + f"\nПодробности в журнале: {log_hint}"
                )

    def _on_refresh_lmstudio_models(self) -> None:
        base_url = self.lmstudio_url_input.text().strip() or "http://localhost:1234/v1"
        api_key = self.lmstudio_api_key_input.text().strip()
        self.refresh_lmstudio_models_button.setEnabled(False)
        self.settings_status_label.setText("Запрашиваю список моделей у LM Studio…")
        self._lmstudio_models_worker = LMStudioModelsWorker(base_url, api_key=api_key, parent=self)
        self._lmstudio_models_worker.finished_models.connect(self._on_lmstudio_models_finished)
        self._lmstudio_models_worker.failed.connect(self._on_lmstudio_models_failed)
        self._lmstudio_models_worker.start()

    def _on_lmstudio_models_finished(self, models: list[str]) -> None:
        self.refresh_lmstudio_models_button.setEnabled(True)
        current = self.lmstudio_model_combo.currentText()
        self.lmstudio_model_combo.clear()
        self.lmstudio_model_combo.addItems(models)
        if current:
            idx = self.lmstudio_model_combo.findText(current)
            if idx >= 0:
                self.lmstudio_model_combo.setCurrentIndex(idx)
            else:
                self.lmstudio_model_combo.setEditText(current)
        if models:
            self.settings_status_label.setText(f"Обновлено: моделей в LM Studio — {len(models)}.")
        else:
            self.settings_status_label.setText(
                "LM Studio ответил, но не отдал ни одной модели — загрузите модель в приложении."
            )

    def _on_lmstudio_models_failed(self, message: str) -> None:
        self.refresh_lmstudio_models_button.setEnabled(True)
        text = f"❌ Не удалось получить список моделей LM Studio: {message}"
        log_hint = getattr(self.agent, "log_path", None)
        if log_hint:
            text += f"\nПодробности в журнале: {log_hint}"
        self.settings_status_label.setText(text)

    def _local_gguf_files(self) -> list[Path]:
        models_dir = self.agent.skill_context.workspace_root / "models"
        if not models_dir.exists():
            return []
        return sorted(models_dir.rglob("*.gguf"))

    def _refresh_local_model_combo(self) -> None:
        current = self.local_model_combo.currentText()
        self.local_model_combo.clear()
        files = self._local_gguf_files()
        for f in files:
            self.local_model_combo.addItem(str(f.relative_to(self.agent.skill_context.workspace_root)), str(f))
        if not files:
            self.local_model_combo.addItem("(в workspace/models нет .gguf-файлов)", "")
        elif current:
            idx = self.local_model_combo.findText(current)
            if idx >= 0:
                self.local_model_combo.setCurrentIndex(idx)
        # На первом вызове (при построении вкладки) этой метки ещё нет —
        # обратную связь показываем только по нажатию кнопки «Обновить».
        if hasattr(self, "settings_status_label"):
            self.settings_status_label.setText(
                f"Обновлено: найдено .gguf-файлов — {len(files)}." if files else "Обновлено: .gguf-файлов не найдено."
            )

    def _on_save_llm_settings(self) -> None:
        from ai_agent.core.llm_settings import DEFAULT_LMSTUDIO_BASE_URL, LLMSettings

        provider = self.provider_combo.currentData()
        local_path = self.local_model_combo.currentData() or self.local_model_combo.currentText().strip()
        settings = LLMSettings(
            provider=provider,
            anthropic_model=self.anthropic_model_input.text().strip() or "claude-sonnet-5",
            anthropic_api_key=self.anthropic_key_input.text().strip(),
            local_model_path=local_path,
            local_n_ctx=self.agent.llm_settings.local_n_ctx,
            lmstudio_base_url=self.lmstudio_url_input.text().strip() or DEFAULT_LMSTUDIO_BASE_URL,
            lmstudio_model=self.lmstudio_model_combo.currentText().strip(),
            lmstudio_api_key=self.lmstudio_api_key_input.text().strip(),
            system_prompt=self.system_prompt_edit.toPlainText(),
        )
        settings.save(self.agent.llm_settings_path)

        new_llm, error = build_llm_provider_safe(settings)
        self.agent.llm = new_llm
        self.agent.llm_settings = settings
        self.agent.llm_setup_error = error
        self.agent.system_prompt = combine_system_prompt(settings)

        self._refresh_current_provider_label()
        if error:
            self.settings_status_label.setText(
                f"⚠ Настройки сохранены, но провайдер не запустился (используется тестовый режим): {error}"
            )
        else:
            self.settings_status_label.setText("Настройки сохранены и применены — можно возвращаться в чат.")

    # ---- обработчики: права доступа / диагностика ------------------------------------

    def _on_open_log_file(self) -> None:
        log_path = self.agent.log_path
        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_path)))

    # ---- обработчики: модели (Hugging Face + локальные) ------------------------------

    def _on_toggle_model_download(self, checked: bool) -> None:
        self.agent.skill_context.policy.config.model_download_enabled = checked
        self.agent.skill_context.policy.config.save(self._policy_path)
        state = "включено" if checked else "выключено"
        self.model_status_label.setText(
            f"Скачивание моделей с Hugging Face {state} (сохранено в policy.yaml). "
            "Каждое скачивание всё равно попросит отдельное подтверждение."
        )

    def _refresh_hf_auth_status(self) -> None:
        # Локальная проверка (без сети) — есть ли уже сохранённый токен.
        token = hf_login_status()
        if token:
            self.hf_auth_status_label.setText("токен сохранён локально")
        else:
            self.hf_auth_status_label.setText("не авторизован (анонимный доступ)")

    def _on_open_hf_auth_page(self) -> None:
        QDesktopServices.openUrl(QUrl(HF_TOKENS_URL))

    def _on_save_hf_token(self) -> None:
        token = self.hf_token_input.text().strip()
        if not token:
            self.model_status_label.setText("Вставьте токен перед сохранением.")
            return
        self.model_status_label.setText("Проверяю токен…")
        self._run_model_skill("models.set_hf_token", {"token": token}, self._on_save_hf_token_finished)

    def _on_save_hf_token_finished(self, result: SkillResult) -> None:
        if not result.ok:
            self.model_status_label.setText(f"Токен не принят: {result.error}")
            return
        self.hf_token_input.clear()
        self._refresh_hf_auth_status()
        self.model_status_label.setText(result.output)

    def _on_hf_logout(self) -> None:
        self._run_model_skill("models.clear_hf_token", {}, self._on_hf_logout_finished)

    def _on_hf_logout_finished(self, result: SkillResult) -> None:
        if not result.ok:
            self.model_status_label.setText(f"Ошибка: {result.error}")
            return
        self._refresh_hf_auth_status()
        self.model_status_label.setText("Вы вышли из аккаунта Hugging Face.")

    def _run_model_skill(self, name: str, kwargs: dict, on_done) -> None:
        """Запускает навык в фоне. Несколько таких вызовов могут идти
        параллельно (например при открытии вкладки), поэтому кнопки
        разблокируются по счётчику, а не по последнему завершившемуся."""
        self._models_busy_count += 1
        self._set_models_busy(True)

        worker = SkillWorker(self.agent.skills, self.agent.skill_context, name, kwargs, parent=self)

        def _on_finished_skill(result: SkillResult) -> None:
            self._models_busy_count = max(0, self._models_busy_count - 1)
            if self._models_busy_count == 0:
                self._set_models_busy(False)
            on_done(result)

        def _on_failed(message: str) -> None:
            self._models_busy_count = max(0, self._models_busy_count - 1)
            if self._models_busy_count == 0:
                self._set_models_busy(False)
            self.model_status_label.setText(f"Ошибка: {message}")

        worker.finished_skill.connect(_on_finished_skill)
        worker.failed.connect(_on_failed)
        # Держим ссылку, пока поток жив, иначе Python может собрать объект раньше времени.
        self._model_workers.append(worker)
        worker.finished.connect(lambda: self._model_workers.remove(worker) if worker in self._model_workers else None)
        worker.start()

    def _on_recommend_models(self) -> None:
        self.model_status_label.setText("Подбираю модели под ваше железо…")
        self._run_model_skill("models.recommend", {}, self._on_recommend_finished)

    def _on_recommend_finished(self, result: SkillResult) -> None:
        if not result.ok:
            self.model_status_label.setText(f"Ошибка: {result.error}")
            return
        hw = result.data["hardware"]
        self.hw_summary_label.setText(
            f"🖥 CPU: {hw['cpu_cores']} ядер   💾 RAM: {hw['total_ram_gb']} ГБ   "
            f"🎮 GPU: {'есть' if hw['has_gpu'] else 'нет'}   "
            f"⚙ Профиль производительности: {self.agent.skill_context.profile.tier}"
        )
        self.recommend_list.clear()
        for m in result.data["models"]:
            item = QListWidgetItem(f"{m['label']} — ~{m['approx_size_gb']} ГБ ({m['quantization']})\n{m['notes']}")
            item.setData(Qt.UserRole, m["repo_id"])
            self.recommend_list.addItem(item)
        if result.data["models"]:
            self.model_status_label.setText(f"Найдено рекомендаций: {len(result.data['models'])}")
        else:
            self.model_status_label.setText("Подходящих моделей под текущее железо не нашлось.")

    def _on_test_hf_connection(self) -> None:
        self.model_status_label.setText("Проверяю связь с Hugging Face…")
        self._run_model_skill(
            "models.search_huggingface", {"query": "gguf", "max_results": 1}, self._on_test_hf_connection_finished
        )

    def _on_test_hf_connection_finished(self, result: SkillResult) -> None:
        if not result.ok:
            if result.data.get("auth_required"):
                self.model_status_label.setText(
                    "❌ Связь есть, но нужна авторизация на Hugging Face — откройте страницу токенов выше, "
                    "получите токен и сохраните его в разделе «Авторизация»."
                )
            else:
                text = f"❌ Связи с Hugging Face нет: {result.error}"
                log_hint = getattr(self.agent, "log_path", None)
                if log_hint:
                    text += f"\nПодробности в журнале: {log_hint}"
                self.model_status_label.setText(text)
            return
        self.model_status_label.setText("✅ Связь с Hugging Face есть — поиск моделей работает.")

    def _on_search_models(self) -> None:
        query = self.model_search_input.text().strip()
        if not query:
            return
        self.model_status_label.setText(f"Ищу «{query}» на Hugging Face…")
        self._run_model_skill("models.search_huggingface", {"query": query}, self._on_search_finished)

    def _on_search_finished(self, result: SkillResult) -> None:
        if not result.ok:
            if result.data.get("auth_required"):
                self.model_status_label.setText(
                    "Нужна авторизация на Hugging Face — откройте страницу токенов выше, "
                    "получите токен и сохраните его в разделе «Авторизация»."
                )
            else:
                self.model_status_label.setText(f"Ошибка поиска: {result.error}")
            return
        self.search_list.clear()
        for r in result.data["results"]:
            item = QListWidgetItem(f"{r['id']}   ⬇ {r['downloads'] or 0}   ♥ {r['likes'] or 0}")
            item.setData(Qt.UserRole, r["id"])
            self.search_list.addItem(item)
        self.model_status_label.setText(f"Найдено: {len(result.data['results'])}")

    def _on_download_selected(self, list_widget: QListWidget) -> None:
        item = list_widget.currentItem()
        if item is None:
            self.model_status_label.setText("Сначала выберите модель в списке.")
            return
        if not self.enable_download_checkbox.isChecked():
            self.model_status_label.setText(
                "Скачивание моделей выключено — включите галочку "
                "«Разрешить скачивание моделей с Hugging Face» выше."
            )
            return
        repo_id = item.data(Qt.UserRole)
        self.model_status_label.setText(
            f"Скачиваю {repo_id}… может занять время и потребует подтверждения."
        )
        self._run_model_skill("models.download_huggingface", {"repo_id": repo_id}, self._on_download_finished)

    def _on_download_finished(self, result: SkillResult) -> None:
        if not result.ok:
            if result.data.get("auth_required"):
                self.model_status_label.setText(
                    "Эта модель требует авторизации на Hugging Face (закрытый доступ/лицензия) — "
                    "откройте страницу токенов выше, получите токен и сохраните его в разделе «Авторизация», "
                    "затем повторите скачивание."
                )
            else:
                self.model_status_label.setText(f"Ошибка скачивания: {result.error}")
            return
        self.model_status_label.setText(result.output)
        self._on_list_local_models()  # скачанное сразу появится в разделе локальных моделей

    def _on_list_local_models(self) -> None:
        self._run_model_skill("models.list_local", {}, self._on_list_local_finished)

    def _on_list_local_finished(self, result: SkillResult) -> None:
        if not result.ok:
            self.model_status_label.setText(f"Ошибка: {result.error}")
            return
        self.local_models_list.clear()
        for m in result.data["models"]:
            icon = "📁" if m["is_dir"] else "📄"
            self.local_models_list.addItem(f"{icon} {m['name']} — ~{m['size_gb']} ГБ")
        count = len(result.data["models"])
        self.model_status_label.setText(
            f"Обновлено: локальных моделей — {count}." if count else "Обновлено: локальных моделей нет."
        )

    def _on_import_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл модели")
        if not path:
            return
        self._import_path(path)

    def _on_import_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите папку с моделью")
        if not path:
            return
        self._import_path(path)

    def _import_path(self, path: str) -> None:
        self.model_status_label.setText(f"Копирую «{path}» в workspace/models…")
        self._run_model_skill("models.import_local", {"source_path": path}, self._on_import_finished)

    def _on_import_finished(self, result: SkillResult) -> None:
        if not result.ok:
            self.model_status_label.setText(f"Ошибка импорта: {result.error}")
            return
        self.model_status_label.setText(result.output)
        self._on_list_local_models()

    def _set_models_busy(self, busy: bool) -> None:
        for w in (
            self.recommend_button,
            self.search_button,
            self.download_recommend_button,
            self.download_search_button,
            self.refresh_local_button,
            self.import_file_button,
            self.import_folder_button,
            self.model_search_input,
            self.save_hf_token_button,
            self.hf_token_input,
            self.logout_hf_button,
            self.test_hf_button,
        ):
            w.setEnabled(not busy)

    # ---- обработчики: диктофон -------------------------------------------------------

    def _on_toggle_mic_enabled(self, checked: bool) -> None:
        self.agent.skill_context.policy.config.microphone_enabled = checked
        self.agent.skill_context.policy.config.save(self._policy_path)

    def _on_toggle_mic_continuous_enabled(self, checked: bool) -> None:
        self.agent.skill_context.policy.config.microphone_continuous_enabled = checked
        self.agent.skill_context.policy.config.save(self._policy_path)

    def _on_toggle_mic_retain_audio(self, checked: bool) -> None:
        self.agent.skill_context.policy.config.microphone_retain_audio = checked
        self.agent.skill_context.policy.config.save(self._policy_path)

    def _run_voice_skill(self, name: str, kwargs: dict, on_done) -> None:
        """Как _run_model_skill: несколько таких вызовов могут идти
        параллельно, поэтому кнопки разблокируются по счётчику."""
        self._voice_busy_count += 1
        self._set_voice_busy(True)

        worker = SkillWorker(self.agent.skills, self.agent.skill_context, name, kwargs, parent=self)

        def _on_finished_skill(result: SkillResult) -> None:
            self._voice_busy_count = max(0, self._voice_busy_count - 1)
            if self._voice_busy_count == 0:
                self._set_voice_busy(False)
            on_done(result)

        def _on_failed(message: str) -> None:
            self._voice_busy_count = max(0, self._voice_busy_count - 1)
            if self._voice_busy_count == 0:
                self._set_voice_busy(False)
            self.voice_status_label.setText(f"Ошибка: {message}")

        worker.finished_skill.connect(_on_finished_skill)
        worker.failed.connect(_on_failed)
        self._voice_workers.append(worker)
        worker.finished.connect(lambda: self._voice_workers.remove(worker) if worker in self._voice_workers else None)
        worker.start()

    def _set_voice_busy(self, busy: bool) -> None:
        for w in (
            self.record_once_button,
            self.create_task_button,
            self.set_reminder_button,
            self.install_voice_button,
            self.refresh_devices_button,
        ):
            w.setEnabled(not busy)

    def _on_record_once(self) -> None:
        self.voice_status_label.setText("Запись… (остановится по паузе в речи или через 30 секунд).")
        self._run_voice_skill("voice.record_once", {"max_seconds": 30.0}, self._on_voice_action_done)

    def _on_toggle_continuous(self) -> None:
        voice = self.agent.skill_context.voice
        active = voice is not None and voice.is_continuous_active
        skill = "voice.stop_continuous" if active else "voice.start_continuous"
        self._run_voice_skill(skill, {}, self._on_voice_action_done)

    def _on_create_task(self) -> None:
        title = self.new_task_title_input.text().strip()
        if not title:
            self.voice_status_label.setText("Введите название задачи.")
            return
        description = self.new_task_desc_input.text().strip()
        self._run_voice_skill(
            "tasks.create", {"title": title, "description": description}, self._on_task_created_done
        )

    def _on_task_created_done(self, result: SkillResult) -> None:
        if result.ok:
            self.new_task_title_input.clear()
            self.new_task_desc_input.clear()
        self._on_voice_action_done(result)

    def _on_set_reminder(self) -> None:
        task_index = self.confirm_task_combo.currentIndex()
        if task_index < 0:
            self.voice_status_label.setText("Сначала создайте и выберите задачу в списке ниже.")
            return
        task_id = self.confirm_task_combo.itemData(task_index)
        try:
            minutes = float(self.reminder_minutes_input.text().strip())
        except ValueError:
            self.voice_status_label.setText("Введите число минут для напоминания.")
            return
        fire_at = time.time() + minutes * 60
        self._run_voice_skill(
            "tasks.set_reminder", {"task_id": task_id, "fire_at": fire_at}, self._on_voice_action_done
        )

    def _on_confirm_utterance(self, belongs: bool) -> None:
        utterance_item = self.utterances_list.currentItem()
        task_index = self.confirm_task_combo.currentIndex()
        if utterance_item is None or task_index < 0:
            self.voice_status_label.setText("Выберите реплику в списке и задачу в выпадающем списке.")
            return
        utterance_id = utterance_item.data(Qt.UserRole)
        task_id = self.confirm_task_combo.itemData(task_index)
        self._run_voice_skill(
            "voice.confirm_task",
            {"utterance_id": utterance_id, "task_id": task_id, "belongs": belongs},
            self._on_voice_action_done,
        )

    def _on_generate_protocol(self) -> None:
        self._run_voice_skill("voice.generate_protocol", {"filename": "protocol.docx"}, self._on_voice_action_done)

    def _on_voice_action_done(self, result: SkillResult) -> None:
        self.voice_status_label.setText(result.output if result.ok else f"Ошибка: {result.error}")
        self._refresh_tasks_and_utterances()
        self._refresh_continuous_status()

    def _refresh_continuous_status(self) -> None:
        voice = self.agent.skill_context.voice
        active = voice is not None and voice.is_continuous_active
        if active:
            self.continuous_status_label.setText("🔴 Идёт постоянная запись.")
            self.continuous_button.setText("⏹ Выключить постоянную запись")
        else:
            self.continuous_status_label.setText("Постоянная запись выключена.")
            self.continuous_button.setText("⏺ Включить постоянную запись")

    def _refresh_tasks_and_utterances(self) -> None:
        voice = self.agent.skill_context.voice
        self.tasks_list.clear()
        self.confirm_task_combo.clear()
        self.utterances_list.clear()
        if voice is None:
            return

        for task in voice.store.list_tasks():
            profile = voice.classifier.profiles.get(task.id)
            confidence = f"{profile.confidence:.0%}" if profile else "—"
            self.tasks_list.addItem(f"#{task.id} «{task.title}» — уверенность классификатора: {confidence}")
            self.confirm_task_combo.addItem(f"#{task.id} «{task.title}»", task.id)

        for utterance in voice.store.list_utterances()[-50:]:
            label = utterance.text + (f" → задача #{utterance.task_id}" if utterance.task_id else " → не отнесено")
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, utterance.id)
            self.utterances_list.addItem(item)

    def _on_check_reminders(self) -> None:
        voice = self.agent.skill_context.voice
        if voice is None:
            return
        for due in self._reminder_checker.check():
            self.chat_log.append(
                theme.system_message_html(f"⏰ Напоминание по задаче «{due.task.title}»: {due.reminder.message}")
            )

    def _on_level_tick(self) -> None:
        voice = self.agent.skill_context.voice
        level = voice.current_level if voice is not None else 0.0
        level = min(max(level, 0.0), 1.0)
        self.level_meter.setValue(int(level * 100))
        color = theme.level_meter_color(level)
        self.level_meter.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }}")

    def _on_refresh_input_devices(self) -> None:
        self._run_voice_skill("voice.list_input_devices", {}, self._on_input_devices_listed)

    def _on_input_devices_listed(self, result: SkillResult) -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("Устройство по умолчанию", None)
        if result.ok:
            for d in result.data.get("devices", []):
                label = d["name"] + (" (по умолчанию)" if d["is_default"] else "")
                self.device_combo.addItem(label, d["index"])
            settings = getattr(self.agent, "voice_settings", None)
            selected = settings.input_device if settings is not None else None
            idx = self.device_combo.findData(selected)
            self.device_combo.setCurrentIndex(idx if idx >= 0 else 0)
            if len(result.data.get("devices", [])) == 0:
                self.voice_status_label.setText("Устройства записи не найдены.")
        else:
            self.voice_status_label.setText(f"Не удалось получить список микрофонов: {result.error}")
        self.device_combo.blockSignals(False)

    def _on_device_changed(self, index: int) -> None:
        if index < 0:
            return
        device_index = self.device_combo.itemData(index)
        voice = self.agent.skill_context.voice
        if voice is not None:
            voice.set_input_device(device_index)
        settings = getattr(self.agent, "voice_settings", None)
        if settings is not None:
            settings.input_device = device_index
            settings.save(self.agent.voice_settings_path)

    def _on_check_voice_setup(self) -> None:
        self._run_voice_skill("voice.check_setup", {}, self._on_voice_setup_checked)

    def _on_voice_setup_checked(self, result: SkillResult) -> None:
        if not result.ok:
            self.setup_status_label.setText(f"Не удалось проверить компоненты: {result.error}")
            return
        self.setup_status_label.setText(result.output)
        if result.data.get("ready"):
            self.install_voice_button.setText("✅ Компоненты установлены (переустановить)")
        else:
            self.install_voice_button.setText("⬇ Установить / обновить компоненты")

    def _on_install_voice_dependencies(self) -> None:
        self.voice_status_label.setText("Устанавливаю компоненты диктофона…")
        self._run_voice_skill("voice.install_dependencies", {}, self._on_voice_dependencies_installed)

    def _on_voice_dependencies_installed(self, result: SkillResult) -> None:
        if not result.ok:
            self.voice_status_label.setText(f"Ошибка установки: {result.error}")
            return
        self.voice_status_label.setText(f"{result.output} Загружаю модель распознавания речи под ваш ПК…")
        self._run_voice_skill("voice.download_stt_model", {}, self._on_voice_stt_model_downloaded)

    def _on_voice_stt_model_downloaded(self, result: SkillResult) -> None:
        if result.ok:
            settings = getattr(self.agent, "voice_settings", None)
            model_size = result.data.get("model_size", "")
            if settings is not None and model_size:
                settings.stt_model_size = model_size
                settings.save(self.agent.voice_settings_path)
        self._on_voice_action_done(result)
        self._on_check_voice_setup()
