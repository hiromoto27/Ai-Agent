"""Главное окно десктоп-приложения (чат, навыки, память, права доступа, модели)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai_agent.app import build_agent
from ai_agent.core.orchestrator import AgentResult
from ai_agent.core.skills.base import SkillResult

from . import theme
from .confirm_bridge import ConfirmBridge
from .skill_worker import SkillWorker
from .worker import AgentWorker


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

        self._worker: AgentWorker | None = None
        self._model_workers: list[SkillWorker] = []
        self._build_ui()

    # ---- построение интерфейса --------------------------------------------------

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_chat_tab(), theme.TAB_TITLES["chat"])
        tabs.addTab(self._build_models_tab(), theme.TAB_TITLES["models"])
        tabs.addTab(self._build_skills_tab(), theme.TAB_TITLES["skills"])
        tabs.addTab(self._build_memory_tab(), theme.TAB_TITLES["memory"])
        tabs.addTab(self._build_permissions_tab(), theme.TAB_TITLES["permissions"])
        self.setCentralWidget(tabs)

    def _build_chat_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

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
        view = QTextEdit()
        view.setReadOnly(True)
        config = self.agent.skill_context.policy.config
        policy_path = self.agent.skill_context.policy.audit_log_path.parent / "policy.yaml"
        view.setPlainText(
            "Текущая политика прав доступа (config/policy.yaml):\n\n"
            f"shell.enabled: {config.shell_enabled}\n"
            f"scripting.execute_enabled: {config.scripting_execute_enabled}\n"
            f"package_install.enabled: {config.package_install_enabled}\n"
            f"model_download.enabled: {config.model_download_enabled}\n"
            f"workspace_only: {config.workspace_only}\n"
            f"network.enabled: {config.network_enabled}\n\n"
            f"Файл: {policy_path}\n"
            "Отредактируйте его и перезапустите приложение, чтобы изменить права.\n"
            "Действия, требующие подтверждения, покажут диалог во время выполнения задачи."
        )
        layout.addWidget(view)
        return widget

    def _build_models_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.hw_summary_label = QLabel("Определяю характеристики ПК…")
        self.hw_summary_label.setObjectName("hwSummary")
        self.hw_summary_label.setWordWrap(True)
        layout.addWidget(self.hw_summary_label)

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

        self.model_status_label = QLabel("")
        self.model_status_label.setObjectName("statusLabel")
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

        self._on_recommend_models()
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

    # ---- обработчики: модели (Hugging Face) -----------------------------------------

    def _run_model_skill(self, name: str, kwargs: dict, on_done) -> None:
        self._set_models_busy(True)
        worker = SkillWorker(self.agent.skills, self.agent.skill_context, name, kwargs, parent=self)
        worker.finished_skill.connect(on_done)
        worker.failed.connect(self._on_models_failed)
        # Держим ссылку, пока поток жив, иначе Python может собрать объект раньше времени.
        self._model_workers.append(worker)
        worker.finished.connect(lambda: self._model_workers.remove(worker) if worker in self._model_workers else None)
        worker.start()

    def _on_recommend_models(self) -> None:
        self.model_status_label.setText("Подбираю модели под ваше железо…")
        self._run_model_skill("models.recommend", {}, self._on_recommend_finished)

    def _on_recommend_finished(self, result: SkillResult) -> None:
        self._set_models_busy(False)
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

    def _on_search_models(self) -> None:
        query = self.model_search_input.text().strip()
        if not query:
            return
        self.model_status_label.setText(f"Ищу «{query}» на Hugging Face…")
        self._run_model_skill("models.search_huggingface", {"query": query}, self._on_search_finished)

    def _on_search_finished(self, result: SkillResult) -> None:
        self._set_models_busy(False)
        if not result.ok:
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
        repo_id = item.data(Qt.UserRole)
        self.model_status_label.setText(
            f"Скачиваю {repo_id}… может занять время и потребует подтверждения."
        )
        self._run_model_skill("models.download_huggingface", {"repo_id": repo_id}, self._on_download_finished)

    def _on_download_finished(self, result: SkillResult) -> None:
        self._set_models_busy(False)
        if not result.ok:
            self.model_status_label.setText(f"Ошибка скачивания: {result.error}")
            return
        self.model_status_label.setText(result.output)

    def _on_models_failed(self, message: str) -> None:
        self._set_models_busy(False)
        self.model_status_label.setText(f"Ошибка: {message}")

    def _set_models_busy(self, busy: bool) -> None:
        for w in (
            self.recommend_button,
            self.search_button,
            self.download_recommend_button,
            self.download_search_button,
            self.model_search_input,
        ):
            w.setEnabled(not busy)
