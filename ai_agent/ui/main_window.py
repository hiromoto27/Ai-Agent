"""Главное окно десктоп-приложения (чат, навыки, память, права доступа)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai_agent.app import build_agent
from ai_agent.core.orchestrator import AgentResult

from .confirm_bridge import ConfirmBridge
from .worker import AgentWorker


class MainWindow(QMainWindow):
    def __init__(self, state_dir: Path | None = None, workspace_root: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Локальный ИИ-агент")
        self.resize(900, 620)

        self.confirm_bridge = ConfirmBridge(self)
        kwargs = {}
        if state_dir is not None:
            kwargs["state_dir"] = state_dir
        if workspace_root is not None:
            kwargs["workspace_root"] = workspace_root
        self.agent = build_agent(confirm_callback=self.confirm_bridge.confirm_callback, **kwargs)

        self._worker: AgentWorker | None = None
        self._build_ui()

    # ---- построение интерфейса --------------------------------------------------

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_chat_tab(), "Чат")
        tabs.addTab(self._build_skills_tab(), "Навыки")
        tabs.addTab(self._build_memory_tab(), "Память")
        tabs.addTab(self._build_permissions_tab(), "Права доступа")
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
            f"workspace_only: {config.workspace_only}\n"
            f"network.enabled: {config.network_enabled}\n\n"
            f"Файл: {policy_path}\n"
            "Отредактируйте его и перезапустите приложение, чтобы изменить права.\n"
            "Действия, требующие подтверждения, покажут диалог во время выполнения задачи."
        )
        layout.addWidget(view)
        return widget

    # ---- обработчики --------------------------------------------------------------

    def _on_send(self) -> None:
        task = self.input_line.text().strip()
        if not task:
            return
        self.input_line.clear()
        self._append_chat(f"Вы: {task}")
        self._set_busy(True)

        self._worker = AgentWorker(self.agent, task, parent=self)
        self._worker.finished_task.connect(self._on_task_finished)
        self._worker.failed.connect(self._on_task_failed)
        self._worker.start()

    def _on_task_finished(self, result: AgentResult) -> None:
        self._append_chat(f"Агент: {result.final_text}")
        for step in result.steps:
            status = "OK" if step.result.ok else "ОШИБКА"
            self._append_chat(f"    [{status}] {step.tool_name}({step.arguments})")
        self._set_busy(False)
        self._refresh_memory_tab()

    def _on_task_failed(self, message: str) -> None:
        self._append_chat(f"[Ошибка агента] {message}")
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self.input_line.setEnabled(not busy)
        self.send_button.setEnabled(not busy)

    def _append_chat(self, text: str) -> None:
        self.chat_log.append(text)

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
