"""Оформление интерфейса: тёмная тема (QSS) + рендер сообщений чата в виде
"пузырей" и подписи вкладок с иконками-эмодзи (без внешних ассетов —
надёжно переживает упаковку в .exe без лишних data-файлов)."""

from __future__ import annotations

import html

BG = "#1e1f26"
BG_PANEL = "#262832"
BG_INPUT = "#2f3140"
BORDER = "#3a3d4d"
TEXT = "#e6e6ec"
TEXT_DIM = "#9a9cad"
ACCENT = "#7c6cf0"
ACCENT_HOVER = "#8f81f5"
ACCENT_PRESSED = "#6b5ce0"
USER_BUBBLE = "#4b3f9e"
AGENT_BUBBLE = "#31333f"
OK_COLOR = "#4fd18b"
ERROR_COLOR = "#f16565"

TAB_TITLES = {
    "chat": "💬  Чат",
    "settings": "⚙  Настройки",
    "skills": "🧩  Навыки",
    "memory": "🧠  Память",
    "permissions": "🔒  Права доступа",
    "models": "🤗  Модели",
    "voice": "🎙  Диктофон",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: "Segoe UI", "SF Pro Text", "Ubuntu", sans-serif;
    font-size: 14px;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background-color: {BG_PANEL};
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_DIM};
    padding: 10px 18px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}

QTabBar::tab:selected {{
    background-color: {BG_PANEL};
    color: {TEXT};
    font-weight: 600;
}}

QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}

QTextEdit, QListWidget {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 8px;
    selection-background-color: {ACCENT};
}}

QListWidget::item {{
    padding: 8px;
    border-radius: 6px;
    margin: 2px 0;
}}

QListWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}

QLineEdit {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 9px 12px;
    color: {TEXT};
}}

QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    color: {TEXT};
}}

QComboBox:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
}}

QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
}}

QPushButton:disabled {{
    background-color: {BORDER};
    color: {TEXT_DIM};
}}

QPushButton#secondary {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid {BORDER};
}}

QPushButton#secondary:hover {{
    background-color: {BORDER};
}}

QCheckBox {{
    spacing: 8px;
    padding: 4px 0;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background-color: {BG_INPUT};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
}}

QGroupBox {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 16px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 2px;
    padding: 0 6px;
    color: {TEXT};
    background-color: {BG_PANEL};
}}

QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 7px;
    text-align: center;
    color: {TEXT_DIM};
    height: 14px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 6px;
}}

QLabel#hwSummary {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
}}

QLabel#statusLabel {{
    color: {TEXT_DIM};
    padding: 4px 2px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


def level_meter_color(level: float) -> str:
    """Цвет полосы индикатора уровня сигнала микрофона: зелёный — нормально,
    жёлтый — громко, красный — вероятное клиппирование (0.0..1.0)."""
    if level >= 0.9:
        return ERROR_COLOR
    if level >= 0.6:
        return "#e0c341"
    return OK_COLOR


def _bubble(text: str, *, align: str, bg: str, label: str = "") -> str:
    safe = html.escape(text).replace("\n", "<br>")
    prefix = f"<b>{label}:</b> " if label else ""
    return (
        f'<div align="{align}" style="margin:6px 0;">'
        f'<table cellpadding="0" cellspacing="0" style="display:inline;"><tr><td '
        f'style="background-color:{bg}; color:{TEXT}; border-radius:10px; padding:8px 12px;">'
        f"{prefix}{safe}</td></tr></table></div>"
    )


def user_message_html(text: str) -> str:
    return _bubble(text, align="right", bg=USER_BUBBLE, label="Вы")


def agent_message_html(text: str) -> str:
    return _bubble(text, align="left", bg=AGENT_BUBBLE, label="Агент")


def tool_step_html(tool_name: str, arguments: dict, ok: bool) -> str:
    color = OK_COLOR if ok else ERROR_COLOR
    badge = "OK" if ok else "ОШИБКА"
    args = html.escape(str(arguments))
    return (
        '<div align="left" style="margin:2px 0 8px 0;">'
        f'<span style="font-family:Consolas,monospace; font-size:12px; color:{color};">'
        f"&nbsp;&nbsp;[{badge}] {html.escape(tool_name)}({args})</span></div>"
    )


def system_message_html(text: str) -> str:
    safe = html.escape(text)
    return f'<div align="center" style="margin:6px 0; color:{ERROR_COLOR}; font-size:12px;">{safe}</div>'
