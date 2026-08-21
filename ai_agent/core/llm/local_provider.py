"""Провайдер на локальной GGUF-модели через ``llama-cpp-python``.

В отличие от Claude API, произвольные локальные модели (особенно
маленькие квантованные) ненадёжно поддерживают "нативный" протокол
function calling — он зависит от конкретного шаблона чата, зашитого в
GGUF, и версии llama.cpp. Поэтому вместо этого используется текстовый
протокол в духе ReAct: модели через системный промпт объясняется, что
для вызова инструмента нужно ответить строкой
``TOOL_CALL: <имя> <JSON-аргументы>``, а обычный ответ — просто текст.
Это работает с любой инструкт-моделью, а не только с теми, что обучены
на конкретном формате function calling.

Экспериментальный путь: качество соблюдения протокола зависит от
конкретной модели (крупные/новые следуют инструкциям надёжнее мелких).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Protocol

from .base import LLMProvider, LLMResponse, Message, ToolCall

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3

_TOOL_CALL_RE = re.compile(r"^\s*TOOL_CALL\s*:\s*(\S+)\s*(\{.*\})?\s*$", re.IGNORECASE | re.DOTALL)


class LlamaClient(Protocol):
    def create_chat_completion(self, *, messages: list[dict], max_tokens: int, temperature: float) -> dict: ...


def _default_client(model_path: str, n_ctx: int, n_threads: Optional[int]) -> LlamaClient:
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise ImportError(
            "Пакет 'llama-cpp-python' не установлен. Установите extras: "
            "pip install -e '.[local-llm]' (требует компилятор C++ на большинстве систем — "
            "готовых wheel-пакетов для всех платформ на PyPI нет)."
        ) from e

    kwargs: dict[str, Any] = {"model_path": model_path, "n_ctx": n_ctx, "verbose": False}
    if n_threads:
        kwargs["n_threads"] = n_threads
    return Llama(**kwargs)


def _format_tools_block(tools: list[dict]) -> str:
    if not tools:
        return ""
    lines = ["Доступные инструменты:"]
    for t in tools:
        props = t.get("input_schema", {}).get("properties", {})
        params = ", ".join(f"{name} ({info.get('type', 'any')})" for name, info in props.items()) or "без параметров"
        lines.append(f"- {t['name']}: {t['description']} Параметры: {params}")
    lines.append(
        "\nЕсли для ответа нужен инструмент — ответь ОДНОЙ строкой в формате:\n"
        "TOOL_CALL: <имя_инструмента> <JSON с аргументами>\n"
        'Например: TOOL_CALL: files.write {"path": "note.txt", "content": "привет"}\n'
        "Ничего больше в эту строку не добавляй. Если инструмент не нужен — "
        "ответь пользователю обычным текстом, без TOOL_CALL."
    )
    return "\n".join(lines)


def _to_chat_messages(messages: list[Message], system: str) -> list[dict]:
    chat: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m.role == "tool":
            chat.append({"role": "user", "content": f"[Результат инструмента {m.tool_name}]: {m.content}"})
        elif m.role == "assistant" and m.tool_calls:
            calls_text = "\n".join(
                f"TOOL_CALL: {tc.name} {json.dumps(tc.arguments, ensure_ascii=False)}" for tc in m.tool_calls
            )
            chat.append({"role": "assistant", "content": calls_text})
        else:
            chat.append({"role": m.role, "content": m.content})
    return chat


class LocalLlamaProvider(LLMProvider):
    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        client: Optional[LlamaClient] = None,
    ) -> None:
        model_path = Path(model_path)
        if client is None and not model_path.exists():
            raise ValueError(f"файл модели не найден: {model_path}")

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client or _default_client(str(model_path), n_ctx, n_threads)
        self._call_counter = 0

    def complete(self, messages: list[Message], tools: list[dict], system: str = "") -> LLMResponse:
        full_system = "\n\n".join(part for part in (system, _format_tools_block(tools)) if part)
        chat_messages = _to_chat_messages(messages, full_system)

        response = self._client.create_chat_completion(
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response["choices"][0]["message"]["content"] or ""

        match = _TOOL_CALL_RE.match(content.strip())
        if match:
            name = match.group(1)
            raw_args = match.group(2)
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                # Модель не соблюла формат — безопаснее вернуть как обычный текст,
                # чем угадывать аргументы вызова инструмента.
                return LLMResponse(content=content.strip(), stop_reason="end_turn")
            self._call_counter += 1
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id=f"local_call_{self._call_counter}", name=name, arguments=arguments)],
                stop_reason="tool_use",
            )

        return LLMResponse(content=content.strip(), stop_reason="end_turn")
