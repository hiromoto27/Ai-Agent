"""Провайдер для локального сервера LM Studio.

LM Studio — отдельное приложение с собственным интерфейсом для поиска,
скачивания и запуска GGUF-моделей, которое (в отличие от
``LocalLlamaProvider``, грузящего модель прямо в процесс агента через
``llama-cpp-python``, включая необходимость компилятора C++) отдаёт
модель через собственный локальный HTTP-сервер, совместимый по API с
OpenAI Chat Completions (Settings → Developer → Enable Local Server,
по умолчанию ``http://localhost:1234/v1``). Пользователю достаточно
запустить сервер и загрузить модель в самом LM Studio — агенту нужен
только адрес сервера.

В отличие от ``LocalLlamaProvider`` (свой текстовый протокол вызова
инструментов) здесь используется нативный tool calling из OpenAI-формата
API — LM Studio сам транслирует его в то, что понимает загруженная
модель (для моделей, которые это поддерживают).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from ai_agent.core.logging_setup import get_logger

from .base import LLMProvider, LLMResponse, Message, ToolCall

logger = get_logger("llm.lmstudio")

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 120.0

_HINT = (
    "Убедитесь, что в LM Studio включён локальный сервер "
    "(Developer → Enable Local Server) и загружена модель."
)
_AUTH_HINT = (
    "LM Studio отклонил запрос как неавторизованный (в Developer → Local Server включено "
    "«Require API Key»). Укажите тот же ключ в настройках агента в поле «API-ключ LM Studio»."
)


class LMStudioConnectionError(ConnectionError):
    """Не удалось обратиться к локальному серверу LM Studio."""


class HttpClient(Protocol):
    def get(self, path: str) -> Any: ...
    def post(self, path: str, *, json: dict) -> Any: ...


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _default_http_client(base_url: str, timeout: float, api_key: str = "") -> HttpClient:
    try:
        import httpx
    except ImportError as e:
        raise ImportError("Пакет 'httpx' не установлен. Установите extras: pip install -e '.[web]'") from e
    return httpx.Client(base_url=base_url, timeout=timeout, headers=_auth_headers(api_key))


def list_models(
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 10.0,
    api_key: str = "",
    http_client: Optional[HttpClient] = None,
) -> list[str]:
    """GET /models — идентификаторы моделей, которые LM Studio отдаёт прямо сейчас."""
    owns_client = http_client is None
    client = http_client or _default_http_client(base_url, timeout, api_key)
    try:
        response = client.get("/models")
        if getattr(response, "status_code", 200) in (401, 403):
            raise LMStudioConnectionError(_AUTH_HINT)
        response.raise_for_status()
    except LMStudioConnectionError:
        logger.exception("Не удалось получить список моделей LM Studio (%s)", base_url)
        raise
    except Exception as e:
        logger.exception("Не удалось получить список моделей LM Studio (%s)", base_url)
        raise LMStudioConnectionError(f"не удалось получить список моделей от LM Studio ({base_url}): {e}. {_HINT}") from e
    finally:
        if owns_client:
            client.close()
    data = response.json()
    return [m["id"] for m in data.get("data", [])]


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in tools
    ]


def _to_openai_messages(messages: list[Message], system: str) -> list[dict]:
    chat: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m.role == "tool":
            chat.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            chat.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        else:
            chat.append({"role": m.role, "content": m.content})
    return chat


class LMStudioProvider(LLMProvider):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        api_key: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: Optional[HttpClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") or DEFAULT_BASE_URL
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Соединение не проверяется здесь — конструктор явного выбора
        # провайдера не должен блокироваться на сетевом запросе (сервер
        # LM Studio может быть ещё не запущен на момент старта агента).
        # Проверка реальной доступности — кнопка "Проверить подключение"
        # в настройках, либо первый реальный вызов complete().
        self._client = http_client or _default_http_client(self.base_url, timeout, api_key)

    def complete(self, messages: list[Message], tools: list[dict], system: str = "") -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model or "local-model",
            "messages": _to_openai_messages(messages, system),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = _to_openai_tools(tools)

        try:
            response = self._client.post("/chat/completions", json=payload)
            if getattr(response, "status_code", 200) in (401, 403):
                raise LMStudioConnectionError(_AUTH_HINT)
            response.raise_for_status()
        except LMStudioConnectionError:
            raise
        except Exception as e:
            # Не логируем здесь: complete() всегда вызывается либо из
            # Agent.run_task(), либо из test_llm_connection() — оба уже
            # пишут traceback в лог сами, дублировать незачем.
            raise LMStudioConnectionError(f"не удалось обратиться к LM Studio ({self.base_url}): {e}. {_HINT}") from e

        return self._convert_response(response.json())

    @staticmethod
    def _convert_response(data: dict) -> LLMResponse:
        choice = data["choices"][0]
        message = choice["message"]
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        for tc in raw_tool_calls:
            fn = tc["function"]
            try:
                arguments = json.loads(fn["arguments"]) if fn.get("arguments") else {}
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=arguments))

        if tool_calls:
            stop_reason = "tool_use"
        elif choice.get("finish_reason") == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return LLMResponse(content=message.get("content") or "", tool_calls=tool_calls, stop_reason=stop_reason)
