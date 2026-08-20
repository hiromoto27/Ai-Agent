"""Провайдер на базе Claude API (Anthropic). Требует пакет `anthropic`
(extras: llm) и переменную окружения ANTHROPIC_API_KEY (либо явный ключ)."""

from __future__ import annotations

import os

from .base import LLMProvider, LLMResponse, Message, ToolCall

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Пакет 'anthropic' не установлен. Установите extras: pip install -e '.[llm]'"
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Не задан ANTHROPIC_API_KEY. Укажите ключ в настройках приложения или переменной окружения."
            )

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model

    def complete(self, messages: list[Message], tools: list[dict], system: str = "") -> LLMResponse:
        kwargs = dict(
            model=self.model,
            max_tokens=DEFAULT_MAX_TOKENS,
            messages=self._convert_messages(messages),
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)
        return self._convert_response(response)

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                content: list[dict] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    @staticmethod
    def _convert_response(response) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        if tool_calls:
            stop_reason = "tool_use"
        elif response.stop_reason == "max_tokens":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        return LLMResponse(content="".join(text_parts), tool_calls=tool_calls, stop_reason=stop_reason)
