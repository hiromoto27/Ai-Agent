"""Детерминированный офлайн-провайдер для разработки и автотестов —
не требует сети/API-ключа. Понимает простой протокол в тексте
пользователя, что позволяет тестировать полный цикл оркестратора
(включая вызовы инструментов) без реальной LLM.

Протокол:
- Если сообщение пользователя начинается с ``TOOL: <имя> <json-аргументы>``,
  провайдер эмитирует вызов этого инструмента.
- После получения результата инструмента (role="tool") провайдер завершает
  ход финальным текстовым ответом.
- Любое другое сообщение — просто эхо-ответ финальным текстом.
"""

from __future__ import annotations

import json

from .base import LLMProvider, LLMResponse, Message, ToolCall

_TOOL_PREFIX = "TOOL:"


class EchoProvider(LLMProvider):
    def __init__(self) -> None:
        self._call_counter = 0

    def complete(self, messages: list[Message], tools: list[dict], system: str = "") -> LLMResponse:
        if not messages:
            return LLMResponse(content="(пусто)")

        last = messages[-1]

        if last.role == "tool":
            return LLMResponse(
                content=f"Готово. Результат последнего действия: {last.content}",
                stop_reason="end_turn",
            )

        if last.role == "user" and last.content.strip().startswith(_TOOL_PREFIX):
            remainder = last.content.strip()[len(_TOOL_PREFIX):].strip()
            name, _, raw_args = remainder.partition(" ")
            try:
                arguments = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                arguments = {}
            self._call_counter += 1
            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id=f"echo_call_{self._call_counter}", name=name, arguments=arguments)],
                stop_reason="tool_use",
            )

        return LLMResponse(content=f"Понял задачу: {last.content}", stop_reason="end_turn")
