"""Абстракция поставщика LLM — общий интерфейс, за которым скрыты
конкретные провайдеры (Claude API, локальная модель, тестовый Echo)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

StopReason = Literal["end_turn", "tool_use", "max_tokens"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = "end_turn"


class LLMProvider(ABC):
    @abstractmethod
    def complete(
        self, messages: list[Message], tools: list[dict], system: str = ""
    ) -> LLMResponse:
        raise NotImplementedError
