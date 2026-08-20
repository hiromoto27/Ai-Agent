import pytest

from ai_agent.core.llm.anthropic_provider import AnthropicProvider
from ai_agent.core.llm.base import Message, ToolCall
from ai_agent.core.llm.echo_provider import EchoProvider


def test_echo_plain_message_returns_text():
    provider = EchoProvider()
    result = provider.complete([Message(role="user", content="привет")], tools=[])
    assert result.stop_reason == "end_turn"
    assert "привет" in result.content
    assert result.tool_calls == []


def test_echo_tool_protocol_emits_tool_call():
    provider = EchoProvider()
    msg = Message(role="user", content='TOOL: files.write {"path": "a.txt", "content": "hi"}')
    result = provider.complete([msg], tools=[])
    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "files.write"
    assert call.arguments == {"path": "a.txt", "content": "hi"}


def test_echo_finalizes_after_tool_result():
    provider = EchoProvider()
    messages = [
        Message(role="user", content="TOOL: files.write {}"),
        Message(role="tool", content="Записано 2 символа", tool_call_id="echo_call_1"),
    ]
    result = provider.complete(messages, tools=[])
    assert result.stop_reason == "end_turn"
    assert "Записано 2 символа" in result.content


def test_anthropic_provider_requires_package_or_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        AnthropicProvider(api_key=None)


def test_anthropic_convert_messages_handles_tool_roundtrip():
    messages = [
        Message(role="user", content="сделай что-нибудь"),
        Message(
            role="assistant",
            content="запускаю инструмент",
            tool_calls=[ToolCall(id="call_1", name="files.write", arguments={"path": "a.txt"})],
        ),
        Message(role="tool", content="готово", tool_call_id="call_1"),
    ]
    converted = AnthropicProvider._convert_messages(messages)
    assert converted[0] == {"role": "user", "content": "сделай что-нибудь"}
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"][0] == {"type": "text", "text": "запускаю инструмент"}
    assert converted[1]["content"][1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "files.write",
        "input": {"path": "a.txt"},
    }
    assert converted[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "готово"}],
    }


class _FakeBlock:
    def __init__(self, type_, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeResponse:
    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason


def test_anthropic_convert_response_text_only():
    response = _FakeResponse([_FakeBlock("text", text="привет")])
    result = AnthropicProvider._convert_response(response)
    assert result.content == "привет"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"


def test_anthropic_convert_response_with_tool_use():
    response = _FakeResponse(
        [_FakeBlock("tool_use", id="call_1", name="files.read", input={"path": "a.txt"})],
        stop_reason="tool_use",
    )
    result = AnthropicProvider._convert_response(response)
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "files.read"
    assert result.tool_calls[0].arguments == {"path": "a.txt"}
