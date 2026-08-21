from pathlib import Path

import pytest

from ai_agent.core.llm.base import Message, ToolCall
from ai_agent.core.llm.local_provider import LocalLlamaProvider, _format_tools_block, _to_chat_messages

TOOLS = [
    {
        "name": "files.write",
        "description": "Записать текст в файл.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    }
]


class _FakeLlama:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_call: dict | None = None

    def create_chat_completion(self, *, messages, max_tokens, temperature):
        self.last_call = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        return {"choices": [{"message": {"content": self.reply}}]}


def test_missing_model_file_without_client_raises_value_error(tmp_path: Path):
    with pytest.raises(ValueError):
        LocalLlamaProvider(model_path=tmp_path / "does-not-exist.gguf")


def test_plain_text_reply(tmp_path: Path):
    fake = _FakeLlama("Привет! Чем могу помочь?")
    provider = LocalLlamaProvider(model_path=tmp_path / "fake.gguf", client=fake)

    result = provider.complete([Message(role="user", content="привет")], tools=[], system="Ты — агент.")

    assert result.stop_reason == "end_turn"
    assert result.content == "Привет! Чем могу помочь?"
    assert result.tool_calls == []


def test_tool_call_reply_is_parsed(tmp_path: Path):
    fake = _FakeLlama('TOOL_CALL: files.write {"path": "note.txt", "content": "hi"}')
    provider = LocalLlamaProvider(model_path=tmp_path / "fake.gguf", client=fake)

    result = provider.complete([Message(role="user", content="создай файл")], tools=TOOLS, system="")

    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "files.write"
    assert call.arguments == {"path": "note.txt", "content": "hi"}


def test_tool_call_with_malformed_json_falls_back_to_text(tmp_path: Path):
    fake = _FakeLlama("TOOL_CALL: files.write {not valid json}")
    provider = LocalLlamaProvider(model_path=tmp_path / "fake.gguf", client=fake)

    result = provider.complete([Message(role="user", content="создай файл")], tools=TOOLS, system="")

    assert result.stop_reason == "end_turn"
    assert result.tool_calls == []
    assert "TOOL_CALL" in result.content


def test_system_prompt_includes_tools_block(tmp_path: Path):
    fake = _FakeLlama("ок")
    provider = LocalLlamaProvider(model_path=tmp_path / "fake.gguf", client=fake)

    provider.complete([Message(role="user", content="привет")], tools=TOOLS, system="Базовый промпт.")

    system_message = fake.last_call["messages"][0]
    assert system_message["role"] == "system"
    assert "Базовый промпт." in system_message["content"]
    assert "files.write" in system_message["content"]
    assert "TOOL_CALL" in system_message["content"]


def test_tool_result_message_becomes_user_turn():
    messages = [
        Message(role="user", content="создай файл"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="1", name="files.write", arguments={"path": "a.txt"})]),
        Message(role="tool", content="Записано 0 символов", tool_call_id="1", tool_name="files.write"),
    ]
    chat = _to_chat_messages(messages, system="")

    assert chat[0] == {"role": "user", "content": "создай файл"}
    assert chat[1]["role"] == "assistant"
    assert "TOOL_CALL: files.write" in chat[1]["content"]
    assert chat[2] == {"role": "user", "content": "[Результат инструмента files.write]: Записано 0 символов"}


def test_format_tools_block_empty_when_no_tools():
    assert _format_tools_block([]) == ""


def test_multiple_calls_get_unique_ids(tmp_path: Path):
    fake = _FakeLlama('TOOL_CALL: files.write {"path": "a.txt", "content": "x"}')
    provider = LocalLlamaProvider(model_path=tmp_path / "fake.gguf", client=fake)

    r1 = provider.complete([Message(role="user", content="1")], tools=TOOLS, system="")
    r2 = provider.complete([Message(role="user", content="2")], tools=TOOLS, system="")

    assert r1.tool_calls[0].id != r2.tool_calls[0].id
