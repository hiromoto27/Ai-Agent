import pytest

from ai_agent.core.llm.base import Message, ToolCall
from ai_agent.core.llm.lmstudio_provider import (
    DEFAULT_BASE_URL,
    LMStudioConnectionError,
    LMStudioProvider,
    list_models,
)

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


class _FakeResponse:
    def __init__(self, data, status_ok=True, status_code=200) -> None:
        self._data = data
        self._status_ok = status_ok
        self.status_code = status_code

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._data


class _FakeHttpClient:
    def __init__(self, chat_response=None, models_response=None, raise_on="", status_code=200) -> None:
        self.chat_response = chat_response
        self.models_response = models_response
        self.raise_on = raise_on
        self.status_code = status_code
        self.last_post: dict | None = None
        self.closed = False

    def post(self, path, *, json):
        if self.raise_on == "post":
            raise ConnectionError("сервер недоступен")
        self.last_post = {"path": path, "json": json}
        return _FakeResponse(self.chat_response, status_code=self.status_code)

    def get(self, path):
        if self.raise_on == "get":
            raise ConnectionError("сервер недоступен")
        return _FakeResponse(self.models_response, status_code=self.status_code)

    def close(self):
        self.closed = True


def _text_response(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def _tool_call_response(name: str, arguments: dict) -> dict:
    import json as _json

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": _json.dumps(arguments)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def test_plain_text_reply():
    fake = _FakeHttpClient(chat_response=_text_response("Привет! Чем могу помочь?"))
    provider = LMStudioProvider(http_client=fake)

    result = provider.complete([Message(role="user", content="привет")], tools=[], system="Ты — агент.")

    assert result.stop_reason == "end_turn"
    assert result.content == "Привет! Чем могу помочь?"
    assert result.tool_calls == []


def test_native_tool_call_is_parsed():
    fake = _FakeHttpClient(chat_response=_tool_call_response("files.write", {"path": "note.txt", "content": "hi"}))
    provider = LMStudioProvider(http_client=fake)

    result = provider.complete([Message(role="user", content="создай файл")], tools=TOOLS, system="")

    assert result.stop_reason == "tool_use"
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "files.write"
    assert call.arguments == {"path": "note.txt", "content": "hi"}
    assert call.id == "call_1"


def test_tools_sent_in_openai_format_when_present():
    fake = _FakeHttpClient(chat_response=_text_response("ок"))
    provider = LMStudioProvider(http_client=fake)

    provider.complete([Message(role="user", content="привет")], tools=TOOLS, system="")

    sent_tools = fake.last_post["json"]["tools"]
    assert sent_tools[0]["type"] == "function"
    assert sent_tools[0]["function"]["name"] == "files.write"
    assert sent_tools[0]["function"]["parameters"] == TOOLS[0]["input_schema"]


def test_no_tools_key_sent_when_no_tools():
    fake = _FakeHttpClient(chat_response=_text_response("ок"))
    provider = LMStudioProvider(http_client=fake)

    provider.complete([Message(role="user", content="привет")], tools=[], system="")

    assert "tools" not in fake.last_post["json"]


def test_system_prompt_becomes_system_message():
    fake = _FakeHttpClient(chat_response=_text_response("ок"))
    provider = LMStudioProvider(http_client=fake)

    provider.complete([Message(role="user", content="привет")], tools=[], system="Базовый промпт.")

    sent_messages = fake.last_post["json"]["messages"]
    assert sent_messages[0] == {"role": "system", "content": "Базовый промпт."}


def test_tool_result_message_becomes_tool_role():
    fake = _FakeHttpClient(chat_response=_text_response("ок"))
    provider = LMStudioProvider(http_client=fake)

    messages = [
        Message(role="user", content="создай файл"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="files.write", arguments={"path": "a.txt"})],
        ),
        Message(role="tool", content="Записано 0 символов", tool_call_id="1", tool_name="files.write"),
    ]
    provider.complete(messages, tools=[], system="")

    sent_messages = fake.last_post["json"]["messages"]
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[1]["tool_calls"][0]["function"]["name"] == "files.write"
    assert sent_messages[2] == {"role": "tool", "tool_call_id": "1", "content": "Записано 0 символов"}


def test_connection_failure_raises_lmstudio_connection_error():
    fake = _FakeHttpClient(raise_on="post")
    provider = LMStudioProvider(http_client=fake)

    with pytest.raises(LMStudioConnectionError):
        provider.complete([Message(role="user", content="привет")], tools=[], system="")


def test_401_response_raises_with_api_key_hint():
    fake = _FakeHttpClient(chat_response=_text_response("не важно"), status_code=401)
    provider = LMStudioProvider(http_client=fake)

    with pytest.raises(LMStudioConnectionError, match="API"):
        provider.complete([Message(role="user", content="привет")], tools=[], system="")


def test_403_response_raises_with_api_key_hint():
    fake = _FakeHttpClient(chat_response=_text_response("не важно"), status_code=403)
    provider = LMStudioProvider(http_client=fake)

    with pytest.raises(LMStudioConnectionError, match="API"):
        provider.complete([Message(role="user", content="привет")], tools=[], system="")


def test_list_models_401_raises_with_api_key_hint():
    fake = _FakeHttpClient(models_response={"data": []}, status_code=401)

    with pytest.raises(LMStudioConnectionError, match="API"):
        list_models(http_client=fake)


def test_api_key_sent_as_bearer_header_via_default_client():
    from ai_agent.core.llm.lmstudio_provider import _auth_headers, _default_http_client

    assert _auth_headers("secret-key") == {"Authorization": "Bearer secret-key"}
    assert _auth_headers("") == {}

    client = _default_http_client(DEFAULT_BASE_URL, 10.0, "secret-key")
    try:
        assert client.headers["Authorization"] == "Bearer secret-key"
    finally:
        client.close()


def test_default_model_placeholder_used_when_not_configured():
    fake = _FakeHttpClient(chat_response=_text_response("ок"))
    provider = LMStudioProvider(http_client=fake)

    provider.complete([Message(role="user", content="привет")], tools=[], system="")

    assert fake.last_post["json"]["model"] == "local-model"


def test_configured_model_is_sent():
    fake = _FakeHttpClient(chat_response=_text_response("ок"))
    provider = LMStudioProvider(model="qwen2.5-1.5b-instruct", http_client=fake)

    provider.complete([Message(role="user", content="привет")], tools=[], system="")

    assert fake.last_post["json"]["model"] == "qwen2.5-1.5b-instruct"


def test_list_models_returns_ids():
    fake = _FakeHttpClient(models_response={"data": [{"id": "qwen2.5-1.5b-instruct"}, {"id": "llama-3.1-8b"}]})

    models = list_models(http_client=fake)

    assert models == ["qwen2.5-1.5b-instruct", "llama-3.1-8b"]
    assert fake.closed is False  # клиент передан извне — провайдер его не закрывает


def test_list_models_raises_on_connection_failure():
    fake = _FakeHttpClient(raise_on="get")

    with pytest.raises(LMStudioConnectionError):
        list_models(http_client=fake)


def test_list_models_empty_when_no_data_key():
    fake = _FakeHttpClient(models_response={})

    assert list_models(http_client=fake) == []
