import sys
from pathlib import Path

from ai_agent import app


def test_default_policy_source_dev_mode():
    path = app._default_policy_source()
    assert path.name == "policy.default.yaml"
    assert path.exists()  # реальный шаблон должен существовать в репозитории


def test_default_policy_source_frozen_mode(monkeypatch, tmp_path: Path):
    fake_bundle = tmp_path / "bundle"
    (fake_bundle / "config").mkdir(parents=True)
    (fake_bundle / "config" / "policy.default.yaml").write_text("version: 1\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(fake_bundle), raising=False)

    path = app._default_policy_source()
    assert path == fake_bundle / "config" / "policy.default.yaml"
    assert path.exists()


def test_ensure_state_dirs_bootstraps_policy_from_template(tmp_path: Path):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "ws"
    app.ensure_state_dirs(state_dir, workspace)

    assert (state_dir / "policy.yaml").exists()
    assert (workspace / "documents").is_dir()
    assert (workspace / "scripts").is_dir()
    assert (workspace / "models").is_dir()
    assert (workspace / "models" / "README.txt").exists()

    # повторный вызов не должен перезаписывать уже существующий policy.yaml
    # или README в models/ (пользователь мог там что-то уже разместить рядом)
    (state_dir / "policy.yaml").write_text("custom: true\n", encoding="utf-8")
    (workspace / "models" / "README.txt").write_text("моё\n", encoding="utf-8")
    app.ensure_state_dirs(state_dir, workspace)
    assert "custom: true" in (state_dir / "policy.yaml").read_text(encoding="utf-8")
    assert (workspace / "models" / "README.txt").read_text(encoding="utf-8") == "моё\n"


def test_build_llm_provider_falls_back_to_echo_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from ai_agent.core.llm.echo_provider import EchoProvider

    provider = app.build_llm_provider()
    assert isinstance(provider, EchoProvider)


def test_build_llm_provider_auto_prefers_anthropic_when_key_present(monkeypatch):
    from ai_agent.core.llm.anthropic_provider import AnthropicProvider
    from ai_agent.core.llm_settings import LLMSettings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    provider = app.build_llm_provider(LLMSettings())
    assert isinstance(provider, AnthropicProvider)


def test_build_llm_provider_explicit_anthropic_without_key_raises(monkeypatch):
    from ai_agent.core.llm_settings import LLMSettings

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        app.build_llm_provider(LLMSettings(provider="anthropic"))
        assert False, "должно было поднять исключение"
    except ValueError:
        pass


def test_build_llm_provider_explicit_local_without_path_raises():
    from ai_agent.core.llm_settings import LLMSettings

    try:
        app.build_llm_provider(LLMSettings(provider="local"))
        assert False, "должно было поднять исключение"
    except ValueError:
        pass


def test_build_llm_provider_explicit_lmstudio_constructs_without_network():
    from ai_agent.core.llm.lmstudio_provider import LMStudioProvider
    from ai_agent.core.llm_settings import LLMSettings

    # Конструктор явного выбора LM Studio не должен блокироваться на
    # сетевом запросе — соединение проверяется отдельно (кнопка
    # "Проверить подключение"), не на старте агента.
    provider = app.build_llm_provider(LLMSettings(provider="lmstudio", lmstudio_base_url="http://localhost:1234/v1"))
    assert isinstance(provider, LMStudioProvider)
    assert provider.base_url == "http://localhost:1234/v1"


def test_build_llm_provider_explicit_lmstudio_passes_model():
    from ai_agent.core.llm_settings import LLMSettings

    provider = app.build_llm_provider(LLMSettings(provider="lmstudio", lmstudio_model="qwen2.5-1.5b-instruct"))
    assert provider.model == "qwen2.5-1.5b-instruct"


def test_build_llm_provider_explicit_lmstudio_passes_api_key():
    from ai_agent.core.llm_settings import LLMSettings

    provider = app.build_llm_provider(LLMSettings(provider="lmstudio", lmstudio_api_key="secret-key"))
    assert provider.api_key == "secret-key"


def test_build_llm_provider_safe_never_raises_and_reports_reason(monkeypatch):
    from ai_agent.core.llm.echo_provider import EchoProvider
    from ai_agent.core.llm_settings import LLMSettings

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider, error = app.build_llm_provider_safe(LLMSettings(provider="anthropic"))
    assert isinstance(provider, EchoProvider)
    assert error  # причина сбоя не потеряна


def test_build_llm_provider_safe_no_error_when_fine(monkeypatch):
    from ai_agent.core.llm_settings import LLMSettings

    provider, error = app.build_llm_provider_safe(LLMSettings(provider="echo"))
    assert error == ""


def test_build_agent_attaches_llm_settings(tmp_path: Path):
    agent = app.build_agent(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")
    assert agent.llm_settings_path == tmp_path / "state" / "llm_settings.yaml"
    assert agent.llm_settings.provider == "auto"
    assert agent.llm_setup_error == ""


def test_build_agent_surfaces_setup_error_without_crashing(tmp_path: Path, monkeypatch):
    from ai_agent.core.llm.echo_provider import EchoProvider
    from ai_agent.core.llm_settings import LLMSettings

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    LLMSettings(provider="anthropic").save(state_dir / "llm_settings.yaml")

    agent = app.build_agent(state_dir=state_dir, workspace_root=tmp_path / "ws")

    assert isinstance(agent.llm, EchoProvider)
    assert agent.llm_setup_error != ""


def test_llm_connection_success_with_echo_provider():
    from ai_agent.core.llm.echo_provider import EchoProvider

    ok, message = app.test_llm_connection(EchoProvider())
    assert ok is True
    assert message


def test_llm_connection_failure_reports_exception():
    class _BrokenProvider:
        def complete(self, messages, tools, system=""):
            raise ConnectionError("сервер недоступен")

    ok, message = app.test_llm_connection(_BrokenProvider())
    assert ok is False
    assert "сервер недоступен" in message


def test_llm_connection_failure_on_empty_response():
    from ai_agent.core.llm.base import LLMResponse

    class _EmptyProvider:
        def complete(self, messages, tools, system=""):
            return LLMResponse(content="   ")

    ok, message = app.test_llm_connection(_EmptyProvider())
    assert ok is False
    assert "пустым" in message


def test_build_agent_combines_custom_system_prompt(tmp_path: Path):
    from ai_agent.core.llm_settings import LLMSettings
    from ai_agent.core.orchestrator import DEFAULT_SYSTEM_PROMPT

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    LLMSettings(system_prompt="Отвечай только эмодзи.").save(state_dir / "llm_settings.yaml")

    agent = app.build_agent(state_dir=state_dir, workspace_root=tmp_path / "ws")

    assert agent.system_prompt.startswith(DEFAULT_SYSTEM_PROMPT)
    assert "Отвечай только эмодзи." in agent.system_prompt


def test_build_agent_wires_spawn_subagent_skill(tmp_path: Path):
    agent = app.build_agent(state_dir=tmp_path / "state", workspace_root=tmp_path / "ws")

    spawn_skill = agent.skills.get("agents.spawn_subagent")
    assert spawn_skill._agent is agent
