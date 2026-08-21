from pathlib import Path

from ai_agent.core.llm_settings import LLMSettings


def test_defaults():
    settings = LLMSettings()
    assert settings.provider == "auto"
    assert settings.anthropic_model == "claude-sonnet-5"
    assert settings.anthropic_api_key == ""
    assert settings.local_model_path == ""
    assert settings.lmstudio_base_url == "http://localhost:1234/v1"
    assert settings.lmstudio_model == ""
    assert settings.system_prompt == ""


def test_load_missing_file_returns_defaults(tmp_path: Path):
    settings = LLMSettings.load(tmp_path / "does-not-exist.yaml")
    assert settings == LLMSettings()


def test_save_then_load_roundtrip(tmp_path: Path):
    path = tmp_path / "llm_settings.yaml"
    original = LLMSettings(
        provider="local",
        anthropic_model="claude-opus-5",
        anthropic_api_key="sk-ant-secret",
        local_model_path="/home/user/AiAgentWorkspace/models/Qwen__Qwen2.5-1.5B-Instruct-GGUF/model.gguf",
        local_n_ctx=8192,
        lmstudio_base_url="http://192.168.1.10:1234/v1",
        lmstudio_model="qwen2.5-1.5b-instruct",
        system_prompt="Отвечай только на русском и всегда предлагай план из шагов.",
    )
    original.save(path)

    loaded = LLMSettings.load(path)
    assert loaded == original


def test_load_partial_yaml_fills_defaults(tmp_path: Path):
    path = tmp_path / "llm_settings.yaml"
    path.write_text("provider: anthropic\n", encoding="utf-8")

    loaded = LLMSettings.load(path)
    assert loaded.provider == "anthropic"
    assert loaded.anthropic_model == "claude-sonnet-5"
    assert loaded.system_prompt == ""
