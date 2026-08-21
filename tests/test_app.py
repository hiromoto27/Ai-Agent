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
