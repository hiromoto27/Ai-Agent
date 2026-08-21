from pathlib import Path

import pytest

from ai_agent.core.policy import PermissionDenied, PolicyConfig, PolicyEngine
from ai_agent.core.policy.engine import always_allow, always_deny


def make_engine(tmp_path: Path, **overrides) -> tuple[PolicyEngine, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = PolicyConfig.default()
    for key, value in overrides.items():
        setattr(config, key, value)
    engine = PolicyEngine(
        config,
        workspace_root=workspace,
        confirm_callback=always_allow,
        audit_log_path=tmp_path / "audit.jsonl",
    )
    return engine, workspace


def test_write_inside_workspace_allowed(tmp_path):
    engine, workspace = make_engine(tmp_path)
    target = workspace / "notes.txt"
    engine.enforce("files.write", path=target)  # не должно бросить исключение


def test_write_outside_workspace_denied_without_confirmation(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.confirm_callback = always_deny
    outside = tmp_path / "outside.txt"
    with pytest.raises(PermissionDenied):
        engine.enforce("files.write", path=outside)


def test_write_outside_workspace_allowed_with_confirmation(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.confirm_callback = always_allow
    outside = tmp_path / "outside.txt"
    engine.enforce("files.write", path=outside)  # подтверждено -> не бросает


def test_deny_pattern_always_blocks_even_inside_workspace(tmp_path):
    engine, workspace = make_engine(tmp_path)
    secret = workspace / ".ssh" / "id_rsa"
    with pytest.raises(PermissionDenied):
        engine.enforce("files.read", path=secret)


def test_workspace_under_windows_appdata_temp_is_not_denied(tmp_path):
    """Регрессия: на Windows pytest (и многие приложения) кладут временные/
    рабочие каталоги под %LOCALAPPDATA%\\Temp, т.е. путь буквально содержит
    "AppData". Старый deny-паттерн "**/AppData/**" ошибочно блокировал
    запись даже в собственный workspace агента. Актуальный deny-список
    (.ssh/.aws/*.pem/*.key) не должен трогать такие пути."""
    windows_like_root = tmp_path / "Users" / "runneradmin" / "AppData" / "Local" / "Temp" / "workspace"
    windows_like_root.mkdir(parents=True)
    config = PolicyConfig.default()
    engine = PolicyEngine(
        config,
        workspace_root=windows_like_root,
        confirm_callback=always_deny,
        audit_log_path=tmp_path / "audit.jsonl",
    )
    engine.enforce("files.write", path=windows_like_root / "notes.txt")


def test_shell_disabled_by_default(tmp_path):
    engine, _ = make_engine(tmp_path)
    with pytest.raises(PermissionDenied):
        engine.enforce("shell.execute", command="echo hi")


def test_shell_enabled_requires_confirmation(tmp_path):
    engine, _ = make_engine(tmp_path, shell_enabled=True)
    engine.confirm_callback = always_deny
    with pytest.raises(PermissionDenied):
        engine.enforce("shell.execute", command="echo hi")

    engine.confirm_callback = always_allow
    engine.enforce("shell.execute", command="echo hi")


def test_shell_allow_commands_whitelist(tmp_path):
    engine, _ = make_engine(tmp_path, shell_enabled=True, shell_allow_commands=["git "])
    engine.confirm_callback = always_allow
    engine.enforce("shell.execute", command="git status")
    with pytest.raises(PermissionDenied):
        engine.enforce("shell.execute", command="rm -rf /")


def test_scripting_write_allowed_by_default(tmp_path):
    engine, _ = make_engine(tmp_path)
    engine.enforce("scripting.write")


def test_scripting_execute_disabled_by_default(tmp_path):
    engine, _ = make_engine(tmp_path)
    with pytest.raises(PermissionDenied):
        engine.enforce("scripting.execute")


def test_scripting_execute_enabled_requires_confirmation(tmp_path):
    engine, _ = make_engine(tmp_path, scripting_execute_enabled=True)
    engine.confirm_callback = always_deny
    with pytest.raises(PermissionDenied):
        engine.enforce("scripting.execute")
    engine.confirm_callback = always_allow
    engine.enforce("scripting.execute")


def test_network_domain_allow_list(tmp_path):
    engine, _ = make_engine(tmp_path, network_allow_domains=["example.com", "*.trusted.org"])
    engine.enforce("web.fetch", domain="example.com")
    engine.enforce("web.fetch", domain="api.trusted.org")
    with pytest.raises(PermissionDenied):
        engine.enforce("web.fetch", domain="evil.com")


def test_network_deny_overrides_allow(tmp_path):
    engine, _ = make_engine(
        tmp_path, network_allow_domains=["*"], network_deny_domains=["evil.com"]
    )
    with pytest.raises(PermissionDenied):
        engine.enforce("web.fetch", domain="evil.com")
    engine.enforce("web.fetch", domain="example.com")


def test_package_install_disabled_by_default(tmp_path):
    engine, _ = make_engine(tmp_path)
    with pytest.raises(PermissionDenied):
        engine.enforce("package.install", package="requests")


def test_package_install_enabled_requires_confirmation(tmp_path):
    engine, _ = make_engine(tmp_path, package_install_enabled=True)
    engine.confirm_callback = always_deny
    with pytest.raises(PermissionDenied):
        engine.enforce("package.install", package="requests")
    engine.confirm_callback = always_allow
    engine.enforce("package.install", package="requests")


def test_model_download_disabled_by_default(tmp_path):
    engine, _ = make_engine(tmp_path)
    with pytest.raises(PermissionDenied):
        engine.enforce("model.download", repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")


def test_model_download_enabled_requires_confirmation(tmp_path):
    engine, _ = make_engine(tmp_path, model_download_enabled=True)
    engine.confirm_callback = always_deny
    with pytest.raises(PermissionDenied):
        engine.enforce("model.download", repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    engine.confirm_callback = always_allow
    engine.enforce("model.download", repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")


def test_audit_log_written(tmp_path):
    engine, workspace = make_engine(tmp_path)
    engine.enforce("files.write", path=workspace / "a.txt")
    audit_path = tmp_path / "audit.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"allowed": true' in lines[0]


def test_config_load_and_save_roundtrip(tmp_path):
    path = tmp_path / "policy.yaml"
    config = PolicyConfig.default()
    config.shell_enabled = True
    config.network_allow_domains = ["example.com"]
    config.save(path)

    loaded = PolicyConfig.load(path)
    assert loaded.shell_enabled is True
    assert loaded.network_allow_domains == ["example.com"]


def test_config_load_missing_file_returns_default(tmp_path):
    config = PolicyConfig.load(tmp_path / "does-not-exist.yaml")
    assert config.shell_enabled is False
    assert config.workspace_only is True
