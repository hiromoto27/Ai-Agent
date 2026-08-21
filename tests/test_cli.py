from pathlib import Path

import pytest

from ai_agent.cli import main


@pytest.fixture(autouse=True)
def no_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_once_plain_task_exits_zero(tmp_path: Path, capsys):
    exit_code = main(
        [
            "--once",
            "привет, агент",
            "--workspace",
            str(tmp_path / "ws"),
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "привет, агент" in out
    assert "Профиль производительности" in out


def test_bootstrap_creates_config_and_workspace_dirs(tmp_path: Path):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "ws"
    main(["--once", "тест", "--workspace", str(workspace), "--state-dir", str(state_dir)])

    assert (state_dir / "policy.yaml").exists()
    assert (state_dir / "settings.yaml").exists()
    assert (state_dir / "memory.sqlite3").exists()
    assert (workspace / "documents").is_dir()
    assert (workspace / "scripts").is_dir()


def test_once_tool_call_with_auto_approve_creates_file(tmp_path: Path):
    workspace = tmp_path / "ws"
    exit_code = main(
        [
            "--once",
            'TOOL: files.write {"path": "notes.txt", "content": "created by agent"}',
            "--workspace",
            str(workspace),
            "--state-dir",
            str(tmp_path / "state"),
            "--yes",
        ]
    )
    assert exit_code == 0
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "created by agent"


def test_once_tool_call_denied_without_yes_flag(tmp_path: Path, monkeypatch):
    # без --yes используется интерактивный confirm, который читает stdin;
    # для shell.execute это отключено политикой по умолчанию и вовсе не
    # доходит до confirm, так что результат — явный отказ, exit code 1
    exit_code = main(
        [
            "--once",
            'TOOL: system.run_command {"command": "echo hi"}',
            "--workspace",
            str(tmp_path / "ws"),
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )
    assert exit_code == 1


def test_memory_persists_across_cli_invocations(tmp_path: Path, capsys):
    state_dir = tmp_path / "state"
    workspace = tmp_path / "ws"
    main(["--once", "Сделать отчёт по продажам за март", "--workspace", str(workspace), "--state-dir", str(state_dir)])
    capsys.readouterr()

    from ai_agent.core.memory import MemoryStore

    memory = MemoryStore(state_dir / "memory.sqlite3")
    assert memory.stats()["total_episodes"] == 1
