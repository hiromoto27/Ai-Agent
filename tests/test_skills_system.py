from ai_agent.core.skills.system import RunCommandSkill


def test_run_command_allowed_and_captures_output(permissive_context):
    result = RunCommandSkill().run(permissive_context, command="echo hello-agent")
    assert result.ok
    assert "hello-agent" in result.output
    assert result.data["exit_code"] == 0


def test_run_command_nonzero_exit_reported(permissive_context):
    result = RunCommandSkill().run(permissive_context, command="python -c \"import sys; sys.exit(3)\"")
    assert not result.ok
    assert result.data["exit_code"] == 3


def test_run_command_denied_by_default(locked_context):
    result = RunCommandSkill().run(locked_context, command="echo hi")
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_run_command_missing_binary(permissive_context):
    result = RunCommandSkill().run(permissive_context, command="this-binary-does-not-exist-xyz")
    assert not result.ok
    assert "не найдена" in result.error


def test_run_command_timeout(permissive_context):
    result = RunCommandSkill().run(
        permissive_context, command="python -c \"import time; time.sleep(5)\"", timeout_sec=1
    )
    assert not result.ok
    assert "таймаут" in result.error
