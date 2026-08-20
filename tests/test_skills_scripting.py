from ai_agent.core.skills.scripting import RunScriptSkill, WriteScriptSkill


def test_write_script_creates_file_in_scripts_dir(permissive_context):
    result = WriteScriptSkill().run(
        permissive_context,
        filename="helper.py",
        content="print('from helper script')",
    )
    assert result.ok
    target = permissive_context.workspace_root / "scripts" / "helper.py"
    assert target.exists()
    assert "from helper script" in target.read_text()


def test_write_script_allowed_even_when_execute_locked(locked_context):
    # запись скрипта разрешена по умолчанию (безопасная файловая операция),
    # даже если сам запуск скриптов запрещён
    result = WriteScriptSkill().run(locked_context, filename="helper.py", content="print(1)")
    assert result.ok


def test_write_then_run_script(permissive_context):
    WriteScriptSkill().run(
        permissive_context,
        filename="greet.py",
        content="import sys; print('hi', sys.argv[1])",
    )
    result = RunScriptSkill().run(permissive_context, filename="greet.py", args=["world"])
    assert result.ok
    assert "hi world" in result.output


def test_run_script_denied_by_default(locked_context):
    WriteScriptSkill().run(locked_context, filename="greet.py", content="print('hi')")
    result = RunScriptSkill().run(locked_context, filename="greet.py")
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_run_script_path_traversal_blocked(permissive_context):
    result = RunScriptSkill().run(permissive_context, filename="../../etc/passwd")
    assert not result.ok


def test_run_script_missing_file(permissive_context):
    result = RunScriptSkill().run(permissive_context, filename="does-not-exist.py")
    assert not result.ok
    assert "не найден" in result.error


def test_run_script_nonzero_exit(permissive_context):
    WriteScriptSkill().run(
        permissive_context, filename="fail.py", content="import sys; sys.exit(2)"
    )
    result = RunScriptSkill().run(permissive_context, filename="fail.py")
    assert not result.ok
    assert result.data["exit_code"] == 2
