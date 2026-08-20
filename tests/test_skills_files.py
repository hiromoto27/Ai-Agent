from ai_agent.core.skills.files import ListDirSkill, ReadFileSkill, WriteFileSkill


def test_write_then_read_roundtrip(permissive_context):
    write = WriteFileSkill()
    read = ReadFileSkill()

    result = write.run(permissive_context, path="notes/todo.txt", content="hello world")
    assert result.ok
    assert (permissive_context.workspace_root / "notes" / "todo.txt").exists()

    result = read.run(permissive_context, path="notes/todo.txt")
    assert result.ok
    assert result.output == "hello world"


def test_read_missing_file(permissive_context):
    read = ReadFileSkill()
    result = read.run(permissive_context, path="does-not-exist.txt")
    assert not result.ok
    assert "не найден" in result.error


def test_list_dir(permissive_context):
    write = WriteFileSkill()
    write.run(permissive_context, path="a.txt", content="x")
    write.run(permissive_context, path="sub/b.txt", content="y")

    result = ListDirSkill().run(permissive_context, path=".")
    assert result.ok
    assert "a.txt" in result.output
    assert "sub/" in result.output


def test_write_outside_workspace_denied_when_locked(locked_context):
    write = WriteFileSkill()
    outside = locked_context.workspace_root.parent / "outside.txt"
    result = write.run(locked_context, path=str(outside), content="x")
    assert not result.ok
    assert "доступ запрещён" in result.error
