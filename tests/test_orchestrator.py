from pathlib import Path

from ai_agent.core.llm.base import LLMResponse, Message, ToolCall
from ai_agent.core.llm.echo_provider import EchoProvider
from ai_agent.core.memory import MemoryStore
from ai_agent.core.orchestrator import Agent
from ai_agent.core.skills import build_default_registry


def make_agent(tmp_path: Path, permissive_context, llm=None):
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    registry = build_default_registry()
    return Agent(
        llm=llm or EchoProvider(),
        skills=registry,
        memory=memory,
        skill_context=permissive_context,
    ), memory


def test_plain_task_no_tool_use(tmp_path, permissive_context):
    agent, memory = make_agent(tmp_path, permissive_context)
    result = agent.run_task("расскажи о себе")
    assert result.steps == []
    assert result.success is True
    assert "расскажи о себе" in result.final_text

    stats = memory.stats()
    assert stats["total_episodes"] == 1
    assert stats["successes"] == 1


def test_task_with_successful_tool_call(tmp_path, permissive_context):
    agent, memory = make_agent(tmp_path, permissive_context)
    task = 'TOOL: files.write {"path": "notes.txt", "content": "hello"}'
    result = agent.run_task(task)

    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "files.write"
    assert result.steps[0].result.ok
    assert result.success is True
    assert (permissive_context.workspace_root / "notes.txt").read_text(encoding="utf-8") == "hello"

    recent = memory.recent(1)
    assert recent[0].success is True
    assert "files.write" in recent[0].plan_summary


def test_task_with_denied_tool_call_records_failure(tmp_path, locked_context):
    agent, memory = make_agent(tmp_path, locked_context)
    task = 'TOOL: system.run_command {"command": "echo hi"}'
    result = agent.run_task(task)

    assert len(result.steps) == 1
    assert not result.steps[0].result.ok
    assert result.success is False

    recent = memory.recent(1)
    assert recent[0].success is False
    assert "доступ запрещён" in recent[0].reflection


def test_memory_context_included_in_system_prompt_on_similar_task(tmp_path, permissive_context):
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    memory.add_episode(
        task="Сделать презентацию про квартальные продажи",
        outcome="Готово, 8 слайдов",
        success=True,
        reflection="В следующий раз добавить график динамики",
    )
    registry = build_default_registry()

    captured_system = {}

    class CapturingProvider(EchoProvider):
        def complete(self, messages, tools, system=""):
            captured_system["value"] = system
            return super().complete(messages, tools, system)

    agent = Agent(
        llm=CapturingProvider(),
        skills=registry,
        memory=memory,
        skill_context=permissive_context,
    )
    agent.run_task("Подготовь презентацию по продажам за этот квартал")

    assert "квартальные продажи" in captured_system["value"]
    assert "график динамики" in captured_system["value"]


class _AlwaysToolCallProvider:
    """Провайдер, который никогда не завершает ход — для проверки лимита шагов."""

    def complete(self, messages, tools, system=""):
        return LLMResponse(
            content="",
            tool_calls=[ToolCall(id="c", name="files.list", arguments={"path": "."})],
            stop_reason="tool_use",
        )


def test_step_limit_is_respected(tmp_path, permissive_context):
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    registry = build_default_registry()
    agent = Agent(
        llm=_AlwaysToolCallProvider(),
        skills=registry,
        memory=memory,
        skill_context=permissive_context,
        max_steps=3,
    )
    result = agent.run_task("бесконечная задача")
    assert result.hit_step_limit is True
    assert len(result.steps) == 3
    assert result.success is False
