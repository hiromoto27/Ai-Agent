from pathlib import Path

from ai_agent.core.llm.base import LLMResponse, Message
from ai_agent.core.llm.echo_provider import EchoProvider
from ai_agent.core.memory import MemoryStore
from ai_agent.core.orchestrator import Agent
from ai_agent.core.skills import build_default_registry
from ai_agent.core.skills.agents import SpawnSubagentSkill


def make_agent_with_full_registry(tmp_path: Path, permissive_context, llm=None) -> Agent:
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    registry = build_default_registry()
    agent = Agent(
        llm=llm or EchoProvider(),
        skills=registry,
        memory=memory,
        skill_context=permissive_context,
    )
    registry.get("agents.spawn_subagent").configure(agent)
    return agent


def test_spawn_subagent_not_configured_returns_error(permissive_context):
    skill = SpawnSubagentSkill()
    result = skill.run(permissive_context, task="что-нибудь", skills=[])
    assert not result.ok
    assert "не инициализирована" in result.error


def test_spawn_subagent_runs_scoped_task_and_writes_file(tmp_path: Path, permissive_context):
    agent = make_agent_with_full_registry(tmp_path, permissive_context)
    task = 'TOOL: files.write {"path": "from_subagent.txt", "content": "hi"}'

    result = agent.skills.invoke(
        "agents.spawn_subagent",
        permissive_context,
        task=task,
        skills=["files.write"],
    )

    assert result.ok
    assert (permissive_context.workspace_root / "from_subagent.txt").read_text(encoding="utf-8") == "hi"
    assert "files.write" in result.data["steps"]


def test_spawn_subagent_rejects_recursive_self_inclusion(tmp_path: Path, permissive_context):
    agent = make_agent_with_full_registry(tmp_path, permissive_context)

    result = agent.skills.invoke(
        "agents.spawn_subagent",
        permissive_context,
        task="что-нибудь",
        skills=["agents.spawn_subagent"],
    )

    assert not result.ok
    assert "рекурсия запрещена" in result.error


def test_spawn_subagent_rejects_unknown_skill_name(tmp_path: Path, permissive_context):
    agent = make_agent_with_full_registry(tmp_path, permissive_context)

    result = agent.skills.invoke(
        "agents.spawn_subagent",
        permissive_context,
        task="что-нибудь",
        skills=["no.such.skill"],
    )

    assert not result.ok
    assert "неизвестные навыки" in result.error


def test_spawn_subagent_denied_when_policy_disabled(tmp_path: Path, permissive_context):
    permissive_context.policy.config.subagents_enabled = False
    agent = make_agent_with_full_registry(tmp_path, permissive_context)

    result = agent.skills.invoke(
        "agents.spawn_subagent",
        permissive_context,
        task="что-нибудь",
        skills=["files.write"],
    )

    assert not result.ok
    assert "доступ запрещён" in result.error


def test_spawn_subagent_picks_up_llm_swapped_after_configure(tmp_path: Path, permissive_context):
    class _RecordingProvider:
        def __init__(self) -> None:
            self.called = False

        def complete(self, messages, tools, system=""):
            self.called = True
            return LLMResponse(content="ответ от новой модели", stop_reason="end_turn")

    agent = make_agent_with_full_registry(tmp_path, permissive_context, llm=EchoProvider())
    new_llm = _RecordingProvider()
    agent.llm = new_llm  # эмулирует переключение провайдера через вкладку "Настройки"

    result = agent.skills.invoke(
        "agents.spawn_subagent",
        permissive_context,
        task="что-нибудь",
        skills=["files.write"],
    )

    assert new_llm.called is True
    assert result.output == "ответ от новой модели"


def test_spawn_subagent_role_appended_to_system_prompt(tmp_path: Path, permissive_context):
    captured = {}

    class _CapturingProvider:
        def complete(self, messages, tools, system=""):
            captured["system"] = system
            return LLMResponse(content="ок", stop_reason="end_turn")

    agent = make_agent_with_full_registry(tmp_path, permissive_context, llm=_CapturingProvider())

    agent.skills.invoke(
        "agents.spawn_subagent",
        permissive_context,
        task="что-нибудь",
        skills=["files.write"],
        role="специалист по Excel",
    )

    assert "специалист по Excel" in captured["system"]


def test_spawn_subagent_respects_max_steps_override(tmp_path: Path, permissive_context):
    class _AlwaysToolCallProvider:
        def complete(self, messages, tools, system=""):
            from ai_agent.core.llm.base import ToolCall

            return LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c", name="files.list", arguments={"path": "."})],
                stop_reason="tool_use",
            )

    agent = make_agent_with_full_registry(tmp_path, permissive_context, llm=_AlwaysToolCallProvider())

    result = agent.skills.invoke(
        "agents.spawn_subagent",
        permissive_context,
        task="бесконечная задача",
        skills=["files.list"],
        max_steps=2,
    )

    assert result.data["hit_step_limit"] is True
    assert result.data["steps"].count("files.list") == 2
