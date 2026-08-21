import logging
from pathlib import Path

from ai_agent.core.llm.base import LLMResponse, Message, ToolCall
from ai_agent.core.llm.echo_provider import EchoProvider
from ai_agent.core.logging_setup import setup_logging
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


class _RecordingProvider:
    """Провайдер, который записывает, что видел на входе, и отвечает эхом
    последнего пользовательского сообщения — для проверки, что история
    диалога реально передаётся между вызовами run_task."""

    def __init__(self) -> None:
        self.seen_message_counts: list[int] = []

    def complete(self, messages, tools, system=""):
        self.seen_message_counts.append(len(messages))
        return LLMResponse(content=f"ответ на: {messages[-1].content}", stop_reason="end_turn")


def test_conversation_history_persists_across_run_task_calls(tmp_path, permissive_context):
    agent, _ = make_agent(tmp_path, permissive_context, llm=_RecordingProvider())

    agent.run_task("первое сообщение")
    agent.run_task("второе сообщение")

    # На втором вызове модель должна была увидеть уже 3 сообщения в истории
    # (user1, assistant1, user2), а не только новое.
    assert agent.llm.seen_message_counts == [1, 3]
    assert agent.conversation[0].content == "первое сообщение"
    assert agent.conversation[1].role == "assistant"
    assert agent.conversation[2].content == "второе сообщение"


def test_reset_conversation_clears_history(tmp_path, permissive_context):
    agent, _ = make_agent(tmp_path, permissive_context, llm=_RecordingProvider())

    agent.run_task("первое сообщение")
    agent.reset_conversation()
    agent.run_task("новое сообщение после сброса")

    assert agent.conversation[0].content == "новое сообщение после сброса"
    assert agent.llm.seen_message_counts == [1, 1]  # оба раза "с нуля"


class _FailingProvider:
    """Провайдер, у которого complete() всегда падает — эмулирует сетевую
    ошибку/сбой API, чтобы проверить, что оркестратор не роняет агента."""

    def complete(self, messages, tools, system=""):
        raise ConnectionError("не удалось соединиться с сервером провайдера")


def _flush_log_handlers():
    for handler in logging.getLogger("ai_agent").handlers:
        handler.flush()


def test_llm_failure_returns_graceful_result_instead_of_raising(tmp_path, permissive_context):
    agent, memory = make_agent(tmp_path, permissive_context, llm=_FailingProvider())

    result = agent.run_task("сделай что-нибудь")

    assert result.success is False
    assert result.steps == []
    assert "не удалось соединиться с сервером провайдера" in result.final_text
    assert "_FailingProvider" in result.final_text


def test_llm_failure_keeps_conversation_consistent(tmp_path, permissive_context):
    agent, _ = make_agent(tmp_path, permissive_context, llm=_FailingProvider())

    agent.run_task("сделай что-нибудь")

    assert len(agent.conversation) == 2
    assert agent.conversation[0].role == "user"
    assert agent.conversation[0].content == "сделай что-нибудь"
    assert agent.conversation[1].role == "assistant"
    assert "не удалось соединиться с сервером провайдера" in agent.conversation[1].content


def test_llm_failure_is_recorded_in_memory_as_unsuccessful_episode(tmp_path, permissive_context):
    agent, memory = make_agent(tmp_path, permissive_context, llm=_FailingProvider())

    agent.run_task("сделай что-нибудь")

    recent = memory.recent(1)
    assert recent[0].success is False


def test_llm_failure_is_logged_with_traceback(tmp_path, permissive_context):
    log_path = setup_logging(tmp_path)
    agent, _ = make_agent(tmp_path, permissive_context, llm=_FailingProvider())

    agent.run_task("сделай что-нибудь")
    _flush_log_handlers()

    content = log_path.read_text(encoding="utf-8")
    assert "ConnectionError" in content
    assert "Traceback" in content


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
