"""Agent Core: цикл "спланировать → выполнить → рефлексия".

На каждом шаге LLM либо просит вызвать один или несколько инструментов
(навыков), либо даёт финальный ответ. Результаты инструментов
скармливаются обратно модели, пока она не завершит ход или не будет
достигнут лимит шагов (из профиля автонастройки под мощность ПК).

После завершения задачи агент пишет эпизод в долговременную память —
это и есть механизм "самообучения": в следующий раз похожий опыт будет
подмешан в системный промпт через retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai_agent.core.llm.base import LLMProvider, Message
from ai_agent.core.memory import MemoryStore
from ai_agent.core.skills.base import SkillContext, SkillRegistry, SkillResult

DEFAULT_SYSTEM_PROMPT = (
    "Ты — локальный ИИ-агент, помогающий пользователю на его личном компьютере. "
    "У тебя есть набор инструментов (навыков) для работы с файлами, документами, "
    "интернетом и системой. Все потенциально опасные действия (запуск команд, "
    "запуск скриптов, установка пакетов, выход за пределы рабочей папки) "
    "проходят проверку прав доступа и могут требовать подтверждения пользователя — "
    "если инструмент вернул отказ в доступе, не пытайся обойти его, а сообщи "
    "пользователю, что нужно разрешение. Используй инструмент поиска в интернете, "
    "если не хватает знаний или подходящего навыка для задачи. Отвечай кратко и по делу.\n\n"
    "Если формулировка задачи нечёткая, неполная или допускает разное толкование — "
    "не угадывай и не действуй наугад. Задай пользователю один уточняющий вопрос "
    "(или несколько по пунктам, если неясностей много) обычным текстовым ответом, "
    "без вызова инструментов, и дождись ответа. Уточняй даже небольшие "
    "неоднозначности — переспросить дешевле, чем сделать не то, что нужно. Это "
    "продолжающийся диалог: используй контекст предыдущих сообщений."
)


@dataclass
class StepRecord:
    tool_name: str
    arguments: dict
    result: SkillResult


@dataclass
class AgentResult:
    final_text: str
    steps: list[StepRecord] = field(default_factory=list)
    success: bool | None = None
    hit_step_limit: bool = False


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        skills: SkillRegistry,
        memory: MemoryStore,
        skill_context: SkillContext,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int | None = None,
    ) -> None:
        self.llm = llm
        self.skills = skills
        self.memory = memory
        self.skill_context = skill_context
        self.system_prompt = system_prompt
        self.max_steps = max_steps or skill_context.profile.max_agent_steps
        # История диалога сохраняется между вызовами run_task в пределах
        # одного Agent — иначе уточняющий вопрос агента ("что именно
        # сделать?") был бы бессмысленным: следующее сообщение пользователя
        # обрабатывалось бы как отдельная, не связанная с ним задача.
        self.conversation: list[Message] = []

    def run_task(self, task: str) -> AgentResult:
        system = self.system_prompt + self._build_memory_context(task)
        self.conversation.append(Message(role="user", content=task))
        messages = self.conversation
        steps: list[StepRecord] = []
        tools = self.skills.tool_schemas()

        for _ in range(self.max_steps):
            response = self.llm.complete(messages, tools=tools, system=system)

            if not response.tool_calls:
                messages.append(Message(role="assistant", content=response.content))
                success = self._infer_success(steps)
                self._reflect(task, steps, response.content, success)
                return AgentResult(final_text=response.content, steps=steps, success=success)

            messages.append(
                Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                result = self.skills.invoke(call.name, self.skill_context, **call.arguments)
                steps.append(StepRecord(tool_name=call.name, arguments=call.arguments, result=result))
                messages.append(
                    Message(
                        role="tool",
                        content=result.to_tool_message(),
                        tool_call_id=call.id,
                        tool_name=call.name,
                    )
                )

        final_text = "Достигнут лимит шагов — задача не была завершена полностью."
        messages.append(Message(role="assistant", content=final_text))
        self._reflect(task, steps, final_text, success=False)
        return AgentResult(final_text=final_text, steps=steps, success=False, hit_step_limit=True)

    def reset_conversation(self) -> None:
        """Начать диалог заново (не трогает долговременную память-эпизоды)."""
        self.conversation = []

    def _build_memory_context(self, task: str) -> str:
        k = self.skill_context.profile.memory_retrieval_k
        episodes = self.memory.retrieve_similar(task, k=k)
        if not episodes:
            return ""
        lines = ["\n\nПохожие прошлые задачи из памяти (учти их при планировании):"]
        for episode in episodes:
            lines.append("- " + episode.to_context_snippet().replace("\n", " | "))
        return "\n".join(lines)

    @staticmethod
    def _infer_success(steps: list[StepRecord]) -> bool:
        if not steps:
            return True
        return all(step.result.ok for step in steps)

    def _reflect(self, task: str, steps: list[StepRecord], outcome: str, success: bool) -> None:
        plan_summary = (
            "; ".join(f"{s.tool_name}({s.arguments})" for s in steps) or "(без использования инструментов)"
        )
        reflection = ""
        failed_steps = [s for s in steps if not s.result.ok]
        if failed_steps:
            reflection = "Проблемы: " + "; ".join(
                f"{s.tool_name} -> {s.result.error}" for s in failed_steps
            )
        self.memory.add_episode(
            task=task,
            plan_summary=plan_summary,
            outcome=outcome,
            success=success,
            reflection=reflection,
        )
