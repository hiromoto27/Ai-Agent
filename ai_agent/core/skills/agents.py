"""Навык, позволяющий агенту делегировать подзадачу временному
вспомогательному агенту с ограниченным набором навыков.

Это способ агента "создавать себе других агентов с выделенными
навыками" из исходного запроса: не отдельный процесс/сервис, а
кратковременный ``Agent`` (см. ``ai_agent.core.orchestrator``) с той же
рабочей директорией/политикой/памятью, но урезанным реестром навыков —
удобно, когда для части работы нужна другая специализация, либо чтобы
не путать основной диалог промежуточными шагами.

Рекурсия запрещена намеренно (вспомогательный агент не может сам
получить ``agents.spawn_subagent``) — это ограничивает и глубину
вложенности, и риск «побега» из-под контроля политики: реальная
опасность в любом случае определяется тем, какие конкретные навыки
получит вспомогательный агент, а не самим фактом его создания.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec, SkillRegistry

if TYPE_CHECKING:
    from ai_agent.core.orchestrator import Agent

DEFAULT_MAX_STEPS = 6


class SpawnSubagentSkill(Skill):
    spec = SkillSpec(
        name="agents.spawn_subagent",
        description=(
            "Создать временного вспомогательного агента с ограниченным набором навыков для "
            "решения конкретной подзадачи — полезно для узкоспециализированного шага или "
            "чтобы не смешивать его с основным диалогом. Подзадача выполняется сразу, "
            "результат возвращается как обычный результат навыка. Вспомогательный агент не "
            "может сам создавать других агентов."
        ),
        parameters=[
            SkillParam("task", "string", "Формулировка подзадачи для вспомогательного агента"),
            SkillParam(
                "skills",
                "array",
                "Имена навыков, которые получит вспомогательный агент (подмножество уже доступных)",
                items={"type": "string"},
            ),
            SkillParam(
                "role",
                "string",
                "Короткое описание специализации/роли вспомогательного агента (для его системного промпта)",
                required=False,
            ),
            SkillParam(
                "max_steps", "integer", "Ограничение числа шагов вспомогательного агента", required=False
            ),
        ],
    )

    def __init__(self) -> None:
        self._agent: Optional["Agent"] = None
        self._default_max_steps = DEFAULT_MAX_STEPS

    def configure(self, agent: "Agent", default_max_steps: int = DEFAULT_MAX_STEPS) -> None:
        """Довязывается после сборки основного Agent — см. ``app.build_agent()``.

        На момент регистрации навыка в реестре (``build_default_registry``)
        сам Agent ещё не существует (реестр как раз собирается для него),
        поэтому конфигурация происходит отдельным шагом сразу после того,
        как Agent готов. Держим ссылку на сам ``agent``, а не на снимок его
        ``llm``/``skills``/``memory`` по отдельности — LLM-провайдер агента
        можно поменять на лету (вкладка «Настройки»), и вспомогательные
        агенты должны сразу видеть актуальный, а не замороженный на момент
        конфигурации.
        """
        self._agent = agent
        self._default_max_steps = default_max_steps

    def _run(
        self,
        context: SkillContext,
        task: str,
        skills: list[str],
        role: str = "",
        max_steps: Optional[int] = None,
    ) -> SkillResult:
        if self._agent is None:
            return SkillResult(ok=False, error="подсистема вспомогательных агентов не инициализирована")

        context.policy.enforce("agents.spawn")

        if self.spec.name in skills:
            return SkillResult(
                ok=False,
                error="вспомогательному агенту нельзя выдать agents.spawn_subagent — рекурсия запрещена",
            )

        full_registry = self._agent.skills
        unknown = [name for name in skills if not full_registry.has(name)]
        if unknown:
            return SkillResult(ok=False, error=f"неизвестные навыки: {', '.join(unknown)}")

        sub_registry = SkillRegistry()
        for name in skills:
            sub_registry.register(full_registry.get(name))

        # Отдельный (лениво импортируемый) модуль — Agent, в свою очередь,
        # опирается на ai_agent.core.skills.base, поэтому импорт на уровне
        # модуля рискует круговой зависимостью при сборке пакета skills.
        from ai_agent.core.orchestrator import Agent, DEFAULT_SYSTEM_PROMPT

        system_prompt = DEFAULT_SYSTEM_PROMPT
        if role.strip():
            system_prompt += f"\n\nТвоя специализация в этой подзадаче: {role.strip()}"

        sub_agent = Agent(
            llm=self._agent.llm,
            skills=sub_registry,
            memory=self._agent.memory,
            skill_context=context,
            system_prompt=system_prompt,
            max_steps=max_steps or self._default_max_steps,
        )
        result = sub_agent.run_task(task)

        steps_summary = (
            "; ".join(f"{s.tool_name}({s.arguments})" for s in result.steps) or "(без вызова инструментов)"
        )
        return SkillResult(
            ok=result.success is not False,
            output=result.final_text,
            data={
                "steps": steps_summary,
                "hit_step_limit": result.hit_step_limit,
                "success": result.success,
            },
        )


def register_agent_skills(registry: SkillRegistry) -> None:
    registry.register(SpawnSubagentSkill())
