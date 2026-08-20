"""Сборка приложения: общая точка входа для CLI и GUI.

Создаёт рабочее окружение (workspace для файлов агента, отдельную
директорию состояния для конфигов/памяти/аудит-лога, которую сам агент
не видит через свои файловые навыки) и собирает готовый Agent с
автонастройкой под мощность текущего ПК.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ai_agent.core.autotune import ensure_settings
from ai_agent.core.llm.base import LLMProvider
from ai_agent.core.llm.echo_provider import EchoProvider
from ai_agent.core.memory import MemoryStore
from ai_agent.core.orchestrator import Agent
from ai_agent.core.policy import PolicyConfig, PolicyEngine
from ai_agent.core.policy.engine import ConfirmCallback, always_deny
from ai_agent.core.skills import build_default_registry
from ai_agent.core.skills.base import SkillContext

DEFAULT_STATE_DIR = Path.home() / ".ai-agent"
DEFAULT_WORKSPACE = Path.home() / "AiAgentWorkspace"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _default_policy_source() -> Path:
    return _PACKAGE_ROOT / "config" / "policy.default.yaml"


def ensure_state_dirs(state_dir: Path, workspace_root: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "documents").mkdir(exist_ok=True)
    (workspace_root / "scripts").mkdir(exist_ok=True)

    policy_path = state_dir / "policy.yaml"
    if not policy_path.exists():
        default_source = _default_policy_source()
        if default_source.exists():
            shutil.copy(default_source, policy_path)
        else:  # pragma: no cover - защитный путь на случай отсутствия шаблона
            PolicyConfig.default().save(policy_path)


def build_llm_provider() -> LLMProvider:
    """Claude API, если задан ANTHROPIC_API_KEY, иначе офлайн EchoProvider."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from ai_agent.core.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(api_key=api_key)
        except ImportError:
            pass
    return EchoProvider()


def build_agent(
    state_dir: Path = DEFAULT_STATE_DIR,
    workspace_root: Path = DEFAULT_WORKSPACE,
    confirm_callback: ConfirmCallback | None = None,
    llm: LLMProvider | None = None,
) -> Agent:
    ensure_state_dirs(state_dir, workspace_root)

    profile = ensure_settings(state_dir / "settings.yaml")
    policy_config = PolicyConfig.load(state_dir / "policy.yaml")
    policy = PolicyEngine(
        policy_config,
        workspace_root=workspace_root,
        confirm_callback=confirm_callback or always_deny,
        audit_log_path=state_dir / "audit.jsonl",
    )
    skill_context = SkillContext(workspace_root=workspace_root, policy=policy, profile=profile)
    memory = MemoryStore(state_dir / "memory.sqlite3")
    registry = build_default_registry()

    return Agent(
        llm=llm or build_llm_provider(),
        skills=registry,
        memory=memory,
        skill_context=skill_context,
    )
