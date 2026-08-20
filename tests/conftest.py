from pathlib import Path

import pytest

from ai_agent.core.autotune import Profile
from ai_agent.core.policy import PolicyConfig, PolicyEngine
from ai_agent.core.policy.engine import always_allow, always_deny
from ai_agent.core.skills.base import SkillContext

TEST_PROFILE = Profile(
    tier="medium",
    max_parallel_tool_calls=2,
    max_agent_steps=10,
    memory_retrieval_k=4,
    max_context_episodes=100,
    enable_local_llm_default=False,
    max_script_timeout_sec=10,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def permissive_context(workspace: Path) -> SkillContext:
    config = PolicyConfig.default()
    config.shell_enabled = True
    config.scripting_execute_enabled = True
    config.package_install_enabled = True
    engine = PolicyEngine(config, workspace_root=workspace, confirm_callback=always_allow)
    return SkillContext(workspace_root=workspace, policy=engine, profile=TEST_PROFILE)


@pytest.fixture
def locked_context(workspace: Path) -> SkillContext:
    config = PolicyConfig.default()
    engine = PolicyEngine(config, workspace_root=workspace, confirm_callback=always_deny)
    return SkillContext(workspace_root=workspace, policy=engine, profile=TEST_PROFILE)
