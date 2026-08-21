import logging
from pathlib import Path

from ai_agent.core.logging_setup import setup_logging
from ai_agent.core.policy import PermissionDenied
from ai_agent.core.skills.base import Skill, SkillContext, SkillResult, SkillSpec, _redact_kwargs


class _BoomSkill(Skill):
    spec = SkillSpec(name="test.boom", description="Всегда падает.", parameters=[])

    def _run(self, context: SkillContext, **kwargs) -> SkillResult:
        raise RuntimeError("сломалось специально")


class _DeniedSkill(Skill):
    spec = SkillSpec(name="test.denied", description="Всегда отказ.", parameters=[])

    def _run(self, context: SkillContext, **kwargs) -> SkillResult:
        raise PermissionDenied("нельзя")


def _flush():
    for handler in logging.getLogger("ai_agent").handlers:
        handler.flush()


def test_redact_kwargs_hides_secret_like_values():
    redacted = _redact_kwargs({"token": "hf_super_secret", "path": "notes.txt", "api_key": "sk-ant-x"})
    assert redacted["token"] == "***"
    assert redacted["api_key"] == "***"
    assert redacted["path"] == "notes.txt"


def test_generic_exception_is_logged_with_traceback(tmp_path: Path, permissive_context):
    log_path = setup_logging(tmp_path)
    _BoomSkill().run(permissive_context, foo="bar")
    _flush()

    content = log_path.read_text(encoding="utf-8")
    assert "test.boom" in content
    assert "RuntimeError" in content
    assert "Traceback" in content


def test_secret_kwarg_is_redacted_in_log(tmp_path: Path, permissive_context):
    log_path = setup_logging(tmp_path)
    _BoomSkill().run(permissive_context, token="hf_super_secret_value")
    _flush()

    content = log_path.read_text(encoding="utf-8")
    assert "hf_super_secret_value" not in content
    assert "***" in content


def test_permission_denied_is_logged_without_traceback_noise(tmp_path: Path, permissive_context):
    log_path = setup_logging(tmp_path)
    _DeniedSkill().run(permissive_context)
    _flush()

    content = log_path.read_text(encoding="utf-8")
    assert "test.denied" in content
    assert "нельзя" in content
