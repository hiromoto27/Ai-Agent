from .base import Skill, SkillContext, SkillParam, SkillResult, SkillSpec, SkillRegistry
from .documents import register_document_skills
from .files import register_file_skills
from .models import register_model_skills
from .package_manager import register_package_skills
from .scripting import register_scripting_skills
from .system import register_system_skills
from .voice import register_voice_skills
from .web import register_web_skills


def build_default_registry(search_endpoint: str | None = None) -> SkillRegistry:
    """Собирает реестр со всеми встроенными навыками агента."""
    registry = SkillRegistry()
    register_file_skills(registry)
    register_system_skills(registry)
    register_scripting_skills(registry)
    register_document_skills(registry)
    if search_endpoint:
        register_web_skills(registry, search_endpoint=search_endpoint)
    else:
        register_web_skills(registry)
    register_package_skills(registry)
    register_model_skills(registry)
    register_voice_skills(registry)
    return registry


__all__ = [
    "Skill",
    "SkillContext",
    "SkillParam",
    "SkillResult",
    "SkillSpec",
    "SkillRegistry",
    "build_default_registry",
]
