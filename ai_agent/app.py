"""Сборка приложения: общая точка входа для CLI и GUI.

Создаёт рабочее окружение (workspace для файлов агента, отдельную
директорию состояния для конфигов/памяти/аудит-лога, которую сам агент
не видит через свои файловые навыки) и собирает готовый Agent с
автонастройкой под мощность текущего ПК.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from ai_agent.core.autotune import ensure_settings
from ai_agent.core.llm.base import LLMProvider, Message
from ai_agent.core.llm.echo_provider import EchoProvider
from ai_agent.core.llm_settings import LLMSettings
from ai_agent.core.logging_setup import get_logger, setup_logging
from ai_agent.core.memory import MemoryStore
from ai_agent.core.orchestrator import DEFAULT_SYSTEM_PROMPT, Agent
from ai_agent.core.policy import PolicyConfig, PolicyEngine
from ai_agent.core.policy.engine import ConfirmCallback, always_deny
from ai_agent.core.skills import build_default_registry
from ai_agent.core.skills.base import SkillContext

logger = get_logger("app")

DEFAULT_STATE_DIR = Path.home() / ".ai-agent"
DEFAULT_WORKSPACE = Path.home() / "AiAgentWorkspace"

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _default_policy_source() -> Path:
    """Путь к шаблону политики по умолчанию.

    В обычном запуске (из исходников) это ``<repo>/config``. Под
    PyInstaller (frozen .exe) PyInstaller распаковывает файлы из ``datas``
    во временную/бандл-директорию ``sys._MEIPASS`` — см. packaging/build.spec,
    где ``config/policy.default.yaml`` включён с тем же относительным путём.
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = _PACKAGE_ROOT
    return base / "config" / "policy.default.yaml"


def ensure_state_dirs(state_dir: Path, workspace_root: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "documents").mkdir(exist_ok=True)
    (workspace_root / "scripts").mkdir(exist_ok=True)

    models_dir = workspace_root / "models"
    models_dir.mkdir(exist_ok=True)
    models_readme = models_dir / "README.txt"
    if not models_readme.exists():
        models_readme.write_text(
            "Сюда можно вручную положить модели, скачанные откуда угодно "
            "(не только через встроенный поиск по Hugging Face) — файлы "
            "(.gguf и т.п.) или целые папки репозиториев. Агент их не "
            "запускает автоматически, но видит во вкладке «Модели» в "
            "разделе «Локальные модели» (кнопка «Обновить»).\n",
            encoding="utf-8",
        )

    policy_path = state_dir / "policy.yaml"
    if not policy_path.exists():
        default_source = _default_policy_source()
        if default_source.exists():
            shutil.copy(default_source, policy_path)
        else:  # pragma: no cover - защитный путь на случай отсутствия шаблона
            PolicyConfig.default().save(policy_path)


def _build_anthropic(settings: LLMSettings) -> LLMProvider:
    from ai_agent.core.llm.anthropic_provider import AnthropicProvider

    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    return AnthropicProvider(api_key=api_key, model=settings.anthropic_model)


def _build_local(settings: LLMSettings) -> LLMProvider:
    if not settings.local_model_path:
        raise ValueError("не указан путь к локальной модели (.gguf) в настройках")
    from ai_agent.core.llm.local_provider import LocalLlamaProvider

    return LocalLlamaProvider(model_path=settings.local_model_path, n_ctx=settings.local_n_ctx)


def _build_lmstudio(settings: LLMSettings) -> LLMProvider:
    from ai_agent.core.llm.lmstudio_provider import LMStudioProvider

    return LMStudioProvider(base_url=settings.lmstudio_base_url, model=settings.lmstudio_model)


def build_llm_provider(settings: LLMSettings | None = None) -> LLMProvider:
    """Строит LLM-провайдер по настройкам пользователя.

    Явный выбор (``provider`` != "auto") не деградирует молча: если он не
    может стартовать (нет ключа/пакета/модели), исключение поднимается —
    так пользователь ясно видит проблему вместо незаметного отката на
    офлайн-заглушку (именно это раньше выглядело как "агент не отвечает,
    просто повторяет вопрос"). Только "auto" перебирает варианты тихо.
    """
    settings = settings or LLMSettings()

    if settings.provider == "echo":
        return EchoProvider()
    if settings.provider == "anthropic":
        return _build_anthropic(settings)
    if settings.provider == "local":
        return _build_local(settings)
    if settings.provider == "lmstudio":
        return _build_lmstudio(settings)

    # auto: облако, если есть ключ, иначе локальная модель, если указана, иначе эхо.
    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return _build_anthropic(settings)
        except ImportError:
            pass
    if settings.local_model_path:
        try:
            return _build_local(settings)
        except (ImportError, ValueError):
            pass
    return EchoProvider()


def build_llm_provider_safe(settings: LLMSettings | None = None) -> tuple[LLMProvider, str]:
    """Как build_llm_provider, но никогда не бросает исключение — при сбое
    явного выбора возвращает EchoProvider и текст причины (вторым
    элементом), чтобы вызывающий код (GUI/CLI) мог показать это
    пользователю, а не просто уронить всё приложение при старте."""
    try:
        return build_llm_provider(settings), ""
    except Exception as e:
        return EchoProvider(), f"{type(e).__name__}: {e}"


def test_llm_connection(llm: LLMProvider) -> tuple[bool, str]:
    """Реальная проверка связи с провайдером: короткий тестовый запрос без
    инструментов, вне истории диалога агента. Раньше единственным
    индикатором был статичный ярлык с именем класса провайдера — он не
    показывал, отвечает ли провайдер вообще (неверный ключ/API недоступен/
    файл модели битый видны только при первом реальном сообщении в чате)."""
    probe = [Message(role="user", content="Ответь одним словом: OK")]
    try:
        response = llm.complete(probe, tools=[], system="Отвечай только словом OK, без пояснений.")
    except Exception as e:
        logger.exception("Проверка связи с LLM-провайдером не удалась")
        return False, f"{type(e).__name__}: {e}"
    if not response.content.strip():
        return False, "провайдер ответил пустым сообщением"
    return True, response.content.strip()


def combine_system_prompt(llm_settings: LLMSettings) -> str:
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if llm_settings.system_prompt.strip():
        system_prompt += "\n\nДополнительные инструкции от пользователя:\n" + llm_settings.system_prompt.strip()
    return system_prompt


def build_agent(
    state_dir: Path = DEFAULT_STATE_DIR,
    workspace_root: Path = DEFAULT_WORKSPACE,
    confirm_callback: ConfirmCallback | None = None,
    llm: LLMProvider | None = None,
) -> Agent:
    ensure_state_dirs(state_dir, workspace_root)
    log_path = setup_logging(state_dir)

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

    llm_settings_path = state_dir / "llm_settings.yaml"
    llm_settings = LLMSettings.load(llm_settings_path)
    setup_error = ""
    if llm is None:
        llm, setup_error = build_llm_provider_safe(llm_settings)

    agent = Agent(
        llm=llm,
        skills=registry,
        memory=memory,
        skill_context=skill_context,
        system_prompt=combine_system_prompt(llm_settings),
    )
    # Не часть контракта Agent — просто удобное место для GUI/CLI хранить
    # текущие настройки провайдера и путь для их сохранения без отдельного
    # объекта-контейнера.
    agent.llm_settings = llm_settings
    agent.llm_settings_path = llm_settings_path
    agent.llm_setup_error = setup_error
    agent.log_path = log_path

    logger.info("Агент собран: provider=%s -> %s", llm_settings.provider, type(llm).__name__)
    if setup_error:
        logger.warning("Выбранный провайдер не запустился, откат на EchoProvider: %s", setup_error)
    return agent
