"""Система прав доступа (запретов) агента.

Каждое потенциально опасное действие навыка (файлы, команды системы,
скрипты, сеть, установка пакетов) проходит через ``PolicyEngine.enforce``
до фактического выполнения. Решение основано на конфиге allow/deny,
а действия из ``confirmation_required_for`` дополнительно требуют
подтверждения через внешний callback (CLI-вопрос, диалог в GUI и т.п.).

По умолчанию агент "заперт" в рабочую директорию (workspace) и не может
выполнять команды/устанавливать пакеты, пока пользователь явно это не
разрешит в конфиге.
"""

from __future__ import annotations

import fnmatch
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

ConfirmCallback = Callable[[str, dict], bool]


class PermissionDenied(Exception):
    """Действие запрещено политикой или отклонено пользователем."""


@dataclass
class PolicyDecision:
    allowed: bool
    requires_confirmation: bool = False
    reason: str = ""


@dataclass
class PolicyConfig:
    workspace_only: bool = True
    filesystem_allow_read: list[str] = field(default_factory=list)
    filesystem_allow_write: list[str] = field(default_factory=list)
    filesystem_deny: list[str] = field(
        default_factory=lambda: [
            "**/.ssh/**",
            "**/.aws/**",
            "**/*.pem",
            "**/*.key",
        ]
    )
    shell_enabled: bool = False
    shell_allow_commands: list[str] = field(default_factory=list)
    scripting_enabled: bool = True
    scripting_execute_enabled: bool = False
    network_enabled: bool = True
    network_allow_domains: list[str] = field(default_factory=lambda: ["*"])
    network_deny_domains: list[str] = field(default_factory=list)
    package_install_enabled: bool = False
    model_download_enabled: bool = False
    subagents_enabled: bool = True
    # Разовая запись — как остальные навыки: включил здесь один раз, дальше
    # работает без переспроса. Постоянная (фоновая) запись — отдельный, более
    # строгий флаг: подтверждение запрашивается при КАЖДОМ включении режима
    # (см. PolicyEngine.check_microphone), а не через confirmation_required_for.
    microphone_enabled: bool = False
    microphone_continuous_enabled: bool = False
    microphone_retain_audio: bool = False
    confirmation_required_for: list[str] = field(
        default_factory=lambda: [
            "shell.execute",
            "scripting.execute",
            "package.install",
            "model.download",
            "files.write_outside_workspace",
            "files.read_outside_workspace",
        ]
    )
    audit_log: bool = True

    @classmethod
    def default(cls) -> "PolicyConfig":
        return cls()

    @classmethod
    def load(cls, path: Path) -> "PolicyConfig":
        if not path.exists():
            return cls.default()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fs = raw.get("filesystem", {})
        shell = raw.get("shell", {})
        scripting = raw.get("scripting", {})
        network = raw.get("network", {})
        package_install = raw.get("package_install", {})
        model_download = raw.get("model_download", {})
        agents = raw.get("agents", {})
        microphone = raw.get("microphone", {})
        return cls(
            workspace_only=raw.get("workspace_only", True),
            filesystem_allow_read=fs.get("allow_read", []),
            filesystem_allow_write=fs.get("allow_write", []),
            filesystem_deny=fs.get("deny", cls().filesystem_deny),
            shell_enabled=shell.get("enabled", False),
            shell_allow_commands=shell.get("allow_commands", []),
            scripting_enabled=scripting.get("enabled", True),
            scripting_execute_enabled=scripting.get("execute_enabled", False),
            network_enabled=network.get("enabled", True),
            network_allow_domains=network.get("allow_domains", ["*"]),
            network_deny_domains=network.get("deny_domains", []),
            package_install_enabled=package_install.get("enabled", False),
            model_download_enabled=model_download.get("enabled", False),
            subagents_enabled=agents.get("enabled", True),
            microphone_enabled=microphone.get("enabled", False),
            microphone_continuous_enabled=microphone.get("continuous_enabled", False),
            microphone_retain_audio=microphone.get("retain_audio", False),
            confirmation_required_for=raw.get(
                "confirmation_required_for", cls().confirmation_required_for
            ),
            audit_log=raw.get("audit_log", True),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "workspace_only": self.workspace_only,
            "filesystem": {
                "allow_read": self.filesystem_allow_read,
                "allow_write": self.filesystem_allow_write,
                "deny": self.filesystem_deny,
            },
            "shell": {
                "enabled": self.shell_enabled,
                "allow_commands": self.shell_allow_commands,
            },
            "scripting": {
                "enabled": self.scripting_enabled,
                "execute_enabled": self.scripting_execute_enabled,
            },
            "network": {
                "enabled": self.network_enabled,
                "allow_domains": self.network_allow_domains,
                "deny_domains": self.network_deny_domains,
            },
            "package_install": {"enabled": self.package_install_enabled},
            "model_download": {"enabled": self.model_download_enabled},
            "agents": {"enabled": self.subagents_enabled},
            "microphone": {
                "enabled": self.microphone_enabled,
                "continuous_enabled": self.microphone_continuous_enabled,
                "retain_audio": self.microphone_retain_audio,
            },
            "confirmation_required_for": self.confirmation_required_for,
            "audit_log": self.audit_log,
        }
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def always_deny(action: str, context: dict) -> bool:
    return False


def always_allow(action: str, context: dict) -> bool:
    return True


def _match_any(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(value, pattern) for pattern in patterns)


def _match_domain(domain: str, patterns: list[str]) -> bool:
    domain = domain.lower()
    for pattern in patterns:
        pattern = pattern.lower()
        if pattern == "*":
            return True
        if fnmatch.fnmatch(domain, pattern):
            return True
    return False


class PolicyEngine:
    def __init__(
        self,
        config: PolicyConfig,
        workspace_root: Path,
        confirm_callback: Optional[ConfirmCallback] = None,
        audit_log_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.workspace_root = workspace_root.resolve()
        self.confirm_callback = confirm_callback or always_deny
        self.audit_log_path = audit_log_path

    # ---- проверки по категориям -------------------------------------------------

    def check_filesystem(self, path: Path, mode: str) -> PolicyDecision:
        assert mode in ("read", "write")
        resolved = path.resolve()
        posix = resolved.as_posix()

        if _match_any(posix, self.config.filesystem_deny):
            return PolicyDecision(False, reason=f"путь запрещён политикой: {posix}")

        try:
            resolved.relative_to(self.workspace_root)
            in_workspace = True
        except ValueError:
            in_workspace = False

        if in_workspace:
            return PolicyDecision(True)

        extra_allow = (
            self.config.filesystem_allow_write
            if mode == "write"
            else self.config.filesystem_allow_read
        )
        if _match_any(posix, extra_allow):
            return PolicyDecision(True)

        if not self.config.workspace_only:
            return PolicyDecision(True)

        action = f"files.{mode}_outside_workspace"
        if action in self.config.confirmation_required_for:
            return PolicyDecision(True, requires_confirmation=True, reason="путь вне рабочей директории")

        return PolicyDecision(False, reason=f"путь вне рабочей директории и не в allow-list: {posix}")

    def check_shell(self, command: str) -> PolicyDecision:
        if not self.config.shell_enabled:
            return PolicyDecision(False, reason="выполнение команд отключено в политике")
        if self.config.shell_allow_commands:
            head = command.strip().split()[0] if command.strip() else ""
            if not any(command.strip().startswith(prefix) for prefix in self.config.shell_allow_commands):
                return PolicyDecision(
                    False,
                    reason=f"команда '{head}' не входит в allow_commands",
                )
        requires_confirmation = "shell.execute" in self.config.confirmation_required_for
        return PolicyDecision(True, requires_confirmation=requires_confirmation)

    def check_scripting_write(self) -> PolicyDecision:
        if not self.config.scripting_enabled:
            return PolicyDecision(False, reason="создание скриптов отключено в политике")
        return PolicyDecision(True)

    def check_scripting_execute(self) -> PolicyDecision:
        if not self.config.scripting_execute_enabled:
            return PolicyDecision(
                False, reason="выполнение скриптов отключено (scripting.execute_enabled=false)"
            )
        requires_confirmation = "scripting.execute" in self.config.confirmation_required_for
        return PolicyDecision(True, requires_confirmation=requires_confirmation)

    def check_network(self, domain: str) -> PolicyDecision:
        if not self.config.network_enabled:
            return PolicyDecision(False, reason="сеть отключена в политике")
        if _match_domain(domain, self.config.network_deny_domains):
            return PolicyDecision(False, reason=f"домен запрещён: {domain}")
        if not _match_domain(domain, self.config.network_allow_domains):
            return PolicyDecision(False, reason=f"домен не в allow-list: {domain}")
        return PolicyDecision(True)

    def check_package_install(self, package: str) -> PolicyDecision:
        if not self.config.package_install_enabled:
            return PolicyDecision(
                False, reason="установка пакетов отключена (package_install.enabled=false)"
            )
        requires_confirmation = "package.install" in self.config.confirmation_required_for
        return PolicyDecision(True, requires_confirmation=requires_confirmation)

    def check_model_download(self) -> PolicyDecision:
        if not self.config.model_download_enabled:
            return PolicyDecision(
                False, reason="скачивание моделей отключено (model_download.enabled=false)"
            )
        requires_confirmation = "model.download" in self.config.confirmation_required_for
        return PolicyDecision(True, requires_confirmation=requires_confirmation)

    def check_agents_spawn(self) -> PolicyDecision:
        if not self.config.subagents_enabled:
            return PolicyDecision(
                False, reason="создание вспомогательных агентов отключено (agents.enabled=false)"
            )
        requires_confirmation = "agents.spawn" in self.config.confirmation_required_for
        return PolicyDecision(True, requires_confirmation=requires_confirmation)

    def check_microphone(self, continuous: bool) -> PolicyDecision:
        if continuous:
            if not self.config.microphone_continuous_enabled:
                return PolicyDecision(
                    False, reason="постоянная запись микрофона отключена (microphone.continuous_enabled=false)"
                )
            # Риск постоянной записи качественно другой (агент слышит всё,
            # что происходит рядом, а не только явно надиктованное) — в
            # отличие от разовой записи, подтверждение запрашивается ВСЕГДА
            # при включении режима, а не по списку confirmation_required_for.
            return PolicyDecision(True, requires_confirmation=True)
        if not self.config.microphone_enabled:
            return PolicyDecision(False, reason="доступ к микрофону отключён (microphone.enabled=false)")
        requires_confirmation = "mic.record" in self.config.confirmation_required_for
        return PolicyDecision(True, requires_confirmation=requires_confirmation)

    # ---- единая точка входа для навыков ------------------------------------------

    def enforce(self, action: str, **context) -> None:
        decision = self._decide(action, context)
        confirmed: Optional[bool] = None

        if decision.allowed and decision.requires_confirmation:
            confirmed = self.confirm_callback(action, context)
            if not confirmed:
                decision = PolicyDecision(False, reason="пользователь не подтвердил действие")

        self._audit(action, context, decision, confirmed)

        if not decision.allowed:
            raise PermissionDenied(f"{action}: {decision.reason or 'запрещено политикой'}")

    def _decide(self, action: str, context: dict) -> PolicyDecision:
        if action in ("files.read", "files.write"):
            mode = "read" if action == "files.read" else "write"
            return self.check_filesystem(Path(context["path"]), mode)
        if action == "shell.execute":
            return self.check_shell(context.get("command", ""))
        if action == "scripting.write":
            return self.check_scripting_write()
        if action == "scripting.execute":
            return self.check_scripting_execute()
        if action == "web.fetch":
            return self.check_network(context.get("domain", ""))
        if action == "package.install":
            return self.check_package_install(context.get("package", ""))
        if action == "model.download":
            return self.check_model_download()
        if action == "agents.spawn":
            return self.check_agents_spawn()
        if action == "mic.record":
            return self.check_microphone(continuous=False)
        if action == "mic.continuous_start":
            return self.check_microphone(continuous=True)
        return PolicyDecision(False, reason=f"неизвестное действие: {action}")

    def _audit(self, action: str, context: dict, decision: PolicyDecision, confirmed: Optional[bool]) -> None:
        if not self.config.audit_log or self.audit_log_path is None:
            return
        entry = {
            "ts": time.time(),
            "action": action,
            "context": {k: str(v) for k, v in context.items()},
            "allowed": decision.allowed,
            "requires_confirmation": decision.requires_confirmation,
            "confirmed": confirmed,
            "reason": decision.reason,
        }
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
