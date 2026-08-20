"""Автонастройка агента под мощность текущего ПК.

Определяет доступные ресурсы (CPU, RAM, GPU) и строит профиль
параметров исполнения (лимиты параллелизма, размер контекста памяти,
доступность локальной LLM и т.д.). Всё через чистые функции, чтобы
их было легко тестировать с подставными данными оборудования.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import yaml

try:
    import psutil
except ImportError:  # pragma: no cover - psutil объявлен обязательной зависимостью
    psutil = None  # type: ignore[assignment]

Tier = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class HardwareInfo:
    cpu_cores: int
    total_ram_gb: float
    has_gpu: bool
    os_name: str


@dataclass(frozen=True)
class Profile:
    tier: Tier
    max_parallel_tool_calls: int
    max_agent_steps: int
    memory_retrieval_k: int
    max_context_episodes: int
    enable_local_llm_default: bool
    max_script_timeout_sec: int

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_gpu() -> bool:
    """Грубая эвристика наличия GPU: nvidia-smi или наличие CUDA через torch."""
    if shutil.which("nvidia-smi") is not None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, timeout=3, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except (subprocess.SubprocessError, OSError):
            pass
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    return False


def detect_hardware() -> HardwareInfo:
    cpu_cores = os.cpu_count() or 1
    if psutil is not None:
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    else:  # pragma: no cover - защитный путь без psutil
        total_ram_gb = 4.0
    return HardwareInfo(
        cpu_cores=cpu_cores,
        total_ram_gb=round(total_ram_gb, 2),
        has_gpu=_detect_gpu(),
        os_name=platform.system(),
    )


def _classify_tier(hw: HardwareInfo) -> Tier:
    if hw.cpu_cores <= 4 or hw.total_ram_gb < 8:
        return "low"
    if hw.cpu_cores <= 8 or hw.total_ram_gb < 16:
        return "medium"
    return "high"


def build_profile(hw: HardwareInfo) -> Profile:
    tier = _classify_tier(hw)

    if tier == "low":
        return Profile(
            tier=tier,
            max_parallel_tool_calls=1,
            max_agent_steps=6,
            memory_retrieval_k=2,
            max_context_episodes=20,
            enable_local_llm_default=False,
            max_script_timeout_sec=15,
        )
    if tier == "medium":
        return Profile(
            tier=tier,
            max_parallel_tool_calls=2,
            max_agent_steps=10,
            memory_retrieval_k=4,
            max_context_episodes=100,
            enable_local_llm_default=False,
            max_script_timeout_sec=30,
        )
    return Profile(
        tier=tier,
        max_parallel_tool_calls=4,
        max_agent_steps=16,
        memory_retrieval_k=6,
        max_context_episodes=500,
        enable_local_llm_default=hw.has_gpu,
        max_script_timeout_sec=60,
    )


def ensure_settings(settings_path: Path, force: bool = False) -> Profile:
    """Создаёт (если нет) config/settings.yaml с автоопределённым профилем.

    Если файл уже существует и force=False, профиль перечитывается из него
    (пользователь мог отредактировать вручную) — автонастройка не
    перетирает ручные правки при обычном запуске.
    """
    if settings_path.exists() and not force:
        data = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        profile_data = data.get("profile")
        if profile_data:
            return Profile(**profile_data)

    hw = detect_hardware()
    profile = build_profile(hw)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        yaml.safe_dump(
            {"hardware": asdict(hw), "profile": profile.to_dict()},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return profile
