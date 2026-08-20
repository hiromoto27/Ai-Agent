from pathlib import Path

from ai_agent.core.autotune import HardwareInfo, build_profile, detect_hardware, ensure_settings


def test_detect_hardware_returns_sane_values():
    hw = detect_hardware()
    assert hw.cpu_cores >= 1
    assert hw.total_ram_gb > 0
    assert hw.os_name


def test_low_tier_for_weak_hardware():
    hw = HardwareInfo(cpu_cores=2, total_ram_gb=4, has_gpu=False, os_name="Windows")
    profile = build_profile(hw)
    assert profile.tier == "low"
    assert profile.max_parallel_tool_calls == 1
    assert profile.enable_local_llm_default is False


def test_medium_tier():
    hw = HardwareInfo(cpu_cores=6, total_ram_gb=12, has_gpu=False, os_name="Windows")
    profile = build_profile(hw)
    assert profile.tier == "medium"


def test_high_tier_with_gpu_enables_local_llm():
    hw = HardwareInfo(cpu_cores=16, total_ram_gb=32, has_gpu=True, os_name="Windows")
    profile = build_profile(hw)
    assert profile.tier == "high"
    assert profile.enable_local_llm_default is True
    assert profile.max_parallel_tool_calls == 4


def test_high_tier_without_gpu_keeps_local_llm_disabled():
    hw = HardwareInfo(cpu_cores=16, total_ram_gb=32, has_gpu=False, os_name="Linux")
    profile = build_profile(hw)
    assert profile.enable_local_llm_default is False


def test_ensure_settings_creates_file_and_is_idempotent(tmp_path: Path):
    settings_path = tmp_path / "config" / "settings.yaml"
    profile1 = ensure_settings(settings_path)
    assert settings_path.exists()

    # Второй вызов должен читать уже сохранённый профиль, а не пересоздавать
    settings_path.write_text(
        settings_path.read_text(encoding="utf-8").replace(
            f"tier: {profile1.tier}", "tier: low"
        ),
        encoding="utf-8",
    )
    profile2 = ensure_settings(settings_path)
    assert profile2.tier == "low"


def test_ensure_settings_force_overwrites(tmp_path: Path):
    settings_path = tmp_path / "settings.yaml"
    ensure_settings(settings_path)
    original_mtime = settings_path.stat().st_mtime_ns
    profile = ensure_settings(settings_path, force=True)
    assert settings_path.stat().st_mtime_ns >= original_mtime
    assert profile.tier in ("low", "medium", "high")
