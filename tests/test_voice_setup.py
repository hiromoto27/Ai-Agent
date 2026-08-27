from ai_agent.core.autotune import HardwareInfo, Profile
from ai_agent.core.voice.setup import DependencyStatus, check_dependencies, recommend_stt_model_size


def _profile(tier: str) -> Profile:
    return Profile(
        tier=tier,
        max_parallel_tool_calls=1,
        max_agent_steps=1,
        memory_retrieval_k=1,
        max_context_episodes=1,
        enable_local_llm_default=False,
        max_script_timeout_sec=1,
    )


def test_check_dependencies_returns_consistent_status():
    status = check_dependencies()
    assert isinstance(status, DependencyStatus)
    assert status.ready == (status.audio_ok and status.stt_ok)


def test_recommend_stt_model_size_by_tier_without_gpu():
    hw = HardwareInfo(cpu_cores=4, total_ram_gb=8, has_gpu=False, os_name="Linux")
    assert recommend_stt_model_size(hw, _profile("low")) == "tiny"
    assert recommend_stt_model_size(hw, _profile("medium")) == "base"
    assert recommend_stt_model_size(hw, _profile("high")) == "small"


def test_recommend_stt_model_size_bumps_up_with_gpu_on_high_tier():
    hw = HardwareInfo(cpu_cores=16, total_ram_gb=32, has_gpu=True, os_name="Linux")
    assert recommend_stt_model_size(hw, _profile("high")) == "medium"


def test_recommend_stt_model_size_gpu_does_not_affect_lower_tiers():
    hw = HardwareInfo(cpu_cores=2, total_ram_gb=4, has_gpu=True, os_name="Linux")
    assert recommend_stt_model_size(hw, _profile("low")) == "tiny"
