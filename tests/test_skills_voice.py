from pathlib import Path

from ai_agent.core.autotune import Profile
from ai_agent.core.policy import PolicyConfig, PolicyEngine
from ai_agent.core.policy.engine import always_allow, always_deny
from ai_agent.core.skills.base import SkillContext
from ai_agent.core.skills.voice import (
    TasksCreateSkill,
    TasksListSkill,
    TasksSetReminderSkill,
    VoiceCheckSetupSkill,
    VoiceConfirmTaskSkill,
    VoiceDownloadSttModelSkill,
    VoiceGenerateProtocolSkill,
    VoiceInstallDependenciesSkill,
    VoiceListInputDevicesSkill,
    VoiceRecordOnceSkill,
    VoiceSetInputDeviceSkill,
    VoiceStartContinuousSkill,
    VoiceStopContinuousSkill,
)
from ai_agent.core.voice.service import VoiceService
from ai_agent.core.voice.store import VoiceStore
from test_voice_service import FakeCapture, FakeSTT

_TEST_PROFILE = Profile(
    tier="medium",
    max_parallel_tool_calls=2,
    max_agent_steps=10,
    memory_retrieval_k=4,
    max_context_episodes=100,
    enable_local_llm_default=False,
    max_script_timeout_sec=10,
)


def make_context(
    tmp_path: Path,
    *,
    microphone_enabled: bool = False,
    microphone_continuous_enabled: bool = False,
    confirm_callback=always_deny,
    voice: VoiceService | None = None,
) -> SkillContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config = PolicyConfig.default()
    config.microphone_enabled = microphone_enabled
    config.microphone_continuous_enabled = microphone_continuous_enabled
    engine = PolicyEngine(
        config, workspace_root=workspace, confirm_callback=confirm_callback, audit_log_path=tmp_path / "audit.jsonl"
    )
    return SkillContext(workspace_root=workspace, policy=engine, profile=_TEST_PROFILE, voice=voice)


def make_fake_voice(tmp_path: Path, texts: list[str], chunks: list[bytes] | None = None) -> VoiceService:
    """VoiceService на фейковых захвате/STT. FakeCapture — конечный генератор
    (в отличие от FakeContinuousCapture из test_voice_service.py, которая
    "висит" до stop()) — безопасный вариант по умолчанию, чтобы навык
    record_once не подвешивал тест, если забыть явно остановить запись."""
    store = VoiceStore(tmp_path / "voice.sqlite3")
    chunks = chunks if chunks is not None else [b"chunk"] * len(texts)
    return VoiceService(
        store=store,
        audio_capture_factory=lambda: FakeCapture(chunks),
        stt_factory=lambda: FakeSTT(list(texts)),
    )


def test_record_once_denied_when_microphone_disabled(tmp_path):
    context = make_context(tmp_path, microphone_enabled=False)
    result = VoiceRecordOnceSkill().run(context, max_seconds=1)
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_record_once_fails_cleanly_without_voice_service(tmp_path):
    context = make_context(tmp_path, microphone_enabled=True, voice=None)
    result = VoiceRecordOnceSkill().run(context, max_seconds=1)
    assert not result.ok
    assert "диктофон" in result.error


def test_record_once_returns_transcribed_utterances(tmp_path):
    voice = make_fake_voice(tmp_path, ["Обсудили бюджет квартала"])
    context = make_context(tmp_path, microphone_enabled=True, voice=voice)
    result = VoiceRecordOnceSkill().run(context, max_seconds=1)
    assert result.ok
    assert len(result.data["utterance_ids"]) == 1


def test_start_continuous_requires_confirmation_denied(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(
        tmp_path, microphone_continuous_enabled=True, confirm_callback=always_deny, voice=voice
    )
    result = VoiceStartContinuousSkill().run(context)
    assert not result.ok
    assert voice.is_continuous_active is False


def test_start_continuous_requires_confirmation_every_call(tmp_path):
    calls = []

    def counting_confirm(action, ctx):
        calls.append(action)
        return True

    voice = make_fake_voice(tmp_path, [])
    context = make_context(
        tmp_path, microphone_continuous_enabled=True, confirm_callback=counting_confirm, voice=voice
    )
    try:
        result1 = VoiceStartContinuousSkill().run(context)
        assert result1.ok
        VoiceStopContinuousSkill().run(context)
        result2 = VoiceStartContinuousSkill().run(context)
        assert result2.ok
    finally:
        voice.stop_continuous()

    # Каждое включение постоянной записи запрашивает подтверждение заново.
    assert calls == ["mic.continuous_start", "mic.continuous_start"]


def test_stop_continuous_without_voice_service_errors(tmp_path):
    context = make_context(tmp_path, voice=None)
    result = VoiceStopContinuousSkill().run(context)
    assert not result.ok


def test_tasks_create_and_list(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)
    created = TasksCreateSkill().run(context, title="Бюджет", description="Финансы квартала")
    assert created.ok
    task_id = created.data["task_id"]

    listed = TasksListSkill().run(context)
    assert listed.ok
    assert f"#{task_id}" in listed.output


def test_tasks_set_reminder_unknown_task(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)
    result = TasksSetReminderSkill().run(context, task_id=999, fire_at=123.0)
    assert not result.ok


def test_tasks_set_reminder_success(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)
    task = TasksCreateSkill().run(context, title="Бюджет")
    result = TasksSetReminderSkill().run(context, task_id=task.data["task_id"], fire_at=123.0)
    assert result.ok
    assert "reminder_id" in result.data


def test_voice_confirm_task_unknown_ids(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)
    result = VoiceConfirmTaskSkill().run(context, utterance_id=1, task_id=1, belongs=True)
    assert not result.ok


def test_voice_confirm_task_updates_classifier(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)
    task = TasksCreateSkill().run(context, title="Проект").data["task_id"]
    recording = voice.store.start_recording(mode="once")
    utterance = voice.store.add_utterance(recording.id, "Реплика про проект")

    result = VoiceConfirmTaskSkill().run(context, utterance_id=utterance.id, task_id=task, belongs=True)
    assert result.ok
    assert voice.store.get_utterance(utterance.id).task_id == task


def test_generate_protocol_creates_docx(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)
    task_id = TasksCreateSkill().run(context, title="Бюджет").data["task_id"]
    recording = voice.store.start_recording(mode="once")
    voice.store.add_utterance(recording.id, "Нужно согласовать бюджет", task_id=task_id)

    result = VoiceGenerateProtocolSkill().run(context, filename="protocol")
    assert result.ok
    path = context.workspace_root / "documents" / "protocol.docx"
    assert path.exists()


# ---- устройства записи -------------------------------------------------------------------


def test_list_input_devices_reports_missing_dependencies(monkeypatch, tmp_path):
    import ai_agent.core.voice.audio as audio_module

    def _raise():
        raise RuntimeError("sounddevice не установлен (extras: voice)")

    monkeypatch.setattr(audio_module, "list_input_devices", _raise)
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)

    result = VoiceListInputDevicesSkill().run(context)
    assert not result.ok
    assert "sounddevice" in result.error


def test_list_input_devices_returns_devices(monkeypatch, tmp_path):
    import ai_agent.core.voice.audio as audio_module

    devices = [{"index": 0, "name": "Микрофон", "channels": 1, "default_samplerate": 16000.0, "is_default": True}]
    monkeypatch.setattr(audio_module, "list_input_devices", lambda: devices)
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)

    result = VoiceListInputDevicesSkill().run(context)
    assert result.ok
    assert result.data["devices"] == devices


def test_set_input_device_updates_service(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)

    result = VoiceSetInputDeviceSkill().run(context, device_index=3)
    assert result.ok
    assert voice.input_device == 3


def test_set_input_device_without_voice_service_errors(tmp_path):
    context = make_context(tmp_path, voice=None)
    result = VoiceSetInputDeviceSkill().run(context, device_index=1)
    assert not result.ok


# ---- установка компонентов -----------------------------------------------------------------


def test_check_setup_reports_status_and_recommendation(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)

    result = VoiceCheckSetupSkill().run(context)
    assert result.ok
    # _TEST_PROFILE.tier == "medium" -> "base" (см. setup.recommend_stt_model_size)
    assert result.data["recommended_model"] == "base"
    assert result.data["hardware_tier"] == "medium"


def test_install_dependencies_denied_without_policy(tmp_path, monkeypatch):
    import ai_agent.core.voice.setup as setup_module

    monkeypatch.setattr(setup_module, "check_dependencies", lambda: setup_module.DependencyStatus(False, "x", False, "y"))
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)  # package_install выключен по умолчанию

    result = VoiceInstallDependenciesSkill().run(context)
    assert not result.ok


def test_install_dependencies_already_ready_short_circuits(tmp_path, monkeypatch):
    import ai_agent.core.voice.setup as setup_module

    monkeypatch.setattr(setup_module, "check_dependencies", lambda: setup_module.DependencyStatus(True, "", True, ""))
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)

    result = VoiceInstallDependenciesSkill().run(context)
    assert result.ok
    assert "уже установлены" in result.output


def test_install_dependencies_runs_pip_with_confirmation(tmp_path, monkeypatch):
    import ai_agent.core.skills.voice as voice_skills
    import ai_agent.core.voice.setup as setup_module

    monkeypatch.setattr(setup_module, "check_dependencies", lambda: setup_module.DependencyStatus(False, "x", False, "y"))
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice, confirm_callback=always_allow)
    context.policy.config.package_install_enabled = True

    calls = []

    class _FakeCompleted:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeCompleted()

    monkeypatch.setattr(voice_skills.subprocess, "run", fake_run)

    result = VoiceInstallDependenciesSkill(pip_cmd=["fake-pip"]).run(context)
    assert result.ok
    assert calls[0][:2] == ["fake-pip", "install"]


def test_install_dependencies_reports_pip_failure(tmp_path, monkeypatch):
    import ai_agent.core.skills.voice as voice_skills
    import ai_agent.core.voice.setup as setup_module

    monkeypatch.setattr(setup_module, "check_dependencies", lambda: setup_module.DependencyStatus(False, "x", False, "y"))
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice, confirm_callback=always_allow)
    context.policy.config.package_install_enabled = True

    class _FakeFailed:
        returncode = 1
        stderr = "боль и страдание"

    monkeypatch.setattr(voice_skills.subprocess, "run", lambda cmd, **kwargs: _FakeFailed())

    result = VoiceInstallDependenciesSkill().run(context)
    assert not result.ok
    assert "боль и страдание" in result.error


def test_download_stt_model_denied_without_policy(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    context = make_context(tmp_path, voice=voice)  # model_download выключен по умолчанию

    result = VoiceDownloadSttModelSkill().run(context)
    assert not result.ok


def test_download_stt_model_uses_recommended_size_and_preloads(tmp_path):
    voice = make_fake_voice(tmp_path, [])
    preload_calls = []
    voice.preload_stt = lambda model_size=None: preload_calls.append(model_size)
    context = make_context(tmp_path, voice=voice, confirm_callback=always_allow)
    context.policy.config.model_download_enabled = True

    result = VoiceDownloadSttModelSkill().run(context)
    assert result.ok
    assert preload_calls == ["base"]  # _TEST_PROFILE.tier == "medium" -> "base"
    assert result.data["model_size"] == "base"


def test_download_stt_model_reports_load_error(tmp_path):
    voice = make_fake_voice(tmp_path, [])

    def boom(model_size=None):
        raise RuntimeError("faster-whisper не установлен (extras: voice)")

    voice.preload_stt = boom
    context = make_context(tmp_path, voice=voice, confirm_callback=always_allow)
    context.policy.config.model_download_enabled = True

    result = VoiceDownloadSttModelSkill().run(context)
    assert not result.ok
    assert "faster-whisper" in result.error
