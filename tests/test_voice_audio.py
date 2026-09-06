import struct

import pytest

from ai_agent.core.voice.audio import dependencies_available, frame_level, list_input_devices


def test_dependencies_available_returns_bool_and_reason():
    ok, reason = dependencies_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)
    assert ok or reason  # если не готово — обязана быть понятная причина


def test_list_input_devices_raises_clean_error_without_dependencies(monkeypatch):
    import ai_agent.core.voice.audio as audio_module

    monkeypatch.setattr(audio_module, "dependencies_available", lambda: (False, "sounddevice не установлен (extras: voice)"))
    with pytest.raises(RuntimeError, match="sounddevice"):
        audio_module.list_input_devices()


def test_frame_level_silence_is_zero():
    silence = b"\x00\x00" * 480
    assert frame_level(silence) == 0.0


def test_frame_level_empty_bytes_is_zero():
    assert frame_level(b"") == 0.0


def test_frame_level_full_scale_is_near_one():
    loud = struct.pack("<" + "h" * 480, *([32767] * 480))
    assert frame_level(loud) > 0.99


def test_frame_level_scales_with_amplitude():
    quiet = struct.pack("<" + "h" * 480, *([1000] * 480))
    loud = struct.pack("<" + "h" * 480, *([20000] * 480))
    assert frame_level(quiet) < frame_level(loud)


def test_frame_level_never_exceeds_one():
    loud = struct.pack("<" + "h" * 480, *([32767] * 480))
    assert frame_level(loud) <= 1.0
