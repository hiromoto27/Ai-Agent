from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from ai_agent.core.autotune import HardwareInfo
from ai_agent.core.hf_models import (
    CURATED_MODELS,
    HFSearchResult,
    clear_token,
    download_model,
    get_saved_token,
    is_auth_error,
    recommend_models,
    save_token,
    search_models,
)


def _http_error(cls, status_code: int, message: str = "error"):
    response = httpx.Response(status_code, request=httpx.Request("GET", "https://huggingface.co/x"))
    return cls(message, response=response)


def test_xet_backend_disabled_by_default():
    """Регрессия: скачивание больших файлов через ускоренный бэкенд Xet
    молча обрывалось на некоторых сетях, оставляя пустую папку модели
    (см. историю коммитов). Модуль должен выставлять HF_HUB_DISABLE_XET
    до первого импорта huggingface_hub, чтобы качать обычным HTTPS."""
    import huggingface_hub.constants as hf_constants

    assert hf_constants.HF_HUB_DISABLE_XET is True


def test_recommend_models_weak_pc_gets_only_smallest():
    hw = HardwareInfo(cpu_cores=2, total_ram_gb=4, has_gpu=False, os_name="Windows")
    recs = recommend_models(hw)
    assert recs  # хоть что-то должно найтись даже на слабом ПК
    assert all(m.approx_size_gb <= hw.total_ram_gb * 0.7 for m in recs)
    assert all(not m.requires_gpu for m in recs)


def test_recommend_models_tiny_ram_gets_nothing():
    hw = HardwareInfo(cpu_cores=1, total_ram_gb=1, has_gpu=False, os_name="Linux")
    recs = recommend_models(hw)
    assert recs == []


def test_recommend_models_excludes_gpu_models_without_gpu():
    hw = HardwareInfo(cpu_cores=16, total_ram_gb=64, has_gpu=False, os_name="Linux")
    recs = recommend_models(hw)
    assert all(not m.requires_gpu for m in recs)
    assert len(recs) >= 1


def test_recommend_models_includes_gpu_models_with_gpu():
    hw = HardwareInfo(cpu_cores=16, total_ram_gb=64, has_gpu=True, os_name="Linux")
    recs = recommend_models(hw)
    assert any(m.requires_gpu for m in recs)


def test_recommend_models_sorted_ascending_by_size():
    hw = HardwareInfo(cpu_cores=16, total_ram_gb=64, has_gpu=True, os_name="Linux")
    recs = recommend_models(hw)
    sizes = [m.approx_size_gb for m in recs]
    assert sizes == sorted(sizes)


def test_curated_models_have_unique_repo_ids():
    ids = [m.repo_id for m in CURATED_MODELS]
    assert len(ids) == len(set(ids))


class _FakeModelInfo:
    def __init__(self, id: str, downloads: int, likes: int, tags: list[str]) -> None:
        self.id = id
        self.downloads = downloads
        self.likes = likes
        self.tags = tags


class _FakeHFApi:
    def __init__(self, results: list[_FakeModelInfo]) -> None:
        self.results = results
        self.last_call: dict | None = None

    def list_models(self, *, search, limit, sort):
        self.last_call = {"search": search, "limit": limit, "sort": sort}
        return self.results[:limit]


def test_search_models_maps_results_and_forwards_query():
    fake = _FakeHFApi([_FakeModelInfo("Qwen/Qwen2.5-1.5B-Instruct-GGUF", 1000, 50, ["gguf", "text-generation"])])
    results = search_models("qwen instruct gguf", limit=5, api=fake)
    assert results == [HFSearchResult("Qwen/Qwen2.5-1.5B-Instruct-GGUF", 1000, 50, ["gguf", "text-generation"])]
    assert fake.last_call == {"search": "qwen instruct gguf", "limit": 5, "sort": "downloads"}


def test_search_models_handles_missing_optional_fields():
    class _Bare:
        id = "some/repo"

    fake = _FakeHFApi([_Bare()])
    results = search_models("x", api=fake)
    assert results[0].downloads is None
    assert results[0].likes is None
    assert results[0].tags == []


def test_download_model_calls_injected_fn_with_expected_args(tmp_path: Path):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        out = Path(kwargs["local_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    target = tmp_path / "models"
    result = download_model("Qwen/Qwen2.5-1.5B-Instruct-GGUF", target, download_fn=fake_download)

    assert result == target / "Qwen__Qwen2.5-1.5B-Instruct-GGUF"
    assert calls[0]["repo_id"] == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    assert calls[0]["allow_patterns"] is None


def test_download_model_forwards_allow_patterns(tmp_path: Path):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    download_model(
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        tmp_path,
        allow_patterns=["*q4_k_m.gguf"],
        download_fn=fake_download,
    )
    assert calls[0]["allow_patterns"] == ["*q4_k_m.gguf"]


def test_is_auth_error_for_403():
    assert is_auth_error(_http_error(HfHubHTTPError, 403, "forbidden"))


def test_is_auth_error_for_401():
    assert is_auth_error(_http_error(HfHubHTTPError, 401, "unauthorized"))


def test_is_auth_error_for_gated_repo():
    assert is_auth_error(_http_error(GatedRepoError, 403, "gated"))


def test_is_auth_error_false_for_404():
    assert not is_auth_error(_http_error(HfHubHTTPError, 404, "not found"))


def test_is_auth_error_false_for_unrelated_exception():
    assert not is_auth_error(ValueError("something else"))


def test_save_token_validates_then_logs_in(monkeypatch):
    calls = {}

    def fake_whoami(token=None):
        calls["whoami_token"] = token
        return {"name": "alice"}

    def fake_login(token=None, add_to_git_credential=False, skip_if_logged_in=True):
        calls["login_token"] = token

    monkeypatch.setattr("huggingface_hub.whoami", fake_whoami)
    monkeypatch.setattr("huggingface_hub.login", fake_login)

    info = save_token("hf_abc123")

    assert info == {"name": "alice"}
    assert calls["whoami_token"] == "hf_abc123"
    assert calls["login_token"] == "hf_abc123"


def test_save_token_rejects_empty_string():
    with pytest.raises(ValueError):
        save_token("   ")


def test_save_token_propagates_invalid_token_error(monkeypatch):
    def fake_whoami(token=None):
        raise _http_error(HfHubHTTPError, 401, "invalid token")

    monkeypatch.setattr("huggingface_hub.whoami", fake_whoami)
    with pytest.raises(HfHubHTTPError):
        save_token("bad-token")


def test_get_saved_token_wraps_get_token(monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "cached-token")
    assert get_saved_token() == "cached-token"


def test_clear_token_calls_logout(monkeypatch):
    called = []
    monkeypatch.setattr("huggingface_hub.logout", lambda token_name=None: called.append(token_name))
    clear_token()
    assert called == [None]
