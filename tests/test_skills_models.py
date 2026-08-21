from pathlib import Path

from ai_agent.core.skills.models import (
    DownloadHuggingFaceModelSkill,
    RecommendModelsSkill,
    SearchHuggingFaceSkill,
)


def test_recommend_models_needs_no_permissions(locked_context):
    """Рекомендации — чисто локальная операция, работает даже при
    полностью запертой политике (нет сети, нет shell)."""
    result = RecommendModelsSkill().run(locked_context)
    assert result.ok
    assert "hardware" in result.data
    assert isinstance(result.data["models"], list)


class _FakeModelInfo:
    def __init__(self, id: str, downloads: int, likes: int) -> None:
        self.id = id
        self.downloads = downloads
        self.likes = likes
        self.tags: list[str] = []


class _FakeApi:
    def list_models(self, *, search, limit, sort):
        return [_FakeModelInfo("Qwen/Qwen2.5-1.5B-Instruct-GGUF", 500, 10)][:limit]


def test_search_huggingface_allowed_by_default(permissive_context):
    skill = SearchHuggingFaceSkill(api=_FakeApi())
    result = skill.run(permissive_context, query="qwen instruct")
    assert result.ok
    assert result.data["results"][0]["id"] == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"


def test_search_huggingface_denied_when_network_disabled(locked_context):
    locked_context.policy.config.network_enabled = False
    skill = SearchHuggingFaceSkill(api=_FakeApi())
    result = skill.run(locked_context, query="qwen")
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_download_huggingface_denied_by_default(locked_context):
    skill = DownloadHuggingFaceModelSkill(download_fn=lambda **kw: kw["local_dir"])
    result = skill.run(locked_context, repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_download_huggingface_allowed_with_permissive_context(permissive_context, workspace: Path):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        out = Path(kwargs["local_dir"])
        out.mkdir(parents=True, exist_ok=True)
        return str(out)

    skill = DownloadHuggingFaceModelSkill(download_fn=fake_download)
    result = skill.run(permissive_context, repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")

    assert result.ok
    expected_dir = workspace / "models" / "Qwen__Qwen2.5-1.5B-Instruct-GGUF"
    assert Path(result.data["path"]) == expected_dir
    assert expected_dir.exists()
    assert calls[0]["repo_id"] == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
