import sys

from ai_agent.core.skills.package_manager import (
    EstimatePackageSizeSkill,
    InstallPackageSkill,
    fetch_pypi_size,
)


class _FakeResponse:
    def __init__(self, data, status_ok=True) -> None:
        self._data = data
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._data


class _FakeHttpClient:
    def __init__(self, data=None, raise_on_get=False) -> None:
        self.data = data
        self.raise_on_get = raise_on_get
        self.last_url: str | None = None
        self.closed = False

    def get(self, url):
        self.last_url = url
        if self.raise_on_get:
            raise ConnectionError("сеть недоступна")
        return _FakeResponse(self.data)

    def close(self):
        self.closed = True


def _pypi_payload(version="1.2.3", size=1_048_576):
    return {"info": {"version": version}, "urls": [{"size": size}]}


def test_install_package_invokes_pip_with_confirmation(permissive_context):
    # Подменяем pip на echo-скрипт, чтобы не ходить в сеть в тестах.
    fake_pip = [sys.executable, "-c", "import sys; print('installed', sys.argv[-1])"]
    fake_http = _FakeHttpClient(data=_pypi_payload())
    skill = InstallPackageSkill(pip_cmd=fake_pip, http_client=fake_http)
    result = skill.run(permissive_context, package="some-fake-package")
    assert result.ok
    assert result.data["package"] == "some-fake-package"


def test_install_package_denied_by_default(locked_context):
    skill = InstallPackageSkill(http_client=_FakeHttpClient(raise_on_get=True))
    result = skill.run(locked_context, package="requests")
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_install_package_rejects_malicious_name(permissive_context):
    skill = InstallPackageSkill(http_client=_FakeHttpClient(raise_on_get=True))
    result = skill.run(permissive_context, package="requests; rm -rf /")
    assert not result.ok
    assert "некорректное имя пакета" in result.error


def test_install_package_pip_failure_reported(permissive_context):
    fake_pip = [sys.executable, "-c", "import sys; sys.exit(1)"]
    skill = InstallPackageSkill(pip_cmd=fake_pip, http_client=_FakeHttpClient(raise_on_get=True))
    result = skill.run(permissive_context, package="whatever")
    assert not result.ok


def test_install_package_proceeds_when_size_lookup_fails(permissive_context):
    """Недоступность PyPI — не повод блокировать саму установку, это лишь подсказка."""
    fake_pip = [sys.executable, "-c", "import sys; print('installed', sys.argv[-1])"]
    skill = InstallPackageSkill(pip_cmd=fake_pip, http_client=_FakeHttpClient(raise_on_get=True))
    result = skill.run(permissive_context, package="requests")
    assert result.ok


def test_fetch_pypi_size_returns_bytes_and_version():
    fake_http = _FakeHttpClient(data=_pypi_payload(version="2.32.0", size=2_000_000))
    size_bytes, version = fetch_pypi_size("requests", "", http_client=fake_http)
    assert size_bytes == 2_000_000
    assert version == "2.32.0"
    assert fake_http.closed is False  # клиент передан извне — не закрываем его сами


def test_fetch_pypi_size_uses_versioned_url_when_pinned():
    fake_http = _FakeHttpClient(data=_pypi_payload(version="2.32.0", size=2_000_000))
    fetch_pypi_size("requests", "2.32.0", http_client=fake_http)
    assert "requests/2.32.0/json" in fake_http.last_url


def test_estimate_package_size_skill_reports_mb(permissive_context):
    fake_http = _FakeHttpClient(data=_pypi_payload(version="2.32.0", size=1_048_576))
    skill = EstimatePackageSizeSkill(http_client=fake_http)
    result = skill.run(permissive_context, package="requests")
    assert result.ok
    assert result.data["size_bytes"] == 1_048_576
    assert result.data["version"] == "2.32.0"
    assert "1.0 МБ" in result.output


def test_estimate_package_size_skill_rejects_malicious_name(permissive_context):
    skill = EstimatePackageSizeSkill(http_client=_FakeHttpClient(raise_on_get=True))
    result = skill.run(permissive_context, package="requests; rm -rf /")
    assert not result.ok
    assert "некорректное имя пакета" in result.error


def test_estimate_package_size_skill_reports_network_failure(permissive_context):
    skill = EstimatePackageSizeSkill(http_client=_FakeHttpClient(raise_on_get=True))
    result = skill.run(permissive_context, package="requests")
    assert not result.ok
    assert "не удалось получить размер пакета" in result.error


def test_estimate_package_size_skill_denied_without_network(locked_context):
    locked_context.policy.config.network_enabled = False
    skill = EstimatePackageSizeSkill(http_client=_FakeHttpClient(data=_pypi_payload()))
    result = skill.run(locked_context, package="requests")
    assert not result.ok
    assert "доступ запрещён" in result.error
