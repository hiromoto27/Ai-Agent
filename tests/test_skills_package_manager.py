import sys

from ai_agent.core.skills.package_manager import InstallPackageSkill


def test_install_package_invokes_pip_with_confirmation(permissive_context):
    # Подменяем pip на echo-скрипт, чтобы не ходить в сеть в тестах.
    fake_pip = [sys.executable, "-c", "import sys; print('installed', sys.argv[-1])"]
    skill = InstallPackageSkill(pip_cmd=fake_pip)
    result = skill.run(permissive_context, package="some-fake-package")
    assert result.ok
    assert result.data["package"] == "some-fake-package"


def test_install_package_denied_by_default(locked_context):
    skill = InstallPackageSkill()
    result = skill.run(locked_context, package="requests")
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_install_package_rejects_malicious_name(permissive_context):
    skill = InstallPackageSkill()
    result = skill.run(permissive_context, package="requests; rm -rf /")
    assert not result.ok
    assert "некорректное имя пакета" in result.error


def test_install_package_pip_failure_reported(permissive_context):
    fake_pip = [sys.executable, "-c", "import sys; sys.exit(1)"]
    skill = InstallPackageSkill(pip_cmd=fake_pip)
    result = skill.run(permissive_context, package="whatever")
    assert not result.ok
