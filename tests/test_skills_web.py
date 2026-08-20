import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ai_agent.core.autotune import Profile
from ai_agent.core.policy import PolicyConfig, PolicyEngine
from ai_agent.core.policy.engine import always_allow
from ai_agent.core.skills.base import SkillContext
from ai_agent.core.skills.web import FetchUrlSkill, SearchWebSkill

SEARCH_HTML = """
<html><body>
<a rel="nofollow" class="result__a" href="https://example.org/tool-one">Cool Tool One</a>
<a rel="nofollow" class="result__a" href="https://example.org/tool-two">Cool Tool Two</a>
</body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (стандартное имя метода http.server)
        if self.path.startswith("/page"):
            body = b"<html><body>Hello from test page</body></html>"
        elif self.path.startswith("/search"):
            body = SEARCH_HTML.encode("utf-8")
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # подавляем шум в выводе тестов
        pass


@pytest.fixture(scope="module")
def local_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def web_context(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = PolicyConfig.default()
    config.network_allow_domains = ["127.0.0.1"]
    engine = PolicyEngine(config, workspace_root=workspace, confirm_callback=always_allow)
    profile = Profile("medium", 2, 10, 4, 100, False, 10)
    return SkillContext(workspace_root=workspace, policy=engine, profile=profile)


def test_fetch_url(local_server, web_context):
    port = local_server.server_port
    result = FetchUrlSkill().run(web_context, url=f"http://127.0.0.1:{port}/page")
    assert result.ok
    assert "Hello from test page" in result.output


def test_fetch_url_domain_denied(local_server, web_context):
    web_context.policy.config.network_allow_domains = ["only-this-domain.example"]
    port = local_server.server_port
    result = FetchUrlSkill().run(web_context, url=f"http://127.0.0.1:{port}/page")
    assert not result.ok
    assert "доступ запрещён" in result.error


def test_search_web(local_server, web_context):
    port = local_server.server_port
    skill = SearchWebSkill(endpoint=f"http://127.0.0.1:{port}/search")
    result = skill.run(web_context, query="agent tools")
    assert result.ok
    assert len(result.data["results"]) == 2
    assert result.data["results"][0]["title"] == "Cool Tool One"
    assert result.data["results"][0]["url"] == "https://example.org/tool-one"
