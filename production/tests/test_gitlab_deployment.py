"""GitLab protocol, archive safety and installed notebook deployment."""

import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import nbformat
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
TOKEN = "deployment-test-token"
SPEC = importlib.util.spec_from_file_location(
    "gitlab_bootstrap", ROOT / "runtime/bootstrap.py",
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def archive_bytes(entries=None):
    """Build a GitLab-shaped archive or a supplied malformed fixture."""
    if entries is None:
        entries = [("repo/production/pyproject.toml", b"[project]\n")]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


@pytest.fixture
def gitlab():
    """Serve deterministic GitLab responses and retain protocol evidence."""
    state = {"archive": archive_bytes(), "status": 200, "calls": []}

    class Handler(BaseHTTPRequestHandler):
        """Serve commit, raw bootstrap and archive requests on loopback."""

        def log_message(self, *args):
            """Keep HTTP logging out of notebook/test output."""

        def do_GET(self):
            """Require authentication and return the selected fixture."""
            state["calls"].append((self.path, dict(self.headers)))
            if self.headers.get("PRIVATE-TOKEN") != TOKEN:
                status, body = 401, b"Unauthorized"
            elif "/repository/commits/" in self.path:
                status, body = 200, json.dumps({"id": SHA}).encode()
            elif "/repository/files/" in self.path:
                status = 200
                body = (ROOT / "runtime/bootstrap.py").read_bytes()
            else:
                status, body = state["status"], state["archive"]
            self.send_response(status)
            if status == 302:
                self.send_header("Location", "/should-not-follow")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["api"] = f"http://127.0.0.1:{server.server_port}/api/v4"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def session():
    """Provide caller-owned authentication and clean it after each test."""
    with requests.Session() as client:
        client.headers["PRIVATE-TOKEN"] = TOKEN
        try:
            yield client
        finally:
            client.headers.clear()


@pytest.fixture
def pip_stub(monkeypatch):
    """Inspect installation arguments/files without changing this venv."""
    calls = []

    def run(command, **options):
        """Confirm installation source exists until pip has returned."""
        calls.append(command)
        assert options == {
            "capture_output": True, "text": True, "check": False,
        }
        assert TOKEN not in repr(command)
        if "install" in command:
            assert (Path(command[-1]) / "pyproject.toml").is_file()
            output = "installed"
        else:
            output = json.dumps({
                "package_version": "0.2.0", "python": sys.executable,
            })
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(bootstrap.subprocess, "run", run)
    return calls


def test_install_pins_archive_and_cleans_source(gitlab, session, pip_stub):
    """Use an encoded project path, same commit and current interpreter."""
    summary = bootstrap.install_project(
        gitlab["api"], "group/nested/project", SHA, session,
    )
    assert summary == {
        "commit": SHA, "package_version": "0.2.0",
        "python": sys.executable,
    }
    url, headers = gitlab["calls"][0]
    assert "/projects/group%2Fnested%2Fproject/" in url
    assert parse_qs(urlsplit(url).query) == {
        "sha": [SHA], "path": ["production"],
        "include_lfs_blobs": ["false"],
    }
    assert headers["PRIVATE-TOKEN"] == TOKEN
    assert pip_stub[0][:3] == [sys.executable, "-m", "pip"]
    assert "-e" not in pip_stub[0]
    assert not Path(pip_stub[0][-1]).exists()
    assert TOKEN not in json.dumps(summary)
    assert session.headers["PRIVATE-TOKEN"] == TOKEN


@pytest.mark.parametrize("status", [401, 403, 404, 302, 500])
def test_http_failure_does_not_install(gitlab, session, pip_stub, status):
    """Fail closed on authentication errors, missing refs and redirects."""
    gitlab.update(status=status, archive=TOKEN.encode())
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        bootstrap.install_project(gitlab["api"], 123, SHA, session)
    assert TOKEN not in str(error.value)
    assert len(gitlab["calls"]) == 1
    assert pip_stub == []


@pytest.mark.parametrize("name", [
    "../escape", "/absolute", "repo/../escape", "repo\\escape",
    "C:/escape",
])
def test_unsafe_archive_paths(gitlab, session, pip_stub, name):
    """Reject traversal and platform-specific absolute archive paths."""
    gitlab["archive"] = archive_bytes([(name, b"bad")])
    with pytest.raises(ValueError, match="safe relative paths"):
        bootstrap.install_project(gitlab["api"], 123, SHA, session)
    assert not pip_stub


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_archive_links_and_special_files(gitlab, session, pip_stub, kind):
    """Reject links and special entries without writing them to disk."""
    info = zipfile.ZipInfo("repo/link")
    info.create_system = 3
    info.external_attr = (kind | 0o777) << 16
    gitlab["archive"] = archive_bytes([(info, b"../../escape")])
    with pytest.raises(ValueError, match="links/special files"):
        bootstrap.install_project(gitlab["api"], 123, SHA, session)
    assert not pip_stub


@pytest.mark.parametrize("body, message", [
    (b"<html>Login</html>", "ZIP archive"),
    (archive_bytes([]), "one GitLab archive root"),
    (archive_bytes([("repo/README.md", b"hi")]), "pyproject.toml"),
    (archive_bytes([("one/a", b"a"), ("two/b", b"b")]),
     "one GitLab archive root"),
])
def test_invalid_archives(gitlab, session, pip_stub, body, message):
    """Reject nonarchives and incomplete or ambiguous package layouts."""
    gitlab["archive"] = body
    with pytest.raises(ValueError, match=message):
        bootstrap.install_project(gitlab["api"], 123, SHA, session)
    assert not pip_stub


def test_root_package_layout(gitlab, session, pip_stub):
    """Support uploading only production/ as the GitLab repository root."""
    gitlab["archive"] = archive_bytes([("repo/pyproject.toml", b"ok")])
    bootstrap.install_project(gitlab["api"], 123, SHA, session, ".")
    assert len(pip_stub) == 2


@pytest.mark.parametrize("location", ["user_site", "pythonpath"])
def test_verification_preserves_install_paths(
    gitlab, session, monkeypatch, tmp_path, location,
):
    """Discover pip's installed metadata using normal hosted Python paths."""
    interpreter = str(Path(sys._base_executable).resolve())
    real_run = subprocess.run
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "user"))
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    if location == "user_site":
        checked = real_run(
            [interpreter, "-c",
             "import site; print(site.getusersitepackages())"],
            capture_output=True, text=True, check=True,
        )
        destination = Path(checked.stdout.strip())
    else:
        destination = tmp_path / "hosted-packages"
        monkeypatch.setenv("PYTHONPATH", str(destination))

    def install_then_verify(command, **options):
        """Simulate pip writing metadata, then run real Python verification."""
        if "install" in command:
            metadata = destination / "labor_cost_model-9.8.7.dist-info"
            metadata.mkdir(parents=True)
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\n"
                "Name: labor-cost-model\nVersion: 9.8.7\n"
            )
            return subprocess.CompletedProcess(command, 0, "installed", "")
        return real_run(command, **options)

    monkeypatch.setattr(bootstrap.sys, "executable", interpreter)
    monkeypatch.setattr(bootstrap.subprocess, "run", install_then_verify)
    summary = bootstrap.install_project(gitlab["api"], 123, SHA, session)
    assert summary["package_version"] == "9.8.7"
    assert summary["python"] == interpreter


def test_pip_failure_redacts_and_cleans(gitlab, session, monkeypatch):
    """Do not return success or retain source after partial pip failure."""
    paths = []

    def fail(command, **options):
        """Simulate a failing installer that echoes a sensitive value."""
        paths.append(Path(command[-1]))
        return subprocess.CompletedProcess(command, 1, "", TOKEN)

    monkeypatch.setattr(bootstrap.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="partially updated") as error:
        bootstrap.install_project(gitlab["api"], 123, SHA, session)
    assert TOKEN not in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert all(not path.exists() for path in paths)


def test_request_failure_is_sanitized(session, monkeypatch):
    """Omit HTTP exception internals that could contain credentials."""
    def fail(*args, **kwargs):
        """Simulate a network/proxy failure with sensitive details."""
        raise requests.ConnectionError(TOKEN)

    monkeypatch.setattr(session, "get", fail)
    with pytest.raises(RuntimeError, match="connectivity") as error:
        bootstrap.install_project("https://gitlab.test/api/v4", 1,
                                  SHA, session)
    assert TOKEN not in str(error.value)


@pytest.mark.parametrize("changes", [
    {"commit_sha": "main"}, {"package_subdir": "../production"},
    {"api_url": "http://gitlab.test/api/v4"},
    {"api_url": "https://user:secret@gitlab.test/api/v4"},
    {"api_url": "https://gitlab.test/api/v4?token=secret"},
])
def test_invalid_configuration(session, changes):
    """Reject mutable refs and credential-bearing or insecure URLs."""
    options = dict(api_url="https://gitlab.test/api/v4", project=1,
                   commit_sha=SHA, session=session)
    options.update(changes)
    with pytest.raises(ValueError):
        bootstrap.install_project(**options)


def test_disabled_tls_is_rejected(session):
    """Keep certificate validation on for corporate HTTPS endpoints."""
    session.verify = False
    with pytest.raises(ValueError, match="TLS verification"):
        bootstrap.install_project("https://gitlab.test/api/v4", 1,
                                  SHA, session)


def test_notebook_pins_both_downloads(gitlab, pip_stub, monkeypatch):
    """Execute the real install cell using a loopback GitLab transport."""
    notebook = nbformat.read(ROOT / "examples/gitlab_install.ipynb", 4)
    nbformat.validate(notebook)
    sessions = []
    original_session = requests.Session

    class LocalSession(original_session):
        """Route the example HTTPS origin to our local protocol fixture."""

        def __init__(self):
            """Retain the session to verify its credentials are cleared."""
            super().__init__()
            sessions.append(self)

        def get(self, url, **kwargs):
            """Preserve paths/headers while substituting only transport."""
            url = url.replace("https://gitlab.test/api/v4", gitlab["api"])
            return super().get(url, **kwargs)

    monkeypatch.setattr(requests, "Session", LocalSession)
    namespace = {}
    exec(notebook.cells[1].source, namespace)
    namespace.update(
        GITLAB_API="https://gitlab.test/api/v4",
        GITLAB_PROJECT="group/project", GITLAB_REF="release/2026",
        Z_ENV_DAA_GITLAB_TOKEN=TOKEN,
    )
    exec(notebook.cells[2].source, namespace)
    calls = [path for path, headers in gitlab["calls"]]
    assert "commits/release%2F2026" in calls[0]
    assert "production%2Fruntime%2Fbootstrap.py/raw" in calls[1]
    assert parse_qs(urlsplit(calls[1]).query)["ref"] == [SHA]
    assert parse_qs(urlsplit(calls[2]).query)["sha"] == [SHA]
    assert all(not client.headers for client in sessions)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []


def test_deployment_requirements_match_metadata():
    """Prevent the environment recipe from drifting from runtime needs."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    from packaging.requirements import Requirement

    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = {
        str(Requirement(item)) for item in metadata["project"]["dependencies"]
    }
    lines = (ROOT / "runtime/requirements.txt").read_text().splitlines()
    actual = {
        str(Requirement(line))
        for line in lines
        if line and not line.startswith("#")
    }
    assert actual == expected | {str(Requirement("requests>=2.31,<3"))}


@pytest.mark.skipif(
    not os.environ.get("LABOR_DEPLOYMENT_TEST_PYTHON"),
    reason="Set LABOR_DEPLOYMENT_TEST_PYTHON to a disposable prepared venv.",
)
def test_real_install_and_post_restart_analysis(gitlab, tmp_path):
    """Install the actual source and analyze from a fresh isolated process."""
    interpreter = os.environ["LABOR_DEPLOYMENT_TEST_PYTHON"]
    entries = [("repo/production/pyproject.toml",
                (ROOT / "pyproject.toml").read_bytes())]
    entries.extend(
        ("repo/production/" + str(path.relative_to(ROOT)), path.read_bytes())
        for path in (ROOT / "src/labor_cost").rglob("*.py")
    )
    gitlab["archive"] = archive_bytes(entries)
    # The bootstrap is fetched source in real use, not an installed module.
    script = (ROOT / "runtime/bootstrap.py").read_text()
    script += (
        "\nwith requests.Session() as session:\n"
        f"    session.headers['PRIVATE-TOKEN'] = {TOKEN!r}\n"
        f"    summary = install_project({gitlab['api']!r}, 123, "
        f"{SHA!r}, session)\n"
        "    session.headers.clear()\n"
        "print(json.dumps(summary))\n"
    )
    installed = subprocess.run(
        [interpreter, "-I", "-c", script], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    assert json.loads(installed.stdout)["commit"] == SHA
    notebook = nbformat.read(ROOT / "examples/gitlab_install.ipynb", 4)
    smoke = '''
import sys
from pathlib import Path

def audit(event, args):
    """Prohibit all notebook reads during installed-package operation."""
    if event == "open" and str(args[0]).endswith(".ipynb"):
        raise AssertionError("Installed execution read a notebook")

sys.addaudithook(audit)
'''
    smoke += notebook.cells[4].source + notebook.cells[5].source
    smoke += (
        "\nassert Path(labor_cost.__file__).is_relative_to("
        "Path(sys.prefix))\n"
        "assert version('labor-cost-model') == labor_cost.__version__\n"
        "from importlib.metadata import distribution\n"
        "dist = distribution('labor-cost-model')\n"
        "for item in dist.files:\n"
        "    if '.dist-info/' in str(item):\n"
        f"        assert {TOKEN!r} not in "
        "dist.locate_file(item).read_text()\n"
    )
    checked = subprocess.run(
        [interpreter, "-I", "-c", smoke], cwd=tmp_path,
        capture_output=True, text=True, check=True,
    )
    assert "reconciled change +250" in checked.stdout
    assert TOKEN not in installed.stdout + checked.stdout
