"""Private source runtimes, bounded archives and hosted notebook loading."""

import hashlib
import importlib
import json
import stat
import subprocess
import sys
import sysconfig
import textwrap
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import nbformat
import numpy as np
import pytest
import test_gitlab_deployment as deployment
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

bootstrap = deployment.bootstrap
gitlab = deployment.gitlab
session = deployment.session
ROOT = deployment.ROOT
SHA = deployment.SHA
TOKEN = deployment.TOKEN
PACKAGE = "production/src/labor_cost"


def source_archive(files=None, package=PACKAGE):
    """Build a small package with a relative import and a known version."""
    if files is None:
        files = {
            "__init__.py": b"from .api import answer\n__version__ = '1.2.3'\n",
            "api.py": b"answer = 42\n",
        }
    return deployment.archive_bytes([
        (f"repo/{package}/{name}", content)
        for name, content in files.items()
    ])


def real_archive():
    """Archive only the actual package's source without local bytecode."""
    return deployment.archive_bytes([
        ("repo/production/src/" + str(p.relative_to(ROOT / "src")),
         p.read_bytes())
        for p in (ROOT / "src/labor_cost").rglob("*.py")
    ])


def offline_load(data=None, **kwargs):
    """Load without network while retaining normal validation/provenance."""
    return bootstrap.load_project(
        "https://gitlab.test/api/v4", "group/project", SHA,
        archive_bytes=source_archive() if data is None else data, **kwargs,
    )


def test_pinned_download_and_private_lifetime(
    gitlab, session, monkeypatch, tmp_path,
):
    """Pin the archive, preserve imports, and clean only this runtime."""
    gitlab["archive"] = source_archive()
    installed = importlib.import_module("labor_cost")
    original_path = list(sys.path)

    def forbidden(*args, **kwargs):
        """Source loading must never install anything or launch a process."""
        raise AssertionError("Unexpected installer/subprocess call")

    monkeypatch.setattr(bootstrap, "install_project", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    runtime = bootstrap.load_project(
        gitlab["api"], "group/nested/project", SHA.upper(), session,
        ref="release/2026", work_dir=tmp_path,
    )
    with_runtime = runtime.package.__name__
    target = Path(runtime.package.__file__).parents[1]
    try:
        assert runtime.import_module("api").answer == 42
        assert runtime.package.answer == 42
        url, headers = gitlab["calls"][0]
        assert "/projects/group%2Fnested%2Fproject/" in url
        assert parse_qs(urlsplit(url).query) == {
            "sha": [SHA], "path": [PACKAGE], "include_lfs_blobs": ["false"],
        }
        assert headers["PRIVATE-TOKEN"] == TOKEN
        assert runtime.provenance["commit"] == SHA
        assert runtime.provenance["ref"] == "release/2026"
        assert runtime.provenance["package_version"] == "1.2.3"
        assert runtime.provenance["archive_sha256"] == hashlib.sha256(
            gitlab["archive"],
        ).hexdigest()
        assert runtime.provenance["files"]["api.py"] == hashlib.sha256(
            b"answer = 42\n",
        ).hexdigest()
        assert TOKEN not in json.dumps(runtime.provenance)
        assert sys.path == original_path
        assert sys.modules["labor_cost"] is installed
        assert session.headers["PRIVATE-TOKEN"] == TOKEN
    finally:
        runtime.close()
    runtime.close()
    assert not target.exists()
    assert not any(n.startswith(with_runtime) for n in sys.modules)
    assert sys.modules["labor_cost"] is installed
    with pytest.raises(bootstrap.ProjectLoadError, match="closed"):
        runtime.import_module("api")
    with pytest.raises(bootstrap.ProjectLoadError, match="closed"):
        _ = runtime.package


def test_two_handles_and_root_layout():
    """Separate versions have independent modules and cleanup ownership."""
    one = offline_load()
    two = offline_load(
        source_archive(package="src/labor_cost"),
        package_path="src/labor_cost",
    )
    try:
        assert one.package is not two.package
        one.import_module("api").answer = -1
        assert two.import_module("api").answer == 42
        one.close()
        assert two.package.answer == 42
        for invalid in ("../api", "api/other", ".api", "", 123):
            with pytest.raises(bootstrap.ProjectLoadError, match="module"):
                two.import_module(invalid)
    finally:
        one.close()
        two.close()


@pytest.mark.parametrize("status", [401, 403, 404, 302, 500])
def test_source_http_errors(gitlab, session, tmp_path, status):
    """HTTP failures neither leak response tokens nor leave source files."""
    gitlab.update(status=status, archive=TOKEN.encode())
    with pytest.raises(bootstrap.ProjectLoadError, match=f"HTTP {status}"):
        bootstrap.load_project(
            gitlab["api"], 123, SHA, session, work_dir=tmp_path,
        )
    assert len(gitlab["calls"]) == 1
    assert not list(tmp_path.iterdir())


def test_stream_limit_and_session_failures(monkeypatch, session, tmp_path):
    """Stop reading on the first oversized chunk and sanitize transport."""
    chunks_read = []

    class Response:
        """A bounded fake transport whose body must not be read eagerly."""

        status_code = 200

        def __enter__(self):
            """Return the streaming response."""
            return self

        def __exit__(self, *args):
            """No resources are allocated by this transport."""

        def iter_content(self, chunk_size):
            """Fail the test if the loader continues after crossing its cap."""
            chunks_read.append(1)
            yield b"1234"
            chunks_read.append(2)
            yield b"5678"
            raise AssertionError("Read beyond limit")

    monkeypatch.setattr(session, "get", lambda *a, **kw: Response())
    with pytest.raises(bootstrap.ProjectLoadError, match="too large"):
        bootstrap.load_project(
            "https://gitlab.test/api/v4", 123, SHA, session,
            max_archive_bytes=5, work_dir=tmp_path,
        )
    assert chunks_read == [1, 2]
    assert not list(tmp_path.iterdir())

    def fail(*args, **kwargs):
        """Simulate a request error containing authentication details."""
        raise RuntimeError(TOKEN)

    monkeypatch.setattr(session, "get", fail)
    with pytest.raises(bootstrap.ProjectLoadError) as error:
        bootstrap.load_project("https://gitlab.test/api/v4", 123, SHA,
                               session)
    assert TOKEN not in str(error.value)
    session.verify = False
    with pytest.raises(bootstrap.ProjectLoadError, match="TLS"):
        bootstrap.load_project("https://gitlab.test/api/v4", 123, SHA,
                               session)


@pytest.mark.parametrize("filename", [
    "../escape", "/absolute", "repo/../escape", "repo//empty",
    "repo/./dot", "repo\\escape", "repo/C:/drive", "repo/trailing//",
])
def test_source_paths_rejected(filename, tmp_path):
    """Reject unsafe paths even outside the requested subtree."""
    data = deployment.archive_bytes([
        (f"repo/{PACKAGE}/__init__.py", b"__version__ = '1'"),
        (filename, b"bad"),
    ])
    with pytest.raises(bootstrap.ProjectLoadError):
        offline_load(data, work_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_source_special_files(kind):
    """Reject nonregular entries before importing any code."""
    info = zipfile.ZipInfo(f"repo/{PACKAGE}/__init__.py")
    info.create_system = 3
    info.external_attr = (kind | 0o777) << 16
    with pytest.raises(bootstrap.ProjectLoadError, match="special"):
        offline_load(deployment.archive_bytes([(info, b"bad")]))


def test_duplicate_encrypted_missing_and_corrupt_archives(tmp_path):
    """Reject duplicates, encryption, wrong layout and invalid ZIP bytes."""
    name = f"repo/{PACKAGE}/__init__.py"
    with pytest.warns(UserWarning, match="Duplicate"):
        duplicate = deployment.archive_bytes([(name, b"x"), (name, b"x")])
    encrypted = bytearray(source_archive())
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        at = encrypted.index(signature) + offset
        encrypted[at] |= 1
    cases = [duplicate, bytes(encrypted), b"not a zip",
             source_archive({"api.py": b"x"}),
             deployment.archive_bytes([
                 (name, b"x"), ("other/file", b"x"),
             ])]
    for data in cases:
        with pytest.raises(bootstrap.ProjectLoadError):
            offline_load(data, work_dir=tmp_path)
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("limits", [
    {"max_files": 1}, {"max_unpacked_bytes": 2}, {"max_archive_bytes": 2},
])
def test_archive_size_limits(limits, tmp_path):
    """Apply byte and selected-file limits to offline archives as well."""
    with pytest.raises(bootstrap.ProjectLoadError, match="large"):
        offline_load(work_dir=tmp_path, **limits)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [
    {"timeout": 0}, {"timeout": float("nan")}, {"timeout": True},
    {"max_files": False}, {"max_unpacked_bytes": -1},
    {"ref": 12}, {"package_path": "../src"}, {"package_path": 42},
])
def test_source_invalid_options(options):
    """Validate configuration before creating a runtime."""
    with pytest.raises(bootstrap.ProjectLoadError):
        offline_load(**options)


@pytest.mark.parametrize("source, match", [
    (b"from . import api\nraise RuntimeError('secret')", "import failed"),
    (b"import missing_source_loader_dependency", "dependency"),
])
def test_import_failure_cleans_namespace(source, match, tmp_path):
    """Remove partially imported submodules and extracted files on failure."""
    modules = set(sys.modules)
    with pytest.raises(bootstrap.ProjectLoadError, match=match):
        offline_load(source_archive({
            "__init__.py": source, "api.py": b"answer = 42\n",
        }), work_dir=tmp_path)
    assert not list(tmp_path.iterdir())
    assert not any(n.startswith("_labor_cost_runtime_")
                   for n in set(sys.modules) - modules)


@pytest.mark.parametrize("mode", ["exact", "permutation"])
def test_experiment_through_private_runtime(mode, tmp_path):
    """Preserve the user's model/data across loading, costing and reporting."""
    runtime = offline_load(real_archive())
    other = None
    try:
        lc = runtime.package
        source = (ROOT / "examples/poc_extented_model_test.py").read_text()
        source = source.replace("from labor_cost import (",
                                f"from {lc.__name__} import (")
        source = source.replace('explainer="exact"', f'explainer="{mode}"')
        scope = {}
        exec(compile(source, "experiment-source-runtime", "exec"), scope)
        result, compiled = scope["result"], scope["compiled"]
        np.testing.assert_allclose(
            result.table("driver_bridge").impact.sum(), 1799.0,
            rtol=1e-10, atol=1e-6,
        )
        assert lc.render_model(compiled).markdown
        report = lc.report(result, teams=["Operations"], show=False)
        bundle = lc.export_run(result, tmp_path / "run")
        # Transfer through portable data, not through foreign class instances.
        other = offline_load(real_archive())
        restored = other.package.load_run(bundle)
        assert restored.states is None
        charts = other.package.report(
            restored, teams=["Operations"], show=False,
        )
        assert charts.figures["organization"].to_json() == (
            report.figures["organization"].to_json()
        )
    finally:
        runtime.close()
        if other:
            other.close()


def test_source_notebook_without_installed_distribution(gitlab, tmp_path):
    """Run a fresh dependency-only kernel without installed library or pip."""
    gitlab["archive"] = real_archive()
    # Build a throwaway environment with existing third-party dependencies,
    # excluding our editable package, its metadata and all .pth injection.
    environment = tmp_path / "env"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    python = environment / "bin/python"
    site = Path(subprocess.check_output(
        [str(python), "-c",
         "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip())
    for item in Path(sysconfig.get_path("purelib")).iterdir():
        if (
            "labor_cost" in item.name or item.suffix == ".pth"
            or item.name.startswith(("pip", "__editable__", "__pycache__"))
        ):
            continue
        (site / item.name).symlink_to(item, target_is_directory=item.is_dir())
    notebook = nbformat.read(ROOT / "examples/gitlab_load.ipynb", 4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
    # Route only transport; run the delivered notebook's HTTPS checks intact.
    setup = textwrap.dedent('''
        import importlib.util
        import importlib.metadata
        import sys
        import requests
        assert importlib.util.find_spec("labor_cost") is None
        try:
            importlib.metadata.version("labor-cost-model")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise AssertionError("labor-cost-model must not be installed")

        def audit(event, args):
            """Prohibit installers; dependencies may query system metadata."""
            if event == "subprocess.Popen":
                command = str(args[1]).lower()
                if any(word in command for word in (
                    "pip", "install", "setup.py", "ensurepip",
                )):
                    raise AssertionError("Unexpected installation process")
        sys.addaudithook(audit)
        sessions = []
        OriginalSession = requests.Session
        class LocalSession(OriginalSession):
            """Route the notebook HTTPS URL to the simulated GitLab server."""
            def __init__(self):
                """Keep sessions for credential cleanup assertions."""
                super().__init__()
                sessions.append(self)
            def get(self, url, **kwargs):
                """Replace transport while preserving requests and headers."""
                url = url.replace(GITLAB_API, TEST_API)
                return super().get(url, **kwargs)
        requests.Session = LocalSession
    ''')
    notebook.cells.insert(2, nbformat.v4.new_code_cell(
        setup + f"\nTEST_API = {gitlab['api']!r}\n"
        f"Z_ENV_GITLAB_TOKEN = {TOKEN!r}\n"
        "GITLAB_REF = 'release/2026'\nGITLAB_PROJECT = 'group/project'\n",
    ))
    notebook.cells.append(nbformat.v4.new_code_cell('''
assert all(not session.headers for session in sessions)
assert CODE.provenance["commit"] == CODE_COMMIT
assert "labor_cost" not in sys.modules
assert api.analyze is lc.analyze
CODE.close()
assert CODE.closed
'''))
    kernels = tmp_path / "kernels"
    kernel = kernels / "source-test"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [str(python), "-m", "ipykernel_launcher", "-f",
                 "{connection_file}"],
        "display_name": "Source loader test", "language": "python",
    }))
    manager = KernelManager(
        kernel_name="source-test",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
    )
    try:
        NotebookClient(
            notebook, km=manager, timeout=180, kernel_name="source-test",
            resources={"metadata": {"path": str(tmp_path)}},
        ).execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    outputs = [output for c in notebook.cells if c.cell_type == "code"
               for output in c.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    assert sum("application/vnd.plotly.v1+json" in o.get("data", {})
               for o in outputs) == 2
    assert any("Cost: organization" in o.get("data", {}).get(
        "text/markdown", "",
    ) for o in outputs)
    calls = [path for path, _ in gitlab["calls"]]
    assert "commits/release%2F2026" in calls[0]
    assert "production%2Fruntime%2Fbootstrap.py/raw" in calls[1]
    assert parse_qs(urlsplit(calls[1]).query)["ref"] == [SHA]
    assert parse_qs(urlsplit(calls[2]).query)["sha"] == [SHA]
