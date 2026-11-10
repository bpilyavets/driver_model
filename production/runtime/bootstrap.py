"""Load or install labor-cost-model from a pinned GitLab archive."""

import hashlib
import importlib
import importlib.util
import io
import json
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit
from uuid import uuid4

import requests


def _project_url(api_url, project):
    """Validate the API origin and encode a numeric ID or project path."""
    parsed = urlsplit(api_url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not parsed.hostname
        or (parsed.scheme != "https" and not (
            local and parsed.scheme == "http"
        ))
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an HTTPS GitLab API URL without credentials.")
    if not str(project).strip():
        raise ValueError("A GitLab project ID or unencoded path is required.")
    return api_url.rstrip("/") + "/projects/" + quote(str(project), safe="")


def _relative_parts(name):
    """Accept only portable relative paths with no traversal components."""
    path = PurePosixPath(name)
    if (
        not name or path.is_absolute() or "\\" in name or ":" in name
        or ".." in path.parts or "\x00" in name
    ):
        raise ValueError("Archive/package paths must be safe relative paths.")
    return path.parts


def _download(url, session, archive, commit_sha, package_subdir):
    """Fetch a commit-pinned archive without following credential redirects."""
    try:
        with session.get(
            url + "/repository/archive.zip",
            params={
                "sha": commit_sha, "path": package_subdir,
                "include_lfs_blobs": "false",
            },
            timeout=(15, 120), stream=True, allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    "GitLab archive download failed "
                    f"(HTTP {response.status_code}). Check project, token "
                    "permissions and API URL; redirects are not followed."
                )
            with archive.open("wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    target.write(chunk)
    except requests.RequestException:
        raise RuntimeError(
            "GitLab archive request failed. Check connectivity, corporate "
            "CA configuration and timeouts."
        ) from None


def _extract(archive, destination, package_parts):
    """Extract regular files only and locate GitLab's single archive root."""
    try:
        with zipfile.ZipFile(archive) as source:
            entries = []
            roots = set()
            seen = set()
            for info in source.infolist():
                parts = _relative_parts(info.filename)
                kind = stat.S_IFMT(info.external_attr >> 16)
                if not parts or kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError("Archive links/special files rejected.")
                target = destination.joinpath(*parts)
                if target in seen:
                    raise ValueError("Duplicate archive paths rejected.")
                seen.add(target)
                roots.add(parts[0])
                entries.append((info, target))
            if len(roots) != 1:
                raise ValueError("Expected one GitLab archive root directory.")
            for info, target in entries:
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(info) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
    except (zipfile.BadZipFile, NotImplementedError):
        raise ValueError("GitLab did not return a supported ZIP archive.") \
            from None
    package = destination.joinpath(next(iter(roots)), *package_parts)
    if not (package / "pyproject.toml").is_file():
        raise ValueError(
            "No pyproject.toml at package_subdir in the GitLab archive. "
            "Check the repository layout and selected commit."
        )
    return package


def _run(command, session):
    """Run pip/metadata checks and sanitize failures before displaying them."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False,
        )
    except OSError:
        raise RuntimeError("Could not start the notebook Python process.") \
            from None
    if result.returncode:
        detail = result.stdout + "\n" + result.stderr
        for key in ("PRIVATE-TOKEN", "Authorization", "JOB-TOKEN"):
            secret = session.headers.get(key)
            if secret:
                detail = detail.replace(secret, "[REDACTED]")
        raise RuntimeError(
            "Package installation/verification failed. Check pip, package "
            "index access, dependency compatibility and environment write "
            "permissions. The environment may be partially updated.\n"
            + detail[-6000:]
        )
    return result.stdout


def install_project(
    api_url, project, commit_sha, session, package_subdir="production",
):
    """Install a pinned GitLab package in the current Python environment.

    Parameters
    ----------
    api_url : str
        GitLab API base, e.g. ``https://gitlab.example.com/api/v4``.
    project : str or int
        Numeric project ID or unencoded ``group/project`` path.
    commit_sha : str
        Full immutable commit hash, resolved before fetching this bootstrap.
    session : requests.Session
        Caller-owned authenticated session with TLS verification enabled.
    package_subdir : str
        Repository directory containing pyproject.toml; ``.`` for repo root.

    Returns
    -------
    dict
        Commit, distribution version and interpreter. Restart the kernel
        before importing the installed library. No source loader is used.

    Notes
    -----
    HTTP is accepted only for loopback integration tests. Temporary source
    files are removed even on failure; pip changes are not transactional.
    The caller owns the session and must clear its authentication headers.
    """
    if sys.version_info < (3, 10):
        raise RuntimeError("labor-cost-model requires Python 3.10 or later.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit_sha):
        raise ValueError("Resolve the reference to a full commit SHA first.")
    if session.verify is False:
        raise ValueError("TLS verification must remain enabled.")
    url = _project_url(api_url, project)
    parts = _relative_parts(package_subdir)
    with tempfile.TemporaryDirectory(prefix="labor-gitlab-") as temporary:
        root = Path(temporary)
        archive = root / "project.zip"
        _download(url, session, archive, commit_sha, "/".join(parts))
        package = _extract(archive, root / "source", parts)
        _run([
            sys.executable, "-m", "pip", "--disable-pip-version-check",
            "--no-input", "install", str(package),
        ], session)
        # Match pip's normal site/environment lookup. Isolated mode (-I)
        # hides user-site installs and paths configured by hosted kernels.
        metadata = _run([
            sys.executable, "-c",
            "import json, sys; from importlib.metadata import version; "
            "print(json.dumps({'package_version': "
            "version('labor-cost-model'), 'python': sys.executable}))",
        ], session)
        installed = json.loads(metadata)
    return {"commit": commit_sha, **installed}


class ProjectLoadError(RuntimeError):
    """A pinned source runtime could not be loaded or is already closed."""


class ProjectRuntime:
    """Own one private package namespace and its temporary source files.

    Keep this handle alive while using its package, models and results.
    ``close`` removes the namespace and files, but cannot revoke Python
    references already held by callers. Do not use those after closing.
    Third-party dependencies are shared with the notebook environment.
    """

    def __init__(self, temporary, namespace, provenance):
        """Retain local files, namespace and credential-free provenance."""
        self._temporary = temporary
        self._namespace = namespace
        self._package = None
        self.provenance = provenance
        self.closed = False

    @property
    def package(self):
        """Return the public package API while this runtime is open."""
        if self.closed:
            raise ProjectLoadError("This code runtime is closed.")
        return self._package

    def import_module(self, name):
        """Import a dotted module name relative to this private package."""
        if self.closed:
            raise ProjectLoadError("This code runtime is closed.")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", name,
        ):
            raise ProjectLoadError("Use a module name inside this package.")
        return importlib.import_module(self._namespace + "." + name)

    def close(self):
        """Remove this runtime's modules and files; repeated calls are safe."""
        if self.closed:
            return
        for name in list(sys.modules):
            if name == self._namespace or name.startswith(
                self._namespace + ".",
            ):
                del sys.modules[name]
        self._package = None
        self._temporary.cleanup()
        self.closed = True


def _source_parts(name, *, directory=False):
    """Validate paths without silently normalizing ambiguous components."""
    if not isinstance(name, str):
        raise ProjectLoadError("Package/archive paths must be strings.")
    stripped = name[:-1] if directory and name.endswith("/") else name
    parts = stripped.split("/")
    if (
        PurePosixPath(name).is_absolute() or "\\" in name or "\x00" in name
        or ":" in name or any(p in {"", ".", ".."} for p in parts)
    ):
        raise ProjectLoadError("Unsafe package/archive path.")
    return tuple(parts)


def _fetch_source(url, session, commit, package_path, timeout, limit):
    """Download a bounded archive without retaining or exposing credentials."""
    if session is None or not callable(getattr(session, "get", None)):
        raise ProjectLoadError("Pass an authenticated requests session.")
    if getattr(session, "verify", True) is False:
        raise ProjectLoadError("TLS verification must remain enabled.")
    try:
        with session.get(
            url + "/repository/archive.zip",
            params={"sha": commit, "path": package_path,
                    "include_lfs_blobs": "false"},
            timeout=timeout, stream=True, allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise ProjectLoadError(
                    f"GitLab archive request failed: HTTP "
                    f"{response.status_code}. Redirects are not followed."
                )
            data = io.BytesIO()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if data.tell() + len(chunk) > limit:
                    raise ProjectLoadError("GitLab archive is too large.")
                data.write(chunk)
            return data.getvalue()
    except ProjectLoadError:
        raise
    except Exception:
        raise ProjectLoadError(
            "Could not download GitLab archive. Check connectivity, "
            "session credentials, CA configuration and timeout."
        ) from None


def _source_members(archive, package_parts, max_files, max_unpacked_bytes):
    """Validate the archive and select only the requested package subtree."""
    seen, roots, members = set(), set(), []
    total = 0
    for info in archive.infolist():
        parts = _source_parts(info.orig_filename, directory=info.is_dir())
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ProjectLoadError("Archive links/special files rejected.")
        if (kind == stat.S_IFDIR and not info.is_dir()) or (
            kind == stat.S_IFREG and info.is_dir()
        ):
            raise ProjectLoadError("Conflicting archive file type.")
        if parts in seen or info.flag_bits & 1:
            raise ProjectLoadError("Duplicate/encrypted archive file.")
        seen.add(parts)
        roots.add(parts[0])
        if info.is_dir() or parts[1:1 + len(package_parts)] != package_parts:
            continue
        relative = parts[1 + len(package_parts):]
        if not relative:
            raise ProjectLoadError("The package path must be a directory.")
        total += info.file_size
        members.append((info, relative))
        if len(members) > max_files or total > max_unpacked_bytes:
            raise ProjectLoadError("Extracted GitLab package is too large.")
    if len(roots) != 1 or not any(
        relative == ("__init__.py",) for _, relative in members
    ):
        raise ProjectLoadError(
            "Expected one archive root and __init__.py at package_path."
        )
    return members


def _extract_source(data, target, package_parts, max_files, unpacked_limit):
    """Extract validated source with bounded writes and per-file hashes."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = _source_members(
                archive, package_parts, max_files, unpacked_limit,
            )
            hashes, total = {}, 0
            for info, relative in members:
                destination = target.joinpath(*relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest, size = hashlib.sha256(), 0
                with archive.open(info) as src, destination.open("xb") as dst:
                    while chunk := src.read(64 * 1024):
                        size += len(chunk)
                        total += len(chunk)
                        if size > info.file_size or total > unpacked_limit:
                            raise ProjectLoadError("Unpacked size exceeded.")
                        digest.update(chunk)
                        dst.write(chunk)
                if size != info.file_size:
                    raise ProjectLoadError("Incomplete GitLab archive file.")
                hashes["/".join(relative)] = digest.hexdigest()
            return hashes
    except ProjectLoadError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        raise ProjectLoadError("Could not extract the GitLab package.") \
            from None


def load_project(
    api_url, project_id, commit, session=None, *, ref=None,
    archive_bytes=None, work_dir=None,
    package_path="production/src/labor_cost", timeout=30,
    max_archive_bytes=20_000_000, max_unpacked_bytes=80_000_000,
    max_files=5000,
):
    """Load pinned source directly into the current kernel without pip.

    Parameters
    ----------
    api_url, project_id, commit : str or int
        HTTPS API base, project ID/path and full immutable commit SHA.
        Loopback HTTP is accepted for local integration tests only.
    session : requests.Session, optional
        Caller-owned authenticated session; required for network downloads.
        Authentication is not retained in the returned runtime.
    ref : str, optional
        Requested branch/tag for provenance; downloads use commit only.
    archive_bytes : bytes, optional
        Offline GitLab-shaped ZIP with one root containing package_path.
    work_dir : path-like, optional
        Parent of the temporary source directory; must already exist.
    package_path : str
        Repository-relative package directory containing __init__.py.
    timeout : float
        Positive requests timeout in seconds, at most 120.
    max_archive_bytes, max_unpacked_bytes, max_files : int
        Positive download, extracted-package byte and package file limits.

    Returns
    -------
    ProjectRuntime
        Use ``runtime.package`` for the public API and close after use.
        Dependencies must already exist in the notebook environment.

    Notes
    -----
    No sys.path edits, global labor_cost alias, subprocess, installation or
    labor-cost-model distribution metadata are required. Different runtime
    handles own different class identities: do not mix their objects.
    """
    if sys.version_info < (3, 10):
        raise ProjectLoadError("labor-cost-model requires Python 3.10+.")
    if not isinstance(api_url, str) or not isinstance(
        project_id, (str, int),
    ) or isinstance(project_id, bool):
        raise ProjectLoadError("Provide a GitLab API URL and project ID.")
    try:
        url = _project_url(api_url, project_id)
    except ValueError:
        raise ProjectLoadError("Invalid GitLab API URL or project ID.") \
            from None
    if not isinstance(commit, str) or not re.fullmatch(
        r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", commit,
    ):
        raise ProjectLoadError("Resolve the ref to a full commit SHA first.")
    if ref is not None and not isinstance(ref, str):
        raise ProjectLoadError("The requested ref must be a string.")
    if (
        not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
        or not 0 < timeout <= 120
    ):
        raise ProjectLoadError("Timeout must be between 0 and 120 seconds.")
    for limit in (max_archive_bytes, max_unpacked_bytes, max_files):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ProjectLoadError("Archive limits must be positive integers.")
    parts = _source_parts(package_path)
    commit = commit.lower()
    if archive_bytes is None:
        archive_bytes = _fetch_source(
            url, session, commit, package_path, timeout, max_archive_bytes,
        )
    if not isinstance(archive_bytes, bytes):
        raise ProjectLoadError("The archive must be bytes.")
    if len(archive_bytes) > max_archive_bytes:
        raise ProjectLoadError("GitLab archive is too large.")
    try:
        temporary = tempfile.TemporaryDirectory(
            prefix="labor-gitlab-source-", dir=work_dir,
        )
    except OSError:
        raise ProjectLoadError("A writable temporary directory is required.") \
            from None
    target = Path(temporary.name) / "labor_cost"
    namespace = "_labor_cost_runtime_" + commit + "_" + uuid4().hex
    provenance = {
        "project": str(project_id), "commit": commit, "ref": ref,
        "package_path": package_path,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
    }
    runtime = ProjectRuntime(temporary, namespace, provenance)
    try:
        provenance["files"] = _extract_source(
            archive_bytes, target, parts, max_files, max_unpacked_bytes,
        )
        spec = importlib.util.spec_from_file_location(
            namespace, target / "__init__.py",
            submodule_search_locations=[str(target)],
        )
        if spec is None or spec.loader is None:
            raise ProjectLoadError("Could not create a package import spec.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
        runtime._package = module
        provenance["package_version"] = module.__version__
        return runtime
    except BaseException as error:
        runtime.close()
        if isinstance(error, ProjectLoadError):
            raise
        if isinstance(error, ImportError):
            missing = error.name or "unknown"
            raise ProjectLoadError(
                f"Package import failed ({missing}). Check the notebook "
                "dependency requirements and package source."
            ) from None
        if isinstance(error, Exception):
            raise ProjectLoadError(
                "Package import failed. Check the selected source and "
                "notebook dependency compatibility."
            ) from None
        raise
