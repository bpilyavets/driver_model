"""Install labor-cost-model from a pinned GitLab archive, without Git."""

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
        metadata = _run([
            sys.executable, "-I", "-c",
            "import json, sys; from importlib.metadata import version; "
            "print(json.dumps({'package_version': "
            "version('labor-cost-model'), 'python': sys.executable}))",
        ], session)
        installed = json.loads(metadata)
    return {"commit": commit_sha, **installed}
