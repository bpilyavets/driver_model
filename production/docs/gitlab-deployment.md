# Load from GitLab in hosted Jupyter

The recommended hosted workflow loads a pinned copy of the library directly
into the current kernel. It uses a private runtime handle, following the
colleague's loader pattern, and does not invoke pip or require installed
`labor-cost-model` metadata. GitLab supplies code; payroll is supplied
separately through `DatasetSnapshot`.

Start with [the direct-loading notebook](../examples/gitlab_load.ipynb).
It contains configuration, every request helper, source loading, a small
analysis, formula previews and optional Plotly reporting. Your supplied
GitLab host, project `278363`, ref `master` and secret variable
`Z_ENV_GITLAB_TOKEN` are already reflected in its configuration.

## Direct-loading setup

Keep `production/runtime/bootstrap.py` and `production/src/labor_cost/`
in GitLab. Upload the updated bootstrap as well as the package: an older
bootstrap that exposes only `install_project` cannot handle `load_project`.
The package directory must include `__init__.py` and its submodules. No
wheel, build configuration, Git executable or package registry is needed
by this loader. Upload source, not `.venv`, credentials or payroll.

Use your company's environment process with
[runtime/requirements.txt](../runtime/requirements.txt), or
[runtime/requirements-reporting.txt](../runtime/requirements-reporting.txt)
for Plotly charts. These supply third-party dependencies, not our library.
The loader does not install or upgrade dependencies. Select that kernel;
source loading itself needs no restart, but dependency changes may do so.
The requirements' interpreter compatibility still applies (initial target
Python 3.10–3.12). The kernel needs a writable temporary directory.

Run the notebook from the beginning. It resolves `GITLAB_REF` once to a full
SHA, fetches the bootstrap and package archive from that SHA, and loads the
source into a unique private namespace. Authentication remains in the
requests session, whose headers are cleared in `finally`. TLS stays enabled;
redirects are refused. Use your company's existing CA/proxy configuration.

For a repository containing only the contents of `production/`, set
`PACKAGE_SUBDIR = "."` in the notebook; it derives `runtime/bootstrap.py`
and `src/labor_cost` instead. The loader itself accepts a repository-relative
`package_path`, defaulting to `production/src/labor_cost`.

## Public API through a runtime handle

After your existing cell fetches and executes the bootstrap into `loader`,
replace its `install_project(...)` call with:

```python
CODE = loader["load_project"](
    GITLAB_API, GITLAB_PROJECT, CODE_COMMIT, gitlab_session,
    ref=GITLAB_REF,
)
lc = CODE.package
```

Keep that call **inside** the authenticated session's `try` block. The
complete notebook shows the surrounding session management. No token is
retained in `CODE`, passed to a subprocess, or written into provenance.

Use all library APIs through `lc`:

```python
compiled = lc.compile_model(model)  # Model authored with lc.ModelSpec, etc.
snapshot = lc.DatasetSnapshot("payroll", "2026-09-close", payroll_frame)
request = lc.AnalysisRequest("2026-08", "2026-09")
result = lc.analyze(compiled, snapshot, request)
charts = lc.report(result, teams=["Operations"])
```

`model` and `payroll_frame` above are your separately supplied model and data;
Operations must exist in that result. The notebook includes a complete
synthetic model/data example. For existing notebooks, replace their library
import block with explicit bindings such as `ModelSpec = lc.ModelSpec` and
`analyze = lc.analyze`, or prefix public names with `lc.`. Do not use
`from labor_cost import ...`: it accesses another installed package, if any.
`lc` is an object, not an importable alias, so `from lc import ...` is also
incorrect. The loader never adds an alias to `sys.modules['labor_cost']` or
changes `sys.path`.

`CODE.import_module("api")` returns that runtime's API submodule, matching
the colleague's pattern. Prefer `CODE.package` for the full public API,
including model declarations, formula previews, bundles and reporting.

## Lifetime, updates and provenance

Keep `CODE` alive throughout the analysis. `CODE.close()` removes only its
private modules and temporary files; repeated calls are safe. Already-held
Python references cannot be revoked: do not use them after closing. Third-party
modules are shared with the kernel and are not unloaded. Partial import failures
clean up the new runtime without removing existing package instances.

The notebook prevents accidentally overwriting an open `CODE`. Finish/export
work, close it, choose a new ref and rerun the loading and analysis cells.
Recreate all models, snapshots and results through the new `lc`. Separate
handles have separate class identities and must not exchange Python objects.
To move results between handles, export with the original runtime and load
with the destination runtime's `lc.load_run`. Namespace isolation does not
make concurrent permutation execution safe; it still requires process
isolation as documented by the engine.

`CODE.provenance` contains project, requested ref, resolved commit,
`package_path`, archive SHA-256, individual file SHA-256 hashes and the loaded
package's `__version__`. It requires no distribution metadata for our library.
Record it alongside the analysis bundle, as shown in the notebook; it does
not automatically change analysis manifests or bundle schemas. Hashes identify
the downloaded bytes; repository trust and HTTPS remain necessary.

`load_project` also accepts `archive_bytes` for offline GitLab-shaped ZIPs
and `work_dir` for a writable temporary parent. The default limits are 20 MB
of downloaded bytes, 80 MB of extracted package files and 5,000 package files;
`timeout=30` is the requests timeout. Limits are configurable positive values,
with timeout capped at 120 seconds. Downloads stop when the byte cap is
crossed. Extraction validates paths, rejects links/special files, duplicates
and encryption, and requires one archive root containing the selected package.
No files outside that package subtree are extracted. HTTP is permitted only
for loopback integration tests; deployment uses HTTPS.

## Direct-loading acceptance and troubleshooting

In your hosted notebook, confirm the displayed commit, `lc.__version__` and
`lc.__file__` (a temporary source path). The example checks signed costs
−100 → 150, a reconciled +250 change, a formula preview and optional charts.
It runs without an installed `labor-cost-model`. A missing installed-package
metadata entry is expected and is not a failure of this workflow.

- Missing `load_project`: upload the new bootstrap and select its commit.
- Import/dependency error: use the requirements with your company's environment
  process; loading source does not install NumPy, pandas, SHAP or PyArrow.
- Reporting dependency error: use the reporting requirements, or disable the
  notebook's optional reporting cell while checking core functionality.
- HTTP errors: check the final API URL, token permissions, project and ref.
- Missing package or invalid archive: check `package_path` and repository layout.
- Type/model mismatch after switching versions: recreate objects through one
  handle or transfer saved results using bundles.
- No writable temporary directory: pass `work_dir` pointing to an existing
  writable directory approved for your notebook.

Local tests use a simulated GitLab endpoint; they cannot establish connectivity
to your internal GitLab. The existing analytical gap register remains open.

## Alternative: install with pip

The original [installation notebook](../examples/gitlab_install.ipynb) and
`install_project` remain supported where installation into the notebook's
Python environment works. This path uses ordinary `import labor_cost`,
requires pip/build access, and needs a kernel restart after installation.
The remainder of this guide describes **that alternative**, not the direct
source-loading workflow above.

### 1. Put the project in GitLab

Retain this layout in the selected commit:

```text
production/
  pyproject.toml
  src/labor_cost/
  runtime/bootstrap.py
  runtime/requirements.txt
  examples/gitlab_install.ipynb
```

Upload source files, not `.venv`, tokens, local datasets or generated run
bundles. The PoC notebook is not needed for installation or execution.
If you upload only the contents of `production/` as the repository root,
set `PACKAGE_SUBDIR = "."` in the deployment notebook. Otherwise retain
the default `"production"`.

The bootstrap is deployment source kept in GitLab, outside the installed
calculation package. It has no dependency on an already installed `labor_cost`.

### 2. Prepare the notebook environment

Use your company's existing virtual-environment instructions with
[runtime/requirements.txt](../runtime/requirements.txt). The file includes
the package's four core dependency constraints and `requests` for the
GitLab connection. Its agreement with `pyproject.toml` is tested.

For Plotly result reports, use
[runtime/requirements-reporting.txt](../runtime/requirements-reporting.txt)
instead. It extends that core recipe with the reporting dependencies;
the installation procedure stays the same. See the
[reporting guide](reporting.md) for hosted notebook renderer settings.

Select that environment's Jupyter kernel. It must provide pip and permit
installation into the selected environment. Keep the company's existing
kernel-registration process; this guide does not replace it. An existing
hosted Jupyter server does not require installing our `[notebook]` extra.

Python 3.10 is the package minimum. Python 3.10–3.12 is the initial
deployment target because the pinned NumPy/SHAP stack must also support
the interpreter. Do not assume every newer Python is compatible merely
because the package minimum is satisfied. The index must provide both
runtime dependencies and pip's isolated build requirements (`setuptools`
and `wheel`). Existing pip/mirror/proxy configuration is respected.

### 3. Configure and install

Open the deployment notebook in your hosted Jupyter and fill in:

| Variable | Value |
| --- | --- |
| `GITLAB_API` | Your HTTPS API base, e.g. `https://gitlab.company.example/api/v4`. |
| `GITLAB_PROJECT` | Numeric project ID or unencoded `group/subgroup/project` path. |
| `GITLAB_REF` | Branch, release tag or full commit SHA. |
| `PACKAGE_SUBDIR` | `production`, or `.` when that directory is the repository root. |
| `Z_ENV_DAA_GITLAB_TOKEN` | Existing secret-provided notebook variable or environment variable. |

Use a token permitted to read the repository through the GitLab API.
Do not paste it into notebook source or a URL. The notebook supplies it
through a `PRIVATE-TOKEN` session header, clears that header in `finally`,
and closes the session. It does not remove your original secret variable.

Run the configuration and installation cells. They:

1. Display the kernel's interpreter and Python version.
2. Resolve the selected ref through `repository/commits/<encoded-ref>`.
3. Fetch `runtime/bootstrap.py` under `PACKAGE_SUBDIR` using that full SHA.
4. Execute the bootstrap from that trusted repository commit.
5. Download `repository/archive.zip` with the same SHA and package path.
6. Validate/extract regular files into a temporary directory, then run
   `sys.executable -m pip ... install <local-package-directory>`.
7. Display the resolved commit, installed distribution version and Python
   executable after successful installation. Remove temporary source files.

GitLab's [repository archive API](https://docs.gitlab.com/api/repositories/#retrieve-file-archive-from-a-repository)
supports the commit and subdirectory parameters. Paths are URL-encoded by
the notebook/bootstrap; supply ordinary unencoded configuration values.
Redirects are not followed, so authentication stays at the configured
endpoint. Use the final API URL if a gateway redirects an old address.

The bootstrap installs non-editably. Dependencies are resolved by pip using
your configured package index; package-index credentials remain managed by
your existing pip configuration, separate from the GitLab session token.
No GitLab credential is passed to pip or included in the returned summary.

Record the displayed commit with your deployment/run notes. The package
version alone does not distinguish commits that retain the same version.
This commit record is separate from the model version and dataset snapshot
identity; the installer does not alter analysis manifests.

### 4. Restart, then verify

Restart the kernel after installation and run only the verification cells
below the notebook's restart marker. They do not need GitLab access or the
token. The first displays the Python executable, installed package version
and `labor_cost.__file__`; the module should come from the selected
environment's `site-packages`, not a checkout's `src/` directory.

The second compiles a tiny signed-payment model, builds an observed state,
checks its cost of **−100**, then analyzes a change to **150**. Expected
output is a reconciled **+250** change and passing recorded validations.
This verifies installation, compilation, costing and analysis through the
normal public API, without loading the PoC notebook.

Your actual workload then uses a published model and data supplied by your
existing process:

```python
from labor_cost import analyze, compile_model, DatasetSnapshot
```

Continue with the [authoring walkthrough](../examples/GUIDE.MD) or the
[existing catalog client](../examples/notebook_client.ipynb). Installing
code does not acquire payroll, publish a model or create a catalog.

For an update, choose the intended tag/commit, rerun installation and
restart again. Reinstalling an older commit uses the same procedure;
dependency versions may still satisfy ranges rather than reproducing an
old environment exactly. Keep the company's environment lock/snapshot if
exact dependency reproduction is needed. Do not hot-reload analysis modules
or rerun installation midway through an analysis.

### Troubleshooting

| Symptom | Next check |
| --- | --- |
| HTTP 401/403 | Token availability and repository API permissions. |
| HTTP 404 | Project ID/path, ref, package layout and bootstrap path; inaccessible projects may also return 404. |
| HTTP 3xx | Configure the final GitLab API URL; redirects are deliberately refused. |
| Certificate/network error | Use the approved proxy/CA configuration, including `REQUESTS_CA_BUNDLE` for requests when needed. Do not disable TLS verification. |
| No pip / permission denied | Select the writable virtual environment created by the company's procedure. |
| Dependency/build failure | Check Python compatibility and availability on the configured index, including build requirements. |
| Import fails or shows old code | Restart the kernel and check `sys.executable` and `labor_cost.__file__`; avoid a local file named `labor_cost.py`. |
| Invalid ZIP or missing `pyproject.toml` | Check the archive endpoint response and `PACKAGE_SUBDIR`; authentication HTML is not a package archive. |

### Metadata missing after pip succeeds

Older bootstraps verified installation with `python -I`. That isolated
process hides user-site packages and ignores `PYTHONPATH`, even though the
preceding pip command uses normal Python startup. A hosted environment
can therefore install successfully and then report
`PackageNotFoundError: labor-cost-model` during verification.

The current bootstrap uses the same interpreter and normal path lookup for
both commands. Update `production/runtime/bootstrap.py` in GitLab and rerun
the installation cell at that updated commit. The GitLab authentication
cell does not need to change.

To check an already installed package, restart the notebook kernel and run:

```python
import sys
from importlib.metadata import version

import labor_cost

print("Python:", sys.executable)
print("Version:", version("labor-cost-model"))
print("Module:", labor_cost.__file__)
```

If normal startup still cannot find it, check `sys.executable` and the
destination reported by `sys.executable -m pip show labor-cost-model`.
Pip configuration such as `PIP_TARGET` or `PIP_PREFIX` can install outside
the interpreter's search path; removing isolation does not fix an
unreachable installation destination. Use the company's environment setup
to align the pip destination and selected kernel.

Installation failures raise an error and do not return a success summary.
Temporary downloads are cleaned, but pip may have partially updated the
environment; recreate it with your normal procedure if necessary.

### Verification boundary

Automated tests exercise a simulated GitLab endpoint, pinned requests,
archive validation, failure cleanup and secret handling. An opt-in test
also installs the actual source into a disposable prepared environment and
runs the notebook's verification cells in a fresh isolated Python process.
From `production/`, run the protocol tests with:

```bash
.venv/bin/python -m pytest tests/test_gitlab_deployment.py -q
```

To include real installation, set `LABOR_DEPLOYMENT_TEST_PYTHON` to the
absolute Python path in a **disposable** environment with the deployment
requirements, pip and build access. The test installs into that environment:

```bash
LABOR_DEPLOYMENT_TEST_PYTHON=/absolute/disposable/venv/bin/python \
  .venv/bin/python -m pytest tests/test_gitlab_deployment.py -q
```

Actual cloud acceptance still requires running the installation and smoke
cells in your hosted notebook. Local tests cannot verify internal GitLab
connectivity, company token permissions or your kernel configuration.
Successful deployment does not close the existing
[analytical architecture gaps](architecture-gaps.md).

### Local verification — 2026-11-02

All 30 deployment tests passed, including real installation into a
disposable Python 3.10.12 environment with pip 23.2.1. Runtime dependencies
and isolated-build wheels came from local caches; no internal GitLab or
company credentials were used. The installed package reported version
0.2.0, imported from that environment's `site-packages`, and passed the
−100 → 150 / +250 smoke with notebook reads prohibited. Generated package
metadata was checked for absence of the test token.

Focused Ruff, pycodestyle, named-definition docstrings, clean notebook
syntax/output checks and local documentation links also passed. The full
analytical regression suite was not rerun for this deployment-only change.
