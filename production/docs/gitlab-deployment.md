# Install from GitLab in hosted Jupyter

This pathway installs `labor-cost-model` into your notebook's Python
environment. Afterwards you use ordinary `import labor_cost`; no custom
module loader, Git executable, CI runner or package registry is required.
Calculation still runs inside Jupyter. GitLab stores code, not payroll.

Use the [deployment notebook](../examples/gitlab_install.ipynb). It contains
the complete installation code, including every helper; the colleague's
`gitlab_read`, `load_project` and `CODE.import_module` are not needed.

## 1. Put the project in GitLab

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
calculation package. It deliberately imports no `labor_cost` modules.

## 2. Prepare the notebook environment

Use your company's existing virtual-environment instructions with
[runtime/requirements.txt](../runtime/requirements.txt). The file includes
the package's four core dependency constraints and `requests` for the
GitLab connection. Its agreement with `pyproject.toml` is tested.

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

## 3. Configure and install

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

## 4. Restart, then verify

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

## Troubleshooting

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

Installation failures raise an error and do not return a success summary.
Temporary downloads are cleaned, but pip may have partially updated the
environment; recreate it with your normal procedure if necessary.

## Verification boundary

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
