# Production graph verification evidence

## Jupyter formula previews — 2026-11-01

The formula-preview change adds tree-based mathematical rendering,
notebook-local notation, a data-free authoring notebook, and a published
model preview in the analysis client. The Streamlit example, dependency
extra and dedicated test are retired. Historical Streamlit results below
describe the earlier implementation and are retained as historical evidence.

The complete production suite was run with:

```bash
production/.venv/bin/python -m pytest -c production/pyproject.toml production/tests -q
```

Result: **203 passed, 1 failed** in 361.94 seconds. The sole failure was
`test_quality.py::test_style_and_docstrings`, caused by pre-existing
`examples/poc_extented_model_test.py` formatting: line 194 exceeds 79
characters and line 418 has no final newline. Broad Ruff also flags that
experiment's unused NumPy import. The analyst experiment was left intact.

After final layout refinements, the focused presentation tests and authoring
fresh-kernel test passed together: **21 passed, 5 deselected**. The full
run also passed all analysis-client kernel cases: unconfigured, model-only
preview, exact analysis and permutation analysis. The maintained source,
tests and examples pass Ruff and pycodestyle when excluding the existing
experiment; patch whitespace checks pass.

The tests cover both declaration versions, model identity preservation,
optional-dependency-free rendering, complete and draft models, conditional
axes, paired inputs, multiple memberships and amount bases, selections,
child expansion, symbol collisions, escaping and long-formula wrapping.
They do not close any analytical gaps in the register.

Temporary executed notebooks were inspected in JupyterLab using Chromium.
The final isolated workspace displayed 12 math blocks, including the native
template, a 25-term formula and escaped-label cases, with no MathJax errors
or equations overflowing the available notebook width at a 1440-pixel
viewport. Long products and additions wrap without distributing terms.

Repository notebooks remain unexecuted with empty outputs. The original
PoC notebook remains unchanged, with SHA-256
`03ca24c1e13bbca01587d1c95fe040ebf4a1021f5c9f9e3940fda712d94aea81`.

## Historical implementation verification

The counts, timings and installation checks in this section record the
original v0.2.0 implementation session. They have not been rerun as a full
release gate for every subsequent working-tree change. In particular,
compatibility-engine tests do not imply complete graph-path coverage;
consult the [gap register](architecture-gaps.md).

The v2 implementation preserves employee × employment type × month source
data and exposes scoped row formulas, aggregate expressions, distributions,
atomic switches and affine nested games. Both template declaration
versions execute through this graph runtime. The dictionary compatibility
entry point retains the original extracted implementation.

### Automated coverage

Commands in this historical section were run from `production/`, using
its Python environment with test extras installed.

The final inventory contains 184 tests. Verification was split between
134 compatibility/parity tests in the full regression run and a final
successful run of all 50 tests covering the changed package surfaces:

```bash
python -m pytest tests/test_graph.py tests/test_sdk.py \
  tests/test_bundles.py tests/test_quality.py \
  tests/test_streamlit_client.py -q
```

That final run passed in 100.95 seconds. It includes exact/permutation
notebook parity, salary/time independence, signed payroll, transfers,
supplied exposure, population applicability, missing economics, explicit
references, alternative distribution factorizations, grouped switches,
recursive games and zero-change parents with offsetting children.

It also covers observed-only costing, unchanged caller frames,
presentation-only team filtering, catalog publication/resolution,
schema-two bundle round trips, a genuine schema-one fixture, estimator
limits, RNG preservation, cached/direct aligned arithmetic, and fresh
notebook kernels. Compiler failures cover invalid units, switch ownership,
conditioning cycles, nonlinear child costs and source mappings.

Final Ruff, pycodestyle and named-definition docstring checks passed.
Monetary reconciliation uses `rtol=1e-10`, `atol=1e-6`.

### Deployment verification

Built `labor_cost_model-0.2.0` as a wheel and source distribution using
cached dependencies. Installed the wheel separately from the source tree
and ran Python from `/tmp` with isolated import behavior. The smoke check
verified the imported package came from that installation.

With notebook file reads explicitly blocked, a separately declared signed
payment model compiled, built an observed state, returned a cost of -100,
explained a 250 change, and exported/imported every result table and the
original provenance. Standalone costing imported neither SHAP nor
frontend/plotting modules. Export adds table checksums and schema metadata
to the original manifest.

### Runtime measurement

Reproduce the synthetic workload in separate fresh processes:

```bash
python benchmarks/graph_runtime.py --explainer exact
python benchmarks/graph_runtime.py --explainer permutation
```

The workload contains 267 records, 121 distinct employee IDs, four teams
and two months. It includes all three template employment populations,
multiple grades/coefficient bands, signed payments, overlapping employment
types, a transfer and entry/exit. Source payroll is computed independently
of the cost engine.

| Setting | Full validated analysis | Permanent leaf work per team |
| --- | ---: | --- |
| Exact | 37.99 s | 14 players; 16,384 coalitions; 16,386 evaluations |
| Permutation, 256 cycles, seed 0 | 21.09 s | 14 players; budget 7,424; 7,426 evaluations |

Both runs passed all reconciliation checks. Upper and auxiliary games
remained exact. Each permanent game used eight local aligned-pay cache
entries. Permutation evaluated 3,870–3,928 distinct coalitions per team;
the run audits record actual derived team seeds and counts.

Timing covers the public `analyze` call, including first-use estimator
initialization and validation. Dataset construction and bundle export are
outside the timer. No allocation profiler was enabled. Environment:
Python 3.10.12, NumPy 1.26.4, Pandas 2.3.3, SHAP 0.48.0, PyArrow 23.0.1,
Linux x86-64 under WSL2. The workload fingerprint is
`2b2428d0f499ebdf5154475358397f603e214fa29e19c595c2a4ed91fb8c78e8`.

These are single synthetic observations, not latency commitments or
sampling-accuracy estimates. Measure the intended production workload
before selecting operational limits. Execution is synchronous; any future
concurrent permutation execution must use separate processes.

### Project isolation

During that implementation session, all project additions and changes
were under `production/`. Later approved documentation alignment updates
the root guidance and methodology as well. SHA-256 checks
against the saved baseline passed for all 15 existing source/configuration
files, including the unchanged methodology and original test suite.
The PoC notebook SHA-256 remains:

```text
03ca24c1e13bbca01587d1c95fe040ebf4a1021f5c9f9e3940fda712d94aea81
```


## Documentation alignment verification — 2026-11-01

This was a documentation-only update, not a full analytical regression or
release-certification run. Existing calculation code, tests, notebook and
experimental examples were retained unchanged.

Checks performed for the updated contracts:

- Built a wheel offline and installed it into a temporary target using
  cached dependencies. Imported that installation with isolated Python
  from a temporary working directory; the package was version 0.2.0.
- Executed the root README's production example against independently
  costed synthetic payroll, including negative monetary components.
- Verified observed-only costing before SHAP/frontend modules were loaded,
  then v1 exact, native v2 exact and native v2 permutation analyses.
- Checked actual diagnostic names, both `team_summary` schemas and the
  generic cell-impact columns. Confirmed cell rows are inner-game only.
- Checked source-frame integrity, result-only team selection, validated
  publication, catalog model/dataset resolution and numerical/provenance
  round trips for every emitted table.
- Executed the existing GUIDE.MD Python blocks in order, redirecting only
  its publication destination to a temporary directory. The expected
  6,575 → 6,982 endpoints and +407 bridge were preserved.
- Reproduced the intentionally invalid GAP-001 model: recorded validation
  still passes while population impacts are swapped. The gap remains open.
- Checked the documented lifecycle CLI help, Python snippet syntax,
  relative links/anchors and preservation of all 21 numbered methodology
  headings. Reviewed the diff for documentation-only scope.

Temporary installation and publication outputs were removed after the
checks. The full historical test suite and performance benchmark were not
rerun for this documentation change; their results above remain historical.
See the [gap register](architecture-gaps.md) for outstanding requirements.

## Plotly reporting verification — 2026-11-06

The reporting suite and the updated thin-client fresh-kernel checks passed:
**40 tests**, covering v1/v2 exact/permutation results, nested custom models,
team entry/exit, negative endpoints, zero net changes and offsetting effects,
duplicate labels, precise chart values, selectors and malformed inputs.
Round-trip chart data matched, including the genuine schema-one bundle.
Reporting with calculation entry points blocked passed; tables, nested
object cells and manifest data remained unchanged.

A separately installed wheel compiled, costed and analyzed the deployment
fixture with Plotly/IPython/nbformat absent. Installing its `reporting`
extra enabled charts from a restored bundle with SHAP imports blocked.
Offline HTML charts were rendered in headless Chrome and visually inspected,
including positive/negative endpoints, local allocations, duplicate labels
and offsetting team bars. All 25 GUIDE Python blocks executed, including
reporting and temporary publication; its +407 change was preserved.

Focused Ruff, pycodestyle, named-definition docstrings, clean client-notebook
outputs and documentation-link checks passed. The wider existing quality
suite has two unrelated failures retained unchanged: formatting errors in
`examples/poc_extented_model_test.py` and saved execution outputs in
`examples/model_authoring.ipynb`. Repository-wide Ruff also flags existing
imports in the experimental script and GitLab installation notebook.
Those files were not changed by the reporting work. The complete analytical
regression suite was not rerun, and existing architecture gaps remain open.
