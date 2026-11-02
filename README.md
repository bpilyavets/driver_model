# Labor-cost modeling and attribution

The active implementation is an installable Python library in
[`production/`](production/README.md). Analysts declare payroll formulas,
workforce factors and attribution games. Business users run published
models against predefined datasets through the library or thin Jupyter
notebooks. Calculation and presentation are separate.

The library is progressing toward production readiness. The
[methodology](docs/labor-cost-decomposition.md) defines analytical
requirements; the [gap register](production/docs/architecture-gaps.md)
records outstanding implementation requirements. Passing reconciliation
alone does not establish correct model interpretation or release readiness.

## Start here

- **Analysts:** [complete permanent/temporary walkthrough](production/examples/GUIDE.MD),
  then the [model-authoring reference](production/docs/model-authoring.md)
  and [formula-preview notebook](production/examples/model_authoring.ipynb).
- **Business clients:** [package quickstart](production/README.md) and the
  [publish/resolve/analyze example](production/examples/model_lifecycle.py).
- **Dashboard consumers:** [run bundle and table contract](production/docs/run-bundles.md).
- **Contributors:** [engineering instructions](AGENTS.MD), methodology and
  [verification evidence](production/docs/verification.md).

## Install and choose the interpreter

From the repository root, with Python 3.10 or later:

```bash
python3 -m venv production/.venv
production/.venv/bin/python -m pip install -e './production[test]'
```

If the environment lacks `ensurepip`/pip, use installed `uv` instead:

```bash
uv venv production/.venv
uv pip install --python production/.venv/bin/python -e './production[test]'
```

Use one setup route. The test extra includes the dependencies needed for
repository regression checks; ordinary consumers can install
`./production` without `[test]`. Install `./production[notebook]` for
JupyterLab; it is optional for the calculation core.

Run examples with that interpreter:

```bash
production/.venv/bin/python production/examples/model_lifecycle.py --help
```

In VS Code select `production/.venv/bin/python`. In Jupyter select a kernel
from that environment; register one if necessary:

```bash
production/.venv/bin/python -m ipykernel install --user --name labor-cost-production
```

An import failure usually means a different Python is running. Check:

```bash
production/.venv/bin/python -c 'import sys, labor_cost; print(sys.executable); print(labor_cost.__file__)'
```

## Production workflow

The source grain is employee × employment type × month, with exactly one
team per employee-month. `SourceSchema` maps the columns; populations,
applicable payroll fields and categorical dimensions are model-specific.
The original Russian production column names are required by the
[built-in template](docs/labor-cost-decomposition.md#21-production-schema-signed-values-and-notebook-workflow),
not by every custom model.

For supplied `production_df` using that template's columns:

```python
from labor_cost import (
    AnalysisRequest, DatasetSnapshot, analyze, build_state,
    compile_model, cost, labor_cost_model,
)

model = compile_model(labor_cost_model(schema_version=2))
dataset = DatasetSnapshot("payroll", "2026-09-close-v1", production_df)
request = AnalysisRequest(
    "2026-08", "2026-09",
    source_cost_column="actual_cost",  # Requires this independent total.
    explainer="exact",
)
result = analyze(model, dataset, request)
bridge = result.table("driver_bridge")

state = build_state(
    model, dataset, "2026-08", source_cost_column="actual_cost"
)
observed_cost = cost(state, model)
```

Omit `source_cost_column` to reconstruct the row reference from the model;
that checks aggregation, not independent payroll completeness. Applicable
signed payments remain signed. Exposure defaults to `1/k` across a person's
observed employment types, or uses explicitly selected `(0, 1]` row values
without renormalization. Salary/time inputs remain paired by employee.

Compilation validates declarations without fitting a model or running
Shapley. `analyze` builds comparison states and evaluates the declared
games. `build_state` and `cost` support observed-only costing without SHAP.
New models use v2 declarations; `labor_cost_model()` still defaults to the
v1 declaration for compatibility. Both run through the graph API.

Analysis includes all teams in the selected months. Afterwards,
`result.select_teams(["an existing team"])` selects existing contributions
and separate local bridges. It does not filter matching inputs or rerun
attribution. Never combine organizational and local bridge rows.

Analysts validate and publish a model JSON with `publish_model`. A catalog
is an ordinary filesystem JSON index of explicitly versioned model and
Parquet dataset entries. It resides wherever the application configures
it; it is not a registry service. Business clients resolve those entries
and call `analyze`. The [lifecycle example](production/examples/model_lifecycle.py)
shows the full process with supplied data.

`export_run` writes a new JSON/Parquet run directory. `load_run` restores
its tables and metadata without recalculating attribution. The dashboard
uses that contract, not internal state objects. No dashboard, ingestion,
API service, scheduler or optimizer is included.

## Execution paths and compatibility

| Entry point | Calculation implementation | Result |
| --- | --- | --- |
| `labor_cost.analyze(model, dataset, request)` | Compiled graph runtime; v1 and v2 model declarations | `AnalysisResult`, manifest, named tables, optional in-memory states |
| `labor_cost.analyze_labor_cost(...)` | Extracted legacy engine | Legacy dictionary and diagnostics |
| PoC notebook input cell / `Run All` | Notebook implementation | Legacy dictionary and notebook presentation |

Do not interchange modern and legacy state/reference formats. Production
runtime never loads the notebook. Some compatibility diagnostics have not
yet been implemented for the graph API; see the gap register and exact
[table schemas](production/docs/run-bundles.md).

## Verification and operational limits

From the repository root:

```bash
# Production graph tests plus extracted-engine compatibility regressions.
production/.venv/bin/python -m pytest -c production/pyproject.toml production/tests

# Preserved notebook regressions.
production/.venv/bin/python -m pytest -c pytest.ini tests
```

The notebook-only suite supports `-m 'not kernel'` when local Jupyter socket
access is unavailable. Production fresh-kernel tests are separate. Oracle
regressions require the checkout; installed library execution does not.

Exact attribution is the default with a 15-player safeguard. Permutation
must be requested explicitly; defaults are 256 forward/reverse cycles and
seed 0. Graph leaf games use the selected estimator; parents and auxiliary
games remain exact. Reconciliation is not a sampling-accuracy estimate.
Runs are synchronous. Use separate processes for future concurrent
permutation execution because the estimator temporarily uses global RNG
state. No production latency commitment is implied by synthetic timings.

Historical test counts and runtime measurements are recorded in
[verification evidence](production/docs/verification.md); they do not
certify the current working tree. Required population-level reconciliation,
diagnostics and coverage follow-ups are tracked separately.

## Preserved PoC workflow

For the original model/presentation, open
[`labor_cost_decomposition.ipynb`](labor_cost_decomposition.ipynb), select a
compatible kernel, load your DataFrame and edit **Your data and settings**:

```python
input_df = production_df
period_a = "2026-08"
period_b = "2026-09"
source_cost_column = None
exposure_column = None
```

Then run all cells. Unconfigured execution displays setup guidance. Its
`report`, plotting options and dictionary diagnostics are notebook/legacy
interfaces, not the graph API. Keep the oracle notebook free of stored
payroll outputs. The original `requirements.txt` / `requirements-dev.txt`
remain the PoC dependency files; production dependencies live in the
package configuration. See methodology sections 11, 18 and 21 for the
legacy contracts.
