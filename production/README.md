# Production labor-cost library

An installable, in-process calculation library and declarative model SDK.
Everything in this implementation lives in this directory. The original
PoC notebook remains unchanged and is used only as a test oracle.

The analytical requirements in `../docs/labor-cost-decomposition.md`
remain authoritative. The production package changes the deployment and
authoring interface, not the built-in model's interpretation.

## Install and run

Python 3.10 or later:

```bash
cd production
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest
```

If the system lacks `ensurepip`, `uv venv .venv` followed by
`uv pip install --python .venv/bin/python -e '.[test]'` is equivalent.
The core dependency set contains no notebook, Streamlit or plotting
requirement. SHAP is imported only when attribution is requested.

```python
from labor_cost import (
    AnalysisRequest, DatasetSnapshot, analyze, export_run, labor_cost_model,
)

# production_df is supplied by your existing data-loading process.
dataset = DatasetSnapshot("payroll", "2026-09-close-v1", production_df)
request = AnalysisRequest(
    "2026-08", "2026-09",
    source_cost_column="actual_cost",  # Omit to reconstruct payroll.
    explainer="exact",
)
result = analyze(labor_cost_model(), dataset, request)
bridge = result.table("driver_bridge")
team_view = result.select_teams(["Your team"])
export_run(result, "runs/2026-08-to-09")  # Requires a new directory.
```

Periods are explicit month starts or month strings. Input schema,
exposure assignment and applicability rules match the PoC. Negative
payments and totals are accepted; missing applicable values are errors.
Default exposure splits one person's monthly workforce unit equally
across observed employment types. A selected exposure column bypasses
that assumption without renormalization.

A `DatasetSnapshot` copies its input. Its `frame` property returns another
copy. Dataset/model versions and content fingerprints are included in
results. All teams in the two selected months participate in calculation
and employee matching. `select_teams` selects existing results only.
Organizational contributions and local team bridges must not be combined.

`analyze_labor_cost` remains available as a compatibility entry point
returning the original dictionary structure. New clients should use
`analyze` and `AnalysisResult`.

## Analyst model creation

The v2 API gives analysts control over scoped row formulas, aggregate
cost expressions, distributions, atomic switches and nested games.
The source grain is always employee × employment type × month; team
costs remain additive. Employment categories and dimension names are
model declarations, not special engine branches.

Start with [the complete authoring guide](docs/model-authoring.md) and
[the independent hourly model](examples/author_models.py). It defines
all layers explicitly, including grade-first or region-first composition,
grouped rate/hour switches and optional nested earnings games.

```python
from labor_cost import compile_model, labor_cost_model

model = labor_cost_model(schema_version=2)
compiled = compile_model(model)
print(compiled.inspect())
```

`labor_cost_model()` retains its v1 declaration/fingerprint and legacy
output tables. Both declaration versions run through the graph compiler.
Existing `InputSpec` extensions remain supported; new populations and
nesting use `ModelSpec(..., graph=ModelGraph(...))`.

Run the full analyst-to-business example with supplied hourly data:

```bash
python examples/model_lifecycle.py payroll.parquet new-catalog \
  --period-a 2026-08 --period-b 2026-09 --source-cost-column actual
```

It defines, compiles, validates and publishes a model, writes the catalog
index, resolves the published model/dataset and exports an analysis.
A catalog is an ordinary JSON file in a directory you choose. Business
clients load its declarations and use the installed library; they do not
import analyst authoring scripts. The installed implementation remains
accessible in its execution environment.

## Results and dashboard integration

See [the bundle contract](docs/run-bundles.md).

```python
from labor_cost import load_run

saved = load_run("runs/2026-08-to-09")
source = saved.table("source_snapshot")
canonical = saved.table("canonical_data")
dashboard_bridge = saved.table("bridge")
```

Loading verifies file checksums, model identity and the source snapshot.
It neither runs an analysis nor imports an estimator. The bundle includes
source/canonical tables, hierarchy, separate organizational/local bridges,
assumptions, validation and diagnostics. Internal Python state objects are
not serialized.

For standalone costing, use `build_state(model, dataset, period)` followed
by `cost(state, model)`. No comparison month or attribution is required.
Completion of historical missing economics happens in the comparison
builder; `cost` never silently copies history or fills unknown rates.

## Frontend examples

- `examples/notebook_client.ipynb`: a thin, unexecuted notebook using the
  package and catalog, with no embedded implementation or payroll data.
- `examples/streamlit_app.py`: model/dataset/period selection, execution,
  team result selection and bundle export. Install `.[streamlit]` and run
  `streamlit run examples/streamlit_app.py -- /path/to/catalog.json`.
- `examples/publish_catalog.py`: a command-line publication example using
  an existing Parquet dataset and a successful validation run.

No dashboard, source acquisition, server or scheduler is included.

## Verification and execution limits

```bash
python -m pytest
ruff check src tests examples
pycodestyle src tests examples
```

Tests retain the original compatibility regressions and compare both
compiled template versions against the read-only notebook in exact and
permutation modes. New-model tests exercise unrelated formulas, scoped
inputs, arbitrary dimensions, conditional distributions and nested games.
Additional tests cover model compilation, multiplicative inputs,
publication, bundles and frontend-independent imports. The regression
oracle tests require this repository checkout; deployed packages do not.

Exact remains the default, with the 15-player exact-game guard.
Permutation must be selected explicitly; defaults are 256 forward/reverse
cycles and seed 0. Small upper games and multipliers stay exact. Numerical
reconciliation does not measure sampling accuracy.

Execution is synchronous. Do not run multiple analyses concurrently in
threads: the pinned estimator temporarily uses NumPy's global RNG. Use
separate processes for concurrent runs. Salary caches remain local to a
single game and are not retained in returned states.

`examples/benchmark.py` measures the complete validated public workflow
against a supplied dataset. Its output is workload evidence, not a latency
commitment. No production-sized dataset was supplied with this repository.
See [verification results](docs/verification.md) for regression coverage,
installed-wheel checks and a reproducible synthetic runtime measurement.
