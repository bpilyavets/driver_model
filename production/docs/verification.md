# Production graph verification

The v2 implementation preserves employee × employment type × month source
data and exposes scoped row formulas, aggregate expressions, distributions,
atomic switches and affine nested games. Both template declaration
versions execute through this graph runtime. The dictionary compatibility
entry point retains the original extracted implementation.

## Automated coverage

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

## Deployment verification

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

## Runtime measurement

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

## Project isolation

All project additions and changes are under `production/`. SHA-256 checks
against the saved baseline passed for all 15 existing source/configuration
files, including the unchanged methodology and original test suite.
The PoC notebook SHA-256 remains:

```text
03ca24c1e13bbca01587d1c95fe040ebf4a1021f5c9f9e3940fda712d94aea81
```
