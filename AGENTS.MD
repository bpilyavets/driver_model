# Repository purpose and execution paths

The active implementation is the installable library in `production/`.
It explains labor-cost changes and is progressing toward production
readiness. Standalone costing also leaves room for future planning; do
not implement an optimizer, ingestion service, API server, scheduler or
new dashboard unless requested.

Read [the methodology](docs/labor-cost-decomposition.md) before changing
payroll, workforce states, counterfactuals or attribution. Its general
requirements apply to every model; its explicitly scoped template rules
apply to the built-in labor-cost model. Read the
[implementation-gap register](production/docs/architecture-gaps.md) before
claiming a requirement is fully implemented. Missing requirements remain
requirements until explicitly revised; do not remove them to match code.

Keep these execution paths distinct:

| Path | Role |
| --- | --- |
| `labor_cost.analyze(model, dataset, request)` | Production graph runtime for compiled v1 and v2 declarations; returns `AnalysisResult`. |
| `labor_cost.analyze_labor_cost(...)` | Extracted legacy engine; dictionary compatibility contract. |
| `labor_cost_decomposition.ipynb` | Preserved PoC implementation and read-only regression oracle. |

Production execution must never load or execute the PoC notebook. Keep
its implementation, test fixtures and outputs unchanged unless the user
explicitly requests a notebook change. Do not route new graph features
through the compatibility engine or move package logic into a frontend.

# Package boundaries

Maintain this separation:

```text
ModelSpec -> compiler -> CompiledModel
                              |
DatasetSnapshot + AnalysisRequest
             -> input adaptation and exposure assignment
             -> observed/comparison state construction and completion
             -> pure cost evaluation
             -> attribution and recursive propagation
             -> result tables and manifest
             -> optional run export / frontend presentation
```

`build_state(model, dataset, period)` builds an observed-only state.
`cost(state, model)` must work without SHAP, comparison history or input
acquisition. Custom states need their matching model. Historical economic
completion belongs in the comparison builder, never in cost evaluation.

Keep plotting and Jupyter dependencies outside calculation
code. Catalogs are filesystem JSON indexes; runs are JSON/Parquet bundles.
The dashboard consumes tables and metadata without internal Python states
or recalculating attribution. Maintain synchronous in-process execution.
Permutation currently uses NumPy's global RNG temporarily; future
concurrent runs must use process isolation, not threads.

# General analytical requirements

- Source grain is employee × employment type × month. Require unique
  keys, month-start periods and one team per employee-month. Source
  column names, employment categories and applicable fields are declared
  by the model. Preserve caller DataFrames and actual source amounts.
- Select comparison periods across the full organization before employee
  alignment. Team selection is a view of existing results, never an input
  filter or a new matching population.
- The cost engine defines payroll. Attribution explains it and must not
  introduce different arithmetic. Employees are alignment identities,
  not individual Shapley players.
- Preserve joint employee inputs in products. Never replace a paired
  mean with a product of independent means or manufacture economic
  concepts that cannot be identified from the source data.
- Switch complete probability vectors, conditional tables and membership
  collections atomically. Group structurally coupled factors explicitly.
  Membership includes its positive exposure-conversion factors.
- Accept finite signed monetary amounts, references and costs where the
  model declares signed fields. Never clip corrections or reversals.
  Enforce each declared field's positivity/nonnegativity and units on
  applicable rows. Missing applicable inputs are errors, not absence.
- Default exposure is `1/k` across observed employment types per
  employee-month. Explicit exposure must be in `(0, 1]`; never renormalize
  supplied totals or interpret exposure as hours/FTE without evidence.
- Actual payment factors divide by exposure; actual-basis membership
  supplies `1/exposure` conversion while aligned inputs stay observed.
  Explicit per-exposure factors use their declared basis. Keep raw source,
  canonical row inputs and aggregated state factors distinguishable.
- Missing economics remain unknown, not zero rates. A zero-share unknown
  may be skipped; positive-share unknown economics must fail. Inactive
  branches need no fabricated inputs or games.
- Sparse fallback uses declared support within the same team, population
  and period. Cross-period same-person matching is distinct from fallback.
  Carry missing aligned observations from that person's observed endpoint,
  recording the assumption; never manufacture workers or exposure.
- Complete missing population/team economics according to the selected
  policy. Explicit references affect missing economics only; preserve
  observed economics. `require_reference` must reject unsupported one-sided
  absence. Whole-team copying retains zero exposure at the absent endpoint.
- Organizational and local bridges are distinct allocations. Never combine
  them or sum hierarchy parents with their children. Driver summaries
  aggregate organizational leaves without new attribution games.
- Nested costs must satisfy the compiler's affine-child restrictions;
  organization costs remain additive by team. Propagate with exact
  auxiliary zero-to-one games, never division by a parent's cost change.
  Preserve offsetting children when the parent change is zero.
- Exact is the default, with the 15-player exact-game guard. Leaves may
  explicitly use permutation: 256 cycles and seed 0 by default. Parents
  and auxiliary games stay exact; never change estimators silently.
- Validate every evaluated counterfactual. Exact mode requires exhaustive
  coalitions; sampled cell audits reuse the actual sampled contributions.
  Report methods, budgets/counts, effective seeds, runtime and sampled
  versus exhaustive extrema. Reconciliation is not sampling accuracy.
- Reconcile source, row and aggregate costs at meaningful population/team/
  organization boundaries and every hierarchy boundary, using
  `rtol=1e-10`, `atol=1e-6`, without intermediate rounding. Preserve the
  outstanding population-reconciliation and audit-evidence requirements
  recorded in the gap register; current total-level checks are insufficient.
- Keep assumptions, fallback support, matching/carry provenance and
  effective reference values visible. Coverage/support shares are not
  uncertainty intervals. Do not describe missing graph diagnostics as
  already available because the legacy engine provides them.

# Model authoring and interpretation

Jupyter notebooks are the supported user-facing interface. Mathematical
previews live in the presentation module and derive from expression trees;
notation settings do not change model declarations or fingerprints.
Keep the authoring and analysis notebooks thin, with no stored outputs.

V2 declarations separate `SourceSchema`/`Population`/`FieldSpec`/`RowCost`,
state `Factor`s, aggregate expressions, and `Switch`/`Game` definitions.
Compilation checks supported declarations; analysis constructs states and
runs games. There is no training or fitting stage.

Keep driver extensions localized to model declarations and source payroll
where existing primitives suffice. Verify extensions without unrelated
reporting edits. New mathematical primitives require engine work and
explicit semantic tests; arbitrary callbacks are not a v2 escape hatch.

Support the documented arithmetic, named-axis aggregation, whole
conditional distributions, population scopes and aligned products.
Reject unsupported units, ownership or hierarchy combinations with
contextual errors. Distinguish source category values, population IDs,
factor IDs, game IDs and presentation labels in examples.

Membership measures changes in participating employee profiles and their
weights/conversions. Its scope is the expression inside `aligned_mean`;
it can include bonuses or temporary payroll. Aligned-input drivers measure
changes in matched inputs under the carry assumptions, not every change
in a workforce average. Do not redistribute membership impacts or change
these definitions without an explicit methodology decision.

# Built-in labor-cost template requirements

Preserve the original template semantics in methodology sections 5–10,
14–16 and 21, even though custom v2 models may use different declarations:

- Three employment categories, the documented actual-row payroll formula,
  region-first grade composition and the organization/team hierarchy.
- Positive base salary and regional coefficients; nonnegative worked
  fraction, including values above one. Apply regional adjustment once
  only to eligible components. Coefficient bands do not identify geography.
- Separate salary, worked-fraction and within-cell base-pay membership
  drivers with their existing IDs/labels. Membership affects base pay
  only in this template; supplemental payments keep their cell definitions.
- Same-team/period cell → grade → region → permanent-population fallback;
  complete-copy missing-category/team assumptions and explicit references.
- Fixed regional policy; salary changes include promotions without
  identifying their cause. No inferred grade coefficient or compa-ratio.
- Fourteen permanent players and eight salary/time/membership cache
  scenarios for the unextended template. Caches stay local to a game,
  never in returned states. Verify cached/direct costing equality.
- Independent M/K/L analytical checks and support/matched diagnostics remain
  requirements. Their graph-runtime gaps are explicit, not waived.

Keep v1 declaration fingerprints and compatibility behavior stable unless
an intentional migration is requested. Model version, declaration schema,
engine version and bundle schema are separate identities. Preserve
schema-one bundle readability and record actual settings/provenance.

# Verification and documentation

From the repository root, using the production environment with test
extras installed:

```bash
production/.venv/bin/python -m pytest -c production/pyproject.toml production/tests
production/.venv/bin/python -m pytest -c pytest.ini tests
```

The first suite contains graph-specific tests and regressions against the
extracted legacy engine; the second tests the notebook. Notebook parity
compares against tagged cells read-only. These paths are not interchangeable
coverage. Fresh-kernel tests require local socket access; the root suite's
`-m 'not kernel'` option excludes its kernel tests, not production tests.

For changed calculations, run relevant exact/permutation parity,
independence, signed-input, transfer, exposure, missing-economics,
extension and bundle tests. Test supported new graphs directly, not only
through the compatibility fixture. Verify notebook-free installed costing
and analysis when deployment interfaces change. Record failures and
coverage limitations accurately; do not infer a green current release from
historical test counts or benchmarks.

Document named definitions with meaningful docstrings, using NumPy-style
contracts for substantial public functions. Follow PEP 8, 79-character code
lines and readable prose. From `production/`, run the applicable Ruff and
pycodestyle checks on `src`, `tests` and maintained `examples`. Do not alter
user experiments merely to make an unrelated documentation check pass.

Update methodology, public contracts and affected examples alongside
behavior changes. Preserve published links where practical. Benchmark
real validated workloads separately from accuracy experiments; make no
latency or production-readiness claims without corresponding evidence.
