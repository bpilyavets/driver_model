# Architecture gaps and production-hardening requirements

Status: **open requirements**, reviewed on 2026-11-01 against the current
working tree. This register distinguishes implemented behavior from the
[methodology](../../docs/labor-cost-decomposition.md). It does not waive
requirements or change model semantics. Documentation alignment does not
constitute production-readiness certification.

The graph path means `analyze(...)` for either compiled model schema.
The compatibility path means `analyze_labor_cost(...)` and the extracted
legacy engine. The PoC notebook is a separate preserved oracle. Passing
checks on one path is not evidence that another implements the same checks.

## GAP-001 Population-level aggregate reconciliation

**Priority: high. Status: open. Affected path: graph API, v1/v2 declarations.**

**Requirement.** Independently reconcile modeled population costs to their
applicable row/source costs at declared economic boundaries. Organization
and team efficiency must not conceal offsetting population errors.

**Evidence.** `graph_api.analyze_graph` checks organization/team aggregate
endpoints and row/source totals by team/employment type. The latter does
not compare each child population's aggregate formula to its own row
payroll. An exact in-memory probe used one permanent and one temporary
worker, both with exposure one in each period:

| Population | A actual cost | B actual cost | Actual change |
| --- | ---: | ---: | ---: |
| Permanent | 100 | 120 | +20 |
| Temporary | 50 | 60 | +10 |

The deliberately wrong permanent unit formula used half its payment;
the temporary formula used twice its payment. Both team totals were
still correct (150 → 180). Every recorded validation passed, but the
population leaf impacts were +10 and +20, respectively.

Reproduce from a production environment. This is an intentionally invalid
model demonstrating a missing guard, not an authoring pattern:

```python
import pandas as pd
from labor_cost import (
    AnalysisRequest, DatasetSnapshot, Factor, FieldSpec, Game, ModelGraph,
    ModelSpec, Population, RowCost, SourceSchema, Switch, analyze,
    child, field, local_game, organization_game, outer_factors,
    select, value,
)

mix = value("mix")
model = ModelSpec(
    "population-reconciliation-probe", "1",
    graph=ModelGraph(
        source=SourceSchema(),
        populations=(
            Population("permanent", {"employment_type": "P"}),
            Population("temporary", {"employment_type": "T"}),
        ),
        fields=(FieldSpec("amount", "actual"),),
        row_costs=(RowCost("all", field("amount")),),
        factors=outer_factors() + (
            Factor("mix", "population_mix", dimensions=("kind",),
                   members=("permanent", "temporary")),
            Factor("p", "payment", population="permanent",
                   formula=field("amount")),
            Factor("t", "payment", population="temporary",
                   formula=field("amount")),
        ),
        games=(
            Game("p", 0.5 * value("p"), (Switch("p"),), "permanent"),
            Game("t", 2 * value("t"), (Switch("t"),), "temporary"),
            Game(
                "unit",
                select(mix, "kind", "permanent") * child("p")
                + select(mix, "kind", "temporary") * child("t"),
                (Switch("mix"), Switch("p_cost", child="p"),
                 Switch("t_cost", child="t")),
            ),
        ),
        team_game="unit",
        organization=organization_game("unit"),
        local=local_game("unit"),
    ),
)
frame = pd.DataFrame(
    [
        ("2026-08-01", "a", "P", "A", 100.),
        ("2026-08-01", "b", "T", "A", 50.),
        ("2026-09-01", "a", "P", "A", 120.),
        ("2026-09-01", "b", "T", "A", 60.),
    ],
    columns=["period", "employee_id", "employment_type", "team", "actual"],
)
result = analyze(
    model, DatasetSnapshot("probe", "1", frame),
    AnalysisRequest("2026-08", "2026-09", source_cost_column="actual"),
)
print(result.table("validation").passed.all())  # Currently True.
print(result.table("driver_bridge")[["driver", "impact"]])
```

**Current limitation.** Validated publication checks the recorded validation
result and model fingerprint; it cannot compensate for this missing guard.
Reconciliation success is necessary but insufficient evidence for publishing
a new economic model.

**Acceptance for a later fix.** Introduce explicit, auditable reconciliation
boundaries for complete population economics and reject the probe before
successful publication. Do not assume every nested child represents an
entire population's payroll: partial-cost children and overlapping additive
row components are supported. Cover correct and deliberately offsetting
models, both periods, signed amounts and exposure normalization. Preserve
valid partial-child models and built-in template attribution.

## GAP-002 Diagnostic completeness

**Priority: high for production auditability. Status: open. Affected path:
graph API for both declaration schemas.**

**Requirement.** Preserve coverage, unsupported counterfactual support,
matching/movement and effective-economics provenance required by methodology
sections 16 and 18. Keep source records, people and exposure distinguishable;
movement counts must deduplicate employee-month/team membership.

**Evidence.** On the same built-in fixture, the graph API returns:

- `diagnostics/state_factors`, `diagnostics/aligned_inputs`;
- `diagnostics/membership_profiles`, `diagnostics/employee_alignment`;
- `diagnostics/exposure_allocation`, `diagnostics/fallbacks`;
- `diagnostics/cell_impacts`.

The compatibility engine additionally provides `coverage`, `unsupported`,
`salary_support`, `matched` movement/change summaries and enriched
`salary_cells`, `worked_fraction_cells`, `workforce_mix_cells` diagnostics.
Its `employee_inputs` corresponds in purpose to the graph's aligned-input
table but is not an interchangeable schema.

**Current limitation.** Raw graph profiles/matching records do not equal
the missing summaries. Generic `cell_impacts.impact_unit` is in game-local
units. Current cell rows cover inner games only (currency/exposure); outer
games do not emit rows in this table. It lacks the legacy enriched
organizational/local propagated amounts and support annotations. Renaming
the table does not restore that information.
V1 rendering adapts bridge/hierarchy/team-summary tables, not these audits.

**Acceptance for a later fix.** Provide equivalent diagnostic information
for the template through the graph path and define general schemas for
arbitrary populations/dimensions. Reuse actual estimator samples; never
run an independent exact audit for sampled impacts or rescale samples.
Test copied/referenced economics, sparse fallback, transfers, entry/exit,
multiple employment types and zero-change parents. Test diagnostic units,
reconciliation and provenance through bundle round trips. Label support
shares as support measures, not uncertainty intervals.

## GAP-003 Validation evidence

**Priority: high. Status: open. Affected path: graph API and exported runs.**

**Requirement.** Expose evidence for performed checks, and retain independent
analytical M/K/L checks for the built-in template alongside general
parent-replacement and efficiency checks.

**Evidence.** Graph attribution executes exact/sampled efficiency,
exact-coalition coverage, parent replacement and flattened-bridge checks.
Its returned `validation` table is assembled separately and includes only
a subset of all checks. `coalition_audit` records estimator metadata and
success but is not a complete validation ledger. The compatibility engine
executes and records named `M analytical`, `K analytical`, `L analytical`
checks; the graph path does not implement those independent closed-form
checks for the template.

**Current limitation.** A passed graph validation table does not mean the
legacy validation ledger or every promised independent check was produced.
General graph multipliers remain valid auxiliary-game calculations; M/K/L
closed forms apply to the template, not every configurable hierarchy.

**Acceptance for a later fix.** Record executed validations with stable
scope and relevant numerical error/tolerance information. Restore the
independent template multiplier comparisons without hard-coding its
hierarchy into arbitrary models. Preserve zero-parent behavior and add
negative checks that detect deliberately incorrect multipliers or parent
allocations. Verify exported evidence matches actual execution.

## GAP-004 Coverage by execution path

**Priority: high before readiness claims. Status: open. Affected scope:
verification evidence and future release gates.**

**Requirement.** Every claimed graph guarantee must have direct graph-path
evidence. Notebook/compatibility coverage remains valuable but cannot be
counted as equivalent production graph coverage.

**Evidence map from the existing tests:**

| Guarantee | Direct graph evidence | Other evidence / remaining limitation |
| --- | --- | --- |
| Built-in bridge parity | `test_sdk.py::test_compiled_model_parity`; `test_graph.py::test_v2_template_numerical_parity` | Notebook oracle; output/diagnostic coverage is not exhaustive. |
| Signed inputs, salary/time independence, transfers, exposure, missing branches | `test_graph.py::test_native_semantic_regressions`, `test_declared_fallback_and_reference_policies`, `test_effective_inputs_after_whole_team_copy` | Broader legacy cases run through `test_poc_regressions.py`; do not assume one-to-one graph coverage. |
| New formulas, pairing, scopes, factorization, grouped/nested games | Graph tests and SDK extension tests | These are direct new-engine checks. |
| Standalone costing and presentation-only team selection | `test_graph.py::test_observed_only_costing_and_v2_catalog`; `test_sdk.py::test_input_integrity_and_team_views` | Deployment and frontend checks are distinct from analytical completeness. |
| Bundles, catalog and schema-one import | `test_bundles.py` and graph bundle/catalog tests | Round-trip preservation does not prove all required diagnostics exist. |
| Complete support and movement summaries | No equivalent complete graph diagnostic contract/tests | Legacy diagnostics and root analytical tests; GAP-002. |
| Independent template M/K/L and full validation ledger | No equivalent graph checks/ledger | Compatibility implementation; GAP-003. |
| Offsetting population aggregate errors | The reproducible probe above passes incorrectly | Required future rejection test; GAP-001. |

Test paths in the table are relative to `production/tests/`. That suite's
`model` fixture imports the extracted engine. Its `notebook_model` fixture
loads the notebook oracle, while explicit `analyze(...)` tests exercise
the graph. Root `tests/` target the notebook. This register identifies
coverage by entry point, not merely by test name or test count.

**Acceptance for a later fix.** Maintain a requirements-to-tests map and
add direct graph regression cases for uncovered guarantees. Release
reports must identify tested revision, environment, execution paths and
actual results. Historical counts/timings must remain historical. Closing
GAP-001–003 requires direct graph tests, not only more compatibility tests.

## Change policy

All four gaps remain open after documentation alignment. Resolve them in
separately scoped calculation/test changes. Preserve the notebook oracle,
existing reference-model semantics and numerical tolerances. Any intended
methodology change must be explicit and must not be disguised as a missing
feature being removed from the documentation.
