# Run bundle schema 2 (schema 1 remains readable)

`export_run(result, new_directory)` writes a complete temporary directory
and renames it into place only after all files succeed. Existing
run directories are not overwritten. There are no pickle files or
executable objects.

`manifest.json` contains:

- Bundle schema and engine versions, unique run ID and creation time.
- Full declarative model, schema version and content fingerprint.
- Compiled graph, factor declarations, scopes and game dependency order.
- Dataset ID, version, full snapshot fingerprint and selected-period hash.
- Periods, resolved exposure/completion policy, supplied references and
  explicit estimator settings.
- Dependency versions and full-workflow elapsed time. Actual per-game
  method details are in the separate `coalition_audit` table.
- Table names mapped to generated Parquet filenames, hashes, row counts,
  column names, dtypes and object-column encoding.
- Interpretation of source, normalized and attribution values.

The stored source snapshot contains every team's supplied analysis rows
in the two selected periods. It is distinct from `canonical_data`: payment matrices
use per-exposure economics while observed salary/time retain their units.
The canonical frame retains observed input columns. Named
`per_exposure__<factor>` columns expose normalized row amounts; aggregated
values are in `diagnostics/state_factors`. Source rows remain unchanged.

When analysis receives `PreparedInput` from `prepare_input`,
`source_snapshot` contains retained rows with explicit team replacements.
The additional `input_quality/source_snapshot` preserves selected original
rows before preparation; `input_quality/excluded_rows`, `reasons`, `summary`
and `team_assignments` preserve exclusion and assignment evidence.
`manifest["input_preparation"]` records the raw snapshot fingerprint,
model/period/input settings, row counts, exclusion fraction, fixed 1% limit,
team policy and exposure-after-exclusion assumption. The ordinary dataset
fingerprint describes the retained snapshot. Existing schema-two bundles
without preparation metadata remain unchanged and readable.

## Tables

| Name | Meaning |
| --- | --- |
| `source_snapshot` | Supplied analysis rows selected for the comparison; retained/reassigned rows when using `PreparedInput`. |
| `canonical_data` | Validated inputs after mapping and normalization. |
| `bridge` | Organizational leaf contributions; reconciles as a whole. |
| `driver_bridge` | Organizational leaf contributions summed by driver. |
| `hierarchy` | Parent and leaf rows; do not sum both. |
| `local_bridges/<team>` | Separate bridge for that team's own cost change. |
| `team_summary` | Version-dependent team endpoints; see the schemas below. |
| `validation` | Recorded numerical checks and maximum absolute errors; currently not a complete ledger of all executed checks. |
| `assumptions` | Inactive, copied and explicitly referenced economics, plus declared zero-weight rate completion. |
| `coalition_audit` | Actual estimators, budgets, counts, seeds and runtime. |
| `diagnostics/*` | State factors, aggregate-factor evidence, aligned inputs, profiles, matching/carry, exposure, fallbacks and game-local cell impacts. |
| `input_quality/*` | Only for prepared inputs: `source_snapshot`, `excluded_rows`, `reasons`, `summary`, `team_assignments`. |

The following diagnostic tables are currently emitted by the graph API
for both declaration schemas: `state_factors`, `aligned_inputs`,
`membership_profiles`, `employee_alignment`, `exposure_allocation`,
`fallbacks`, `aggregate_factors` and `cell_impacts`, all under the
`diagnostics/` prefix. Tables may be empty when their primitives are unused.
`diagnostics/state_factors` uses named coordinates for arbitrary dimensions.
`diagnostics/membership_profiles` retains sparse indices, weights and
conversion factors. `diagnostics/cell_impacts` contains actual sampled or
exact game contributions; no independent diagnostic game is rerun.
Its columns are `team`, `game`, `switch`, `cell`, `coordinates`,
`impact_unit`, `method`. Current rows cover inner games only, so
`impact_unit` is currency per exposure. Outer organizational/local games
are audited separately; they do not add rows to this table. These cell
impacts have not been multiplied through parent games and must
not be added directly to organizational or local currency bridges.

Coverage, unsupported-support, salary-support and matched-movement
summaries are not yet emitted by the graph runtime. Its generic cell table
is not equivalent to the enriched legacy salary/time/membership audits.
These remain requirements in
[GAP-002](architecture-gaps.md#gap-002-diagnostic-completeness).

## Team summary schemas

The model declaration schema determines the rendering, independently of
the bundle schema. For a v1 model analyzed through the graph API, columns
are `team`, `period_a_cost`, `period_b_cost`, `local_change`,
`organization_economics_impact`, `local_reconciliation_error`.

For a native v2 model, columns are `team`, `period_A_cost`, `period_B_cost`,
`cost_change`, `exposure_A`, `exposure_B`. V2 does not currently include
v1's organizational economics-impact or reconciliation-error columns.
Both expose local team endpoints; do not treat local cost change as the
team's organizational economics contribution. Choose a documented schema
using the stored model declaration rather than guessing from bundle version.

The `validation` table currently contains aggregate/row/source and bridge
checks assembled by the graph API. Attribution executes additional checks
that are not all recorded there. Population-level aggregate reconciliation
and independent template M/K/L checks are outstanding
[GAP-001](architecture-gaps.md#gap-001-population-level-aggregate-reconciliation)
and [GAP-003](architecture-gaps.md#gap-003-validation-evidence).

## Units and serialization

`result.table(name)` returns a pandas deep copy. Numeric/scalar table edits
do not change the stored table, but nested Python objects in object-dtype
cells are not recursively cloned. Treat nested diagnostic values as
read-only or copy those objects before editing. Reports create their own
copies, including nested object values.

Currency is the input dataset currency; there is no exchange-rate
conversion. Shares are signed ratios of the relevant net change and are
undefined near zero. Explicit unit columns on bridge/hierarchy rows remain
part of the table contract.

Object-dtype columns are JSON-encoded string cells in Parquet, listed in
`json_columns`. This preserves heterogeneous diagnostics reliably. Use
`load_run` to restore them; non-Python readers should apply the same
encoding. Reference tensors and sparse profiles also have explicit typed encodings.
Tuple, ndarray and timestamp values use tagged objects containing
`type` and `value` (and `dtype` for arrays). Missing optional cells encode as
JSON null. Numeric table columns retain their numeric Parquet types.

`load_run` checks schema version, table hashes and source/model identity.
The returned `AnalysisResult` has `states=None`; dashboard consumers need
only tables and metadata. To recompute, compile the stored model, create a
snapshot from `source_snapshot`, and use the stored settings. Source data
outside the selected periods are not needed by this run.

For a prepared run, this reconstructs the retained economic inputs but
does not recreate the `PreparedInput` audit attachment. `load_run` preserves
the existing audit without rerunning preparation. To create a new audited
preparation, use the original selected rows in
`input_quality/source_snapshot` with the recorded model, request and team
policy. That selected subset cannot reconstruct a full original dataset
fingerprint if the original snapshot also contained other periods.

The optional [Plotly reporting layer](reporting.md) consumes a loaded
`AnalysisResult` directly. It uses existing currency bridges and metadata;
neither internal states nor recalculation are needed. Figures are separate
presentation objects, not a new bundle table or schema revision.

## Team views

`result.select_teams([...])` returns selected organizational contributions,
organization-wide scale/mix rows separately, local bridges and team summary.
It performs no new calculation. A subtotal of selected organizational
contributions is not a selected-population bridge. Full organization and
local team allocations must never be combined.

## Catalog configuration

A catalog is an ordinary JSON file with explicit versions:

```json
{
  "models": [{
    "id": "labor-cost",
    "version": "1.0.0",
    "path": "models/labor-cost-1.0.0.json",
    "fingerprint": "the published model content hash"
  }],
  "datasets": [{
    "id": "payroll",
    "version": "2026-09-close-v1",
    "path": "datasets/payroll-2026-09.parquet",
    "fingerprint": "the DatasetSnapshot content hash"
  }]
}
```

Paths resolve relative to the catalog. Model files are written by
`publish_model`; datasets are supplied by existing data processes. Loading
checks fingerprints and refuses changed content under an existing catalog
revision. There is no implicit latest-version selection.

## Version and reference boundaries

Engine/package version, model declaration schema, analyst model version,
and bundle schema are separate identities. A schema-two bundle can contain
a v1 declaration with legacy-rendered tables or a native v2 declaration.
`labor_cost_model()` defaults to v1; passing `schema_version=2` is explicit.

The manifest's graph references contain typed factor values scoped by
population or a complete `team_state`. They are not the legacy dictionary
API's scalar reimbursement / `_employees` reference states. Reference
values are encoded as metadata; exported analysis states themselves are
omitted. See the [authoring reference](model-authoring.md#missing-support-and-explicit-references)
before constructing or replaying references.

Schema-one containers remain loadable without attribution. Native v2
models use scoped driver IDs and generic diagnostic tables; v1 models
retain their legacy bridge/hierarchy/team-summary table shapes.


## Aggregate-factor evidence (engine 0.3.0+)

V2 declarations may include `mean` and `weighted_mean` factors. Model and
bundle schema versions remain 2; reading these declarations requires engine
0.3.0 or later. Existing declarations retain their fingerprints and older
bundles remain readable. Notation settings remain outside model identity.

`diagnostics/aggregate_factors` records each effective cell aggregate:
period/team/population/factor, coordinates, kind, source and weight field IDs,
basis, value, observed numerator/denominator, zero-weight fallback totals,
donor rows/exposure/dimensions, method and source period/team, plus the
whole-population completion method. Reference-only statistics remain null
or absent when not observed; they are never inferred from the supplied rate.
Copied cells retain their donor statistics and identify the source period.
`diagnostics/state_factors` continues exposing the effective numeric tensors.

`assumptions` additionally includes `zero_weight_exposure_mean` rows with
factor and coordinates. These are cell-rate assumptions, not evidence that
an entire population was copied. `canonical_data` includes
`per_exposure__<factor>` columns for `mean` factors as well as payments.
Weighted rates are cell aggregates and are not written as row inputs.
Source snapshot amounts remain unchanged. All new tables round-trip through
ordinary bundle export/load and can be inspected without internal states.
