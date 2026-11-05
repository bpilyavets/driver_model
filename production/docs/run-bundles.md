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
- Actual method details in `coalition_audit`; dependency versions and
  full-workflow elapsed time.
- Table names mapped to generated Parquet filenames, hashes, row counts,
  column names, dtypes and object-column encoding.
- Interpretation of source, normalized and attribution values.

The stored source snapshot contains every team's original rows in the two
selected periods. It is distinct from `canonical_data`: payment matrices
use per-exposure economics while observed salary/time retain their units.
The canonical frame retains observed input columns. Named
`per_exposure__<factor>` columns expose normalized row amounts; aggregated
values are in `diagnostics/state_factors`. Source rows remain unchanged.

## Tables

| Name | Meaning |
| --- | --- |
| `source_snapshot` | Exact source rows selected for the comparison. |
| `canonical_data` | Validated inputs after mapping and normalization. |
| `bridge` | Organizational leaf contributions; reconciles as a whole. |
| `driver_bridge` | Organizational leaf contributions summed by driver. |
| `hierarchy` | Parent and leaf rows; do not sum both. |
| `local_bridges/<team>` | Separate bridge for that team's own cost change. |
| `team_summary` | Version-dependent team endpoints; see the schemas below. |
| `validation` | Recorded numerical checks and maximum absolute errors; currently not a complete ledger of all executed checks. |
| `assumptions` | Inactive, copied and explicitly referenced economics. |
| `coalition_audit` | Actual estimators, budgets, counts, seeds and runtime. |
| `diagnostics/*` | State factors, aligned inputs, profiles, matching/carry, exposure, fallbacks and game-local cell impacts. |

The following diagnostic tables are currently emitted by the graph API
for both declaration schemas: `state_factors`, `aligned_inputs`,
`membership_profiles`, `employee_alignment`, `exposure_allocation`,
`fallbacks` and `cell_impacts`, all under the `diagnostics/` prefix.
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
