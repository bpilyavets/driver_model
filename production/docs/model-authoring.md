# Authoring workforce models

This guide describes native v2 declarations for the production graph API.
V1 compatibility authoring is more restricted; the dictionary compatibility
API uses a separate extracted engine. General analytical requirements live
in the [methodology](../../docs/labor-cost-decomposition.md). Outstanding
runtime guarantees remain explicit in the [gap register](architecture-gaps.md).
For a complete first model, use the
[permanent/temporary walkthrough](../examples/GUIDE.MD).

Every dataset contains employee × employment type × month records.
`SourceSchema` maps those keys and the team column. Each employee-month
belongs to exactly one team. Other dimensions and payroll inputs are
required only where the model declares them applicable.

A model has four distinct layers:

1. Actual row payroll (`RowCost` and `field` expressions).
2. Workforce state (`Factor` declarations).
3. Aggregate cost (`value`, `child`, `aligned_mean`, `sum_over`, `select`).
4. Attribution (`Switch` and `Game` declarations).

`ModelSpec(..., graph=ModelGraph(...))` selects model schema 2.
Put source exposure and missing-economics policies inside `ModelGraph`;
mixing graph declarations with legacy `ModelSpec` options is rejected.
`compile_model` validates declarations without data or Shapley evaluation.
`compiled.inspect()` exposes the source contract, factors, expressions,
players and game dependency order. `analyze` supplies period-specific
states and evaluates the resulting games; there is no training stage.

## A complete independent example

See [`examples/author_models.py`](../examples/author_models.py). Its
`hourly_model` defines all four layers without calling the labor-cost
template. It expects these columns:

| Column | Meaning |
| --- | --- |
| employee_id, employment_type, period, team | Required source keys |
| grade, region | Categorical dimensions, explicitly mapped |
| hourly_rate | Positive currency/time input |
| paid_hours | Nonnegative time input |
| allowance | Signed actual row payment |
| actual | Optional independent actual payroll total |

The row formula is:

```python
field("rate") * field("hours") + field("allowance")
```

The factor declarations include:

```python
Factor("first_mix", "distribution", dimensions=("grade",))
Factor(
    "conditional_mix", "distribution",
    dimensions=("region",), given=("grade",), fallbacks=((),),
)
Factor(
    "members", "membership", dimensions=("grade", "region"),
    fallbacks=(("grade",), ("region",), ()),
)
Factor("rates", "aligned", source="rate")
Factor("hours", "aligned", source="hours")
```

These are P(grade), P(region|grade), the complete sparse membership
collection, and employee-aligned rate/hour vectors. Every factor is a
whole object. `given` conditions within the population and team; it does
not create a separate Shapley game for each conditioning category.

The aggregate unit-cost formula consumes the declared distributions:

```python
paired_pay = aligned_mean(value("rates") * value("hours"), "members")
unit_cost = sum_over(
    value("first_mix") * value("conditional_mix")
    * (paired_pay + value("allowances")),
    "grade", "region",
)
```

Products retain employee pairing before averaging. `allowances` is a
separately declared payment factor with the same cell dimensions. The
organizational formula is declared using `organization_game("unit")`:
N × sum(team_share × team_unit_cost). `local_game("unit")` declares the
separate local exposure/unit-economics game. `outer_factors()` returns
ordinary explicit factor declarations for both expressions.

The example supports three independent changes:

```python
hourly_model(grade_first=False)  # P(region) × P(grade|region)
hourly_model(grouped=True)      # One rate-and-hours atomic switch
hourly_model(nested=True)       # Parent allowance + nested base earnings
```

Each produces a different declared game with the same source grain.
These choices can change attribution even when endpoint costs agree.

## Mathematical previews

Jupyter is the supported user-facing interface. Install `.[notebook]`
from `production/` and select that environment's kernel. The
[authoring notebook](../examples/model_authoring.ipynb) runs without data.

Preview a formula as soon as it exists:

```python
from labor_cost import Notation, field, render_formula, render_model

pay = field("rate") * field("hours") + field("allowance")
notation = Notation(fields={"rate": "RT", "hours": "H", "allowance": "ALW"})
render_formula(pay, notation=notation)
```

The cell displays $\mathrm{RT}\cdot\mathrm{H}+\mathrm{ALW}$ and a symbol
legend. Use `display(...)` when the preview is not the last expression in
the cell. This does not evaluate payroll or require a complete model.

For a complete model, `render_model(model, notation=notation)` shows a
separate equation for each actual-row component and each aggregate game,
including organization and local costs. It compiles a `ModelSpec` using
the existing compiler; an already compiled model is also accepted.
Compilation does not establish business correctness or reconciliation.
V1 aggregate equations are labeled as compiler translations.

Pass `model=model` to `render_formula` to resolve dimensions, units,
conditional distributions and membership bases. A native v2 draft need
not compile for this expression preview. Unbound references retain
unspecified metadata; the preview is not a declaration validator.
Without context, `aligned_mean` appears as a named operator. With context,
it appears as a sum of membership weights, conversion factors, and the
**complete paired employee expression**. The index `i` is an aligned
employee/employment-type record, not an attribution player. For actual
basis, conversion uses inverse donor-row exposure; for per-exposure basis
it is one. Each membership collection retains its own weights/conversions.

Child costs stay named by default. To inspect one expanded expression:

```python
render_formula(
    model.graph.games[-1].formula,
    model=model,
    notation=notation,
    expand_children=True,
)
```

Expansion requires model context. It substitutes the declared child trees,
without polynomial expansion or independent averaging. Long additions and
products wrap at operator boundaries. Conditional dimensions and selections
stay explicit; the local equation evaluates team-axis sums for the selected
team only.
Switch mappings show which complete factors change together, separately
from the mathematical quantities.

### Symbol contract

`Notation` accepts `fields`, `factors`, `games`, `dimensions`, and
`populations` dictionaries keyed by their respective declaration IDs.
It is copied on construction and remains separate from the model.

- Defaults use uppercase initials for underscore-separated names, or the
  first three characters for a single word. These are mnemonic suggestions,
  not inferred business meanings.
- Generated collisions receive namespace superscripts (`fld`, `fac`,
  `cost`, `dim`, `pop`) and, if necessary, deterministic numeric suffixes.
  Multiletter symbols are upright, so `LC` is one quantity.
- Overrides can be longer than three letters. They are plain text, escaped
  for display; raw TeX and control characters are rejected. Explicit symbols
  must be unique across all five namespaces, with errors naming conflicts.
- All declarations in supplied model context participate in resolution.
  Reusing the same model and notation gives consistent symbols in fragments
  and the equation sheet. Without context, only expression references and
  configured overrides participate. Adding declarations can introduce new
  collisions; pin important symbols explicitly.
- The legend maps symbols to identifiers, source mappings, population,
  scope, dimensions, units and amount basis where known. `unspecified`
  means metadata is unavailable or does not apply; it is not an inferred
  unit or economic assumption.

Preview objects expose `.latex` (without dollar delimiters), `.markdown`
(equations and legend), `.legend` (a tuple of read-only metadata mappings),
and `.plain_text`. Jupyter uses Markdown/math through its MIME protocol.
There is no IPython, plotting or estimator import for rendering.

Notation is neither published nor exported in run bundles and never
changes a model fingerprint. Keep reusable overrides in notebook settings.

## Applicability and populations

```python
Population("agency", {"employment_type": "contractor"})
FieldSpec(
    "rate", "vendor_hourly_rate", population="agency",
    units="currency/time", constraint="positive",
)
RowCost("agency", field("rate") * field("hours"))
Factor("rates", "aligned", population="agency", source="rate")
```

The implicit `all` population includes all rows. Predicates are ANDed;
multiple categorical values within one predicate are ORed. Predicates
refer to mapped categorical dimensions, including employment type.
Inputs, state factors and games name their population explicitly.
Unrelated populations need no values in those input columns.

A `population_mix` factor declares a partition:

```python
Factor(
    "employment_mix", "population_mix", dimensions=("kind",),
    members=("employees", "agency"),
)
```

At analysis time the members must be disjoint and exhaustive within the
factor's population. Its coordinates are the declared population names.
Use `select(value("employment_mix"), "kind", "agency")` to weight a
child cost. Selection occurs after switching the entire vector.

Keep these names distinct:

| Name | Example | Referenced by |
| --- | --- | --- |
| Source category value | `temporary` in `employment_type` | Population predicate |
| Population ID | `agency` | Field/factor/game `population`, `RowCost`, partition `members`, partition `select` |
| Game ID | `temporary_cost_game` | `child(...)` and `Switch(child=...)` |
| Field ID | `rate` bound to source `vendor_hourly_rate` | `field(...)`; aligned, policy, mean and weighted-mean `source`; weighted-mean `weight` |
| Factor ID | `rates` | `value(...)` and factor switches |

For example, `Population("agency", {"employment_type": "temporary"})`
does not create a population named `temporary`. A partition containing
`agency` must be selected using `agency`, even if its child game is called
`temporary_cost_game`. A raw category mismatch leaves rows uncovered;
a stale partition member is invalid; a stale selection category fails
when that axis is evaluated. Do not solve these mismatches by replacing
unrelated game IDs or source column names.

Multiple row-cost components may overlap deliberately; their amounts
add. Every source row must have at least one applicable row formula.
Observed row totals, aggregate costs and independent source references
must reconcile. No hidden residual payment is inserted. Current runtime
checks do not independently establish every population's aggregate cost;
offsetting branch errors can pass team totals. This required guard remains
open as [GAP-001](architecture-gaps.md#gap-001-population-level-aggregate-reconciliation).
Use independent expected cases before publishing; successful compilation
and recorded validation alone do not prove a chosen model is correct.

## State primitives and amount bases

| Kind | Value and use |
| --- | --- |
| total | Sum of exposure, scalar or by team |
| distribution | Exposure-weighted probability table over dimensions, conditional on `given` |
| population_mix | Complete exposure-share vector for an explicit partition |
| aligned | Observed input vector, matched by employee/employment type |
| membership | Sparse normalized employee profiles and positive conversion factors |
| payment | Sum of actual row amounts / exposure, or exposure-weighted mean of per-exposure row amounts |
| mean | Quantity per exposure from a declared source field; see basis rules below |
| weighted_mean | Workload-weighted rate from declared source and weight fields, with configurable zero-weight handling (`error` by default) |
| policy | Fixed table from a declared field, consistent across both periods |

Workforce distributions use exposure, not salary or worked time. Default
exposure is 1/k across a person's k observed employment types. A selected
exposure column must contain values in (0, 1]; totals are not rescaled.

`basis="actual"` is the default for payment and membership factors.
Payments divide actual amounts by exposure before weighted aggregation.
Membership carries 1/exposure factors; aligned inputs stay observed.
`basis="per_exposure"` uses supplied normalized values and membership
conversion factors of one. The analyst must declare the basis correctly;
endpoint reconciliation detects discrepancies at the checked boundaries.
It cannot establish that every component has the correct basis: offsetting
population errors can still pass (see [GAP-001](architecture-gaps.md#gap-001-population-level-aggregate-reconciliation)).

Aligned inputs match across the full organization within their declared
population. Team, grade and region transfers retain identity. A missing
employee/type observation carries that person's observed value from the
other period. This does not create workforce records or exposure.

Membership measures changes in participating employee profiles and weights,
not an additional payroll input. Its effect applies to the whole expression
inside `aligned_mean`. If that expression includes bonus, membership affects
bonus as well; the built-in template's base-pay-only label is not universal.
Aligned rate/hour drivers explain changes to matched inputs under the carry
assumptions. They do not include every change in population-average rates
or hours caused by entry/exit. Keep this declared intervention semantics;
redistribution of membership impacts would require a new methodology.

Fixed policies are not attribution players. The labor template preserves
its fixed regional-coefficient assumption and applies adjustment once.
Its default regions are coefficient bands, not identified geography.

## Aggregate formulas and games

Supported arithmetic is finite constants, addition, subtraction and
multiplication. Row units include currency, time, exposure, currency/time,
currency/exposure, time/exposure, 1/exposure and dimensionless values.
Aggregate expressions also
track exposure units and named categorical axes.

`sum_over` explicitly reduces axes; `select` chooses a category;
`aligned_mean` consumes paired employee expressions and one complete
membership collection. Arbitrary division, thresholds, caps, callbacks
and nonlinear nested child costs require engine extensions.

A `Switch(name, factors=(...))` groups complete state objects. A
`Switch(name, child="game_id")` exposes a child cost and requests its
nested explanation. Each changing factor/child has exactly one owner in
a game. Shared atomic state across separate child boundaries is rejected.

Driver IDs default to `game_name::switch_name`. An explicit `driver_id`
can provide a stable reporting identity. Labels are presentation only;
driver summaries do not merge rows merely because labels match.

The root must remain homogeneous and affine in its team-cost family.
Local and organizational team endpoints are checked against each other
and the actual source totals. For example, analysts can replace N and
team-share attribution with direct team-exposure attribution:

```python
organization = Game(
    "organization",
    sum_over(value("team_exposure") * child("unit"), "team"),
    (
        Switch("team_exposures", ("team_exposure",)),
        Switch("team_economics", child="unit"),
    ),
)
```

Every child boundary must be affine in that scalar child cost. Exact
auxiliary zero-to-one games supply propagation multipliers recursively.
Zero-change parents may have offsetting nonzero children; the engine
never divides by a parent's cost change.

Leaf games use the requested estimator. Upper and multiplier games stay
exact. Exact games allow at most 15 active players. Organizational
additive terms are evaluated separately with whole global factors intact.
No larger leaf silently switches estimator or bypasses the safeguard.
Permutation defaults are 256 forward/reverse cycles and seed 0. Audits
record actual samples, budgets, seeds and runtime. Reconciliation is not
an estimate of sampling accuracy.

## Missing support and explicit references

Sparse factors declare ordered `fallbacks`, each a subset of their cell
or conditioning dimensions. `()` means the same team's applicable
population in that period. No fallback pools another team or period.
Empty fallback lists fail if a needed cell lacks support.

Missing populations default to copying the same team's opposite-period
complete economics. Whole-team absence copies unit economics while
retaining zero exposure. `require_reference` disables automatic economic
copying. Inactive branches require no fabricated economics.

References use `(period, team)` keys. Values map population IDs to
complete factor dictionaries, or contain `team_state` for an absent
whole team. Typed reference states can be taken from returned states and
modified with `Tensor`/`Profiles` from `labor_cost.graph_cost`. They must
preserve named axes, aligned employee/type identities, applicable constraints
and complete factor schemas. `Tensor` / `Profiles` are currently imported
from `labor_cost.graph_cost`, not top-level exports. Do not pass the legacy
`analyze_labor_cost` reimbursement scalars or `_employees` dictionaries to
the graph API. A standalone one-period state may have a different employee
axis from a comparison; matching axes must be established explicitly.
Explicit references only replace missing economics. Model/run manifests
encode their values without pickle or executable code.

## Validate, publish, resolve and analyze

Run the complete lifecycle with your own Parquet file:

```bash
python examples/model_lifecycle.py payroll.parquet new-catalog \
  --period-a 2026-08 --period-b 2026-09 --source-cost-column actual
```

The script authors the hourly model, compiles and inspects it, validates
it on supplied data, publishes its JSON, registers model and dataset
versions in `catalog.json`, reloads both through `Catalog`, and exports a
business run. A catalog is a filesystem JSON index whose paths are
relative to its location. It can reside in a local or mounted directory;
there is no registry service or automatic global catalog.

Publication requires a passing analysis of the exact model fingerprint.
Also test known expected business cases: arithmetic reconciliation alone
does not validate a chosen business interpretation.

For observed-only costing:

```python
state = build_state(model, dataset, "2026-08", source_cost_column="actual")
total = cost(state, model)
```

No comparison month or Shapley evaluation is used. `cost` itself performs
no historical completion or input acquisition.

## Compatibility

`labor_cost_model()` retains the v1 declaration and content fingerprint.
`labor_cost_model(schema_version=2)` returns the native version 2.0.0
template. Both compile into the same graph runtime used by new models.
Legacy `InputSpec`/`WorkforceStructure` authoring keeps its previous
validation restrictions; use v2 for different populations or nesting.
Legacy public tables retain their schema through a rendering adapter.
The original dictionary compatibility entry point remains available through
the extracted legacy engine. Its enriched diagnostics are not all emitted
by the graph runtime. Native v2 and v1 graph `team_summary` schemas differ;
see the exact [bundle/table contract](run-bundles.md).

V1 catalog entries and run bundles remain readable. New bundles use
schema 2 and include the compiled declaration graph. Model publication
refuses to overwrite an existing destination file; it does not enforce
global version uniqueness across different files or catalogs. Catalogs
resolve explicitly declared IDs and versions. Deployed execution never
loads the PoC notebook.

## Effective rate and worked time (engine 0.3.0+)

V2 models can replace an employee-aligned base-pay component with effective
rate × worked time per exposure. This changes the attribution definition;
it does not change actual row payroll. Existing aligned models and grouped
switches remain supported. The new types are `mean` and `weighted_mean`:

```python
cell = dict(
    population="permanent",
    dimensions=("coefficient_band", "grade"),
    fallbacks=(("grade",), ("coefficient_band",), ()),
)
hours = Factor(
    "hours_per_exposure", "mean", source="permanent_hours",
    basis="actual", **cell,
)
rate = Factor(
    "effective_rate", "weighted_mean", source="permanent_rate",
    weight="permanent_hours", basis="actual",
    zero_weight="exposure_mean", **cell,
)
base_pay = value("effective_rate") * value("hours_per_exposure")
switches = (
    Switch("effective_rate", label="Effective hourly rate"),
    Switch("hours_per_exposure", label="Hours per exposure"),
)
```

`source` and `weight` name declared fields, not source columns or other
factors. New aggregates accept a source field, not a row formula or
conditional `given` axes. Weight options apply only to `weighted_mean`.
Weight fields must declare `positive` or `nonnegative` constraints.
Use the same population, cell dimensions, basis and fallback order for
the rate/time pair. Multiply `base_pay` by regional policy and composition
shares in the aggregate formula, and reduce both cell axes as usual.
Remove the old rate/time/membership factors and switches for this component.
Unrelated components can retain their own membership factors.

For selected donor rows, let E = sum(e_i), with e_i the assigned exposure:

| Kind | Actual quantity basis | Per-exposure quantity basis |
| --- | --- | --- |
| `mean`, source h | sum(h_i) / E | sum(e_i h_i) / E |
| `weighted_mean`, source r and weight h | sum(h_i r_i) / sum(h_i) | sum(e_i h_i r_i) / sum(e_i h_i) |

The basis applies to the quantity/weight, never to the rate. For actual
hours use `time`; for normalized hours use `time/exposure`. Monthly salary
uses a dimensionless worked fraction, or `1/exposure` if already normalized.
`mean` with actual basis divides the source units by exposure; per-exposure
basis preserves the source units. `weighted_mean` preserves rate units.
Source positivity/nonnegativity constraints also apply to the resulting
state factors, including explicit reference states.

For hourly pay, E × hours-per-exposure × effective-rate equals the sum of
paired actual rate × hours. The rate is weighted by hours: moving hours
between employees can change it even when every contractual rate stays
fixed. It is not an ordinary average contractual rate or a pure raise
measure. Monthly salary × worked fraction follows the same identity with
different units and labels. Exposure is not automatically headcount or FTE.

A populated cell with zero hours has no observed effective rate. Explicit
`zero_weight="exposure_mean"` uses sum(e_i r_i) / E from that same donor
pool. Hours and observed payroll remain zero. This assumption can affect
counterfactuals and is recorded by cell in `assumptions` and
`diagnostics/aggregate_factors`. The default `zero_weight="error"` fails
with context instead. Strictly positive weights, however small, do not
activate the zero-weight policy. Negative weights are invalid.

Sparse fallback chooses a nonempty same-team/period donor pool before
aggregation. A populated zero-hour cell does not search other cells for
positive hours. Whole-population/team completion and explicit references
keep their existing semantics. References for the new factors are typed
cell tensors; unknown underlying numerator/support statistics are not
invented. Standalone `build_state` and `cost` need no comparison history
for the declared zero-hour policy.

Use [model_comparison.ipynb](../examples/model_comparison.ipynb) and its
editable [comparison_models.py](../examples/comparison_models.py) to compare
aligned, grouped rate/membership, and effective-rate variants on the same
catalog dataset. The example follows the permanent/temporary walkthrough,
including regional policy and conditional grade composition. It changes
only permanent base pay; adapt other rate × time components explicitly.
Models retain separate fingerprints and run bundles. Driver tables remain
separate because their definitions differ; absence is not zero impact.

## Explicit input preparation

`prepare_input(model, dataset, request, *, missing_team=None)` accepts a
specification or compiled v1/v2 model and a raw `DatasetSnapshot`. It applies
declared input constraints before state construction and returns
`PreparedInput`, a snapshot with audit evidence. Ordinary analysis remains
strict; this preprocessing is an explicit choice.

Preparation always prints counts/reasons and raises `InputPreparationError`
if more than 1% of rows in the two selected months would be excluded. Its
`.audit` remains available on failure; successful snapshots carry the same
evidence automatically into analysis and run bundles. Exactly 1% is allowed,
multiple defects count once per row, and team replacements are not drops.
Declare `missing_team="Unassigned"` to retain unknown teams under that label.
Input preparation changes the analyzed workforce; default exposure is
assigned after exclusions. See the [production workflow](../examples/GUIDE.MD#14-prepare-the-production-csv-or-dwh-input-explicitly)
for code, policies and exception handling.


## Organizational hierarchy selection

Enable the built-in template with
`labor_cost_model(schema_version=2, line_source="business_line")`.
For a custom graph, set
`hierarchy=OrganizationalHierarchy("business_line", orders=..., default=...)`
on `ModelGraph`. The four supported orders are `("team",)`, `("line",)`,
`("team", "line")` and `("line", "team")`. The default is team alone.
Use the standard outer factors and organization/local games for this
extension. Existing team-scoped payroll games become the economics at
each selected leaf.

Pass the order as `AnalysisRequest(..., hierarchy=order)` or
`build_state(model, dataset, month, hierarchy=order)`. Only orders that
use line require its source column. Conflicting line assignments for an
employee-month fail strict analysis and audited preparation. Preparing a
snapshot binds it to the effective order. The manifest records that
choice alongside the model's allowed orders. Inspect the selection before
running via `compile_model(model).inspect(hierarchy=order)`.

Node paths are tuples of ordered dimension/value pairs, for example
`(("line", "Retail"), ("team", "Alpha"))`. Missing-leaf references use
`(period, path)` keys and either population-to-factor mappings or
`{"node_state": complete_leaf_state}`. A missing parent mix takes
`{"mix": {child_label: share, ...}}` under the parent path (the root is
`()`). Supply the full normalized vector over its observed child axis.
`require_reference` requires explicit values for one-sided missing
economics and mixes. References are in the selected hierarchy's order.

[Run the worked comparison](../examples/organizational_hierarchy.ipynb).
