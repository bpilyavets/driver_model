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
| Field ID | `rate` bound to source `vendor_hourly_rate` | `field(...)` and aligned factor `source` |
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
| payment | Exposure-weighted cell mean of a declared row amount |
| policy | Fixed table from a declared field, consistent across both periods |

Workforce distributions use exposure, not salary or worked time. Default
exposure is 1/k across a person's k observed employment types. A selected
exposure column must contain values in (0, 1]; totals are not rescaled.

`basis="actual"` is the default for payment and membership factors.
Payments divide actual amounts by exposure before weighted aggregation.
Membership carries 1/exposure factors; aligned inputs stay observed.
`basis="per_exposure"` uses supplied normalized values and membership
conversion factors of one. The analyst must declare the basis correctly;
endpoint reconciliation detects inconsistency with the row formula.

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
currency/exposure and dimensionless values. Aggregate expressions also
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
never overwrites an existing version, and deployed execution never loads
the PoC notebook.
