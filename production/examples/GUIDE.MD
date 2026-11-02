# Declare and analyze a permanent/temporary payroll model

This walkthrough uses two payroll formulas. The code blocks can be run in
order in one Python script or notebook. The model was checked against the
library: the example's payroll increases from **6,575 to 6,982**, and its
Shapley contributions reconcile to **407**.

We do not need a catalog to define and test the model. Publication comes
afterward.

## 1. Decide what the model should explain

The row formulas are:

```text
Permanent pay = rate × hours + bonus

Temporary pay = regular hours × regular rate
              + overtime hours × overtime rate
```

For this example, regular hours exclude overtime, and the overtime rate is
the full payment per overtime hour.

We will declare the following aggregation:

- Permanent workers: grade composition, employee-paired rate × hours
  within each grade, and average bonus within each grade.
- Temporary workers: employee-paired regular and overtime earnings across
  the temporary population, without a grade breakdown.
- Each team: permanent/temporary composition multiplied by the
  corresponding unit costs.
- Organization: workforce exposure, team composition, and team unit costs.

The example has one team, so team composition cannot change. The same
declaration works with multiple teams.

**One additional distinction is necessary:** grade composition tells us
how much of the permanent workforce belongs to each grade. Membership
within grade tells us which employees supply that grade's paired rates
and hours. These are separate state objects and, here, separate drivers.

## 2. Import the library

Use a Python environment with the package installed. For a local
installation:

```bash
python -m pip install -e /home/rupilbo/driver_model/production
```

Then:

```python
import numpy as np
import pandas as pd

from labor_cost import (
    AnalysisRequest,
    DatasetSnapshot,
    Factor,
    FieldSpec,
    Game,
    ModelGraph,
    ModelSpec,
    Population,
    RowCost,
    SourceSchema,
    Switch,
    aligned_mean,
    analyze,
    build_state,
    child,
    compile_model,
    cost,
    field,
    local_game,
    organization_game,
    outer_factors,
    select,
    sum_over,
    value,
)
```

The main declarations have distinct jobs:

| Declaration | What you decide |
| --- | --- |
| `Population` | Which rows belong to a worker population |
| `FieldSpec` | Which source inputs apply to that population |
| `RowCost` | How to calculate actual pay for one row |
| `Factor` | What aggregated or employee-aligned state to construct |
| Aggregate expression | How those factors produce population/team cost |
| `Switch` | Which complete objects change together in a counterfactual |
| `Game` | Which switches participate in a particular explanation |

## 3. Declare the source schema and the two populations

Every source row remains one employee × employment type × month record.
We use the library's default key column names:

```python
source = SourceSchema(
    dimensions={"grade": "grade"},
)

populations = (
    Population(
        "permanent",
        {"employment_type": "permanent"},
    ),
    Population(
        "temporary",
        {"employment_type": "temporary"},
    ),
)
```

The default source keys are `employee_id`, `employment_type`, `period`,
and `team`.

In `dimensions={"grade": "grade"}`, the left-hand name is the model
dimension; the right-hand name is its source column. If your source calls
it `job_grade`, write:

```python
# Alternative mapping:
# SourceSchema(dimensions={"grade": "job_grade"})
```

Declaring `grade` here only makes it available. **It does not yet create a
grade-composition driver.** We do that with a `Factor` below.

## 4. Declare the applicable inputs and row formulas

First, bind the payroll fields:

```python
fields = (
    FieldSpec(
        "permanent_rate", "permanent_rate",
        population="permanent",
        units="currency/time", constraint="positive",
    ),
    FieldSpec(
        "permanent_hours", "permanent_hours",
        population="permanent",
        units="time", constraint="nonnegative",
    ),
    FieldSpec(
        "permanent_bonus", "permanent_bonus",
        population="permanent",
        units="currency", constraint="signed",
    ),
    FieldSpec(
        "temporary_hours", "temporary_hours",
        population="temporary",
        units="time", constraint="nonnegative",
    ),
    FieldSpec(
        "temporary_rate", "temporary_rate",
        population="temporary",
        units="currency/time", constraint="positive",
    ),
    FieldSpec(
        "overtime_hours", "overtime_hours",
        population="temporary",
        units="time", constraint="nonnegative",
    ),
    FieldSpec(
        "overtime_rate", "overtime_rate",
        population="temporary",
        units="currency/time", constraint="positive",
    ),
)
```

Each `FieldSpec` starts with the model field name and source column name.
We have deliberately kept them identical.

**`population="temporary"` is how you tell the compiler that an input
applies only to temporary workers.** Permanent rows can leave temporary
inputs null, and vice versa. Applicable inputs must satisfy their
declared constraints.

These constraints are modeling choices: this example requires positive
hourly rates, permits zero hours, and permits negative bonus corrections.
A temporary row still needs its overtime rate when overtime hours are
zero.

Now declare exactly the two formulas:

```python
row_costs = (
    RowCost(
        "permanent",
        field("permanent_rate") * field("permanent_hours")
        + field("permanent_bonus"),
    ),
    RowCost(
        "temporary",
        field("temporary_hours") * field("temporary_rate")
        + field("overtime_hours") * field("overtime_rate"),
    ),
)
```

`field(...)` refers to an input on an individual source row. At this
point, we have defined payroll arithmetic, but we have not defined the
aggregate explanation.

## 5. Declare the factors the engine should construct

This is where you explicitly request grade composition, employment
composition, employee-aligned inputs, and bonus aggregation:

```python
factors = outer_factors() + (
    # Permanent/temporary shares within each team.
    Factor(
        "employment_mix",
        "population_mix",
        dimensions=("employment_kind",),
        members=("permanent", "temporary"),
    ),

    # Grade shares within each team's permanent population.
    Factor(
        "grade_mix",
        "distribution",
        population="permanent",
        dimensions=("grade",),
    ),

    # Which permanent employees contribute within each grade.
    Factor(
        "permanent_members",
        "membership",
        population="permanent",
        dimensions=("grade",),
        fallbacks=((),),
    ),

    # Preserve permanent rate/hour pairing by employee.
    Factor(
        "permanent_rate",
        "aligned",
        population="permanent",
        source="permanent_rate",
    ),
    Factor(
        "permanent_hours",
        "aligned",
        population="permanent",
        source="permanent_hours",
    ),

    # Average bonus economics within each grade.
    Factor(
        "permanent_bonus",
        "payment",
        population="permanent",
        dimensions=("grade",),
        formula=field("permanent_bonus"),
        fallbacks=((),),
    ),

    # Temporary workers have no categorical breakdown.
    Factor(
        "temporary_members",
        "membership",
        population="temporary",
    ),

    # Preserve all four temporary inputs by employee.
    Factor(
        "temporary_hours",
        "aligned",
        population="temporary",
        source="temporary_hours",
    ),
    Factor(
        "temporary_rate",
        "aligned",
        population="temporary",
        source="temporary_rate",
    ),
    Factor(
        "overtime_hours",
        "aligned",
        population="temporary",
        source="overtime_hours",
    ),
    Factor(
        "overtime_rate",
        "aligned",
        population="temporary",
        source="overtime_rate",
    ),
)
```

Several details matter here.

**`grade_mix` is the grade-composition declaration.** It constructs the
complete grade-share vector within each team's permanent population. For
example:

```text
August:    G1 = 2/3, G2 = 1/3
September: G1 = 1/3, G2 = 2/3
```

**`employment_mix` is the permanent/temporary composition declaration.**
Its `employment_kind` axis is created from the two named populations; it
does not require another source column.

**`aligned` preserves employee pairing.** The engine retains each
employee's rate and hours so it can evaluate their product before
averaging. It does not replace the average of `rate × hours` with
`average rate × average hours`.

**`membership` supplies the averaging population and weights.** Permanent
membership is organized by grade. Temporary membership has no dimensions,
so it covers the temporary population within the team.

**`payment` makes a deliberate choice about bonuses:** the driver
represents average bonus economics within each grade. Individual bonus
changes are not separately aligned as employee inputs in this declaration.

`fallbacks=((),)` means: if a grade has no observations in a particular
team and period, use that same team's permanent population in that period
as support. This is an explicit assumption for otherwise unsupported
counterfactual cells. The example below has both grades in both periods,
so it does not need that fallback.

Finally, `outer_factors()` supplies three standard declarations:

| Factor | Meaning |
| --- | --- |
| `workforce_scale` | Total organizational exposure |
| `team_mix` | Complete vector of team exposure shares |
| `team_exposure` | Exposure by team, used for local bridges |

With one employment-type record per employee-month, exposure is one per
row. If an employee has multiple employment types in the same month, the
default allocation is `1/k`; the engine performs the corresponding
payment normalization.

## 6. Write the aggregate cost formulas

For permanent workers, the intended formula is:

```text
Permanent unit cost = sum over grades of:
    grade share × (
        paired mean of rate × hours within grade
        + mean bonus within grade
    )
```

Declare it as:

```python
permanent_unit_cost = sum_over(
    value("grade_mix")
    * (
        aligned_mean(
            value("permanent_rate") * value("permanent_hours"),
            "permanent_members",
        )
        + value("permanent_bonus")
    ),
    "grade",
)
```

For temporary workers:

```python
temporary_unit_cost = aligned_mean(
    value("temporary_hours") * value("temporary_rate")
    + value("overtime_hours") * value("overtime_rate"),
    "temporary_members",
)
```

These expressions return cost per unit of workforce exposure.

The distinction between the expression functions is:

| Expression | Refers to |
| --- | --- |
| `field("permanent_rate")` | Rate on one source row |
| `value("permanent_rate")` | The constructed employee-aligned rate vector |
| `aligned_mean(...)` | Employee-paired arithmetic aggregated using membership |
| `sum_over(..., "grade")` | Summation across the grade axis |
| `child("permanent")` | The scalar cost produced by the permanent game |

The row formula and aggregate formula must reproduce the same observed
payroll. The analysis checks this; it does not assume the declaration is
correct.

## 7. Declare the Shapley games and their switches

The permanent game has five players:

```python
permanent_game = Game(
    name="permanent",
    population="permanent",
    formula=permanent_unit_cost,
    switches=(
        Switch(
            "grade_mix",
            label="Permanent grade composition",
        ),
        Switch(
            "permanent_members",
            label="Workforce mix within grade",
        ),
        Switch(
            "permanent_rate",
            label="Permanent hourly rates",
        ),
        Switch(
            "permanent_hours",
            label="Permanent worked hours",
        ),
        Switch(
            "permanent_bonus",
            label="Permanent bonuses",
        ),
    ),
)
```

`Switch("grade_mix")` means "replace the entire period-A grade-share
vector with the period-B vector." It does not create one player for each
grade.

Similarly, the rate switch replaces the complete aligned rate vector.
Employees themselves are not Shapley players.

The temporary game also has five players:

```python
temporary_game = Game(
    name="temporary",
    population="temporary",
    formula=temporary_unit_cost,
    switches=(
        Switch(
            "temporary_members",
            label="Temporary workforce mix",
        ),
        Switch(
            "temporary_hours",
            label="Temporary regular hours",
        ),
        Switch(
            "temporary_rate",
            label="Temporary regular rates",
        ),
        Switch(
            "overtime_hours",
            label="Temporary overtime hours",
        ),
        Switch(
            "overtime_rate",
            label="Temporary overtime rates",
        ),
    ),
)
```

Now connect both populations through a team game:

```python
mix = value("employment_mix")

team_game = Game(
    name="team",
    formula=(
        select(mix, "employment_kind", "permanent")
        * child("permanent")
        + select(mix, "employment_kind", "temporary")
        * child("temporary")
    ),
    switches=(
        Switch(
            "employment_mix",
            label="Permanent/temporary composition",
        ),
        Switch(
            "permanent_cost",
            child="permanent",
            label="Permanent unit cost",
        ),
        Switch(
            "temporary_cost",
            child="temporary",
            label="Temporary unit cost",
        ),
    ),
)
```

This expresses:

```text
Team unit cost = permanent share × permanent unit cost
               + temporary share × temporary unit cost
```

`select(...)` picks a population's share from the complete vector. The
vector still switches atomically.

The two `child=` switches instruct the engine to explain those parent
contributions using the corresponding child games. You do not calculate
propagation multipliers yourself.

## 8. Assemble and compile the model

```python
model = ModelSpec(
    model_id="simple-payroll",
    version="1.0.0",
    graph=ModelGraph(
        source=source,
        populations=populations,
        fields=fields,
        row_costs=row_costs,
        factors=factors,
        games=(
            permanent_game,
            temporary_game,
            team_game,
        ),
        team_game="team",
        organization=organization_game("team"),
        local=local_game("team"),
        missing_economics="carry_observed",
    ),
)

compiled = compile_model(model)
```

`organization_game("team")` supplies:

```text
Organization cost = total exposure × sum over teams of:
    team share × team unit cost
```

`local_game("team")` supplies a separate local exposure/unit-cost
explanation for each team.

`missing_economics="carry_observed"` means a population absent in one
comparison period uses the same team's observed opposite-period
economics, with the assumption recorded. It does not manufacture workers
or copy their exposure.

Inspect the compiled structure:

```python
description = compiled.inspect()

print(description["game_order"])

for game in description["games"]:
    print(
        game["id"],
        game["population"],
        game["switches"],
    )
```

**Compilation validates the declarations. It does not run Shapley
attribution or train anything.** The actual games are evaluated later,
when you provide the two periods' data.

### Inspect the mathematics before loading data

Keep descriptive field/factor IDs and choose short display symbols in the
notebook. These settings do not change the model's identity:

```python
from labor_cost import Notation, render_formula, render_model

notation = Notation(games={"organization": "LC"})
render_formula(row_costs[0].formula, model=model, notation=notation)
```

Run this in another cell for the complete equation sheet:

```python
preview = render_model(compiled, notation=notation)
preview
```

The sheet separates row components, population/team games, organizational
cost and local cost. It shows conditional distributions, paired employee
products and their membership conversions, plus switch mappings and a
symbol legend. Compilation and preview require no dataset or attribution.

Override `fields`, `factors`, `dimensions`, or `populations` in `Notation`
as needed; three-letter defaults are suggestions, not a hard limit. The
same explicit symbol cannot identify two distinct declarations. For a
partial expression, `render_formula(expression)` also works before the
model is complete; unspecified metadata remains visibly unspecified.
See the [notation contract](../docs/model-authoring.md#mathematical-previews)
and [runnable authoring notebook](model_authoring.ipynb).

## 9. Prepare a small dataset with independently specified payroll totals

This fixture has three permanent workers and one temporary worker in each
month. Employee `p2` moves from G1 to G2.

```python
permanent_data = pd.DataFrame(
    [
        ("2026-08-01", "p1", "G1", 10., 100., 50., 1050.),
        ("2026-08-01", "p2", "G1", 20., 100., 100., 2100.),
        ("2026-08-01", "p3", "G2", 25., 80., 0., 2000.),
        ("2026-09-01", "p1", "G1", 12., 110., 60., 1380.),
        ("2026-09-01", "p2", "G2", 21., 90., 80., 1970.),
        ("2026-09-01", "p3", "G2", 25., 80., 0., 2000.),
    ],
    columns=[
        "period",
        "employee_id",
        "grade",
        "permanent_rate",
        "permanent_hours",
        "permanent_bonus",
        "actual_cost",
    ],
).assign(
    employment_type="permanent",
    team="Operations",
)

temporary_data = pd.DataFrame(
    [
        ("2026-08-01", "t1", 80., 15., 10., 22.5, 1425.),
        ("2026-09-01", "t1", 90., 16., 8., 24., 1632.),
    ],
    columns=[
        "period",
        "employee_id",
        "temporary_hours",
        "temporary_rate",
        "overtime_hours",
        "overtime_rate",
        "actual_cost",
    ],
).assign(
    employment_type="temporary",
    team="Operations",
)

frame = pd.concat(
    [permanent_data, temporary_data],
    ignore_index=True,
)

snapshot = DatasetSnapshot(
    dataset_id="simple-payroll-data",
    version="1",
    frame=frame,
)
```

Concatenation leaves permanent-only fields null on temporary rows and
temporary-only fields null on permanent rows. That is valid because we
declared their applicability.

Temporary rows also have no grade. Their factors never request one.

`actual_cost` is a separate reconciliation reference. For example, the
temporary August row is:

```text
80 × 15 + 10 × 22.5 = 1425
```

The expected totals are:

| Population | August | September | Change |
| --- | ---: | ---: | ---: |
| Permanent | 5,150 | 5,350 | +200 |
| Temporary | 1,425 | 1,632 | +207 |
| **Total** | **6,575** | **6,982** | **+407** |

## 10. Check standalone costing, then run attribution

You can first verify each month without running Shapley:

```python
for period in ("2026-08", "2026-09"):
    state = build_state(
        compiled,
        snapshot,
        period,
        source_cost_column="actual_cost",
    )
    print(period, cost(state, compiled))
```

Expected output:

```text
2026-08 6575.0
2026-09 6982.0
```

Now request the comparison:

```python
request = AnalysisRequest(
    period_a="2026-08",
    period_b="2026-09",
    source_cost_column="actual_cost",
    explainer="exact",
)

result = analyze(compiled, snapshot, request)
```

This is when the engine constructs both endpoint states, evaluates
counterfactuals, runs the declared Shapley games, propagates the nested
contributions, and assembles the results.

Exact attribution is suitable here: each leaf game has only five players.
To request permutation explicitly:

```python
permutation_request = AnalysisRequest(
    period_a="2026-08",
    period_b="2026-09",
    source_cost_column="actual_cost",
    explainer="permutation",
    n_permutations=256,
    random_seed=0,
)
```

Pass `permutation_request` instead of `request` to `analyze` to run that
estimator.

## 11. Read the results and check their interpretation

Start with the validation and organizational driver bridge:

```python
validation = result.table("validation")
assert validation["passed"].all()

bridge = result.table("driver_bridge")

print(bridge[["label", "impact"]].round(6))

np.testing.assert_allclose(
    bridge["impact"].sum(),
    407.0,
    rtol=1e-10,
    atol=1e-6,
)
```

The nonzero impacts for this fixture are approximately:

| Driver | Impact |
| --- | ---: |
| Permanent grade composition | +562.500000 |
| Workforce mix within grade | −655.000000 |
| Permanent hourly rates | +306.666667 |
| Permanent worked hours | −51.666667 |
| Permanent bonuses | +37.500000 |
| Temporary regular hours | +155.000000 |
| Temporary regular rates | +85.000000 |
| Temporary overtime hours | −46.500000 |
| Temporary overtime rates | +13.500000 |
| **Total, before display rounding** | **+407.000000** |

Workforce scale, team mix, permanent/temporary composition, and temporary
membership are unchanged in this fixture. Their impacts are zero within
floating-point tolerance.

The large, offsetting grade and within-grade membership effects deserve
attention. Moving `p2` changes both the grade shares and which employees
supply each grade's rate/hour economics. These are different
counterfactual switches, so they can have opposing contributions. Neither
contribution is a causal estimate of the effect of promotion.

Also, **"Permanent bonuses" means changes in grade-level bonus economics
in this model.** Because we chose a `payment` factor, it is not restricted
to same-person bonus changes. Grade composition also affects how those
grade-level bonus averages enter total cost.

Other useful tables are:

```python
# Full nested explanation, including parent nodes.
hierarchy = result.table("hierarchy")

# Team endpoints and total change.
team_summary = result.table("team_summary")

# Separate local bridge for Operations.
local_bridge = result.table("local_bridges/Operations")

# Completion assumptions and any sparse-cell fallbacks.
assumptions = result.table("assumptions")
fallbacks = result.table("diagnostics/fallbacks")

# Actual estimator methods, budgets and evaluation counts.
estimator_audit = result.table("coalition_audit")
```

Do not add hierarchy parents to their children, or combine organizational
contributions with local bridge contributions: those would double-count
different representations of the explanation.

## 12. Publish only after the declaration behaves as intended

Everything above works without a catalog. Once the model passes
reconciliation and your business-case checks, you can publish its
declaration:

```python
from pathlib import Path
from labor_cost import export_run, publish_model

publication_dir = Path(
    "/home/rupilbo/driver_model/production/"
    "runs/simple-payroll-1.0.0"
)

publish_model(
    compiled,
    publication_dir / "model.json",
    validated_result=result,
)

export_run(
    result,
    publication_dir / "validation-run",
)
```

These destinations must be new; publication does not overwrite an
existing model file.

A catalog is then a JSON index pointing to this published model and an
approved dataset snapshot. Business clients resolve those entries and
call `analyze`; they do not repeat the analyst's declarations. The existing
[catalog lifecycle example](model_lifecycle.py) shows that registration
step.

The places you edit when designing this model are explicit: **`RowCost`
controls individual payroll; `Factor` controls distributions and
representations; aggregate expressions control how they produce cost; and
`Switch`/`Game` control the explanation.** Changing one layer does not
automatically infer your intentions for the others.
