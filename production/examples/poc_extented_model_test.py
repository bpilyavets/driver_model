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

source = SourceSchema(
    dimensions={
        "grade": "grade",
        "reg_coeff": "reg_coeff",
        },
)

populations = (
    Population(
        "permanent",
        {"employment_type": "permanent"},
    ),
    Population(
        "gpd",
        {"employment_type": "gpd"},
    ),
)

fields = (
    FieldSpec(
        "permanent_rate", "stavka_oklad",
        population="permanent",
        units="currency/time", constraint="nonnegative",
    ),
    FieldSpec(
        "permanent_hours", "vyrabotka_act",
        population="permanent",
        units="time", constraint="nonnegative",
    ),
    FieldSpec(
        "permanent_bonus", "bonus_kpi",
        population="permanent",
        units="currency", constraint="signed",
    ),
    FieldSpec(
        "temporary_hours", "temporary_hours",
        population="gpd",
        units="time", constraint="nonnegative",
    ),
    FieldSpec(
        "temporary_rate", "temporary_rate",
        population="gpd",
        units="currency/time", constraint="positive",
    ),
    FieldSpec(
        "overtime_hours", "overtime_hours",
        population="gpd",
        units="time", constraint="nonnegative",
    ),
    FieldSpec(
        "overtime_rate", "overtime_rate",
        population="gpd",
        units="currency/time", constraint="positive",
    ),
)

row_costs = (
    RowCost(
        "permanent",
        field("permanent_rate") * field("permanent_hours")
        + field("permanent_bonus"),
    ),
    RowCost(
        "gpd",
        field("temporary_hours") * field("temporary_rate")
        + field("overtime_hours") * field("overtime_rate"),
    ),
)

factors = outer_factors() + (
    # Permanent/temporary shares within each team.
    Factor(
        "employment_mix",
        "population_mix",
        dimensions=("employment_kind",),
        members=("permanent", "gpd"),
    ),

    # Grade shares within each team's permanent population.
    Factor(
        "grade_mix",
        "distribution",
        population="permanent",
        dimensions=("grade",),
    ),

    Factor(
        "region_within_grade_mix",
        "distribution",
        population="permanent",
        dimensions=("reg_coeff",),
        given=("grade",),
        fallbacks=(()),
    ),

    # Which permanent employees contribute within each cell.
    Factor(
        "permanent_members",
        "membership",
        population="permanent",
        dimensions=("grade", "reg_coeff"),
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
    Factor(
        "permanent_bonus",
        "aligned",
        population="permanent",
        source="permanent_bonus",
    ),

    # Temporary workers have no categorical breakdown.
    Factor(
        "temporary_members",
        "membership",
        population="gpd",
    ),

    # Preserve all four temporary inputs by employee.
    Factor(
        "temporary_hours",
        "aligned",
        population="gpd",
        source="temporary_hours",
    ),
    Factor(
        "temporary_rate",
        "aligned",
        population="gpd",
        source="temporary_rate",
    ),
    Factor(
        "overtime_hours",
        "aligned",
        population="gpd",
        source="overtime_hours",
    ),
    Factor(
        "overtime_rate",
        "aligned",
        population="gpd",
        source="overtime_rate",
    ),
)

permanent_unit_cost = sum_over(
    value("grade_mix")
    * value("region_within_grade_mix")
    * (
        aligned_mean(
            value("permanent_rate") * value("permanent_hours") + value("permanent_bonus"),
            "permanent_members",
        )
    ),
    "grade",
    "reg_coeff",
)

temporary_unit_cost = aligned_mean(
    value("temporary_hours") * value("temporary_rate")
    + value("overtime_hours") * value("overtime_rate"),
    "temporary_members",
)

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
            "region_within_grade_mix",
            label="Permanent region composition within grade",
        ),
        Switch(
            "permanent_members",
            label="Workforce mix within cell",
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

temporary_game = Game(
    name="temp_game",
    population="gpd",
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

mix = value("employment_mix")

team_game = Game(
    name="team",
    formula=(
        select(mix, "employment_kind", "permanent")
        * child("permanent")
        + select(mix, "employment_kind", "gpd")
        * child("temp_game")
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
            child="temp_game",
            label="Temporary unit cost",
        ),
    ),
)

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

description = compiled.inspect()

print(description["game_order"])

for game in description["games"]:
    print(
        game["id"],
        game["population"],
        game["switches"],
    )

permanent_data = pd.DataFrame(
    [
        ("2026-08-01", "p1", "G1", 0.8, 10., 100., 50., 1050.),
        ("2026-08-01", "p2", "G1", 0.8, 20., 100., 100., 2100.),
        ("2026-08-01", "p3", "G2", 1., 25., 80., 0., 2000.),
        ("2026-09-01", "p1", "G1", 0.8, 12., 110., 60., 1380.),
        ("2026-09-01", "p2", "G2", 1., 21., 90., 80., 1970.),
        ("2026-09-01", "p3", "G2", 1., 25., 80., 0., 2000.),
    ],
    columns=[
        "period",
        "employee_id",
        "grade",
        "reg_coeff",
        "stavka_oklad",
        "vyrabotka_act",
        "bonus_kpi",
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
        ("2026-09-01", "t2", 75., 16., 8., 24., 1392.),
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
    employment_type="gpd",
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

for period in ("2026-08", "2026-09"):
    state = build_state(
        compiled,
        snapshot,
        period,
        source_cost_column="actual_cost",
    )
    print(period, cost(state, compiled))

request = AnalysisRequest(
    period_a="2026-08",
    period_b="2026-09",
    source_cost_column="actual_cost",
    explainer="exact",
)

result = analyze(compiled, snapshot, request)

validation = result.table("validation")
assert validation["passed"].all()

bridge = result.table("driver_bridge")

""" print(bridge[["label", "impact"]].round(6))

np.testing.assert_allclose(
    bridge["impact"].sum(),
    407.0,
    rtol=1e-10,
    atol=1e-6,
) """

hierarchy = result.table("hierarchy")
print(hierarchy)