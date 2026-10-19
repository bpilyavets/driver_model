"""Author a complete model without the built-in payroll template.

The input remains employee/employment-type/month records. The model uses
hourly pay and a signed allowance, with analyst-selected grade/region
conditioning. No permanent-worker or salary-specific engine code is used.
"""

from labor_cost import (
    Factor,
    FieldSpec,
    Game,
    ModelGraph,
    ModelSpec,
    RowCost,
    SourceSchema,
    Switch,
    aligned_mean,
    child,
    field,
    local_game,
    organization_game,
    outer_factors,
    sum_over,
    value,
)


def hourly_model(*, grade_first=True, grouped=False, nested=False):
    """Declare both row and aggregate costs, plus explicit atomic games.

    Parameters
    ----------
    grade_first : bool
        Choose P(grade) P(region|grade), or the reversed factorization.
    grouped : bool
        Switch hourly rates and hours together instead of separately.
    nested : bool
        Expand base earnings through a child game; allowance then uses
        one all-population payment aggregate in the parent game.

    Returns
    -------
    ModelSpec
        A v2 model operating on the canonical fixed-grain source schema.
    """
    first, second = ("grade", "region") if grade_first else ("region", "grade")
    dimensions = ("grade", "region")
    factors = list(outer_factors()) + [
        Factor("first_mix", "distribution", dimensions=(first,)),
        Factor(
            "conditional_mix",
            "distribution",
            dimensions=(second,),
            given=(first,),
            fallbacks=((),),
        ),
        Factor(
            "members",
            "membership",
            dimensions=dimensions,
            fallbacks=(("grade",), ("region",), ()),
        ),
        Factor("rates", "aligned", source="rate"),
        Factor("hours", "aligned", source="hours"),
        Factor(
            "allowances",
            "payment",
            dimensions=() if nested else dimensions,
            formula=field("allowance"),
            fallbacks=() if nested else (("grade",), ("region",), ()),
        ),
    ]
    switches = [
        Switch("first_mix"),
        Switch("conditional_mix"),
        Switch("members"),
    ]
    if grouped:
        switches.append(Switch("earned_pay_inputs", ("rates", "hours")))
    else:
        switches.extend((Switch("rates"), Switch("hours")))
    paired = aligned_mean(value("rates") * value("hours"), "members")
    joint = value("first_mix") * value("conditional_mix")
    if nested:
        earnings = Game(
            "earnings", sum_over(joint * paired, *dimensions), tuple(switches)
        )
        unit = Game(
            "unit",
            child("earnings") + value("allowances"),
            (Switch("base_earnings", child="earnings"), Switch("allowances")),
        )
        games = (earnings, unit)
    else:
        switches.append(Switch("allowances"))
        games = (
            Game(
                "unit",
                sum_over(joint * (paired + value("allowances")), *dimensions),
                tuple(switches),
            ),
        )
    return ModelSpec(
        "hourly-workforce",
        f"2.0.{int(grade_first)}-{int(grouped)}-{int(nested)}",
        graph=ModelGraph(
            source=SourceSchema(
                dimensions={"grade": "grade", "region": "region"}
            ),
            populations=(),
            fields=(
                FieldSpec(
                    "rate",
                    "hourly_rate",
                    units="currency/time",
                    constraint="positive",
                ),
                FieldSpec(
                    "hours",
                    "paid_hours",
                    units="time",
                    constraint="nonnegative",
                ),
                FieldSpec("allowance", "allowance"),
            ),
            row_costs=(
                RowCost(
                    "all", field("rate") * field("hours") + field("allowance")
                ),
            ),
            factors=tuple(factors),
            games=games,
            team_game="unit",
            organization=organization_game("unit"),
            local=local_game("unit"),
        ),
    )
