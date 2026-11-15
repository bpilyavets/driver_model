"""Editable declarations for comparing three permanent base-pay models.

These are examples, not an automatic converter for arbitrary models.
Permanent rates and bonuses are pre-regional; temporary pay is unadjusted.
Only permanent base-pay representation varies between the three models.
"""

from labor_cost import (
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
    child,
    field,
    local_game,
    organization_game,
    outer_factors,
    select,
    sum_over,
    value,
)

DIMS = ("coefficient_band", "grade")
FALLBACKS = (("grade",), ("coefficient_band",), ())


def _permanent(representation):
    """Declare the alternative base-pay factors and their atomic switches."""
    common = dict(population="permanent", dimensions=DIMS, fallbacks=FALLBACKS)
    if representation == "effective":
        factors = (
            Factor(
                "hours_per_exposure",
                "mean",
                source="permanent_hours",
                **common,
            ),
            Factor(
                "effective_rate",
                "weighted_mean",
                source="permanent_rate",
                weight="permanent_hours",
                zero_weight="exposure_mean",
                **common,
            ),
        )
        switches = (
            Switch("effective_rate", label="Effective hourly rate"),
            Switch("hours_per_exposure", label="Hours per exposure"),
        )
        formula = value("effective_rate") * value("hours_per_exposure")
    else:
        factors = (
            Factor("permanent_members", "membership", **common),
            Factor(
                "permanent_rate",
                "aligned",
                "permanent",
                source="permanent_rate",
            ),
            Factor(
                "permanent_hours",
                "aligned",
                "permanent",
                source="permanent_hours",
            ),
        )
        switches = (
            (
                Switch("permanent_rate", label="Employee rate changes"),
                Switch("permanent_members", label="Within-cell staffing mix"),
            )
            if representation == "aligned"
            else (
                Switch(
                    "base_pay_and_staffing",
                    ("permanent_rate", "permanent_members"),
                    label="Base-pay level and staffing mix",
                ),
            )
        )
        switches += (Switch("permanent_hours", label="Employee hours"),)
        formula = aligned_mean(
            value("permanent_rate") * value("permanent_hours"),
            "permanent_members",
        )
    return factors, switches, formula


def comparison_models(
    *,
    source=None,
    columns=None,
    permanent_category="permanent",
    temporary_category="temporary",
):
    """Build equivalent row payroll with three distinct attribution games.

    Parameters
    ----------
    source : SourceSchema, optional
        Key and categorical mappings. Dimensions must include grade and
        coefficient_band. The default band maps to source column reg_coef.
    columns : mapping, optional
        Model field IDs to source columns. Unspecified fields use their ID;
        regional_coefficient defaults to reg_coef.
    permanent_category, temporary_category : str
        Source employment-type categories for the two populations.

    Returns
    -------
    dict
        Aligned, grouped and effective v2 ModelSpec declarations. Edit this
        example's row formulas and common components for your own payroll.
    """
    source = source or SourceSchema(
        dimensions={
            "coefficient_band": "reg_coef",
            "grade": "grade",
        }
    )
    columns = {"regional_coefficient": "reg_coef", **(columns or {})}
    definitions = (
        ("permanent_rate", "permanent", "currency/time", "positive"),
        ("permanent_hours", "permanent", "time", "nonnegative"),
        ("permanent_bonus", "permanent", "currency", "signed"),
        ("regional_coefficient", "permanent", "dimensionless", "positive"),
        ("temporary_rate", "temporary", "currency/time", "positive"),
        ("temporary_hours", "temporary", "time", "nonnegative"),
        ("overtime_rate", "temporary", "currency/time", "positive"),
        ("overtime_hours", "temporary", "time", "nonnegative"),
    )
    fields = tuple(
        FieldSpec(name, columns.get(name, name), pop, units, constraint)
        for name, pop, units, constraint in definitions
    )
    populations = (
        Population("permanent", {"employment_type": permanent_category}),
        Population("temporary", {"employment_type": temporary_category}),
    )
    row_costs = (
        RowCost(
            "permanent",
            field("regional_coefficient")
            * (
                field("permanent_rate") * field("permanent_hours")
                + field("permanent_bonus")
            ),
        ),
        RowCost(
            "temporary",
            field("temporary_rate") * field("temporary_hours")
            + field("overtime_rate") * field("overtime_hours"),
        ),
    )
    common = (
        outer_factors()
        + (
            Factor(
                "employment_mix",
                "population_mix",
                dimensions=("employment_kind",),
                members=("permanent", "temporary"),
            ),
            Factor(
                "regional_mix",
                "distribution",
                "permanent",
                dimensions=("coefficient_band",),
            ),
            Factor(
                "grade_mix_within_region",
                "distribution",
                "permanent",
                dimensions=("grade",),
                given=("coefficient_band",),
                fallbacks=((),),
            ),
            Factor(
                "regional_policy",
                "policy",
                "permanent",
                dimensions=("coefficient_band",),
                source="regional_coefficient",
            ),
            Factor(
                "permanent_bonus",
                "payment",
                "permanent",
                dimensions=DIMS,
                formula=field("permanent_bonus"),
                fallbacks=FALLBACKS,
            ),
            Factor("temporary_members", "membership", "temporary"),
        )
        + tuple(
            Factor(name, "aligned", "temporary", source=name)
            for name in (
                "temporary_rate",
                "temporary_hours",
                "overtime_rate",
                "overtime_hours",
            )
        )
    )
    temporary = Game(
        "temporary",
        aligned_mean(
            value("temporary_rate") * value("temporary_hours")
            + value("overtime_rate") * value("overtime_hours"),
            "temporary_members",
        ),
        tuple(
            Switch(name)
            for name in (
                "temporary_rate",
                "temporary_hours",
                "overtime_rate",
                "overtime_hours",
                "temporary_members",
            )
        ),
        "temporary",
    )
    team = Game(
        "team_cost",
        select(value("employment_mix"), "employment_kind", "permanent")
        * child("permanent")
        + select(value("employment_mix"), "employment_kind", "temporary")
        * child("temporary"),
        (
            Switch("employment_mix"),
            Switch("permanent", child="permanent"),
            Switch("temporary", child="temporary"),
        ),
    )
    models = {}
    for representation in ("aligned", "grouped", "effective"):
        factors, switches, base = _permanent(representation)
        permanent = Game(
            "permanent",
            sum_over(
                value("regional_mix")
                * value("grade_mix_within_region")
                * value("regional_policy")
                * (base + value("permanent_bonus")),
                *DIMS,
            ),
            (Switch("regional_mix"), Switch("grade_mix_within_region"))
            + switches
            + (Switch("permanent_bonus"),),
            "permanent",
        )
        models[representation] = ModelSpec(
            "pay-comparison-" + representation,
            "1.0.0",
            graph=ModelGraph(
                source,
                populations,
                fields,
                row_costs,
                common + factors,
                (permanent, temporary, team),
                "team_cost",
                organization_game("team_cost"),
                local_game("team_cost"),
            ),
        )
    return models
