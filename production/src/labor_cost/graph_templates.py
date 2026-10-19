"""Reusable outer games and the v1-to-v2 labor-template translation."""

from functools import reduce
from operator import add, mul

from .expressions import field, monomials
from .graph import (
    Factor,
    FieldSpec,
    Game,
    ModelGraph,
    Population,
    RowCost,
    SourceSchema,
    Switch,
    aligned_mean,
    child,
    select,
    sum_over,
    value,
)


def organization_game(team_game="team"):
    """Declare exposure times a complete team mix and team-unit costs."""
    return Game(
        "organization",
        value("workforce_scale")
        * sum_over(value("team_mix") * child(team_game), "team"),
        (
            Switch(
                "workforce_scale",
                label="Workforce scale",
                driver_id="workforce_scale",
            ),
            Switch("team_mix", label="Team mix", driver_id="team_mix"),
            Switch(
                "team_economics",
                child=team_game,
                label="Team economics",
                driver_id="team_economics",
            ),
        ),
    )


def local_game(team_game="team"):
    """Declare local exposure and unit economics as a separate bridge."""
    return Game(
        "local",
        sum_over(value("team_exposure") * child(team_game), "team"),
        (
            Switch(
                "workforce_scale",
                ("team_exposure",),
                label="Workforce scale",
                driver_id="workforce_scale",
            ),
            Switch(
                "team_economics",
                child=team_game,
                label="Team economics",
                driver_id="team_economics",
            ),
        ),
    )


def outer_factors():
    """Return explicit organization exposure, team exposure and team shares."""
    return (
        Factor("workforce_scale", "total", scope="organization"),
        Factor(
            "team_mix",
            "distribution",
            dimensions=("team",),
            scope="organization",
        ),
        Factor(
            "team_exposure",
            "total",
            dimensions=("team",),
            scope="organization",
        ),
    )


def legacy_graph(spec):
    """Lower validated v1 declarations into the same graph used by v2 models.

    Translation retains source mappings, atomic player order, signed
    payments and observed salary/time semantics. It does not run payroll
    or attribution and does not import the notebook.
    """
    populations = (
        Population("permanent", {"employment_type": "ТК"}),
        Population("temporary", {"employment_type": "ГПД"}),
        Population("outsourced", {"employment_type": "ПКЦ"}),
    )
    grade_source = spec.structure.grade_columns
    if len(grade_source) == 1:
        grade_source = grade_source[0]
    schema = SourceSchema(
        employee="mdm_employee_rk_hash",
        employment_type="motivation_type",
        team="lvl7_mapped_management_unit_nm",
        exposure=spec.exposure_column,
        dimensions={
            "region": spec.structure.region_column or "reg_coef",
            "grade": grade_source,
        },
    )
    fields = [
        FieldSpec(x.name, x.source, "permanent", x.units, x.constraint)
        for x in spec.inputs
    ]
    fields.extend(
        (
            FieldSpec("temporary_pay", "gpd_pay", "temporary"),
            FieldSpec("outsourced_pay", "pkc_pay", "outsourced"),
        )
    )
    pools = {"grade": ("grade",), "region": ("region",), "global": ()}
    fallbacks = tuple(pools[x] for x in spec.structure.fallback_order)
    factors = list(outer_factors()) + [
        Factor(
            "contract_type_mix",
            "population_mix",
            dimensions=("contract",),
            members=("permanent", "temporary", "outsourced"),
        ),
        Factor(
            "regional_mix", "distribution", "permanent", dimensions=("region",)
        ),
        Factor(
            "grade_mix_within_region",
            "distribution",
            "permanent",
            dimensions=("grade",),
            given=("region",),
            fallbacks=((),),
        ),
        Factor(
            "within_cell_workforce_mix",
            "membership",
            "permanent",
            dimensions=("region", "grade"),
            fallbacks=fallbacks,
        ),
        Factor(
            "regional_policy",
            "policy",
            "permanent",
            dimensions=("region",),
            source="regional_coefficient",
        ),
    ]
    inputs = {x.name: x for x in spec.inputs}
    terms = monomials(spec.formula)
    used_payments = {
        name
        for _, names in terms
        for name in names
        if inputs[name].representation == "payment"
    }
    for item in spec.inputs:
        if item.representation == "aligned":
            factors.append(
                Factor(
                    item.driver_id, "aligned", "permanent", source=item.name
                )
            )
        elif item.representation == "payment" and item.name in used_payments:
            factors.append(
                Factor(
                    item.driver_id,
                    "payment",
                    "permanent",
                    dimensions=("region", "grade"),
                    formula=field(item.name),
                    fallbacks=fallbacks,
                )
            )
    cell_terms = []
    for coefficient, names in terms:
        regional = "regional_coefficient" in names
        names = [x for x in names if x != "regional_coefficient"]
        expression = reduce(mul, (value(inputs[x].driver_id) for x in names))
        if all(inputs[x].representation == "aligned" for x in names):
            expression = aligned_mean(expression, "within_cell_workforce_mix")
        if regional:
            expression = value("regional_policy") * expression
        cell_terms.append(coefficient * expression)
    costs = reduce(add, cell_terms)
    labels = {x.driver: x.label for x in spec.structure.distributions}
    switches = [
        Switch(
            "regional_mix",
            label=labels["regional_mix"],
            driver_id="regional_mix",
        ),
        Switch(
            "grade_mix_within_region",
            label=labels["grade_mix_within_region"],
            driver_id="grade_mix_within_region",
        ),
        Switch(
            "within_cell_workforce_mix",
            label=spec.structure.membership_label,
            driver_id="within_cell_workforce_mix",
        ),
    ]
    switches.extend(
        Switch(x.driver_id, label=x.label, driver_id=x.driver_id)
        for x in spec.inputs
        if x.representation == "aligned" or x.name in used_payments
    )
    permanent = Game(
        "permanent",
        sum_over(
            value("regional_mix") * value("grade_mix_within_region") * costs,
            "region",
            "grade",
        ),
        tuple(switches),
        "permanent",
    )
    reimbursement_games = []
    for population, driver, label in (
        ("temporary", "temp_reimbursement", "Temp reimbursement"),
        ("outsourced", "outsource_reimbursement", "Outsource reimbursement"),
    ):
        factors.append(
            Factor(
                driver,
                "payment",
                population,
                formula=field(population + "_pay"),
            )
        )
        reimbursement_games.append(
            Game(
                population,
                value(driver),
                (Switch(driver, label=label, driver_id=driver),),
                population,
            )
        )
    mixture = value("contract_type_mix")
    team_formula = reduce(
        add,
        (
            select(mixture, "contract", pop.name) * child(pop.name)
            for pop in populations
        ),
    )
    team = Game(
        "team",
        team_formula,
        (
            Switch(
                "contract_type_mix",
                label=labels["contract_type_mix"],
                driver_id="contract_type_mix",
            ),
            Switch(
                "temp_reimbursement",
                child="temporary",
                label="Temp reimbursement",
                driver_id="temp_reimbursement",
            ),
            Switch(
                "outsource_reimbursement",
                child="outsourced",
                label="Outsource reimbursement",
                driver_id="outsource_reimbursement",
            ),
            Switch(
                "permanent_unit_cost",
                child="permanent",
                label="Permanent unit cost",
                driver_id="permanent_unit_cost",
            ),
        ),
    )
    return ModelGraph(
        schema,
        populations,
        tuple(fields),
        (
            RowCost("permanent", spec.formula),
            RowCost("temporary", field("temporary_pay")),
            RowCost("outsourced", field("outsourced_pay")),
        ),
        tuple(factors),
        (permanent, *reimbursement_games, team),
        "team",
        organization_game(),
        local_game(),
        spec.missing_economics,
    )
