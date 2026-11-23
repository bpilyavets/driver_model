"""The documented labor-cost model expressed through analyst declarations."""

from dataclasses import replace

from .engine.registry import DRIVERS, PRODUCTION_COLUMNS
from .expressions import field
from .graph import OrganizationalHierarchy
from .spec import InputSpec, ModelSpec


def labor_cost_model(*, schema_version=1, line_source=None):
    """Return the versioned PoC-equivalent payroll model.

    Base salary is observed, not inferred from grade. Grade composition is
    conditional on region; region defaults to a coefficient band. Actual
    row payments receive the regional coefficient exactly where documented.
    """
    sources = {target: source for source, target in PRODUCTION_COLUMNS.items()}
    inputs = [
        InputSpec(
            "base_salary",
            "base_stavka",
            "Base salary changes",
            "aligned",
            "currency",
            "positive",
            "underlying_salary_level",
        ),
        InputSpec(
            "worked_fraction",
            "vyrabotka_percent",
            "Proportion of planned hours worked",
            "aligned",
            "dimensionless",
            "nonnegative",
            "worked_hours",
        ),
        InputSpec(
            "regional_coefficient",
            "reg_coef",
            "Fixed regional coefficient",
            "policy",
            "dimensionless",
            "positive",
        ),
    ]
    regional = field("regional_coefficient")
    formula = regional * field("base_salary") * field("worked_fraction")
    for driver in DRIVERS:
        if driver.kind != "cell_signed":
            continue
        name = driver.columns[0]
        inputs.append(
            InputSpec(
                name,
                sources[name],
                driver.label,
                "payment",
                "currency",
                "signed",
                driver.name,
            )
        )
        term = field(name)
        if driver.apply_regional_coefficient:
            term = regional * term
        formula = formula + term
    model = ModelSpec("labor-cost", "1.0.0", tuple(inputs), formula)
    if line_source is not None and schema_version != 2:
        raise ValueError("Selectable hierarchies require schema_version=2")
    if schema_version == 1:
        return model
    if schema_version == 2:
        from .graph_templates import legacy_graph

        graph = legacy_graph(model)
        if line_source is not None:
            graph = replace(
                graph, hierarchy=OrganizationalHierarchy(line_source)
            )
            return ModelSpec("labor-cost", "2.1.0", graph=graph)
        return ModelSpec("labor-cost", "2.0.0", graph=graph)
    raise ValueError("Supported model schemas are 1 and 2")
