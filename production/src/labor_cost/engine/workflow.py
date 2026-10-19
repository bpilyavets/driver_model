"""Workflow for the standalone workforce cost engine."""

from time import perf_counter

import pandas as pd

from .attribution import (
    decompose,
    validate_explainer_options,
)
from .diagnostics import (
    diagnostics,
    reconcile,
)
from .inputs import (
    adapt_production,
    month_start,
)
from .registry import (
    DRIVERS,
)
from .states import (
    build_comparison_states,
)


def analyze_labor_cost(
    frame,
    period_a,
    period_b,
    *,
    source_cost_column=None,
    exposure_column=None,
    references=None,
    missing_economics="carry_observed",
    registry=DRIVERS,
    explainer="exact",
    n_permutations=256,
    random_seed=0,
):
    """Run the complete labor-cost analysis on a production DataFrame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Employee-month records with the documented production columns.
        Ingestion belongs to the caller; this function never mutates
        frame.
    period_a, period_b : date-like
        Months to compare, in the direction A to B. Other months are
        ignored.
    source_cost_column : str or None
        Optional actual total for each employment-type row. Without it,
        permanent payroll is reconstructed from production components.
    exposure_column : str or None
        Source exposure in (0, 1], used as supplied. None assigns equal
        shares across each person's observed employment types per month.
    references : dict or None
        Optional (period, team) economics overrides for missing
        categories.
    missing_economics : str
        'carry_observed' by default, or 'require_reference' for strict
        mode.
    registry : tuple of Driver
        Declarative payroll/business-driver definitions.
    explainer : {'exact', 'permutation'}
        Permanent attribution method; small upper games remain exact.
    n_permutations : int
        Positive forward/reverse cycle budget, used by permutation only.
    random_seed : int
        Nonnegative sampling seed; repeated inputs/settings reproduce results.

    Returns
    -------
    dict
        Organizational/local bridges, driver_bridge and games; states,
        periods, canonical frame, team_summary, exposure_config,
        validation, assumptions and diagnostics.

    Raises
    ------
    ValueError
        Input, counterfactual support or reconciliation is invalid.

    Notes
    -----
    Signed payments describe recorded monthly payroll, including
    reversals.
    A copied missing rate has zero change by assumption, not
    observation.
    """
    validate_explainer_options(explainer, n_permutations, random_seed)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Assign your pandas DataFrame to input_df")
    if "period" not in frame:
        raise ValueError("The input DataFrame needs a period column")
    periods = (month_start(period_a), month_start(period_b))
    selected = frame.copy()
    selected["period"] = pd.to_datetime(selected.period, errors="raise")
    selected = selected.loc[selected.period.isin(periods)].copy()
    canonical = adapt_production(
        selected,
        source_cost_column=source_cost_column,
        exposure_column=exposure_column,
        registry=registry,
    )
    return analyze_canonical(
        canonical,
        *periods,
        registry=registry,
        references=references,
        missing_economics=missing_economics,
        exposure_column=exposure_column,
        explainer=explainer,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )


def analyze_canonical(
    canonical,
    period_a,
    period_b,
    *,
    registry=DRIVERS,
    references=None,
    missing_economics="carry_observed",
    exposure_column=None,
    explainer="exact",
    n_permutations=256,
    random_seed=0,
    dataset="Your payroll",
    fallback_order=("grade", "region", "global"),
):
    """Analyze validated canonical rows with separate state and cost engines.

    Return reconciled organizational/local results, diagnostics and states.
    Monetary values are signed currency; source rows remain unchanged.
    Completion happens before attribution, never inside standalone costing.
    """
    validate_explainer_options(explainer, n_permutations, random_seed)
    started = perf_counter()
    periods = (month_start(period_a), month_start(period_b))
    comparison = build_comparison_states(
        canonical,
        *periods,
        references=references,
        fallback_order=fallback_order,
        missing_economics=missing_economics,
        registry=registry,
    )
    state_a, state_b = comparison["states"]
    result = decompose(
        state_a,
        state_b,
        registry,
        dataset=dataset,
        explainer=explainer,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    result["validation"] = reconcile(
        canonical, state_a, state_b, result, registry
    )
    result["diagnostics"] = diagnostics(
        canonical,
        state_a,
        state_b,
        result,
        *comparison["fallbacks"],
        *comparison["coverage"],
        registry,
    )
    result["diagnostics"]["employee_alignment"] = comparison[
        "employee_alignment"
    ]
    result.update(
        {
            "states": comparison["states"],
            "periods": periods,
            "frame": canonical,
            "assumptions": comparison["assumptions"],
            "exposure_config": {
                "method": (
                    "equal_split"
                    if exposure_column is None
                    else "source_column"
                ),
                "column": exposure_column,
                "interpretation": (
                    "One assumed workforce unit per person-month, shared "
                    "equally across observed employment types"
                    if exposure_column is None
                    else "Source exposure used as supplied, without rescaling"
                ),
            },
        }
    )
    team_rows = []
    for team, games in result["upper"]["teams"].items():
        local = games["local"]
        change = local["end"] - local["base"]
        team_rows.append(
            {
                "team": team,
                "period_a_cost": local["base"],
                "period_b_cost": local["end"],
                "local_change": change,
                "organization_economics_impact": result["upper"]["impacts"][
                    "team::" + team
                ],
                "local_reconciliation_error": result["local_bridges"][
                    team
                ].impact.sum()
                - change,
            }
        )
    result["team_summary"] = pd.DataFrame(team_rows)
    result["elapsed_seconds"] = perf_counter() - started
    return result
