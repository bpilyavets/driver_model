"""Diagnostics for the standalone workforce cost engine."""

from itertools import product
from math import factorial

import numpy as np
import pandas as pd

from .attribution import (
    active_unit_specs,
)
from .common import (
    ATOL,
    CONTRACTS,
    DIAGNOSTIC_COLUMNS,
    PARENT,
    RTOL,
    SALARY_KEYS,
    TEAM_PARENT,
    check,
)
from .costing import (
    contract_cost_terms,
    cost,
    permanent_cell_costs,
    team_units,
)
from .kernels import (
    prepare_salary_cache,
    profile_mean,
)
from .registry import (
    DRIVERS,
    drivers_at,
)
from .states import (
    COVERAGE_COLUMNS,
)


def salary_cell_audit(
    pa,
    pb,
    game,
    coverage_a,
    coverage_b,
    fallback_a,
    fallback_b,
    team,
    organization_multiplier,
    permanent_multiplier,
    local_multiplier,
    registry=DRIVERS,
    *,
    key="underlying_salary_level",
):
    """Attribute salary, time or membership marginals to permanent cells.

    Additive payment players have zero marginal interaction with these
    three drivers. Shapley additivity and dummy-player invariance allow
    the shared salary contribution's five-player game for this audit.
    Exact audits reconcile to the full exact game. Permutation audits use
    the same sampled cell values as their parent; no exact reallocation
    or post-hoc scaling is allowed. Cells are diagnostics, not players.
    """
    if game["audit"]["method"] == "permutation":
        index = game["impacts"].index.get_loc(key)
        impacts = game["cell_impacts"][index].reshape(
            pa["grade_mix_within_region"].shape
        )
    else:
        names = ["regional_mix", "grade_mix_within_region", *SALARY_KEYS]
        for driver in registry:
            if set(driver.dependencies) & set(SALARY_KEYS):
                for dependency in driver.dependencies:
                    if dependency not in names:
                        names.append(dependency)
        salary_registry = tuple(d for d in registry if d.name in names)
        others = [name for name in names if name != key]
        impacts = np.zeros_like(pa["grade_mix_within_region"])
        cache = prepare_salary_cache(pa, pb)
        for mask in product((0, 1), repeat=len(others)):
            hybrid = dict(pa, _salary_cache=cache)
            hybrid.update(
                {name: (pa, pb)[bit][name] for name, bit in zip(others, mask)}
            )
            before = permanent_cell_costs(hybrid, salary_registry)
            after = permanent_cell_costs(
                dict(hybrid, **{key: pb[key]}), salary_registry
            )
            marginal = after - before
            check(
                "Cell marginal additivity",
                marginal.sum(),
                after.sum() - before.sum(),
            )
            k, n = sum(mask), len(names)
            impacts += (
                factorial(k) * factorial(n - k - 1) / factorial(n) * marginal
            )
    check(
        key + " cell audit vs permanent SHAP",
        impacts.sum(),
        game["impacts"][key],
    )
    index = pd.MultiIndex.from_product(
        [pa["_policy"]["regions"], pa["_policy"]["grades"]],
        names=["region", "grade"],
    )
    audit = index.to_frame(index=False).assign(team=team, driver=key)
    for role, state, cov, fb in [
        ("A", pa, coverage_a, fallback_a),
        ("B", pb, coverage_b, fallback_b),
    ]:
        counts = (
            cov.loc[cov.team.eq(team)]
            .set_index(["region", "grade"])["count"]
            .reindex(index)
        )
        sources = (
            fb.loc[
                fb.team.eq(team) & fb.variable.eq("within_cell_workforce_mix")
            ]
            .drop_duplicates(["region", "grade"], keep="last")
            .set_index(["region", "grade"])
            .fallback_level.reindex(index)
        )
        sources = sources.where(counts.eq(0), "observed cell")
        if counts.isna().any() or sources.isna().any():
            raise ValueError("Missing membership support provenance")
        audit["count_" + role] = counts.to_numpy()
        audit["source_" + role] = sources.to_numpy()
        # These are descriptive means, never the switched salary/time objects.
        audit["salary_" + role] = profile_mean(
            state, state["underlying_salary_level"]
        ).ravel()
        audit["worked_fraction_" + role] = profile_mean(
            state, state["worked_hours"]
        ).ravel()
    audit["unit_impact"] = impacts.ravel()
    audit["organization_impact"] = (
        impacts.ravel() * organization_multiplier * permanent_multiplier
    )
    audit["local_impact"] = (
        impacts.ravel() * local_multiplier * permanent_multiplier
    )
    audit["support"] = np.where(
        audit.count_A.gt(0) & audit.count_B.gt(0),
        "observed in both",
        "imputed in at least one",
    )
    return audit


def employee_profile_diagnostics(a, b):
    """Expose effective salary inputs and sparse membership provenance.

    Includes copied whole-category states and explicit references. The
    recorded source period can differ from the target period; copied
    profiles never imply observed employees in an absent team/category.
    """
    inputs, profiles = [], []
    for organization in (a, b):
        for team, team_state in organization["teams"].items():
            state = team_state["permanent"]
            if state is None:
                continue
            axis = state["_employees"]
            sources = state.get("_employee_sources", (None,) * len(axis))
            teams = state.get("_employee_source_teams", (None,) * len(axis))
            for i, eid in enumerate(axis):
                inputs.append(
                    dict(
                        period=organization["_period"],
                        team=team,
                        employee_id=eid,
                        base_salary=state["underlying_salary_level"][i],
                        worked_fraction=state["worked_hours"][i],
                        source_period=sources[i],
                        source_team=teams[i],
                    )
                )
            for (ri, gi), profile in state[
                "within_cell_workforce_mix"
            ].items():
                factors = profile.get(
                    "base_pay_factors", np.ones(len(profile["indices"]))
                )
                for i, weight, factor in zip(
                    profile["indices"], profile["weights"], factors
                ):
                    profiles.append(
                        dict(
                            period=organization["_period"],
                            team=team,
                            region=state["_policy"]["regions"][ri],
                            grade=state["_policy"]["grades"][gi],
                            employee_id=axis[i],
                            weight=weight,
                            base_pay_factor=factor,
                        )
                    )
    return pd.DataFrame(inputs), pd.DataFrame(profiles)


def same_driver_value(left, right):
    """Compare array-valued or sparse-profile drivers recursively."""
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(
                same_driver_value(value, right[key])
                for key, value in left.items()
            )
        )
    return np.array_equal(left, right)


def matched_salary_diagnostic(frame, periods):
    """Describe matched permanent pay and team/sample movements.

    Periods are explicit. Entries and exits describe observed samples,
    not inferred hiring/termination dates or additional bridge impacts.
    """
    cols = [
        "employee_id",
        "team",
        "region",
        "grade",
        "base_salary",
        "worked_fraction",
    ]
    p = frame.loc[frame.contract_type.eq("permanent")]
    a, b = [p.loc[p.period.eq(period), cols] for period in periods]
    matched = a.merge(
        b, on="employee_id", suffixes=("_A", "_B"), validate="one_to_one"
    )
    matched["salary_change"] = matched.base_salary_B - matched.base_salary_A
    matched["worked_fraction_change"] = (
        matched.worked_fraction_B - matched.worked_fraction_A
    )
    matched["worked_fraction_changed"] = ~np.isclose(
        matched.worked_fraction_A,
        matched.worked_fraction_B,
        rtol=RTOL,
        atol=ATOL,
    )
    for name in ("team", "region", "grade"):
        matched[name + "_changed"] = matched[name + "_A"].ne(
            matched[name + "_B"]
        )
    matched["salary_changed"] = ~np.isclose(
        matched.base_salary_A.to_numpy(dtype=float),
        matched.base_salary_B.to_numpy(dtype=float),
        rtol=RTOL,
        atol=ATOL,
    )
    all_a, all_b = [
        frame.loc[frame.period.eq(period), ["employee_id", "team"]]
        .drop_duplicates("employee_id")
        .set_index("employee_id")
        for period in periods
    ]
    movements = []
    for team in sorted(frame.team.unique()):
        ids_a = set(all_a.index[all_a.team.eq(team)])
        ids_b = set(all_b.index[all_b.team.eq(team)])
        for event, ids, other in [
            ("entered team sample", ids_b - ids_a, all_a),
            ("left team sample", ids_a - ids_b, all_b),
        ]:
            for eid in sorted(ids):
                exists = eid in other.index
                movements.append(
                    dict(
                        team=team,
                        employee_id=eid,
                        event=event,
                        organization_sample_event=(
                            "present in both periods"
                            if exists
                            else (
                                "entered organization sample"
                                if event.startswith("entered")
                                else "left organization sample"
                            )
                        ),
                        other_period_team=(
                            other.loc[eid, "team"] if exists else None
                        ),
                    )
                )
    summary = pd.Series(
        dict(
            matched_permanent_employees=len(matched),
            unchanged_base_salary=int((~matched.salary_changed).sum()),
            higher_base_salary=int(
                (matched.salary_changed & matched.salary_change.gt(0)).sum()
            ),
            lower_base_salary=int(
                (matched.salary_changed & matched.salary_change.lt(0)).sum()
            ),
            changed_worked_fraction=int(matched.worked_fraction_changed.sum()),
            team_transfers=int(matched.team_changed.sum()),
            changed_region=int(matched.region_changed.sum()),
            changed_grade=int(matched.grade_changed.sum()),
            entered_permanent_sample=len(
                set(b.employee_id) - set(a.employee_id)
            ),
            left_permanent_sample=len(set(a.employee_id) - set(b.employee_id)),
        )
    )
    return dict(
        summary=summary,
        matched=matched,
        movements=pd.DataFrame(movements),
        entrants=b.loc[~b.employee_id.isin(a.employee_id)],
        leavers=a.loc[~a.employee_id.isin(b.employee_id)],
    )


def unsupported_mass(a, b, coverage_a, coverage_b, registry=DRIVERS):
    """Report maximum imputed-cell exposure over valid mix switches.

    Team and contract exposure maxima are combined with the maximum
    regional/grade unsupported share. Inactive permanent teams are
    skipped.
    These support diagnostics are not uncertainty intervals.
    """
    rows = []
    for i, team in enumerate(a["_teams"]):
        pa, pb = [s["teams"][team]["permanent"] for s in (a, b)]
        if (
            a["teams"][team]["contract_type_mix"][0]
            + b["teams"][team]["contract_type_mix"][0]
            == 0
        ):
            continue
        shape = pa["grade_mix_within_region"].shape
        max_exposure = (
            max(a["workforce_scale"], b["workforce_scale"])
            * max(a["team_mix"][i], b["team_mix"][i])
            * max(
                a["teams"][team]["contract_type_mix"][0],
                b["teams"][team]["contract_type_mix"][0],
            )
        )
        for role, coverage in [
            (a["_period"], coverage_a),
            (b["_period"], coverage_b),
        ]:
            missing = (
                coverage.loc[coverage.team.eq(team), "count"]
                .to_numpy()
                .reshape(shape)
                == 0
            )
            mass = max(
                float(
                    (
                        (pa, pb)[r]["regional_mix"][:, None]
                        * (pa, pb)[g]["grade_mix_within_region"]
                    )[missing].sum()
                )
                for r, g in product((0, 1), repeat=2)
            )
            for d in drivers_at(PARENT, registry):
                if d.kind.startswith("cell_") or d.name in SALARY_KEYS:
                    rows.append(
                        dict(
                            team=team,
                            value_period=role,
                            variable=d.name,
                            support_basis=(
                                "cell membership; see input provenance"
                                if d.name in SALARY_KEYS
                                else "payment cell"
                            ),
                            unsupported_cells=int(missing.sum()),
                            max_counterfactual_permanent_share=mass,
                            max_counterfactual_permanent_exposure=mass
                            * max_exposure,
                        )
                    )
    return pd.DataFrame(rows)


def exposure_allocation_table(frame):
    """Describe record counts, assigned exposure and base-pay conversion.

    One row represents one employee-month-employment-type observation.
    Factors default to one for canonical callers supplying per-exposure
    economics. Exposure is descriptive here; no assignment is performed.
    """
    columns = ["period", "employee_id", "team", "contract_type", "exposure"]
    result = frame[columns].copy()
    result["employment_type_count"] = frame.groupby(
        ["period", "employee_id"]
    ).contract_type.transform("nunique")
    result["base_pay_factor"] = frame.get("base_pay_factor", 1.0)
    return result


def diagnostics(frame, a, b, result, fa, fb, ca, cb, registry=DRIVERS):
    """Collect salary, fallback and employee-movement diagnostics.

    Periods come from the compared states, never notebook-wide dates.
    Inactive permanent branches have no salary audit or fabricated
    cells.
    """
    audits = {key: [] for key in SALARY_KEYS}
    for team, game in result["permanent"].items():
        multipliers = result["upper"]["teams"][team]
        for key in SALARY_KEYS:
            audits[key].append(
                salary_cell_audit(
                    a["teams"][team]["permanent"],
                    b["teams"][team]["permanent"],
                    game,
                    ca,
                    cb,
                    fa,
                    fb,
                    team,
                    multipliers["M"],
                    multipliers["K"],
                    multipliers["L"],
                    registry,
                    key=key,
                )
            )
    empty_columns = [
        "team",
        "region",
        "grade",
        "unit_impact",
        "organization_impact",
        "local_impact",
        "support",
    ]
    tables = {
        key: (
            pd.concat(parts, ignore_index=True)
            if parts
            else pd.DataFrame(columns=empty_columns)
        )
        for key, parts in audits.items()
    }
    salary = tables["underlying_salary_level"]
    inputs, profiles = employee_profile_diagnostics(a, b)
    support = []
    for team, group in salary.groupby("team"):
        for view in ("organization", "local"):
            column = view + "_impact"
            denominator = group[column].sum()
            absolute = group[column].abs().sum()
            for name, part in group.groupby("support"):
                support.append(
                    {
                        "team": team,
                        "view": view,
                        "support": name,
                        "cells": len(part),
                        "net_impact": part[column].sum(),
                        "absolute_impact": part[column].abs().sum(),
                        "share_of_net": (
                            np.nan
                            if abs(denominator) <= ATOL
                            else part[column].sum() / denominator
                        ),
                        "share_of_absolute": (
                            np.nan
                            if absolute <= ATOL
                            else part[column].abs().sum() / absolute
                        ),
                    }
                )
    return {
        "salary_cells": salary,
        "worked_fraction_cells": tables["worked_hours"],
        "workforce_mix_cells": tables["within_cell_workforce_mix"],
        "exposure_allocation": exposure_allocation_table(frame),
        "employee_inputs": inputs,
        "membership_profiles": profiles,
        "salary_support": pd.DataFrame(support),
        "matched": matched_salary_diagnostic(
            frame, (a["_period"], b["_period"])
        ),
        "coverage": pd.DataFrame(
            ca.to_dict("records") + cb.to_dict("records"),
            columns=COVERAGE_COLUMNS,
        ),
        "fallbacks": pd.DataFrame(
            fa.to_dict("records") + fb.to_dict("records"),
            columns=DIAGNOSTIC_COLUMNS,
        ),
        "unsupported": unsupported_mass(a, b, ca, cb, registry),
    }


def reconcile_games(a, b, result, registry=DRIVERS):
    """Assert every active game, multiplier and parent replacement."""
    upper, permanent = result["upper"], result["permanent"]
    delta = cost(b, registry) - cost(a, registry)
    checks = [
        check("Organizational SHAP", upper["impacts"].sum(), delta),
        check(
            "Organizational flattened bridge",
            result["bridge"].impact.sum(),
            delta,
        ),
    ]
    summary = result["driver_bridge"]
    bridge = result["bridge"]
    if not summary.driver.is_unique or set(summary.driver) != set(
        bridge.driver
    ):
        raise ValueError("Driver summary must contain each leaf driver once")
    checks.append(
        check(
            "Organizational aggregated driver bridge",
            summary.impact.sum(),
            delta,
        )
    )
    for row in summary.itertuples():
        checks.append(
            check(
                "Organizational driver total: " + row.driver,
                row.impact,
                bridge.loc[bridge.driver.eq(row.driver), "impact"].sum(),
            )
        )
    local_changes = []
    for index, team in enumerate(a["_teams"]):
        games = upper["teams"][team]
        units_a, units_b = [
            team_units(s["teams"][team], registry) for s in (a, b)
        ]
        unit_change = games["unit"]["end"] - games["unit"]["base"]
        local_change = games["local"]["end"] - games["local"]["base"]
        checks.extend(
            [
                check(
                    team + " unit SHAP",
                    games["unit"]["impacts"].sum(),
                    unit_change,
                ),
                check(
                    team + " organizational parent",
                    games["M"] * unit_change,
                    upper["impacts"]["team::" + team],
                ),
                check(
                    team + " local parent",
                    games["L"] * unit_change,
                    games["local"]["impacts"][TEAM_PARENT],
                ),
                check(
                    team + " local bridge",
                    result["local_bridges"][team].impact.sum(),
                    local_change,
                ),
            ]
        )
        scale_a, scale_b = a["workforce_scale"], b["workforce_scale"]
        share_a, share_b = a["team_mix"][index], b["team_mix"][index]
        expected_multiplier = (
            scale_a * share_a / 3
            + scale_b * share_b / 3
            + (scale_a * share_b + scale_b * share_a) / 6
        )
        checks.extend(
            [
                check(team + " M analytical", games["M"], expected_multiplier),
                check(
                    team + " K analytical",
                    games["K"],
                    (
                        units_a["contract_type_mix"][0]
                        + units_b["contract_type_mix"][0]
                    )
                    / 2,
                ),
                check(
                    team + " L analytical",
                    games["L"],
                    (scale_a * share_a + scale_b * share_b) / 2,
                ),
            ]
        )
        if team in permanent:
            impacts = permanent[team]["impacts"]
            parent = games["unit"]["impacts"][PARENT]
            checks.extend(
                [
                    check(
                        team + " permanent SHAP",
                        impacts.sum(),
                        units_b[PARENT] - units_a[PARENT],
                    ),
                    check(
                        team + " permanent parent",
                        games["K"] * impacts.sum(),
                        parent,
                    ),
                    check(
                        team + " organization permanent parent",
                        games["M"] * games["K"] * impacts.sum(),
                        games["M"] * parent,
                    ),
                    check(
                        team + " local permanent parent",
                        games["L"] * games["K"] * impacts.sum(),
                        games["L"] * parent,
                    ),
                ]
            )
            for driver in drivers_at(PARENT, registry):
                left = a["teams"][team]["permanent"][driver.name]
                right = b["teams"][team]["permanent"][driver.name]
                if same_driver_value(left, right):
                    checks.append(
                        check(
                            team + " unchanged " + driver.name,
                            impacts[driver.name],
                            0,
                        )
                    )
        for driver in active_unit_specs(units_a, units_b, registry):
            if np.array_equal(units_a[driver.name], units_b[driver.name]):
                checks.append(
                    check(
                        team + " unchanged " + driver.name,
                        games["unit"]["impacts"][driver.name],
                        0,
                    )
                )
        local_changes.append(local_change)
    checks.append(check("Sum of local changes", sum(local_changes), delta))
    if not result["coalition_audit"].passed.all():
        raise ValueError("Counterfactual evaluation audit failed")
    return pd.DataFrame(checks)


def reconcile(frame, a, b, result, registry=DRIVERS):
    """Assert period/team/contract source totals and hierarchical bridges.

    Reconstructed references verify aggregation, not independent payroll
    completeness. Tolerances are RTOL=1e-10 and ATOL=1e-6, with no
    rounding.
    """
    checks = []
    for state in (a, b):
        period = state["_period"]
        rows = frame.loc[frame.period.eq(period)]
        checks.append(
            check(
                f"{period:%Y-%m} organization vs source",
                cost(state, registry),
                rows.source_cost.sum(),
            )
        )
        for index, team in enumerate(state["_teams"]):
            exposure = state["workforce_scale"] * state["team_mix"][index]
            costs = exposure * contract_cost_terms(
                team_units(state["teams"][team], registry), registry
            )
            team_rows = rows.loc[rows.team.eq(team)]
            checks.append(
                check(
                    f"{period:%Y-%m} {team} vs source",
                    costs.sum(),
                    team_rows.source_cost.sum(),
                )
            )
            for contract, value in zip(CONTRACTS, costs):
                source = team_rows.loc[
                    team_rows.contract_type.eq(contract), "source_cost"
                ].sum()
                checks.append(
                    check(f"{period:%Y-%m} {team} {contract}", value, source)
                )
    return pd.concat(
        [pd.DataFrame(checks), reconcile_games(a, b, result, registry)],
        ignore_index=True,
    )
