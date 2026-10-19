"""States for the standalone workforce cost engine."""

from copy import deepcopy

import numpy as np
import pandas as pd

from .common import (
    CONTRACT_KEYS,
    CONTRACTS,
    DIAGNOSTIC_COLUMNS,
    FALLBACK_ORDER,
    PARENT,
    SALARY_KEYS,
    TEAM_PARENT,
)
from .costing import (
    aligned_comparison,
    validate_organization,
    validate_permanent,
    validate_team,
)
from .inputs import (
    month_start,
    validate_input,
)
from .kernels import (
    ensure_same_policy,
    log_fallback,
    validate_policy,
)
from .registry import (
    DRIVERS,
    drivers_at,
)

COVERAGE_COLUMNS = ["region", "grade", "period", "team", "count", "exposure"]


ASSUMPTION_COLUMNS = [
    "team",
    "contract_type",
    "missing_period",
    "source_period",
    "method",
    "affected_economics",
    "interpretation",
]


def make_policy(frame, *, permanent_policies=None):
    """Align team and permanent axes from the selected comparison rows.

    Teams with no permanent observations need no permanent policy.
    Optional policies support explicitly supplied business references.
    """
    policies = deepcopy(permanent_policies or {})
    teams = tuple(sorted(frame.team.unique()))
    if set(policies) - set(teams):
        raise ValueError("Reference policy names a team outside the data")
    for team in teams:
        rows = frame.loc[
            frame.team.eq(team) & frame.contract_type.eq("permanent")
        ]
        if team not in policies:
            if rows.empty:
                policies[team] = None
                continue
            counts = rows.groupby("region").regional_coefficient.nunique()
            if counts.ne(1).any():
                raise ValueError("Differing regional coefficient schedules")
            coefficients = rows.groupby("region").regional_coefficient.first()
            coefficients = coefficients.sort_index()
            policies[team] = {
                "regions": tuple(coefficients.index),
                "grades": tuple(sorted(rows.grade.unique(), key=str)),
                "regional_coefficients": coefficients.to_numpy(dtype=float),
            }
        policy = policies[team]
        validate_policy(policy)
        if not rows.region.isin(policy["regions"]).all():
            raise ValueError("Unknown region outside the aligned policy")
        if not rows.grade.isin(policy["grades"]).all():
            raise ValueError("Unknown grade outside the aligned policy")
        expected = rows.region.map(
            dict(zip(policy["regions"], policy["regional_coefficients"]))
        )
        if not np.array_equal(expected, rows.regional_coefficient):
            raise ValueError("Source coefficients differ from fixed policy")
    return {"teams": teams, "permanent": policies}


def build_state(
    frame, period, policy, registry=DRIVERS, fallback_order=FALLBACK_ORDER
):
    """Build an observed period state, preserving unavailable economics.

    Returns
    -------
    tuple
        State, fallback table and permanent-cell coverage. Missing team
        states or contract economics are None, never zero rates. Such
        states can cost their observed workforce; a comparison must use
        build_comparison_states to complete needed counterfactual
        values.
    """
    validate_input(frame, registry)
    period = month_start(period)
    if len(set(fallback_order)) != len(fallback_order):
        raise ValueError("Duplicate fallback levels")
    if not set(fallback_order) <= {"grade", "region", "global"}:
        raise ValueError("Invalid fallback order")
    if not frame.team.isin(policy["teams"]).all():
        raise ValueError("Unknown team outside aligned policy")
    rows = frame.loc[frame.period.eq(period)]
    if rows.empty:
        raise ValueError(f"No employee-month records for {period:%Y-%m}")
    diagnostics, coverage, teams = [], [], {}
    exposures = (
        rows.groupby("team")
        .exposure.sum()
        .reindex(policy["teams"], fill_value=0)
    )
    for team in policy["teams"]:
        team_rows = rows.loc[rows.team.eq(team)]
        permanent = team_rows.loc[team_rows.contract_type.eq("permanent")]
        permanent_policy = policy["permanent"][team]
        context = {
            "rows": team_rows,
            "perm": permanent,
            "employees": tuple(sorted(permanent.employee_id)),
            "period": period,
            "team": team,
            "policy": permanent_policy,
            "contract_exposure": team_rows.groupby("contract_type")
            .exposure.sum()
            .reindex(CONTRACTS, fill_value=0)
            .to_numpy(),
            "fallback_order": tuple(fallback_order),
            "diagnostics": diagnostics,
        }
        if permanent_policy is not None:
            validate_policy(permanent_policy)
            index = pd.MultiIndex.from_product(
                [permanent_policy["regions"], permanent_policy["grades"]],
                names=["region", "grade"],
            )
            counts = (
                permanent.groupby(["region", "grade"])
                .size()
                .reindex(index, fill_value=0)
            )
            cell_exposure = permanent.groupby(["region", "grade"]).exposure
            cell_exposure = cell_exposure.sum().reindex(index, fill_value=0)
            shape = (
                len(permanent_policy["regions"]),
                len(permanent_policy["grades"]),
            )
            context["cell_exposure"] = cell_exposure.to_numpy().reshape(shape)
            coverage.append(
                index.to_frame(index=False).assign(
                    period=period,
                    team=team,
                    count=counts.to_numpy(),
                    exposure=cell_exposure.to_numpy(),
                )
            )
            if not permanent.empty:
                if not permanent.region.isin(
                    permanent_policy["regions"]
                ).all():
                    raise ValueError("Unknown region outside aligned policy")
                if not permanent.grade.isin(permanent_policy["grades"]).all():
                    raise ValueError("Unknown grade outside aligned policy")
                expected = permanent.region.map(
                    dict(
                        zip(
                            permanent_policy["regions"],
                            permanent_policy["regional_coefficients"],
                        )
                    )
                )
                if not np.array_equal(
                    expected, permanent.regional_coefficient
                ):
                    raise ValueError("Source coefficients differ from policy")
        elif not permanent.empty:
            raise ValueError("Observed permanent employees need a policy")
        if team_rows.empty:
            teams[team] = None
            continue
        team_state = {
            driver.name: driver.builder(context)
            for driver in drivers_at(TEAM_PARENT, registry)
            if driver.builder is not None
        }
        team_state["permanent"] = None
        if not permanent.empty:
            team_state["permanent"] = {
                "_policy": deepcopy(permanent_policy),
                "_employees": context["employees"],
                "_employee_sources": tuple(
                    period for _ in context["employees"]
                ),
                "_employee_source_teams": tuple(
                    team for _ in context["employees"]
                ),
                **{
                    driver.name: driver.builder(context)
                    for driver in drivers_at(PARENT, registry)
                },
            }
        teams[team] = team_state
    state = {
        "_period": period,
        "_teams": tuple(policy["teams"]),
        "workforce_scale": float(exposures.sum()),
        "team_mix": exposures.to_numpy() / exposures.sum(),
        "teams": teams,
    }
    validate_organization(state, registry)
    coverage_frame = (
        pd.concat(coverage, ignore_index=True)
        if coverage
        else pd.DataFrame(columns=COVERAGE_COLUMNS)
    )
    return (
        state,
        pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS),
        coverage_frame,
    )


def log_completed_economics(
    records, period, team, contract, value, method, source_period, registry
):
    """Record copied/reference scalar or matrix values without fake
    rows.
    """
    context = {"period": period, "team": team, "diagnostics": records}
    if contract != "permanent":
        log_fallback(
            context,
            None,
            None,
            CONTRACT_KEYS[contract],
            method,
            value,
            source_period,
        )
        return
    policy = value["_policy"]
    validate_permanent(value, registry)
    for driver in drivers_at(PARENT, registry):
        if driver.kind.startswith("employee_"):
            for eid, amount in zip(value["_employees"], value[driver.name]):
                log_fallback(
                    context,
                    None,
                    None,
                    driver.name,
                    method,
                    amount,
                    source_period,
                )
                records[-1]["employee_id"] = eid
            continue
        for ri, region in enumerate(policy["regions"]):
            if driver.kind == "region_probability":
                log_fallback(
                    context,
                    region,
                    None,
                    driver.name,
                    method,
                    value[driver.name][ri],
                    source_period,
                )
                continue
            for gi, grade in enumerate(policy["grades"]):
                amount = (
                    1.0
                    if driver.kind == "membership_probability"
                    else value[driver.name][ri, gi]
                )
                log_fallback(
                    context,
                    region,
                    grade,
                    driver.name,
                    method,
                    amount,
                    source_period,
                )


def align_observed_employee_inputs(
    states, frame, periods, references, records, registry=DRIVERS
):
    """Align observed team salary inputs by permanent employee identity.

    Match across the organization, including transfers. A missing
    permanent observation uses that person's sole observed salary and
    worked fraction. References can provide values for otherwise unknown
    profile IDs. Never pool other employees' economics across teams.
    Returns detailed matching/carry provenance without adding exposure.
    """
    extra_specs = [
        d
        for d in drivers_at(PARENT, registry)
        if d.kind.startswith("employee_") and d.name not in SALARY_KEYS
    ]
    permanent = frame.loc[frame.contract_type.eq("permanent")]
    lookups = [
        permanent.loc[permanent.period.eq(p)].set_index("employee_id")
        for p in periods
    ]
    provenance = []
    for team in states[0]["_teams"]:
        observed = [state["teams"][team] for state in states]
        observed = [
            None if value is None else value["permanent"] for value in observed
        ]
        reference_states = []
        for period in periods:
            ref = references.get((period, team), {})
            for value in [
                ref.get("permanent"),
                (ref.get("team_state") or {}).get("permanent"),
            ]:
                if value is not None:
                    reference_states.append(value)
        all_states = [
            value for value in observed + reference_states if value is not None
        ]
        if not all_states:
            continue
        axis = tuple(
            sorted(
                {eid for value in all_states for eid in value["_employees"]}
            )
        )
        positions = {eid: i for i, eid in enumerate(axis)}
        for ref in reference_states:
            if tuple(ref["_employees"]) != axis:
                raise ValueError(
                    f"{team}: permanent reference employee axis must be "
                    "aligned to the sorted union of comparison/profile IDs"
                )
        for side, state in enumerate(observed):
            if state is None:
                continue
            old_axis = state["_employees"]
            for profile in state["within_cell_workforce_mix"].values():
                profile["indices"] = np.array(
                    [positions[old_axis[i]] for i in profile["indices"]],
                    dtype=int,
                )
            salary, hours, sources, source_teams = [], [], [], []
            extra_values = {d.name: [] for d in extra_specs}
            for eid in axis:
                if eid in lookups[side].index:
                    donor_side, method = side, "observed_employee"
                elif eid in lookups[1 - side].index:
                    donor_side, method = 1 - side, "carry_employee"
                else:
                    donor_side, method = None, "reference_employee"
                if donor_side is None:
                    ref = next(
                        value
                        for value in reference_states
                        if eid in value["_employees"]
                    )
                    index = ref["_employees"].index(eid)
                    base, worked = (ref[key][index] for key in SALARY_KEYS[1:])
                    source, donor_team = None, None
                else:
                    row = lookups[donor_side].loc[eid]
                    base, worked = row.base_salary, row.worked_fraction
                    source, donor_team = periods[donor_side], row.team
                for driver in extra_specs:
                    amount = (
                        ref[driver.name][index]
                        if donor_side is None
                        else row[driver.columns[0]]
                    )
                    extra_values[driver.name].append(amount)
                    if method != "observed_employee":
                        context = {
                            "period": periods[side],
                            "team": team,
                            "diagnostics": records[side],
                        }
                        log_fallback(
                            context,
                            None,
                            None,
                            driver.name,
                            method,
                            amount,
                            source,
                        )
                        records[side][-1]["employee_id"] = eid
                salary.append(base)
                hours.append(worked)
                sources.append(source)
                source_teams.append(donor_team)
                provenance.append(
                    dict(
                        period=periods[side],
                        team=team,
                        employee_id=eid,
                        source_period=source,
                        source_team=donor_team,
                        method=method,
                        base_salary=base,
                        worked_fraction=worked,
                        interpretation=(
                            "Same employee; no cross-team population pooling"
                            if method == "observed_employee"
                            else "Missing inputs held constant by assumption"
                        ),
                    )
                )
                if method != "observed_employee":
                    context = {
                        "period": periods[side],
                        "team": team,
                        "diagnostics": records[side],
                    }
                    for key, amount in zip(SALARY_KEYS[1:], [base, worked]):
                        log_fallback(
                            context, None, None, key, method, amount, source
                        )
                        records[side][-1]["employee_id"] = eid
            state.update(
                {
                    key: np.asarray(values, dtype=float)
                    for key, values in extra_values.items()
                }
            )
            state.update(
                _employees=axis,
                _employee_sources=tuple(sources),
                _employee_source_teams=tuple(source_teams),
                underlying_salary_level=np.asarray(salary, dtype=float),
                worked_hours=np.asarray(hours, dtype=float),
            )
    return pd.DataFrame(provenance)


def build_comparison_states(
    frame,
    period_a,
    period_b,
    *,
    references=None,
    missing_economics="carry_observed",
    registry=DRIVERS,
    fallback_order=FALLBACK_ORDER,
):
    """Build and complete two workforce states for valid A/B switches.

    Parameters
    ----------
    frame : pandas.DataFrame
        Canonical observations for the selected periods.
    period_a, period_b : date-like
        Explicit month selectors; their order defines change direction.
    references : dict or None
        Keys are (period, team). Values can supply temporary/outsourced
        rates, a complete permanent state or a complete team_state.
        References fill missing economics only, before automatic
        copying.
    missing_economics : {'carry_observed', 'require_reference'}
        Default copies the same team's observed opposite-period
        economics.
        Both-period inactive categories need no rates under either
        policy.
    registry : tuple of Driver
        Drivers used to build and validate payroll states.
    fallback_order : tuple of str
        Same-period sparse-cell fallback order, unchanged by copying.

    Returns
    -------
    dict
        Completed states, fallbacks, coverage, periods and assumptions.
        Observed workforce shares and exposure are never copied.
    """
    if missing_economics not in {"carry_observed", "require_reference"}:
        raise ValueError("Unknown missing_economics policy")
    periods = (month_start(period_a), month_start(period_b))
    selected = frame.loc[frame.period.isin(periods)].copy()
    validate_input(selected, registry)
    refs = {}
    permanent_policies = {}
    for (period, team), values in (references or {}).items():
        key = (month_start(period), team)
        if key in refs:
            raise ValueError("Duplicate normalized reference keys")
        if key[0] not in periods or team not in set(selected.team):
            raise ValueError("Reference must name a selected period and team")
        if not set(values) <= set(CONTRACTS) | {"team_state"}:
            raise ValueError("Unknown reference category")
        refs[key] = deepcopy(values)
        candidates = [values.get("permanent")]
        if values.get("team_state") is not None:
            candidates.append(values["team_state"].get("permanent"))
        for permanent in candidates:
            if permanent is None:
                continue
            validate_permanent(permanent, registry)
            if team in permanent_policies:
                ensure_same_policy(
                    permanent_policies[team], permanent["_policy"]
                )
            permanent_policies[team] = permanent["_policy"]
    policy = make_policy(selected, permanent_policies=permanent_policies)
    built = [
        build_state(selected, period, policy, registry, fallback_order)
        for period in periods
    ]
    states = [item[0] for item in built]
    records = [item[1].to_dict("records") for item in built]
    employee_alignment = align_observed_employee_inputs(
        states, selected, periods, refs, records, registry
    )
    observed = deepcopy(states)
    assumptions = []

    def assume(team, contract, period, source, method, affected):
        """Append one user-facing completion or inactivity
        assumption.
        """
        interpretations = {
            "carry_observed": "Held constant; change is not identifiable",
            "business_reference": "Explicit reference for missing economics",
            "inactive": "No exposure in either state; not applicable",
            "carry_employee": "Missing employee salary/time held constant",
            "reference_employee": "Unobserved profile ID uses reference",
        }
        assumptions.append(
            dict(
                zip(
                    ASSUMPTION_COLUMNS,
                    [
                        team,
                        contract,
                        period,
                        source,
                        method,
                        affected,
                        interpretations[method],
                    ],
                )
            )
        )

    if not employee_alignment.empty:
        missing = employee_alignment.loc[
            employee_alignment.method.ne("observed_employee")
        ]
        for (team, period, source, method), group in missing.groupby(
            ["team", "period", "source_period", "method"], dropna=False
        ):
            assume(
                team,
                "permanent",
                period,
                source,
                method,
                f"salary and worked fraction for {len(group)} profile IDs",
            )
    for team in policy["teams"]:
        for side, period in enumerate(periods):
            other = 1 - side
            reference = refs.get((period, team), {})
            team_state = states[side]["teams"][team]
            if team_state is None:
                source = None
                method = "business_reference"
                candidate = reference.get("team_state")
                if candidate is None:
                    if missing_economics == "require_reference":
                        raise ValueError(
                            f"{period:%Y-%m}/{team}: missing team; "
                            "supply a team_state reference"
                        )
                    candidate = observed[other]["teams"][team]
                    source, method = periods[other], "carry_observed"
                if candidate is None:
                    raise ValueError(f"No observed team economics: {team}")
                team_state = deepcopy(candidate)
                validate_team(team_state, registry)
                states[side]["teams"][team] = team_state
                assume(
                    team, "all", period, source, method, "complete team state"
                )
                context = {
                    "period": period,
                    "team": team,
                    "diagnostics": records[side],
                }
                for index, contract in enumerate(CONTRACTS):
                    log_fallback(
                        context,
                        None,
                        contract,
                        "contract_type_mix",
                        method,
                        team_state["contract_type_mix"][index],
                        source,
                    )
                    value = team_state[CONTRACT_KEYS[contract]]
                    if value is not None:
                        log_completed_economics(
                            records[side],
                            period,
                            team,
                            contract,
                            value,
                            method,
                            source,
                            registry,
                        )
            # Explicit category references may refine a copied whole team,
            # but may never replace an actually observed category.
            original_team = observed[side]["teams"][team]
            for contract in CONTRACTS:
                key = CONTRACT_KEYS[contract]
                original_value = (
                    None if original_team is None else original_team[key]
                )
                if original_value is None and contract in reference:
                    value = deepcopy(reference[contract])
                    if value is None:
                        raise ValueError(
                            "An explicit reference cannot be None"
                        )
                    team_state[key] = value
                    log_completed_economics(
                        records[side],
                        period,
                        team,
                        contract,
                        value,
                        "business_reference",
                        None,
                        registry,
                    )
                    assume(
                        team, contract, period, None, "business_reference", key
                    )
        # Complete category economics after both team-mix vectors exist.
        for index, contract in enumerate(CONTRACTS):
            key = CONTRACT_KEYS[contract]
            active = any(
                state["teams"][team]["contract_type_mix"][index] > 0
                for state in states
            )
            if not active:
                assume(team, contract, None, None, "inactive", key)
                continue
            for side, period in enumerate(periods):
                target = states[side]["teams"][team]
                if target[key] is not None:
                    continue
                if missing_economics == "require_reference":
                    raise ValueError(
                        f"{period:%Y-%m}/{team}: missing {contract}; "
                        "supply an explicit reference"
                    )
                # Only observed donor economics are eligible for copying.
                donor = observed[1 - side]["teams"][team]
                value = None if donor is None else donor[key]
                if value is None:
                    raise ValueError(
                        f"{team}/{contract}: no observed donor; "
                        "supply an explicit reference"
                    )
                target[key] = deepcopy(value)
                source = periods[1 - side]
                log_completed_economics(
                    records[side],
                    period,
                    team,
                    contract,
                    value,
                    "carry_observed",
                    source,
                    registry,
                )
                assume(team, contract, period, source, "carry_observed", key)
        for state in states:
            permanent = state["teams"][team]["permanent"]
            if permanent is not None:
                ensure_same_policy(
                    permanent["_policy"], policy["permanent"][team]
                )
    for state in states:
        validate_organization(state, registry)
    aligned_comparison(*states, registry)
    return {
        "states": tuple(states),
        "periods": periods,
        "fallbacks": tuple(
            pd.DataFrame(items, columns=DIAGNOSTIC_COLUMNS)
            for items in records
        ),
        "coverage": tuple(item[2] for item in built),
        "assumptions": pd.DataFrame(assumptions, columns=ASSUMPTION_COLUMNS),
        "employee_alignment": employee_alignment,
    }
