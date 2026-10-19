"""Costing for the standalone workforce cost engine."""

import numpy as np

from .common import (
    CONTRACT_KEYS,
    CONTRACTS,
    PARENT,
    TEAM_PARENT,
    UNIT_KEYS,
)
from .kernels import (
    ensure_same_policy,
    validate_policy,
)
from .registry import (
    DRIVERS,
    drivers_at,
)
from .validation import validate_value


def validate_permanent(state, registry=DRIVERS):
    """Validate profiles, employee vectors, payment matrices and policy.

    Aggregate-only salary/time references cannot preserve identity
    pairing. Rebuild legacy references using the revised state builder.
    """
    if "_employees" not in state or "within_cell_workforce_mix" not in state:
        raise ValueError(
            "Legacy permanent state/reference: rebuild with build_state; "
            "aligned employee vectors and membership profiles are required"
        )
    axis = state["_employees"]
    if (
        not isinstance(axis, tuple)
        or not axis
        or len(set(axis)) != len(axis)
        or any(not isinstance(eid, str) or not eid.strip() for eid in axis)
    ):
        raise ValueError("Employee axis must contain unique nonempty IDs")
    validate_policy(state["_policy"])
    for driver in drivers_at(PARENT, registry):
        validate_value(
            state[driver.name],
            driver.kind,
            state["_policy"],
            employee_count=len(axis),
        )


def validate_team_units(state, registry=DRIVERS):
    """Require finite signed rates wherever contract shares are
    positive.
    """
    validate_value(state["contract_type_mix"], "contract_probability")
    for index, key in enumerate(UNIT_KEYS):
        value = state[key]
        if value is None:
            if state["contract_type_mix"][index] > 0:
                raise ValueError(f"Positive contract share needs {key}")
        else:
            validate_value(value, "scalar_signed")


def validate_team(state, registry=DRIVERS):
    """Validate a team state, allowing unavailable zero-share
    economics.
    """
    validate_value(state["contract_type_mix"], "contract_probability")
    for index, key in enumerate(CONTRACT_KEYS.values()):
        value = state[key]
        if value is None:
            if state["contract_type_mix"][index] > 0:
                raise ValueError(f"Positive contract share needs {key}")
        elif key == "permanent":
            validate_permanent(value, registry)
        else:
            validate_value(value, "scalar_signed")


def validate_organization(state, registry=DRIVERS):
    """Validate exposure, team shares and all available team
    economics.
    """
    axis = state["_teams"]
    if not axis or len(set(axis)) != len(axis):
        raise ValueError("Team axis must be nonempty and unique")
    if set(axis) != set(state["teams"]):
        raise ValueError("Team states do not match the team axis")
    validate_value(state["workforce_scale"], "scalar_nonnegative")
    validate_value(state["team_mix"], "team_probability", team_count=len(axis))
    for index, team in enumerate(axis):
        team_state = state["teams"][team]
        if team_state is None:
            if state["team_mix"][index] > 0:
                raise ValueError(
                    f"Positive team share needs economics: {team}"
                )
        else:
            validate_team(team_state, registry)


def permanent_cell_costs(state, registry=DRIVERS):
    """Return composition-weighted signed payroll for each permanent cell.

    Registered eligible contributions receive the destination
    coefficient
    once. This shared cell formula also supports salary-impact audits.
    """
    validate_permanent(state, registry)
    joint = state["regional_mix"][:, None] * state["grade_mix_within_region"]
    cell_pay = np.zeros_like(joint)
    for driver in drivers_at(PARENT, registry):
        if driver.cell_cost is None:
            continue
        contribution = np.asarray(driver.cell_cost(state), dtype=float)
        validate_value(contribution, "cell_signed", state["_policy"])
        if driver.apply_regional_coefficient:
            contribution = (
                contribution
                * state["_policy"]["regional_coefficients"][:, None]
            )
        cell_pay += contribution
    result = joint * cell_pay
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite permanent cell costs")
    return result


def permanent_cost(permanent_state, registry=DRIVERS):
    """Return signed currency per permanent exposure, without using
    SHAP.
    """
    return float(permanent_cell_costs(permanent_state, registry).sum())


def team_units(state, registry=DRIVERS):
    """Replace a team's permanent state with its scalar unit cost."""
    permanent = state["permanent"]
    return {
        "contract_type_mix": state["contract_type_mix"],
        PARENT: (
            None if permanent is None else permanent_cost(permanent, registry)
        ),
        "temp_reimbursement": state["temp_reimbursement"],
        "outsource_reimbursement": state["outsource_reimbursement"],
    }


def contract_cost_terms(units, registry=DRIVERS):
    """Return contract contributions; unavailable zero-share terms are
    zero.

    A zero contribution does not imply a zero unit rate. A planning
    state
    that activates an unavailable category must supply its economics.
    """
    validate_team_units(units, registry)
    result = np.zeros(len(CONTRACTS))
    for index, key in enumerate(UNIT_KEYS):
        if units[key] is not None:
            result[index] = units["contract_type_mix"][index] * units[key]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite contract cost")
    return result


def team_unit_from_units(units, registry=DRIVERS):
    """Return blended team currency per exposure from scalar unit
    costs.
    """
    return float(contract_cost_terms(units, registry).sum())


def team_unit_cost(team_state, registry=DRIVERS):
    """Return blended signed team unit cost independently of
    comparison.
    """
    validate_team(team_state, registry)
    return team_unit_from_units(team_units(team_state, registry), registry)


def organization_cost_terms(scale, mix, units):
    """Return signed team costs for an organizational exposure and
    mix.
    """
    validate_value(scale, "scalar_nonnegative")
    validate_value(mix, "team_probability", team_count=len(units))
    result = np.zeros(len(units))
    for index, unit in enumerate(units):
        if unit is None:
            if mix[index] > 0:
                raise ValueError("Positive team share needs unit economics")
        else:
            validate_value(unit, "scalar_signed")
            result[index] = scale * mix[index] * unit
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite organizational costs")
    return result


def cost(workforce_state, registry=DRIVERS):
    """Evaluate a workforce state without history, imputation or SHAP.

    Returns
    -------
    float
        Signed total labor-cost currency. Zero-share unavailable teams
        and categories contribute zero; positive-share unknowns fail.
    """
    validate_organization(workforce_state, registry)
    units = [
        (
            None
            if workforce_state["teams"][team] is None
            else team_unit_cost(workforce_state["teams"][team], registry)
        )
        for team in workforce_state["_teams"]
    ]
    return float(
        organization_cost_terms(
            workforce_state["workforce_scale"],
            workforce_state["team_mix"],
            units,
        ).sum()
    )


def local_cost_from_units(state):
    """Evaluate local scale times unit economics using the shared
    engine.
    """
    return float(
        organization_cost_terms(
            state["workforce_scale"], np.ones(1), [state[TEAM_PARENT]]
        )[0]
    )


def aligned_comparison(a, b, registry=DRIVERS):
    """Require completed, aligned endpoint economics before SHAP."""
    validate_organization(a, registry)
    validate_organization(b, registry)
    if a["_teams"] != b["_teams"]:
        raise ValueError("A/B team axes differ")
    for team in a["_teams"]:
        left, right = a["teams"][team], b["teams"][team]
        if left is None or right is None:
            raise ValueError("Use build_comparison_states for absent teams")
        for index, key in enumerate(CONTRACT_KEYS.values()):
            active = (
                max(
                    left["contract_type_mix"][index],
                    right["contract_type_mix"][index],
                )
                > 0
            )
            if active and (left[key] is None or right[key] is None):
                raise ValueError("Complete missing category economics first")
            if active and key == "permanent":
                ensure_same_policy(left[key]["_policy"], right[key]["_policy"])
                if left[key]["_employees"] != right[key]["_employees"]:
                    raise ValueError("A/B permanent employee axes differ")
