"""Kernels for the standalone workforce cost engine."""

from itertools import product

import numpy as np

from .common import (
    DIAGNOSTIC_COLUMNS,
    SALARY_KEYS,
)
from .validation import (
    validate_value,
)


def weighted_mean(rows, values):
    """Return an exposure-weighted mean, preserving signed payments."""
    return float(
        np.average(np.asarray(values, dtype=float), weights=rows.exposure)
    )


def validate_policy(policy):
    """Validate positive coefficients and unique permanent-state
    axes.
    """
    regions, grades = policy["regions"], policy["grades"]
    coefficients = np.asarray(policy["regional_coefficients"], dtype=float)
    if not regions or not grades:
        raise ValueError("Permanent policy needs nonempty region/grade axes")
    if len(set(regions)) != len(regions) or len(set(grades)) != len(grades):
        raise ValueError("Permanent policy axes must be unique")
    if coefficients.shape != (len(regions),):
        raise ValueError("Regional coefficients do not match region axes")
    if not np.isfinite(coefficients).all() or (coefficients <= 0).any():
        raise ValueError("Regional coefficients must be finite and positive")


def ensure_same_policy(a, b):
    """Reject changed A/B axes or coefficient schedules."""
    validate_policy(a)
    validate_policy(b)
    aligned = (
        a["regions"] == b["regions"]
        and a["grades"] == b["grades"]
        and np.array_equal(
            a["regional_coefficients"], b["regional_coefficients"]
        )
    )
    if not aligned:
        raise ValueError("A/B axes or fixed regional policy differ")


def log_fallback(
    context, region, grade, variable, level, value, source_period=None
):
    """Record one imputed value with team, period and source
    provenance.
    """
    context["diagnostics"].append(
        dict(
            zip(
                DIAGNOSTIC_COLUMNS,
                [
                    context["period"],
                    context["team"],
                    region,
                    grade,
                    variable,
                    level,
                    float(value),
                    source_period,
                ],
            )
        )
    )


def fallback_populations(context):
    """Yield each cell's first supported same-team, same-period population.

    The selected population is shared by salary membership profiles and
    payment estimators. No employee values are replaced with zero.
    """
    permanent = context["perm"]
    for ri, region in enumerate(context["policy"]["regions"]):
        for gi, grade in enumerate(context["policy"]["grades"]):
            populations = {
                "cell": permanent.loc[
                    permanent.region.eq(region) & permanent.grade.eq(grade)
                ],
                "grade": permanent.loc[permanent.grade.eq(grade)],
                "region": permanent.loc[permanent.region.eq(region)],
                "global": permanent,
            }
            for level in ("cell",) + context["fallback_order"]:
                rows = populations[level]
                if not rows.empty:
                    yield ri, gi, region, grade, level, rows
                    break
            else:
                raise ValueError(
                    f"No fallback support: {context['team']}, {region}/{grade}"
                )


def cell_matrix(context, variable, estimator):
    """Estimate signed payment cells on supported within-team populations."""
    result = np.empty_like(context["cell_exposure"], dtype=float)
    for ri, gi, region, grade, level, rows in fallback_populations(context):
        value = estimator(rows)
        if not np.isfinite(value):
            raise ValueError(f"Invalid estimator for {variable}")
        result[ri, gi] = value
        if level != "cell":
            log_fallback(
                context,
                region,
                grade,
                variable,
                level,
                value,
                context["period"],
            )
    return result


def employee_vector(context, column):
    """Return observed employee values on the state's local employee axis."""
    return (
        context["perm"]
        .set_index("employee_id")
        .loc[list(context["employees"]), column]
        .to_numpy(dtype=float)
    )


def employee_profiles(context):
    """Build sparse normalized exposure weights for each permanent cell.

    Keys are region/grade index pairs. Each value holds integer employee
    indices and probability weights. Fallback selects a complete profile
    from the same period and team, preserving salary/time pairing.
    """
    positions = {eid: i for i, eid in enumerate(context["employees"])}
    profiles = {}
    for ri, gi, region, grade, level, rows in fallback_populations(context):
        rows = rows.sort_values("employee_id")
        profiles[ri, gi] = {
            "indices": rows.employee_id.map(positions).to_numpy(dtype=int),
            "weights": rows.exposure.to_numpy(dtype=float)
            / rows.exposure.sum(),
            "base_pay_factors": (
                rows["base_pay_factor"].to_numpy(dtype=float)
                if "base_pay_factor" in rows
                else np.ones(len(rows), dtype=float)
            ),
        }
        if level != "cell":
            # One population provenance row; actual sparse weights remain
            # inspectable in states and in the membership diagnostic table.
            log_fallback(
                context,
                region,
                grade,
                "within_cell_workforce_mix",
                level,
                1.0,
                context["period"],
            )
    return profiles


def profile_mean(state, values, *, base_pay=False):
    """Return cell means, optionally converting earned base pay per exposure.

    Ordinary descriptive means use normalized membership weights only.
    Base pay additionally uses each profile's positive conversion factors,
    which switch with membership. Older per-exposure profiles omit factors
    and retain the established factor-one interpretation.
    """
    shape = (len(state["_policy"]["regions"]), len(state["_policy"]["grades"]))
    result = np.empty(shape, dtype=float)
    for cell, profile in state["within_cell_workforce_mix"].items():
        weights = profile["weights"]
        if base_pay:
            weights = weights * np.asarray(
                profile.get("base_pay_factors", np.ones(len(weights))),
                dtype=float,
            )
        result[cell] = np.dot(weights, values[profile["indices"]])
    return result


def salary_cell_contribution(state):
    """Return pre-regional salary per exposure, preserving employee pairing.

    This formula is also the source for the comparison-local eight-case
    cache. Historical matching and SHAP are not needed to cost a state.
    """
    cache = state.get("_salary_cache")
    if cache is not None:
        key = tuple(id(state[name]) for name in SALARY_KEYS)
        if key in cache:
            return cache[key]
    return profile_mean(
        state,
        state["underlying_salary_level"] * state["worked_hours"],
        base_pay=True,
    )


def prepare_salary_cache(a, b):
    """Precompute the eight salary combinations using the cost contribution.

    Keys refer to endpoint object identities, held alive by the caller.
    Use only inside one synchronous game/audit; never store this cache in
    returned workforce states or reuse it after changing their arrays.
    """
    cache = {}
    for bits in product((0, 1), repeat=3):
        hybrid = dict(a)
        hybrid.pop("_salary_cache", None)
        hybrid.update(
            {key: (a, b)[bit][key] for key, bit in zip(SALARY_KEYS, bits)}
        )
        key = tuple(id(hybrid[name]) for name in SALARY_KEYS)
        value = salary_cell_contribution(hybrid)
        validate_value(value, "cell_signed", hybrid["_policy"])
        cache[key] = value
    return cache


def grade_distribution(context):
    """Build conditional grade rows, including absent-region
    fallback.
    """
    counts = context["cell_exposure"]
    marginal = counts.sum(axis=0) / counts.sum()
    rows = []
    for region, row in zip(context["policy"]["regions"], counts):
        probabilities = row / row.sum() if row.sum() > 0 else marginal.copy()
        if row.sum() == 0:
            for grade, value in zip(
                context["policy"]["grades"], probabilities
            ):
                log_fallback(
                    context,
                    region,
                    grade,
                    "grade_mix_within_region",
                    "team_period_grade_distribution",
                    value,
                    context["period"],
                )
        rows.append(probabilities)
    return np.asarray(rows)


def reimbursement(context, contract):
    """Return observed signed reimbursement per exposure, or None."""
    rows = context["rows"].loc[context["rows"].contract_type.eq(contract)]
    if rows.empty:
        return None
    return weighted_mean(rows, rows.contract_reimbursement)
