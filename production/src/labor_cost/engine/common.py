"""Common for the standalone workforce cost engine."""

import numpy as np

RTOL = 1e-10


ATOL = 1e-6


PROB_ATOL = 1e-12


CONTRACTS = ("permanent", "temporary", "outsourced")


PARENT = "permanent_unit_cost"


TEAM_PARENT = "team_economics"


FALLBACK_ORDER = ("grade", "region", "global")


MAX_EXACT_DRIVERS = 15


SALARY_KEYS = (
    "within_cell_workforce_mix",
    "underlying_salary_level",
    "worked_hours",
)


CONTRACT_KEYS = {
    "permanent": "permanent",
    "temporary": "temp_reimbursement",
    "outsourced": "outsource_reimbursement",
}


UNIT_KEYS = (PARENT, "temp_reimbursement", "outsource_reimbursement")


def check(name, actual, expected):
    """Assert numerical agreement and return a compact validation
    record.
    """
    np.testing.assert_allclose(
        actual, expected, rtol=RTOL, atol=ATOL, err_msg=name
    )
    error = float(
        np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0)
    )
    return {"check": name, "maximum_absolute_error": error, "passed": True}


DIAGNOSTIC_COLUMNS = [
    "period",
    "team",
    "region",
    "grade",
    "variable",
    "fallback_level",
    "fallback_value",
    "source_period",
    "employee_id",
]
