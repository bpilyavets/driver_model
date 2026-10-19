"""Registry for the standalone workforce cost engine."""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .common import (
    PARENT,
    TEAM_PARENT,
)
from .kernels import (
    cell_matrix,
    employee_profiles,
    employee_vector,
    grade_distribution,
    reimbursement,
    salary_cell_contribution,
    weighted_mean,
)

PRODUCTION_COLUMNS = {
    "period": "period",
    "mdm_employee_rk_hash": "employee_id",
    "lvl7_mapped_management_unit_nm": "team",
    "motivation_type": "contract_type",
    "grade": "grade",
    "reg_coef": "regional_coefficient",
    "base_stavka": "base_salary",
    "vyrabotka_percent": "worked_fraction",
    "sum_pay_bl_reg_base": "sick_leave_pay",
    "sum_pay_otpusk_reg_base": "vacation_pay",
    "sum_pay_prazdnik_reg_base": "holiday_pay",
    "sum_pay_oklad_night_reg_base": "night_shift_pay",
    "bonus_kpi_comp": "bonus_kpi",
    "bonus_overtime_comp": "overtime_pay",
    "bonus_sharing_comp_reg_base": "sharing_bonus",
    "bonus_quarterly_comp_reg_base": "quarterly_bonus",
    "sum_pay_compacted": "other_pay",
    "gpd_pay": "gpd_pay",
    "pkc_pay": "pkc_pay",
}


PRODUCTION_CONTRACTS = {
    "ТК": "permanent",
    "ГПД": "temporary",
    "ПКЦ": "outsourced",
}


PAYMENT_COLUMNS = (
    "sick_leave_pay",
    "vacation_pay",
    "holiday_pay",
    "night_shift_pay",
    "bonus_kpi",
    "overtime_pay",
    "sharing_bonus",
    "quarterly_bonus",
    "other_pay",
)


@dataclass(frozen=True)
class Driver:
    """Describe one atomic business driver and its payroll contribution.

    Monetary kinds permit signed values; salary/time and distributions
    retain their own constraints. Builders consume a team-period
    context.
    cell_cost returns an unweighted pre-adjustment region/grade matrix.
    """

    name: str
    label: str
    parent: str
    builder: Callable | None = None
    kind: str = "scalar_signed"
    columns: tuple = ()
    cell_cost: Callable | None = None
    apply_regional_coefficient: bool = False
    dependencies: tuple = ()


def payment_driver(name, label, column, *, apply_regional_coefficient=False):
    """Register one signed additive payment and its regional eligibility.

    The source column supplies per-exposure currency. Its builder uses
    the standard same-period sparse-cell estimator; the cost callback
    returns that completed matrix. Reporting derives its label here.
    """
    return Driver(
        name,
        label,
        PARENT,
        lambda c: cell_matrix(c, name, lambda r: weighted_mean(r, r[column])),
        "cell_signed",
        (column,),
        lambda s: s[name],
        apply_regional_coefficient,
    )


DRIVERS = (
    Driver(
        "workforce_scale",
        "Workforce scale",
        "total",
        kind="scalar_nonnegative",
    ),
    Driver("team_mix", "Team mix", "total", kind="team_probability"),
    Driver(TEAM_PARENT, "Team economics", "total"),
    Driver(
        "contract_type_mix",
        "Contract type mix",
        TEAM_PARENT,
        lambda c: c["contract_exposure"] / c["contract_exposure"].sum(),
        "contract_probability",
    ),
    Driver(
        "temp_reimbursement",
        "Temp reimbursement",
        TEAM_PARENT,
        lambda c: reimbursement(c, "temporary"),
    ),
    Driver(
        "outsource_reimbursement",
        "Outsource reimbursement",
        TEAM_PARENT,
        lambda c: reimbursement(c, "outsourced"),
    ),
    Driver(PARENT, "Permanent unit cost", TEAM_PARENT),
    Driver(
        "regional_mix",
        "Regional coefficient-band mix",
        PARENT,
        lambda c: c["cell_exposure"].sum(axis=1) / c["cell_exposure"].sum(),
        "region_probability",
    ),
    Driver(
        "grade_mix_within_region",
        "Grade composition within region",
        PARENT,
        lambda c: grade_distribution(c),
        "grade_probability",
    ),
    Driver(
        "within_cell_workforce_mix",
        "Within-cell workforce mix — base pay",
        PARENT,
        lambda c: employee_profiles(c),
        "membership_probability",
    ),
    Driver(
        "underlying_salary_level",
        "Base salary changes",
        PARENT,
        lambda c: employee_vector(c, "base_salary"),
        "employee_positive",
        ("base_salary",),
        lambda s: salary_cell_contribution(s),
        True,
    ),
    Driver(
        "worked_hours",
        "Proportion of planned hours worked",
        PARENT,
        lambda c: employee_vector(c, "worked_fraction"),
        "employee_nonnegative",
        ("worked_fraction",),
    ),
    payment_driver(
        "sick_leave_pay",
        "Sick leave pay",
        "sick_leave_pay",
        apply_regional_coefficient=True,
    ),
    payment_driver(
        "vacation_pay",
        "Vacation pay",
        "vacation_pay",
        apply_regional_coefficient=True,
    ),
    payment_driver(
        "holiday_pay",
        "Holiday pay",
        "holiday_pay",
        apply_regional_coefficient=True,
    ),
    payment_driver(
        "night_shift_pay",
        "Night-shift pay",
        "night_shift_pay",
        apply_regional_coefficient=True,
    ),
    payment_driver("bonus_kpi", "Bonus / KPI", "bonus_kpi"),
    payment_driver("overtime", "Overtime", "overtime_pay"),
    payment_driver(
        "sharing_bonus",
        "Sharing bonus",
        "sharing_bonus",
        apply_regional_coefficient=True,
    ),
    payment_driver(
        "quarterly_bonus",
        "Quarterly bonus",
        "quarterly_bonus",
        apply_regional_coefficient=True,
    ),
    payment_driver("other_pay", "Other bundled pay", "other_pay"),
)


def drivers_at(parent, registry=DRIVERS):
    """Validate registry metadata and return drivers under one
    parent.
    """
    names = [d.name for d in registry]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate driver names")
    for d in registry:
        if not isinstance(d.apply_regional_coefficient, bool) or (
            d.apply_regional_coefficient and d.cell_cost is None
        ):
            raise ValueError(
                "Regional adjustment needs a boolean rule "
                "and a cost contribution"
            )
    return tuple(d for d in registry if d.parent == parent)


def registry_table(registry=DRIVERS):
    """Describe registered source columns, constraints and
    adjustment.
    """
    return pd.DataFrame(
        [
            dict(
                driver=d.name,
                label=d.label,
                parent=d.parent,
                source_columns=", ".join(d.columns),
                validation=d.kind,
                payroll_contribution=d.cell_cost is not None,
                apply_regional_coefficient=d.apply_regional_coefficient,
            )
            for d in registry
        ]
    )
