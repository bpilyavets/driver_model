"""Inputs for the standalone workforce cost engine."""

import numpy as np
import pandas as pd

from .common import (
    CONTRACTS,
    PARENT,
)
from .registry import (
    DRIVERS,
    PRODUCTION_COLUMNS,
    PRODUCTION_CONTRACTS,
    drivers_at,
)


def finite_numeric(values, name, *, lower=None, strict=False):
    """Require finite numeric values and an optional lower bound.

    Empty applicable populations are allowed. Money has no lower bound;
    workforce quantities and salary parameters opt into a bound.
    """
    if len(values) == 0:
        return
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError(f"{name} must contain numeric values")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite, nonmissing values")
    if lower is not None:
        invalid = values <= lower if strict else values < lower
        if invalid.any():
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{name} must be {relation} {lower}")


def validate_record_keys(frame):
    """Validate record keys and one team per employee-month.

    Both production adaptation and canonical validation use this grain.
    Multiple employment types are valid; repeated rows of the same type
    and conflicting teams are errors. No records are combined or removed.
    """
    required = {"period", "employee_id", "team", "contract_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing record key columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No employee-month-employment-type records")
    for name in ("employee_id", "team", "contract_type"):
        valid = frame[name].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not valid.all():
            raise ValueError(f"{name} must contain nonempty strings")
    if not pd.api.types.is_datetime64_any_dtype(frame.period):
        raise ValueError("period must contain datetime month starts")
    if frame.period.isna().any() or frame.period.dt.tz is not None:
        raise ValueError("period must be nonmissing and timezone-naive")
    month_starts = frame.period.dt.to_period("M").dt.to_timestamp()
    if not frame.period.eq(month_starts).all():
        raise ValueError("period must contain month starts")
    if not frame.contract_type.isin(CONTRACTS).all():
        raise ValueError("Unknown contract type")
    if frame.duplicated(["period", "employee_id", "contract_type"]).any():
        raise ValueError("Duplicate employee-month-employment-type record")
    teams = frame.groupby(["period", "employee_id"]).team.nunique()
    if teams.gt(1).any():
        raise ValueError("Each employee-month must have one team across types")


def resolve_exposure(frame, *, supplied=None):
    """Assign workforce exposure independently of payroll and worked time.

    Parameters
    ----------
    frame : pandas.DataFrame
        Canonical record identifiers, unique by month/employee/contract.
    supplied : pandas.Series or None
        Optional source exposure aligned to frame.index. Otherwise each
        person's observed employment types share one unit equally.

    Returns
    -------
    pandas.Series
        Positive exposure at most one per row. Supplied values are not
        normalized, even when a person's monthly total differs from one.

    Raises
    ------
    ValueError
        Keys, alignment or exposure values are invalid. Invalid supplied
        values never fall back to equal splitting.
    """
    validate_record_keys(frame)
    if supplied is None:
        counts = frame.groupby(["period", "employee_id"])[
            "contract_type"
        ].transform("nunique")
        values = 1.0 / counts
    else:
        if not isinstance(supplied, pd.Series) or not supplied.index.equals(
            frame.index
        ):
            raise ValueError("Supplied exposure must align with record rows")
        if pd.api.types.is_bool_dtype(supplied):
            raise ValueError("exposure must be numeric, not boolean")
        values = supplied.copy()
    finite_numeric(values, "exposure", lower=0, strict=True)
    if values.gt(1).any():
        raise ValueError("exposure must be in (0, 1]")
    return values.astype(float).rename("exposure")


def validate_input(frame, registry=DRIVERS):
    """Validate canonical employee-month-contract rows without changing them.

    Monetary amounts may be signed. Base salary, regional coefficients,
    exposure and worked-time factors retain their quantity constraints.
    Inapplicable contract-specific values may be missing.
    """
    required = {
        "period",
        "employee_id",
        "team",
        "contract_type",
        "exposure",
        "region",
        "grade",
        "regional_coefficient",
        "contract_reimbursement",
        "source_cost",
    }
    for driver in drivers_at(PARENT, registry):
        required.update(driver.columns)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")
    validate_record_keys(frame)
    finite_numeric(frame.exposure, "exposure", lower=0, strict=True)
    if frame.exposure.gt(1).any():
        raise ValueError("exposure must be in (0, 1]")
    finite_numeric(frame.source_cost, "source_cost")
    permanent = frame.loc[frame.contract_type.eq("permanent")]
    for driver in drivers_at(PARENT, registry):
        for column in driver.columns:
            finite_numeric(permanent[column], column)
    for column in ("base_salary", "regional_coefficient"):
        finite_numeric(permanent[column], column, lower=0, strict=True)
    finite_numeric(permanent.worked_fraction, "worked_fraction", lower=0)
    if "base_pay_factor" in permanent:
        finite_numeric(
            permanent.base_pay_factor, "base_pay_factor", lower=0, strict=True
        )
    if permanent[["region", "grade"]].isna().any().any():
        raise ValueError("Missing permanent region or grade")
    if not permanent.region.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Invalid permanent region")
    schedules = permanent.groupby("region").regional_coefficient.nunique()
    if schedules.gt(1).any():
        raise ValueError("Regional coefficients differ for an aligned region")
    other = frame.loc[~frame.contract_type.eq("permanent")]
    finite_numeric(other.contract_reimbursement, "contract_reimbursement")


def month_start(value):
    """Return a timezone-naive month start for an explicit month
    selector.
    """
    if value is None or pd.isna(value):
        raise ValueError("Choose both comparison months in the input cell")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        raise ValueError("Comparison months must be timezone-naive")
    return timestamp.to_period("M").to_timestamp()


def band_name(value):
    """Name a regional coefficient band without implying geography."""
    return "region_" + str(float(value))


def employee_payroll_reference(frame):
    """Reconstruct employee payroll independently of aggregate costing.

    Returns
    -------
    pandas.Series
        Signed row payroll. Eligible payments are pre-regional per
        exposure. Salary times worked fraction uses base_pay_factor
        (default one); source totals include exposure exactly once.
    """
    permanent = frame.contract_type.eq("permanent")
    result = frame.contract_reimbursement.astype(float).copy()
    rows = frame.loc[permanent]
    if rows.empty:
        return result * frame.exposure
    result.loc[permanent] = (
        rows.regional_coefficient
        * (
            rows.base_salary
            * rows.worked_fraction
            * rows.get("base_pay_factor", 1.0)
            + rows.sick_leave_pay
            + rows.vacation_pay
            + rows.holiday_pay
            + rows.night_shift_pay
        )
        + rows.bonus_kpi
        + rows.overtime_pay
        + rows.regional_coefficient
        * (rows.sharing_bonus + rows.quarterly_bonus)
        + rows.other_pay
    )
    return result * frame.exposure


def adapt_production(
    frame, *, source_cost_column=None, exposure_column=None, registry=DRIVERS
):
    """Adapt actual month/employee/employment-type payroll to model economics.

    Parameters
    ----------
    frame : pandas.DataFrame
        Production columns listed in PRODUCTION_COLUMNS. Each row holds
        actual amounts for one employment type. Salary times worked
        fraction already gives earned base pay before regional adjustment.
        Extra registered payment columns use their canonical names.
    source_cost_column : str or None
        Independent actual payroll total for every row. Otherwise use
        the original production formula before exposure normalization.
    exposure_column : str or None
        Explicit source exposure in (0, 1], used without normalization.
        None splits each person's monthly unit equally across types.
    registry : tuple of Driver
        Business drivers, including locally registered payments.

    Returns
    -------
    pandas.DataFrame
        Validated canonical rows. Payments/reimbursements are per exposure;
        salary/time stay observed, with base_pay_factor converting earned
        base pay to per-exposure economics. Source totals and caller data
        are unchanged. Exposure policy belongs only to this adapter.

    Raises
    ------
    ValueError
        Columns, keys, team consistency, exposure or applicable values
        are invalid. Duplicate employment types are never aggregated.
    """
    required = set(PRODUCTION_COLUMNS)
    canonical = set(PRODUCTION_COLUMNS.values())
    extra = {
        column
        for driver in drivers_at(PARENT, registry)
        for column in driver.columns
    } - canonical
    required.update(extra)
    for column in (source_cost_column, exposure_column):
        if column is not None:
            required.add(column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing production columns: {sorted(missing)}")
    out = frame[list(PRODUCTION_COLUMNS) + sorted(extra)].copy()
    out = out.rename(columns=PRODUCTION_COLUMNS)
    if not out.contract_type.isin(PRODUCTION_CONTRACTS).all():
        raise ValueError("Expected motivation_type values: ТК, ГПД, ПКЦ")
    out["contract_type"] = out.contract_type.map(PRODUCTION_CONTRACTS)
    out["period"] = pd.to_datetime(out.period, errors="raise")
    supplied = None if exposure_column is None else frame[exposure_column]
    exposure = resolve_exposure(out, supplied=supplied)
    # Reconstruct actual row payroll before converting to per-exposure values.
    out["exposure"] = 1.0
    out["base_pay_factor"] = 1.0
    permanent = out.contract_type.eq("permanent")
    factors = out.loc[permanent, "regional_coefficient"]
    finite_numeric(factors, "reg_coef", lower=0, strict=True)
    out["region"] = None
    out.loc[permanent, "region"] = factors.map(band_name)
    out["contract_reimbursement"] = np.where(
        out.contract_type.eq("temporary"), out.gpd_pay, out.pkc_pay
    ).astype(float)
    if source_cost_column is None:
        out["source_cost"] = employee_payroll_reference(out)
        out["reference_basis"] = np.where(
            permanent,
            "reconstructed permanent payroll",
            "supplied contract payroll",
        )
        if extra:
            raise ValueError(
                "Provide an independent source_cost_column when adding "
                "payments beyond the production payroll formula"
            )
    else:
        out["source_cost"] = frame[source_cost_column].copy()
        out["reference_basis"] = "independent supplied source cost"
    validate_input(out, registry)
    out["exposure"] = exposure
    out.loc[permanent, "base_pay_factor"] = 1.0 / exposure.loc[permanent]
    payment_columns = {
        column
        for driver in drivers_at(PARENT, registry)
        if driver.kind == "cell_signed"
        for column in driver.columns
    }
    if permanent.any():
        for column in payment_columns:
            out[column] = out[column].astype(float)
            out.loc[permanent, column] /= exposure.loc[permanent]
    out.loc[~permanent, "contract_reimbursement"] /= exposure.loc[~permanent]
    validate_input(out, registry)
    return out


def normalize_regional_inputs(
    frame, *, already_regionalized=(), registry=DRIVERS
):
    """Remove regional adjustment once from explicitly identified columns.

    Signed payments retain their signs. Only eligible permanent values
    are divided by the positive source-region coefficient; source_cost
    is never modified. Call before state aggregation, using a canonical
    frame whose source reference already has the correct monthly total.
    """
    columns = tuple(already_regionalized)
    eligible = {
        column
        for driver in drivers_at(PARENT, registry)
        if driver.apply_regional_coefficient
        for column in driver.columns
    }
    if len(columns) != len(set(columns)) or not set(columns) <= eligible:
        raise ValueError("Identify unique regionally eligible columns")
    out = frame.copy()
    permanent = out.contract_type.eq("permanent")
    factors = out.loc[permanent, "regional_coefficient"]
    finite_numeric(factors, "regional_coefficient", lower=0, strict=True)
    for column in columns:
        if column not in out:
            raise ValueError(f"Missing normalization column: {column}")
        values = out.loc[permanent, column]
        finite_numeric(values, column)
        out[column] = out[column].astype(float)
        out.loc[permanent, column] = values / factors
    validate_input(out, registry)
    return out
