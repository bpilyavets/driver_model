"""Load the notebook's implementation without executing its user workflow."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "labor_cost_decomposition.ipynb"


@pytest.fixture(scope="session")
def model():
    """Execute tagged implementation cells as the single source of logic."""
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {"__name__": "labor_notebook_tests"}
    for cell in notebook["cells"]:
        if "implementation" in cell.get("metadata", {}).get("tags", []):
            exec("".join(cell["source"]), namespace)
    return namespace


def payroll_row(
    period,
    employee,
    team="Alpha",
    contract="ТК",
    base=100.0,
    coefficient=1.0,
    worked=1.0,
    grade="G1",
    rate=30.0,
    **payments
):
    """Construct one explicit production-shaped record with known payroll."""
    row = {
        "period": period,
        "mdm_employee_rk_hash": employee,
        "lvl7_mapped_management_unit_nm": team,
        "motivation_type": contract,
        "grade": grade if contract == "ТК" else None,
        "reg_coef": coefficient if contract == "ТК" else np.nan,
        "base_stavka": base if contract == "ТК" else np.nan,
        "vyrabotka_percent": worked if contract == "ТК" else np.nan,
        "sum_pay_bl_reg_base": 0.0,
        "sum_pay_otpusk_reg_base": 0.0,
        "sum_pay_prazdnik_reg_base": 0.0,
        "sum_pay_oklad_night_reg_base": 0.0,
        "bonus_kpi_comp": 0.0,
        "bonus_overtime_comp": 0.0,
        "bonus_sharing_comp_reg_base": 0.0,
        "bonus_quarterly_comp_reg_base": 0.0,
        "sum_pay_compacted": 0.0,
        "gpd_pay": rate if contract == "ГПД" else np.nan,
        "pkc_pay": rate if contract == "ПКЦ" else np.nan,
    }
    row.update(payments)
    if contract == "ТК":
        # Independent source calculation, not the model's registry/engine.
        row["actual_cost"] = coefficient * (
            base * worked
            + row["sum_pay_bl_reg_base"]
            + row["sum_pay_otpusk_reg_base"]
            + row["sum_pay_prazdnik_reg_base"]
            + row["sum_pay_oklad_night_reg_base"]
            + row["bonus_sharing_comp_reg_base"]
            + row["bonus_quarterly_comp_reg_base"]
        ) + (
            row["bonus_kpi_comp"]
            + row["bonus_overtime_comp"]
            + row["sum_pay_compacted"]
        )
    else:
        row["actual_cost"] = rate
    return row


@pytest.fixture
def complete_frame():
    """Provide a small sparse fixture with salary/time correlation."""
    return pd.DataFrame(
        [
            payroll_row("2025-03-01", "a", base=100, worked=0.8),
            payroll_row("2025-03-01", "b", base=200, worked=0.5),
            payroll_row(
                "2025-03-01",
                "c",
                base=250,
                coefficient=1.2,
                grade="G2",
                sum_pay_bl_reg_base=-10,
                bonus_kpi_comp=5,
            ),
            payroll_row("2025-03-01", "t", contract="ГПД", rate=-30),
            payroll_row("2025-03-01", "o", contract="ПКЦ", rate=90),
            payroll_row("2025-04-01", "a", base=100, worked=0.7),
            payroll_row("2025-04-01", "b", base=210, worked=0.6),
            payroll_row(
                "2025-04-01",
                "c",
                base=250,
                coefficient=1.3,
                grade="G3",
                sum_pay_bl_reg_base=-20,
                bonus_kpi_comp=8,
            ),
            payroll_row("2025-04-01", "t", contract="ГПД", rate=-50),
            payroll_row("2025-04-01", "o", contract="ПКЦ", rate=120),
        ]
    )


@pytest.fixture(scope="session")
def months():
    """Use non-January dates to detect accidental hard-coded periods."""
    return pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")


def assert_equal_states(left, right):
    """Compare nested state dictionaries without rounding arrays."""
    assert left.keys() == right.keys()
    for key, value in left.items():
        other = right[key]
        if isinstance(value, dict):
            assert_equal_states(value, other)
        elif isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, other, rtol=1e-10, atol=1e-6)
        else:
            assert value == other
