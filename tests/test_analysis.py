"""Check signed payroll, missing economics and hierarchical attribution."""

from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pytest

from conftest import assert_equal_states, payroll_row


def analyze(model, frame, months, **kwargs):
    """Run the public workflow against an independently supplied total."""
    result = model["analyze_labor_cost"](
        frame, *months, source_cost_column="actual_cost", **kwargs
    )
    assert result["validation"].passed.all()
    assert result["coalition_audit"].passed.all()
    return result


def test_complete_reconciliation_and_input_integrity(
    model, complete_frame, months
):
    """Reconcile sparse payroll without changing the caller's records."""
    original = complete_frame.copy(deep=True)
    result = analyze(model, complete_frame, months)
    pd.testing.assert_series_equal(
        result["driver_bridge"].set_index("driver").impact,
        result["bridge"].set_index("driver").impact,
    )
    pd.testing.assert_frame_equal(original, complete_frame)
    assert result["periods"] == months
    assert result["permanent"]["Alpha"]["audit"]["coalitions"] == 16384
    assert set(result["diagnostics"]["unsupported"].value_period) == set(
        months
    )
    assert result["assumptions"].empty
    assert not result["diagnostics"]["fallbacks"].empty
    assert (
        result["diagnostics"]["matched"]["summary"][
            "matched_permanent_employees"
        ]
        == 3
    )
    np.testing.assert_allclose(
        result["diagnostics"]["salary_cells"].organization_impact.sum(),
        result["bridge"].set_index("driver").impact["underlying_salary_level"],
        rtol=1e-10,
        atol=1e-6,
    )


@pytest.mark.parametrize("contract", ["ТК", "ГПД", "ПКЦ"])
@pytest.mark.parametrize("missing_month", ["2025-03-01", "2025-04-01"])
def test_one_sided_category_copy(
    model, complete_frame, months, contract, missing_month
):
    """Copy complete same-team economics in either comparison direction."""
    keep = ~(
        complete_frame.period.eq(missing_month)
        & complete_frame.motivation_type.eq(contract)
    )
    result = analyze(model, complete_frame.loc[keep], months)
    left, right = [s["teams"]["Alpha"] for s in result["states"]]
    category = model["PRODUCTION_CONTRACTS"][contract]
    key = model["CONTRACT_KEYS"][category]
    if contract == "ТК":
        assert_equal_states(left[key], right[key])
        impacts = result["permanent"]["Alpha"]["impacts"]
        np.testing.assert_allclose(impacts, 0, atol=1e-6)
    else:
        assert left[key] == right[key]
        np.testing.assert_allclose(
            result["upper"]["teams"]["Alpha"]["unit"]["impacts"][key],
            0,
            atol=1e-6,
        )
    row = result["assumptions"].iloc[0]
    assert row.method == "carry_observed"
    assert row.missing_period == pd.Timestamp(missing_month)
    assert (
        row.source_period in months and row.source_period != row.missing_period
    )
    counts = result["diagnostics"]["coverage"]
    if contract == "ТК":
        assert (
            counts.loc[counts.period.eq(row.missing_period), "count"].sum()
            == 0
        )


@pytest.mark.parametrize("contract", ["ТК", "ГПД", "ПКЦ"])
def test_both_period_inactive_category(
    model, complete_frame, months, contract
):
    """Do not demand or invent rates for categories absent in both states."""
    frame = complete_frame.loc[complete_frame.motivation_type.ne(contract)]
    result = analyze(model, frame, months)
    category = model["PRODUCTION_CONTRACTS"][contract]
    key = model["CONTRACT_KEYS"][category]
    for state in result["states"]:
        assert state["teams"]["Alpha"][key] is None
    assert result["assumptions"].query("method == 'inactive'").shape[0] == 1
    absent_driver = "permanent_unit_cost" if contract == "ТК" else key
    assert (
        absent_driver
        not in result["upper"]["teams"]["Alpha"]["unit"]["impacts"]
    )
    if contract == "ТК":
        assert not result["permanent"]
        assert result["diagnostics"]["salary_cells"].empty
        assert result["diagnostics"]["coverage"].empty
        assert result["upper"]["teams"]["Alpha"]["K_game"] is None
    activated = deepcopy(result["states"][0])
    index = model["CONTRACTS"].index(category)
    shares = np.zeros(3)
    shares[index] = 1
    activated["teams"]["Alpha"]["contract_type_mix"] = shares
    with pytest.raises(ValueError, match="Positive contract share"):
        model["cost"](activated)


@pytest.mark.parametrize("contract", ["ГПД", "ПКЦ"])
def test_single_nonpermanent_contract_only(model, months, contract):
    """Run with no permanent observations or policy anywhere in the data."""
    frame = pd.DataFrame(
        [
            payroll_row(
                str(months[0].date()), "x", contract=contract, rate=-90
            ),
            payroll_row(
                str(months[1].date()), "x", contract=contract, rate=-50
            ),
        ]
    )
    result = analyze(model, frame, months)
    assert result["upper"]["base"] == -90
    assert result["upper"]["end"] == -50
    assert len(result["assumptions"]) == 2
    assert not result["permanent"]


@pytest.mark.parametrize("missing_side", [0, 1])
def test_whole_team_entry_exit(model, months, missing_side):
    """Carry whole-team economics without copying workforce counts."""
    present = months[1 - missing_side]
    records = [
        payroll_row(str(month.date()), "anchor", contract="ГПД", rate=40)
        for month in months
    ]
    records.append(
        payroll_row(
            str(present.date()),
            "new",
            team="New team",
            contract="ПКЦ",
            rate=-60,
        )
    )
    result = analyze(model, pd.DataFrame(records), months)
    states = result["states"]
    assert_equal_states(
        states[0]["teams"]["New team"], states[1]["teams"]["New team"]
    )
    index = states[0]["_teams"].index("New team")
    assert states[missing_side]["team_mix"][index] == 0
    assert result["upper"]["impacts"]["team::New team"] == pytest.approx(0)
    assert result["assumptions"].query("contract_type == 'all'").shape[0] == 1


@pytest.mark.parametrize("mode", ["carry_observed", "require_reference"])
def test_explicit_signed_reference_overrides_copy(
    model, complete_frame, months, mode
):
    """Use a signed business reference before the automatic donor rate."""
    frame = complete_frame.loc[
        ~(
            complete_frame.period.eq("2025-03-01")
            & complete_frame.motivation_type.eq("ГПД")
        )
    ]
    refs = {(months[0], "Alpha"): {"temporary": -123.0}}
    result = analyze(
        model, frame, months, references=refs, missing_economics=mode
    )
    assert result["states"][0]["teams"]["Alpha"]["temp_reimbursement"] == -123
    assert result["states"][1]["teams"]["Alpha"]["temp_reimbursement"] == -50
    assert result["assumptions"].iloc[0].method == "business_reference"


def test_strict_mode_and_observed_values(model, complete_frame, months):
    """Strict mode rejects one-sided absence, not inactive categories."""
    frame = complete_frame.loc[
        ~(
            complete_frame.period.eq("2025-03-01")
            & complete_frame.motivation_type.eq("ГПД")
        )
    ]
    with pytest.raises(ValueError, match="missing temporary"):
        analyze(model, frame, months, missing_economics="require_reference")
    frame = complete_frame.loc[complete_frame.motivation_type.eq("ГПД")]
    result = analyze(
        model, frame, months, missing_economics="require_reference"
    )
    assert (result["assumptions"].method == "inactive").all()
    result = analyze(
        model,
        frame,
        months,
        references={(months[0], "Alpha"): {"temporary": 99999.0}},
    )
    assert result["states"][0]["teams"]["Alpha"]["temp_reimbursement"] == -30


@pytest.mark.parametrize("rate", [0.0, -50.0])
def test_zero_negative_observed_rates_are_present(model, months, rate):
    """Presence depends on employee exposure, not the amount of payment."""
    frame = pd.DataFrame(
        [
            payroll_row(str(month.date()), "x", contract="ГПД", rate=rate)
            for month in months
        ]
    )
    result = analyze(model, frame, months)
    assert "temporary" not in set(result["assumptions"].contract_type)
    for state in result["states"]:
        assert state["teams"]["Alpha"]["temp_reimbursement"] == rate


SIGNED_COLUMNS = [
    "sum_pay_bl_reg_base",
    "sum_pay_otpusk_reg_base",
    "sum_pay_prazdnik_reg_base",
    "sum_pay_oklad_night_reg_base",
    "bonus_kpi_comp",
    "bonus_overtime_comp",
    "bonus_sharing_comp_reg_base",
    "bonus_quarterly_comp_reg_base",
    "sum_pay_compacted",
]


@pytest.mark.parametrize("column", SIGNED_COLUMNS)
def test_each_component_can_make_payroll_negative(model, months, column):
    """Accept signed components and negative permanent/organization costs."""
    frame = pd.DataFrame(
        [
            payroll_row(
                str(months[0].date()), "x", coefficient=1.5, **{column: -500}
            ),
            payroll_row(
                str(months[1].date()), "x", coefficient=1.5, **{column: -300}
            ),
        ]
    )
    result = analyze(model, frame, months)
    assert result["upper"]["base"] < 0
    assert result["upper"]["end"] < 0
    expected = frame.loc[1, "actual_cost"] - frame.loc[0, "actual_cost"]
    np.testing.assert_allclose(
        result["bridge"].impact.sum(), expected, atol=1e-6
    )


def test_hand_payroll_and_worked_factor_above_one(model, months):
    """Apply the coefficient once to each eligible signed payment."""
    row = payroll_row(
        str(months[0].date()),
        "p",
        base=100,
        worked=1.2,
        coefficient=2,
        sum_pay_bl_reg_base=-1,
        sum_pay_otpusk_reg_base=-2,
        sum_pay_prazdnik_reg_base=-3,
        sum_pay_oklad_night_reg_base=-4,
        bonus_sharing_comp_reg_base=-5,
        bonus_quarterly_comp_reg_base=-6,
        bonus_kpi_comp=-7,
        bonus_overtime_comp=-8,
        sum_pay_compacted=-9,
    )
    frame = model["adapt_production"](pd.DataFrame([row]))
    policy = model["make_policy"](frame)
    state, _, _ = model["build_state"](frame, months[0], policy)
    assert model["cost"](state) == pytest.approx(2 * (120 - 21) - 24)


def test_signed_normalization_and_sparse_fallback(
    model, complete_frame, months
):
    """Regionalized and pre-regional sources yield identical attribution."""
    canonical = model["adapt_production"](
        complete_frame, source_cost_column="actual_cost"
    )
    original = model["build_comparison_states"](canonical, *months)
    adjusted = canonical.copy(deep=True)
    eligible = [
        col
        for driver in model["drivers_at"](model["PARENT"])
        if driver.apply_regional_coefficient
        for col in driver.columns
    ]
    permanent = adjusted.contract_type.eq("permanent")
    for column in eligible:
        adjusted.loc[permanent, column] *= adjusted.loc[
            permanent, "regional_coefficient"
        ]
    normalized = model["normalize_regional_inputs"](
        adjusted, already_regionalized=eligible
    )
    pd.testing.assert_series_equal(
        normalized.source_cost, canonical.source_cost
    )
    rebuilt = model["build_comparison_states"](normalized, *months)
    for left, right in zip(original["states"], rebuilt["states"]):
        assert_equal_states(left, right)
    for left, right in zip(original["fallbacks"], rebuilt["fallbacks"]):
        pd.testing.assert_frame_equal(left, right, check_exact=False)
    first = model["decompose"](*original["states"])
    second = model["decompose"](*rebuilt["states"])
    np.testing.assert_allclose(
        first["bridge"].impact, second["bridge"].impact, atol=1e-6
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("base_stavka", 0),
        ("base_stavka", -1),
        ("reg_coef", 0),
        ("reg_coef", -1),
        ("vyrabotka_percent", -0.1),
        ("bonus_kpi_comp", np.nan),
        ("bonus_kpi_comp", np.inf),
        ("actual_cost", np.nan),
        ("actual_cost", np.inf),
        ("lvl7_mapped_management_unit_nm", None),
        ("motivation_type", "ГПХ"),
    ],
)
def test_invalid_inputs_still_fail(model, months, column, value):
    """Keep missing/nonfinite money and invalid quantity checks strict."""
    frame = pd.DataFrame(
        [payroll_row(str(month.date()), "x") for month in months]
    )
    frame[column] = value
    with pytest.raises(ValueError):
        analyze(model, frame, months)


def test_missing_pay_for_existing_worker_is_not_absence(model, months):
    """Do not carry a rate over missing values for actual employees."""
    frame = pd.DataFrame(
        [
            payroll_row(str(month.date()), "x", contract="ГПД")
            for month in months
        ]
    )
    frame.loc[0, "gpd_pay"] = np.nan
    with pytest.raises(ValueError, match="contract_reimbursement"):
        analyze(model, frame, months)


def test_unexplained_source_total_fails(model, months):
    """A signed independent total still needs components that explain it."""
    frame = pd.DataFrame(
        [payroll_row(str(month.date()), "x") for month in months]
    )
    frame["actual_cost"] = -999
    with pytest.raises(AssertionError):
        analyze(model, frame, months)


def test_reference_and_policy_failures(model, complete_frame, months):
    """Reject unknown reference keys, coefficient changes and invalid mixes."""
    canonical = model["adapt_production"](complete_frame)
    with pytest.raises(ValueError, match="selected period"):
        model["build_comparison_states"](
            canonical, *months, references={(months[0], "Unknown"): {}}
        )
    with pytest.raises(ValueError, match="Unknown missing_economics"):
        model["build_comparison_states"](
            canonical, *months, missing_economics="pool_other_teams"
        )
    comparison = model["build_comparison_states"](canonical, *months)
    state = comparison["states"][0]
    for key in (
        "team_mix",
        "contract_type_mix",
        "regional_mix",
        "grade_mix_within_region",
    ):
        for invalid_value in (0.0, -1.0, np.nan):
            invalid = deepcopy(state)
            target = (
                invalid if key == "team_mix" else invalid["teams"]["Alpha"]
            )
            if key in {"regional_mix", "grade_mix_within_region"}:
                target = target["permanent"]
            target[key] = np.full_like(target[key], invalid_value)
            with pytest.raises(ValueError):
                model["cost"](invalid)
    invalid = deepcopy(comparison["states"][1])
    invalid["teams"]["Alpha"]["permanent"]["_policy"]["regional_coefficients"][
        0
    ] += 0.1
    with pytest.raises(ValueError, match="policy differ"):
        model["decompose"](state, invalid)


def test_duplicates_and_selection(model, complete_frame, months):
    """Validate selected records without treating other months as players."""
    extra = complete_frame.iloc[[0]].copy()
    extra["period"] = "2024-01-01"
    result = analyze(model, pd.concat([complete_frame, extra]), months)
    assert len(result["frame"]) == len(complete_frame)
    with pytest.raises(ValueError, match="Duplicate employee-month"):
        analyze(
            model,
            pd.concat([complete_frame, complete_frame.iloc[[0]]]),
            months,
        )


def test_registry_extension(model, months):
    """One signed payment registration flows through the existing pipeline."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "x", coefficient=1.2),
            payroll_row(str(months[1].date()), "x", coefficient=1.2),
        ]
    )
    frame["transport_allowance"] = [-50.0, 20.0]
    frame["actual_cost"] += frame.reg_coef * frame.transport_allowance
    registry = model["DRIVERS"] + (
        model["payment_driver"](
            "transport_allowance",
            "Transport allowance",
            "transport_allowance",
            apply_regional_coefficient=True,
        ),
    )
    result = analyze(model, frame, months, registry=registry)
    impacts = result["bridge"].set_index("driver").impact
    assert impacts["transport_allowance"] == pytest.approx(84)
    summary = result["driver_bridge"].set_index("driver")
    assert summary.loc["transport_allowance", "impact"] == pytest.approx(84)
    assert summary.loc["transport_allowance", "label"] == "Transport allowance"
    assert summary.index[-1] == "transport_allowance"
    assert result["permanent"]["Alpha"]["audit"]["coalitions"] == 32768
    assert "transport_allowance" in set(
        result["diagnostics"]["unsupported"].variable
    )
    assert (
        "transport_allowance" in result["local_bridges"]["Alpha"].driver.values
    )


def test_reconstruction_with_null_inapplicable_columns(model, months):
    """All-permanent data can leave both contract-pay columns null."""
    frame = pd.DataFrame(
        [
            payroll_row(str(month.date()), "x", bonus_kpi_comp=-150)
            for month in months
        ]
    )
    frame["gpd_pay"] = None
    frame["pkc_pay"] = None
    result = model["analyze_labor_cost"](frame, *months)
    assert result["validation"].passed.all()
    assert result["upper"]["base"] == -50
    assert result["upper"]["end"] == -50
    assert set(result["frame"].reference_basis) == {
        "reconstructed permanent payroll"
    }
