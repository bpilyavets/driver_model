"""Preserve payroll and attribution with multiple employment types."""

from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pytest

from conftest import assert_equal_states, payroll_row
from test_analysis import analyze


@pytest.fixture
def switch_frame(months):
    """Keep earned permanent pay fixed while adding temporary employment."""
    return pd.DataFrame(
        [
            payroll_row(
                str(month.date()), "p", base=100000, worked=0.5,
            )
            for month in months
        ] + [
            payroll_row(
                str(months[1].date()), "p", contract="ГПД", rate=30000,
            ),
        ]
    )


@pytest.mark.parametrize("explainer", ["exact", "permutation"])
@pytest.mark.parametrize("independent", [False, True])
def test_split_preserves_payroll_and_observed_inputs(
    model, months, switch_frame, explainer, independent,
):
    """Adding a contract changes composition, not headcount, salary or time."""
    original = switch_frame.copy(deep=True)
    result = model["analyze_labor_cost"](
        switch_frame, *months,
        source_cost_column="actual_cost" if independent else None,
        explainer=explainer, n_permutations=8,
    )
    assert result["validation"].passed.all()
    pd.testing.assert_frame_equal(switch_frame, original)
    np.testing.assert_allclose(result["frame"].exposure, [1, 0.5, 0.5])
    np.testing.assert_allclose(
        result["frame"].source_cost, [50000, 50000, 30000]
    )
    assert [s["workforce_scale"] for s in result["states"]] == [1, 1]
    assert (result["upper"]["base"], result["upper"]["end"]) == (50000, 80000)
    effects = result["driver_bridge"].set_index("driver").impact
    np.testing.assert_allclose(
        effects[[
            "workforce_scale", "underlying_salary_level", "worked_hours",
        ]],
        0, atol=1e-6,
    )
    assert effects["contract_type_mix"] == pytest.approx(-7500)
    assert effects["within_cell_workforce_mix"] == pytest.approx(37500)
    assert effects.sum() == pytest.approx(30000)
    assert result["local_bridges"]["Alpha"].impact.sum() == pytest.approx(
        30000
    )
    inputs = result["diagnostics"]["employee_inputs"]
    np.testing.assert_allclose(inputs.base_salary, 100000)
    np.testing.assert_allclose(inputs.worked_fraction, 0.5)
    allocation = result["diagnostics"]["exposure_allocation"]
    np.testing.assert_allclose(allocation.employment_type_count, [1, 2, 2])
    assert result["exposure_config"]["method"] == "equal_split"
    assert result["exposure_config"]["column"] is None
    assert len(result["permanent"]["Alpha"]["impacts"]) == 14


def test_mixed_type_counts_and_zero_signed_payments(model, months):
    """Count observed types rather than nonzero or positive payroll rows."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "one"),
            payroll_row(str(months[0].date()), "two"),
            payroll_row(
                str(months[0].date()), "two", contract="ГПД", rate=0,
            ),
            payroll_row(str(months[0].date()), "three", worked=0),
            payroll_row(
                str(months[0].date()), "three", contract="ГПД", rate=-30,
            ),
            payroll_row(
                str(months[0].date()), "three", contract="ПКЦ", rate=60,
            ),
            payroll_row(str(months[1].date()), "three"),
        ]
    )
    canonical = model["adapt_production"](frame)
    np.testing.assert_allclose(
        canonical.exposure, [1, 0.5, 0.5, 1 / 3, 1 / 3, 1 / 3, 1],
    )
    totals = canonical.groupby(["period", "employee_id"]).exposure.sum()
    np.testing.assert_allclose(totals, 1)
    np.testing.assert_allclose(canonical.source_cost, frame.actual_cost)
    policy = model["make_policy"](canonical)
    for period in months:
        state, _, _ = model["build_state"](canonical, period, policy)
        expected = canonical.loc[canonical.period.eq(period), "source_cost"]
        assert model["cost"](state) == pytest.approx(expected.sum())
    assert state["workforce_scale"] == 1


@pytest.mark.parametrize("failure", ["duplicate", "team"])
def test_grain_validation_is_shared(model, switch_frame, failure):
    """Reject duplicate triples and conflicting teams in both entry points."""
    frame = switch_frame.copy()
    canonical = model["adapt_production"](frame)
    if failure == "duplicate":
        frame = pd.concat([frame, frame.iloc[[-1]]])
        canonical = pd.concat([canonical, canonical.iloc[[-1]]])
        message = "Duplicate employee-month-employment-type"
    else:
        frame.loc[2, "lvl7_mapped_management_unit_nm"] = "Beta"
        canonical.loc[2, "team"] = "Beta"
        message = "one team across types"
    with pytest.raises(ValueError, match=message):
        model["adapt_production"](frame)
    with pytest.raises(ValueError, match=message):
        model["validate_input"](canonical)


@pytest.mark.parametrize("bad", [0, -0.1, 1.1, np.nan, np.inf, "0.5", True])
def test_invalid_supplied_exposure_never_falls_back(model, switch_frame, bad):
    """Explicit invalid exposure fails rather than invoking equal splitting."""
    switch_frame["actual_exposure"] = bad
    with pytest.raises(ValueError, match="exposure"):
        model["adapt_production"](
            switch_frame, exposure_column="actual_exposure",
        )


def test_missing_exposure_column_and_explicit_selection(model, switch_frame):
    """An unselected column cannot change the default allocation policy."""
    with pytest.raises(ValueError, match="actual_exposure"):
        model["adapt_production"](
            switch_frame, exposure_column="actual_exposure",
        )
    switch_frame["actual_exposure"] = 0.9
    canonical = model["adapt_production"](switch_frame)
    np.testing.assert_allclose(canonical.exposure, [1, 0.5, 0.5])


def test_source_exposure_bypasses_person_total_normalization(
    model, months, switch_frame, monkeypatch,
):
    """Retain source totals below/above one and show their provenance."""
    frame = pd.concat(
        [switch_frame, pd.DataFrame([payroll_row(
            str(months[0].date()), "p", contract="ГПД", rate=30000,
        )])], ignore_index=True,
    )
    frame["actual_exposure"] = [0.25, 0.75, 0.75, 0.25]
    result = analyze(
        model, frame, months, exposure_column="actual_exposure",
        explainer="permutation", n_permutations=8,
    )
    np.testing.assert_allclose(result["frame"].exposure, frame.actual_exposure)
    np.testing.assert_allclose(result["frame"].source_cost, frame.actual_cost)
    assert [s["workforce_scale"] for s in result["states"]] == [0.5, 1.5]
    assert result["upper"]["base"] == pytest.approx(80000)
    assert result["upper"]["end"] == pytest.approx(80000)
    assert result["driver_bridge"].share_of_net_change.isna().all()
    assert result["exposure_config"]["method"] == "source_column"
    assert result["exposure_config"]["column"] == "actual_exposure"
    shown = {}
    monkeypatch.setitem(
        model, "show", lambda title, value: shown.update({title: value}),
    )
    model["report"](result, organization_view="drivers", plots=False)
    assert shown["Workforce exposure policy"] == result["exposure_config"]
    title = "Period workforce totals — counts and exposure are distinct"
    totals = shown[title]
    np.testing.assert_allclose(totals.source_records, [2, 2])
    np.testing.assert_allclose(totals.distinct_employees, [1, 1])
    np.testing.assert_allclose(totals.exposure, [0.5, 1.5])


@pytest.mark.parametrize("extend_registry", [False, True])
def test_signed_payments_and_regional_normalization(
    model, months, extend_registry,
):
    """Normalize all registered realized payments once without losing signs."""
    frame = pd.DataFrame(
        [row for month in months for row in (
            payroll_row(
                str(month.date()), "p", base=100, worked=0,
                coefficient=1.2, sum_pay_bl_reg_base=-10,
                sum_pay_otpusk_reg_base=20, sum_pay_prazdnik_reg_base=-30,
                sum_pay_oklad_night_reg_base=40,
                bonus_sharing_comp_reg_base=-50,
                bonus_quarterly_comp_reg_base=60, bonus_kpi_comp=-70,
                bonus_overtime_comp=80, sum_pay_compacted=-90,
            ),
            payroll_row(str(month.date()), "p", contract="ГПД", rate=-20),
            payroll_row(str(month.date()), "p", contract="ПКЦ", rate=10),
        )]
    )
    registry = model["DRIVERS"]
    if extend_registry:
        frame["transport_allowance"] = -25.0
        permanent = frame.motivation_type.eq("ТК")
        frame.loc[permanent, "actual_cost"] += -25 * 1.2
        registry += (model["payment_driver"](
            "transport_allowance", "Transport allowance",
            "transport_allowance",
            apply_regional_coefficient=True,
        ),)
    canonical = model["adapt_production"](
        frame, registry=registry,
        source_cost_column="actual_cost" if extend_registry else None,
    )
    np.testing.assert_allclose(canonical.source_cost, frame.actual_cost)
    perm = canonical.contract_type.eq("permanent")
    assert (canonical.loc[perm, "worked_fraction"] == 0).all()
    assert (canonical.loc[perm, "base_salary"] == 100).all()
    assert (canonical.loc[perm, "base_pay_factor"] == 3).all()
    for driver in registry:
        if driver.kind != "cell_signed":
            continue
        for column in driver.columns:
            production = next(
                (key for key, value in model["PRODUCTION_COLUMNS"].items()
                 if value == column), column,
            )
            np.testing.assert_allclose(
                canonical.loc[perm, column], frame.loc[perm, production] * 3,
            )
    regionalized = canonical.copy()
    columns = [
        "base_salary", "sick_leave_pay", "sharing_bonus", "quarterly_bonus",
    ]
    regionalized.loc[perm, columns] *= 1.2
    normalized = model["normalize_regional_inputs"](
        regionalized, already_regionalized=columns, registry=registry,
    )
    pd.testing.assert_frame_equal(normalized, canonical)
    result = analyze(
        model, frame, months, registry=registry,
        explainer="permutation", n_permutations=8,
    )
    if extend_registry:
        assert "transport_allowance" in set(result["driver_bridge"].driver)


@pytest.fixture
def profile_comparison(model, months):
    """Provide unequal fractional membership and sparse region/grade cells."""
    rows = []
    for side, month in enumerate(months):
        period = str(month.date())
        rows.extend([
            payroll_row(period, "p", base=100 + 20 * side, worked=0.5),
            payroll_row(period, "q", base=200, worked=0.25 + 0.1 * side),
            payroll_row(
                period, "r", base=300, coefficient=1.2, grade="G2",
            ),
            payroll_row(
                period, "p" if side == 0 else "q", contract="ГПД", rate=0,
            ),
        ])
    frame = pd.DataFrame(rows)
    canonical = model["adapt_production"](frame)
    comparison = model["build_comparison_states"](canonical, *months)
    return frame, canonical, comparison


def test_profiles_fallbacks_caches_and_sampled_audits(
    model, months, profile_comparison,
):
    """Keep donor factors aligned with profiles through caching and audits."""
    frame, canonical, comparison = profile_comparison
    a, b = [s["teams"]["Alpha"]["permanent"] for s in comparison["states"]]
    profile = a["within_cell_workforce_mix"][0, 0]
    np.testing.assert_allclose(profile["weights"], [1 / 3, 2 / 3])
    np.testing.assert_allclose(profile["base_pay_factors"], [2, 1])
    fallback = a["within_cell_workforce_mix"][1, 0]
    np.testing.assert_array_equal(fallback["indices"], profile["indices"])
    np.testing.assert_allclose(fallback["base_pay_factors"], [2, 1])
    assert model["salary_cell_contribution"](a)[0, 0] == pytest.approx(
        100 / 1.5
    )
    means = model["profile_mean"](a, a["underlying_salary_level"])
    assert means[0, 0] == pytest.approx(500 / 3)
    cache = model["prepare_salary_cache"](a, b)
    for bits in product((0, 1), repeat=3):
        hybrid = dict(a)
        for name, bit in zip(model["SALARY_KEYS"], bits):
            hybrid[name] = (a, b)[bit][name]
        np.testing.assert_allclose(
            model["salary_cell_contribution"](hybrid),
            model["salary_cell_contribution"](
                dict(hybrid, _salary_cache=cache)
            ),
            rtol=1e-10, atol=1e-6,
        )
    result = analyze(
        model, frame, months, explainer="permutation", n_permutations=8,
    )
    profiles = result["diagnostics"]["membership_profiles"]
    assert set(profiles.base_pay_factor) == {1, 2}
    for state in result["states"]:
        rows = canonical.loc[canonical.period.eq(state["_period"])]
        assert model["cost"](state) == pytest.approx(rows.source_cost.sum())


@pytest.mark.parametrize("bad", [[0], [-1], [np.nan], [np.inf], [1, 1]])
def test_invalid_profile_conversion_factors(model, months, switch_frame, bad):
    """Reject nonpositive, nonfinite or misaligned conversion arrays."""
    canonical = model["adapt_production"](switch_frame)
    comparison = model["build_comparison_states"](canonical, *months)
    state = comparison["states"][1]["teams"]["Alpha"]["permanent"]
    profile = next(iter(state["within_cell_workforce_mix"].values()))
    profile["base_pay_factors"] = np.array(bad)
    with pytest.raises(ValueError, match="base_pay_factors"):
        model["permanent_cost"](state)


def test_legacy_per_exposure_factors_default_to_one(
    model, complete_frame, months,
):
    """Keep existing canonical rates and aligned references usable."""
    canonical = model["adapt_production"](complete_frame)
    canonical = canonical.drop(columns="base_pay_factor")
    canonical["exposure"] = 0.5
    canonical["source_cost"] *= 0.5
    comparison = model["build_comparison_states"](canonical, *months)
    state = comparison["states"][0]["teams"]["Alpha"]["permanent"]
    legacy = deepcopy(state)
    for profile in legacy["within_cell_workforce_mix"].values():
        del profile["base_pay_factors"]
    assert model["permanent_cost"](legacy) == pytest.approx(
        model["permanent_cost"](state)
    )
    normalized = model["employee_payroll_reference"](canonical)
    np.testing.assert_allclose(normalized, canonical.source_cost)
    for organization in comparison["states"]:
        rows = canonical.loc[canonical.period.eq(organization["_period"])]
        assert model["cost"](organization) == pytest.approx(
            rows.source_cost.sum()
        )


@pytest.mark.parametrize("whole_team", [False, True])
def test_copied_economics_and_references_keep_factors(
    model, months, whole_team,
):
    """Copy full economics including factors without copying exposure."""
    frame = pd.DataFrame([
        payroll_row(
            str(months[0].date()), "other",
            team="Beta" if whole_team else "Alpha",
            contract="ГПД", rate=10,
        ),
        payroll_row(str(months[1].date()), "p", base=100, worked=0.5),
        payroll_row(str(months[1].date()), "p", contract="ГПД", rate=30),
    ])
    canonical = model["adapt_production"](frame)
    comparison = model["build_comparison_states"](canonical, *months)
    left, right = [s["teams"]["Alpha"] for s in comparison["states"]]
    assert_equal_states(left["permanent"], right["permanent"])
    profiles = left["permanent"]["within_cell_workforce_mix"]
    profile = next(iter(profiles.values()))
    np.testing.assert_allclose(profile["base_pay_factors"], [2])
    key, reference = (
        ("team_state", right)
        if whole_team else ("permanent", right["permanent"])
    )
    explicit = model["build_comparison_states"](
        canonical, *months,
        references={(months[0], "Alpha"): {key: reference}},
    )
    effective = explicit["states"][0]["teams"]["Alpha"]["permanent"]
    assert_equal_states(effective, right["permanent"])
    state_a = explicit["states"][0]
    alpha_share = state_a["team_mix"][state_a["_teams"].index("Alpha")]
    assert alpha_share == (0 if whole_team else 1)
    assert model["cost"](state_a) == pytest.approx(10)


def test_movements_count_people_once(model, months):
    """Repeated types cannot duplicate movements or create Series cells."""
    rows = []
    for period, employee, team in (
        (months[0], "p", "Alpha"), (months[1], "p", "Beta"),
        (months[1], "q", "Alpha"),
    ):
        for contract in ("ТК", "ГПД"):
            rows.append(payroll_row(
                str(period.date()), employee, team=team, contract=contract,
            ))
    canonical = model["adapt_production"](pd.DataFrame(rows))
    diagnostic = model["matched_salary_diagnostic"](canonical, months)
    assert diagnostic["summary"].matched_permanent_employees == 1
    assert diagnostic["summary"].team_transfers == 1
    movements = diagnostic["movements"]
    assert len(movements.query("employee_id == 'p'")) == 2
    assert len(movements.query("employee_id == 'q'")) == 1
    assert movements.other_period_team.map(
        lambda value: value is None or isinstance(value, str)
    ).all()


def test_inactive_permanent_payments_can_be_nullable(model, months):
    """Do not normalize inapplicable permanent fields in contract-only data."""
    frame = pd.DataFrame([
        payroll_row(str(month.date()), "p", contract=contract, rate=30)
        for month in months for contract in ("ГПД", "ПКЦ")
    ])
    for column in model["PAYMENT_COLUMNS"]:
        production = next(
            key for key, value in model["PRODUCTION_COLUMNS"].items()
            if value == column
        )
        frame[production] = pd.NA
    canonical = model["adapt_production"](frame)
    np.testing.assert_allclose(canonical.exposure, 0.5)
    np.testing.assert_allclose(canonical.contract_reimbursement, 60)
    np.testing.assert_allclose(canonical.source_cost, 30)
    comparison = model["build_comparison_states"](canonical, *months)
    for state in comparison["states"]:
        assert state["teams"]["Alpha"]["permanent"] is None
        assert model["cost"](state) == pytest.approx(60)
