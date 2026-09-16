"""Protect the hierarchy, exact games and counterfactual interpretation."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from conftest import assert_equal_states, payroll_row
from test_analysis import analyze


def test_identical_reverse_and_one_team(model, complete_frame, months):
    """Retain symmetry, dummy drivers and one-team local equivalence."""
    result = analyze(model, complete_frame, months)
    state_a, state_b = result["states"]
    reverse = model["decompose"](state_b, state_a)
    np.testing.assert_allclose(
        reverse["bridge"].impact, -result["bridge"].impact, atol=1e-6
    )
    same = model["decompose"](state_a, deepcopy(state_a))
    np.testing.assert_allclose(same["bridge"].impact, 0, atol=1e-6)
    assert same["bridge"].share_of_net_change.isna().all()
    local = result["local_bridges"]["Alpha"].set_index("driver").impact
    organization = result["bridge"].set_index("driver").impact
    np.testing.assert_allclose(organization.loc[local.index], local, atol=1e-6)
    assert organization["team_mix"] == pytest.approx(0, abs=1e-6)


def test_copied_rates_are_symmetric(model, months):
    """Reversing a one-sided category keeps the same donor economics."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "t", contract="ГПД", rate=-20),
            payroll_row(str(months[1].date()), "t", contract="ГПД", rate=-30),
            payroll_row(str(months[1].date()), "o", contract="ПКЦ", rate=40),
        ]
    )
    forward = analyze(model, frame, months)
    backward = analyze(model, frame, tuple(reversed(months)))
    np.testing.assert_allclose(
        forward["bridge"].impact, -backward["bridge"].impact, atol=1e-6
    )
    assert_equal_states(forward["states"][0], backward["states"][1])
    assert_equal_states(forward["states"][1], backward["states"][0])


def test_full_outer_equivalence_and_pure_mix(model, months):
    """Reduced term games equal the complete outer exact game."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "a", contract="ГПД", rate=-20),
            payroll_row(str(months[1].date()), "a", contract="ГПД", rate=-25),
            payroll_row(
                str(months[0].date()),
                "b",
                team="Beta",
                contract="ПКЦ",
                rate=30,
            ),
            payroll_row(
                str(months[1].date()),
                "b",
                team="Beta",
                contract="ПКЦ",
                rate=50,
            ),
            payroll_row(
                str(months[1].date()),
                "c",
                team="Gamma",
                contract="ГПД",
                rate=60,
            ),
        ]
    )
    result = analyze(model, frame, months)
    state_a, state_b = result["states"]
    teams = state_a["_teams"]
    specs = tuple(
        driver
        for driver in model["drivers_at"]("total")
        if driver.name != model["TEAM_PARENT"]
    ) + tuple(
        model["Driver"]("team::" + team, team, "total") for team in teams
    )

    def endpoint(state):
        """Expose each completed team's scalar economics as one player."""
        return {
            "workforce_scale": state["workforce_scale"],
            "team_mix": state["team_mix"],
            **{
                "team::" + team: model["team_unit_cost"](state["teams"][team])
                for team in teams
            },
        }

    def engine(state):
        """Evaluate the full organizational cost using shared terms."""
        return model["organization_cost_terms"](
            state["workforce_scale"],
            state["team_mix"],
            [state["team::" + team] for team in teams],
        ).sum()

    full = model["exact_game"](
        endpoint(state_a), endpoint(state_b), specs, engine
    )
    np.testing.assert_allclose(
        full["impacts"], result["upper"]["impacts"], atol=1e-6
    )
    changed = deepcopy(state_a)
    changed["team_mix"] = np.array([0.2, 0.3, 0.5])
    mix = model["decompose"](state_a, changed)
    delta = model["cost"](changed) - model["cost"](state_a)
    assert mix["upper"]["impacts"]["team_mix"] == pytest.approx(delta)
    np.testing.assert_allclose(
        mix["bridge"].loc[mix["bridge"].team.notna(), "impact"], 0, atol=1e-6
    )


def test_regional_mix_spans_all_signed_eligible_components(model, months):
    """Only mix changes when identical payments move between bands."""
    payments = dict(
        sum_pay_bl_reg_base=-5,
        sum_pay_otpusk_reg_base=-3,
        sum_pay_prazdnik_reg_base=-4,
        sum_pay_oklad_night_reg_base=-2,
        bonus_sharing_comp_reg_base=-1,
        bonus_quarterly_comp_reg_base=-6,
        bonus_kpi_comp=-7,
        bonus_overtime_comp=2,
        sum_pay_compacted=-1,
    )
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "p", coefficient=1, **payments),
            payroll_row(
                str(months[1].date()), "p", coefficient=1.2, **payments
            ),
        ]
    )
    result = analyze(model, frame, months)
    impacts = result["permanent"]["Alpha"]["impacts"]
    assert impacts["regional_mix"] == pytest.approx(0.2 * (100 - 21))
    np.testing.assert_allclose(impacts.drop("regional_mix"), 0, atol=1e-6)


def test_zero_permanent_parent_with_signed_offsetting_children(model, months):
    """Propagate offsetting eligible payments without division by change."""
    frame = pd.DataFrame(
        [
            payroll_row(
                str(months[0].date()),
                "p",
                coefficient=1.2,
                sum_pay_bl_reg_base=50,
                sum_pay_prazdnik_reg_base=50,
            ),
            payroll_row(
                str(months[1].date()),
                "p",
                coefficient=1.2,
                sum_pay_bl_reg_base=-100,
                sum_pay_prazdnik_reg_base=200,
            ),
        ]
    )
    result = analyze(model, frame, months)
    impacts = result["bridge"].set_index("driver").impact
    assert impacts["sick_leave_pay"] == pytest.approx(-180)
    assert impacts["holiday_pay"] == pytest.approx(180)
    assert result["upper"]["end"] == result["upper"]["base"]
    assert result["bridge"].share_of_net_change.isna().all()


def test_zero_team_parent_with_signed_rates(model, months):
    """Keep nonzero contract-rate leaves when team-unit change is zero."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "t", contract="ГПД", rate=50),
            payroll_row(str(months[0].date()), "o", contract="ПКЦ", rate=50),
            payroll_row(str(months[1].date()), "t", contract="ГПД", rate=-50),
            payroll_row(str(months[1].date()), "o", contract="ПКЦ", rate=150),
        ]
    )
    result = analyze(model, frame, months)
    impacts = result["bridge"].set_index("driver").impact
    assert impacts["temp_reimbursement"] == pytest.approx(-100)
    assert impacts["outsource_reimbursement"] == pytest.approx(100)
    assert result["upper"]["impacts"]["team::Alpha"] == pytest.approx(0)


def test_no_cross_team_sparse_pooling(model, complete_frame, months):
    """Sparse-cell estimates are isolated from another team's salary."""
    canonical = model["adapt_production"](complete_frame)
    original = model["build_comparison_states"](canonical, *months)
    other = complete_frame.copy()
    other["lvl7_mapped_management_unit_nm"] = "Beta"
    other["mdm_employee_rk_hash"] += "-beta"
    other["base_stavka"] *= 100
    combined = model["adapt_production"](pd.concat([complete_frame, other]))
    rebuilt = model["build_comparison_states"](combined, *months)
    for left, right in zip(original["states"], rebuilt["states"]):
        assert_equal_states(left["teams"]["Alpha"], right["teams"]["Alpha"])
    with pytest.raises(ValueError, match="No fallback support"):
        model["build_comparison_states"](canonical, *months, fallback_order=())


def test_canonical_exposure_stays_independent_of_worked_time(
    model, complete_frame, months
):
    """Retain exact aggregation with explicit fractional availability."""
    frame = model["adapt_production"](complete_frame)
    frame["exposure"] = np.where(np.arange(len(frame)) % 2, 0.5, 1.0)
    frame["source_cost"] *= frame.exposure
    comparison = model["build_comparison_states"](frame, *months)
    result = model["decompose"](*comparison["states"])
    checks = model["reconcile"](frame, *comparison["states"], result)
    assert checks.passed.all()


def test_explicit_permanent_and_team_references(model, complete_frame, months):
    """References fill absent permanent/whole-team states in strict mode."""
    canonical = model["adapt_production"](complete_frame)
    complete = model["build_comparison_states"](canonical, *months)
    reference = deepcopy(complete["states"][0]["teams"]["Alpha"])
    reference["permanent"]["holiday_pay"] -= 500
    reference["temp_reimbursement"] = -321
    absent_permanent = complete_frame.loc[
        ~(
            complete_frame.period.eq("2025-03-01")
            & complete_frame.motivation_type.eq("ТК")
        )
    ]
    result = analyze(
        model,
        absent_permanent,
        months,
        missing_economics="require_reference",
        references={
            (months[0], "Alpha"): {"permanent": reference["permanent"]}
        },
    )
    assert_equal_states(
        result["states"][0]["teams"]["Alpha"]["permanent"],
        reference["permanent"],
    )
    # Keep the organization present while Alpha is absent in the first month.
    remaining = complete_frame.loc[complete_frame.period.eq("2025-04-01")]
    anchor = pd.DataFrame(
        [
            payroll_row(
                str(month.date()), "anchor", team="Beta", contract="ГПД"
            )
            for month in months
        ]
    )
    frame = pd.concat([remaining, anchor])
    with pytest.raises(ValueError, match="missing team"):
        analyze(model, frame, months, missing_economics="require_reference")
    result = analyze(
        model,
        frame,
        months,
        missing_economics="require_reference",
        references={(months[0], "Alpha"): {"team_state": reference}},
    )
    assert_equal_states(result["states"][0]["teams"]["Alpha"], reference)


def test_observed_state_cost_needs_no_comparison(model, months):
    """Cost can skip an absent team only while its observed share is zero."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "a", contract="ГПД", rate=-20),
            payroll_row(
                str(months[1].date()), "b", team="Beta", contract="ПКЦ"
            ),
        ]
    )
    canonical = model["adapt_production"](frame)
    state, _, _ = model["build_state"](
        canonical, months[0], model["make_policy"](canonical)
    )
    assert model["cost"](state) == -20
    assert state["teams"]["Beta"] is None
    state["team_mix"] = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Positive team share"):
        model["cost"](state)
