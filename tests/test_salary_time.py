"""Verify employee-aligned salary, planned-hours fraction and membership."""

from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import pytest

from conftest import payroll_row
from test_analysis import analyze


def permanent_pair(model, frame, months):
    """Build canonical aligned states without running attribution."""
    canonical = model["adapt_production"](
        frame, source_cost_column="actual_cost"
    )
    comparison = model["build_comparison_states"](canonical, *months)
    return comparison, [
        s["teams"]["Alpha"]["permanent"] for s in comparison["states"]
    ]


@pytest.mark.parametrize(
    "salary_b,time_b,salary_effect,time_effect",
    [
        (110000, 0.625, 6250, 0),
        (100000, 0.75, 0, 12500),
        (110000, 0.75, 6875, 13125),
        (100000, 0, 0, -62500),
        (100000, 1.2, 0, 57500),
    ],
)
def test_salary_time_independence(
    model, months, salary_b, time_b, salary_effect, time_effect
):
    """Separate observed changes and split their real interaction equally."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "p", base=100000, worked=0.625),
            payroll_row(
                str(months[1].date()), "p", base=salary_b, worked=time_b
            ),
        ]
    )
    result = analyze(model, frame, months)
    effects = result["bridge"].set_index("driver").impact
    assert effects["underlying_salary_level"] == pytest.approx(
        salary_effect, abs=1e-6
    )
    assert effects["worked_hours"] == pytest.approx(time_effect, abs=1e-6)
    assert effects["within_cell_workforce_mix"] == pytest.approx(0, abs=1e-6)
    for driver, table in [
        ("underlying_salary_level", "salary_cells"),
        ("worked_hours", "worked_fraction_cells"),
        ("within_cell_workforce_mix", "workforce_mix_cells"),
    ]:
        audit = result["diagnostics"][table]
        assert audit.organization_impact.sum() == pytest.approx(
            effects[driver], abs=1e-6
        )
        assert audit.local_impact.sum() == pytest.approx(
            effects[driver], abs=1e-6
        )


def test_unequal_raise_does_not_invent_time_change(model, months):
    """The previously misleading example now has only a salary impact."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "a", base=100, worked=1),
            payroll_row(str(months[0].date()), "b", base=100, worked=0.5),
            payroll_row(str(months[1].date()), "a", base=100, worked=1),
            payroll_row(str(months[1].date()), "b", base=200, worked=0.5),
        ]
    )
    result = analyze(model, frame, months)
    effects = result["bridge"].set_index("driver").impact
    assert effects["underlying_salary_level"] == pytest.approx(50)
    assert effects["worked_hours"] == pytest.approx(0, abs=1e-6)


def test_salary_time_association_with_unchanged_means(model, months):
    """Preserve pairing even when ordinary mean time does not change."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "a", base=100, worked=1),
            payroll_row(str(months[0].date()), "b", base=200, worked=0.5),
            payroll_row(str(months[1].date()), "a", base=100, worked=0.5),
            payroll_row(str(months[1].date()), "b", base=200, worked=1),
        ]
    )
    result = analyze(model, frame, months)
    effects = result["bridge"].set_index("driver").impact
    assert effects["worked_hours"] == pytest.approx(50)
    assert effects["underlying_salary_level"] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("reverse", [False, True])
def test_entry_exit_carries_values_not_workers(model, months, reverse):
    """Entry/exit cannot be mistaken for a raise or changed worked time."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "a", base=100, worked=1),
            payroll_row(str(months[0].date()), "leaver", base=200, worked=0.3),
            payroll_row(str(months[1].date()), "a", base=100, worked=1),
            payroll_row(str(months[1].date()), "joiner", base=300, worked=0.9),
        ]
    )
    if reverse:
        months = months[::-1]
    result = analyze(model, frame, months)
    effects = result["bridge"].set_index("driver").impact
    np.testing.assert_allclose(
        effects[["worked_hours", "underlying_salary_level"]], 0, atol=1e-6
    )
    assert effects["within_cell_workforce_mix"] == pytest.approx(
        -210 if reverse else 210
    )
    assert set(result["assumptions"].method) >= {"carry_employee"}
    a, b = [state["teams"]["Alpha"]["permanent"] for state in result["states"]]
    np.testing.assert_array_equal(
        a["underlying_salary_level"], b["underlying_salary_level"]
    )
    np.testing.assert_array_equal(a["worked_hours"], b["worked_hours"])
    assert result["states"][0]["workforce_scale"] == 2


def test_transfer_matches_employee_across_teams(model, months):
    """Use a mover's own salary/time history without pooling team averages."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "mover", base=100, worked=0.5),
            payroll_row(
                str(months[1].date()),
                "mover",
                team="Beta",
                base=120,
                worked=0.5,
                grade="G2",
            ),
            *[
                payroll_row(str(month.date()), "a", base=500, worked=1)
                for month in months
            ],
            *[
                payroll_row(
                    str(month.date()), "b", team="Beta", base=1000, worked=1
                )
                for month in months
            ],
        ]
    )
    result = analyze(model, frame, months)
    for team in ("Alpha", "Beta"):
        left, right = [
            state["teams"][team]["permanent"] for state in result["states"]
        ]
        index = left["_employees"].index("mover")
        assert left["underlying_salary_level"][index] == 100
        assert right["underlying_salary_level"][index] == 120
        assert result["permanent"][team]["impacts"][
            "worked_hours"
        ] == pytest.approx(0, abs=1e-6)
    assert result["diagnostics"]["matched"]["summary"]["team_transfers"] == 1
    assert result["bridge"].impact.sum() == pytest.approx(10)


@pytest.mark.parametrize("raise_amount", [0, 20])
def test_promotion_does_not_infer_raise_cause(model, months, raise_amount):
    """Promotions retain actual base changes; grade/membership can offset."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "p", grade="G1", base=100),
            payroll_row(
                str(months[1].date()), "p", grade="G2", base=100 + raise_amount
            ),
        ]
    )
    result = analyze(model, frame, months)
    effects = result["bridge"].set_index("driver").impact
    assert effects["underlying_salary_level"] == pytest.approx(
        raise_amount, abs=1e-6
    )
    assert effects["worked_hours"] == pytest.approx(0, abs=1e-6)


def test_permanent_contract_transition_has_no_inferred_inputs(model, months):
    """A transition supplies one permanent observation, not two pay levels."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "a"),
            payroll_row(str(months[1].date()), "a"),
            payroll_row(
                str(months[0].date()), "switcher", contract="ГПД", rate=70
            ),
            payroll_row(
                str(months[1].date()), "switcher", base=200, worked=0.6
            ),
        ]
    )
    result = analyze(model, frame, months)
    effects = result["permanent"]["Alpha"]["impacts"]
    np.testing.assert_allclose(
        effects[["worked_hours", "underlying_salary_level"]], 0, atol=1e-6
    )
    carries = result["diagnostics"]["employee_alignment"].query(
        "method == 'carry_employee'"
    )
    assert set(carries.employee_id) == {"switcher"}


def test_profiles_and_cached_costs_match_direct(model, complete_frame, months):
    """Check cached/direct salary equality, including sparse fallback."""
    comparison, (a, b) = permanent_pair(model, complete_frame, months)
    cache = model["prepare_salary_cache"](a, b)
    assert len(cache) == 8
    for bits in product((0, 1), repeat=3):
        hybrid = dict(a)
        hybrid.update(
            {
                key: (a, b)[bit][key]
                for key, bit in zip(model["SALARY_KEYS"], bits)
            }
        )
        direct = model["permanent_cell_costs"](hybrid)
        cached = model["permanent_cell_costs"](
            dict(hybrid, _salary_cache=cache)
        )
        np.testing.assert_allclose(direct, cached, rtol=1e-10, atol=1e-6)
    for state in (a, b):
        for profile in state["within_cell_workforce_mix"].values():
            assert profile["weights"].sum() == pytest.approx(1)
            assert (profile["weights"] >= 0).all()
    canonical = model["adapt_production"](complete_frame)
    observed, _, _ = model["build_state"](
        canonical, months[0], model["make_policy"](canonical)
    )
    assert model["cost"](observed) == pytest.approx(
        complete_frame.loc[
            complete_frame.period.eq(str(months[0].date())), "actual_cost"
        ].sum()
    )
    assert "_salary_cache" not in a


@pytest.mark.parametrize(
    "invalid", ["weight", "index", "axis", "legacy", "salary", "time"]
)
def test_invalid_employee_states_fail(model, complete_frame, months, invalid):
    """Reject malformed profiles and unsupported old reference schemas."""
    _, (state, _) = permanent_pair(model, complete_frame, months)
    state = deepcopy(state)
    profile = next(iter(state["within_cell_workforce_mix"].values()))
    if invalid == "weight":
        profile["weights"][0] = -1
    elif invalid == "index":
        profile["indices"][0] = len(state["_employees"])
    elif invalid == "axis":
        state["_employees"] = ("duplicate",) * len(state["_employees"])
    elif invalid == "legacy":
        del state["_employees"]
    elif invalid == "salary":
        state["underlying_salary_level"][0] = 0
    else:
        state["worked_hours"][0] = -0.1
    with pytest.raises(ValueError):
        model["permanent_cost"](state)


def test_exact_limit_fails_above_extension(model):
    """Never silently approximate a game above the documented limit."""
    drivers = tuple(model["Driver"](str(i), str(i), "test") for i in range(16))
    with pytest.raises(ValueError, match="1–15"):
        model["exact_game"]({}, {}, drivers, lambda state: 0)


def test_copied_team_keeps_complete_economics_assumption(model, months):
    """A newly observed team still copies its entire observed economics."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "mover", base=100, worked=0.5),
            payroll_row(
                str(months[1].date()),
                "mover",
                team="Beta",
                base=200,
                worked=0.8,
            ),
            *[payroll_row(str(month.date()), "anchor") for month in months],
        ]
    )
    result = analyze(model, frame, months)
    np.testing.assert_allclose(
        result["permanent"]["Beta"]["impacts"], 0, atol=1e-6
    )
    assert (
        result["assumptions"]
        .query("team == 'Beta' and method == 'carry_observed'")
        .shape[0]
        == 1
    )
    effective = result["diagnostics"]["employee_inputs"].query(
        "team == 'Beta'"
    )
    assert (effective.base_salary == 200).all()
    assert (effective.worked_fraction == 0.8).all()
    assert (effective.source_period == months[1]).all()


def test_legacy_reference_rejected_by_public_workflow(
    model, complete_frame, months
):
    """Do not silently fabricate pairing from an old aggregate reference."""
    canonical = model["adapt_production"](complete_frame)
    comparison = model["build_comparison_states"](canonical, *months)
    reference = deepcopy(
        comparison["states"][0]["teams"]["Alpha"]["permanent"]
    )
    del reference["within_cell_workforce_mix"]
    missing = complete_frame.loc[
        ~(
            complete_frame.period.eq("2025-03-01")
            & complete_frame.motivation_type.eq("ТК")
        )
    ]
    with pytest.raises(ValueError, match="Legacy permanent state/reference"):
        analyze(
            model,
            missing,
            months,
            references={(months[0], "Alpha"): {"permanent": reference}},
        )


def test_profile_audit_matches_full_cell_game(model, complete_frame, months):
    """Verify reduced diagnostic weights against full-player enumeration."""
    canonical = model["adapt_production"](complete_frame)
    pair = model["build_comparison_states"](canonical, *months)
    a, b = [state["teams"]["Alpha"]["permanent"] for state in pair["states"]]
    registry = model["DRIVERS"]
    specs = model["drivers_at"](model["PARENT"], registry)
    # Full exact scalar game for one cell retains all payment players.
    cell = (0, 0)
    cache = model["prepare_salary_cache"](a, b)
    game = model["exact_game"](
        a,
        b,
        specs,
        lambda state: model["permanent_cell_costs"](
            dict(state, _salary_cache=cache)
        )[cell],
    )
    overall = model["permanent_games"](*pair["states"])["Alpha"]
    for key in model["SALARY_KEYS"]:
        audit = model["salary_cell_audit"](
            a,
            b,
            overall,
            *pair["coverage"],
            *pair["fallbacks"],
            "Alpha",
            1,
            1,
            1,
            key=key,
        )
        assert audit.iloc[0].unit_impact == pytest.approx(
            game["impacts"][key], abs=1e-6
        )
