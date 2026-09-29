"""Test estimator selection without weakening the accounting invariants."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from conftest import payroll_row
from test_analysis import analyze


def assert_rng_equal(left, right):
    """Compare NumPy's legacy generator state, including cached values."""
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_sampling_budget_vector_equivalence_and_rng(model):
    """Verify real SHAP cycles, vector additivity and RNG restoration."""
    specs = tuple(model["Driver"](str(i), str(i), "test") for i in range(3))
    a, b = dict.fromkeys("012", 0), dict.fromkeys("012", 1)

    def vector(state):
        """Combine a cubic interaction with an additive contribution."""
        return np.array(
            [
                120 * state["0"] * state["1"] * state["2"],
                15 * state["0"] - 25 * state["1"],
            ]
        )

    rng = np.random.get_state()
    sampled = model["permutation_game"](
        a, b, specs, vector, n_permutations=7, random_seed=19
    )
    assert_rng_equal(rng, np.random.get_state())
    scalar = model["permutation_game"](
        a,
        b,
        specs,
        lambda s: vector(s).sum(),
        n_permutations=7,
        random_seed=19,
    )
    np.testing.assert_allclose(
        sampled["impacts"], scalar["impacts"], atol=1e-6
    )
    np.testing.assert_allclose(
        sampled["cell_impacts"].sum(axis=0), vector(b) - vector(a)
    )
    assert sampled["audit"]["evaluation_budget"] == 7 * 7
    assert sampled["audit"]["model_evaluations"] == 7 * 7 + 2
    assert not sampled["audit"]["exhaustive"]
    exact = model["exact_game"](a, b, specs, lambda s: vector(s).sum())
    assert sampled["impacts"].sum() == pytest.approx(exact["impacts"].sum())
    assert not np.allclose(sampled["impacts"], exact["impacts"])


@pytest.mark.parametrize(
    "option,value",
    [
        ("explainer", "auto"),
        ("explainer", None),
        ("n_permutations", 0),
        ("n_permutations", -1),
        ("n_permutations", 1.5),
        ("n_permutations", True),
        ("random_seed", -1),
        ("random_seed", 1.5),
        ("random_seed", False),
    ],
)
def test_invalid_options_fail_early(model, option, value):
    """Reject bad options even before input adaptation or empty games."""
    with pytest.raises(ValueError, match=option):
        model["analyze_labor_cost"](
            None, "2025-03", "2025-04", **{option: value}
        )
    with pytest.raises(ValueError, match=option):
        model["decompose"]({}, {}, **{option: value})


def test_permutation_hierarchy_cell_audits_and_repeatability(
    model, complete_frame, months
):
    """Use the same sampled allocation in both bridges and all cell audits."""
    rng = np.random.get_state()
    result = analyze(
        model,
        complete_frame,
        months,
        explainer="permutation",
        n_permutations=16,
        random_seed=73,
    )
    assert_rng_equal(rng, np.random.get_state())
    summary = result["driver_bridge"].set_index("driver").impact
    for driver, rows in result["bridge"].groupby("driver"):
        assert summary[driver] == pytest.approx(rows.impact.sum(), abs=1e-6)
    a, b = result["states"]
    repeated = model["decompose"](
        a, b, explainer="permutation", n_permutations=16, random_seed=73
    )
    reverse = model["decompose"](
        b, a, explainer="permutation", n_permutations=16, random_seed=73
    )
    np.testing.assert_allclose(
        result["bridge"].impact, repeated["bridge"].impact
    )
    np.testing.assert_allclose(
        result["bridge"].impact, -reverse["bridge"].impact, atol=1e-6
    )
    np.testing.assert_allclose(
        result["driver_bridge"].impact, repeated["driver_bridge"].impact,
    )
    np.testing.assert_allclose(
        result["driver_bridge"].impact, -reverse["driver_bridge"].impact,
        atol=1e-6,
    )
    for key, table in [
        ("underlying_salary_level", "salary_cells"),
        ("worked_hours", "worked_fraction_cells"),
        ("within_cell_workforce_mix", "workforce_mix_cells"),
    ]:
        audit = result["diagnostics"][table]
        bridge = result["bridge"].query("driver == @key")
        assert audit.organization_impact.sum() == pytest.approx(
            bridge.impact.sum(), abs=1e-6
        )
    upper = result["coalition_audit"].query("game != 'permanent'")
    assert upper.exhaustive.all()
    assert set(upper.method) == {"exact"}
    assert result["permanent"]["Alpha"]["audit"]["method"] == "permutation"
    assert result["explainer_config"]["n_permutations"] == 16
    same = model["decompose"](
        a, deepcopy(a), explainer="permutation", n_permutations=2
    )
    np.testing.assert_allclose(same["bridge"].impact, 0, atol=1e-6)


def test_twenty_drivers_and_default_exact_limit(model, complete_frame, months):
    """Six new registered payments need no attribution/reporting edits."""
    registry = model["DRIVERS"]
    frame = complete_frame.copy()
    permanent = frame.motivation_type.eq("ТК")
    for i in range(6):
        key = f"extra_{i}"
        frame[key] = np.where(
            frame.period.eq("2025-03-01"), -20 - i, 30 + 2 * i
        )
        eligible = i % 2 == 0
        frame.loc[permanent, "actual_cost"] += frame.loc[permanent, key] * (
            frame.loc[permanent, "reg_coef"] if eligible else 1
        )
        registry += (
            model["payment_driver"](
                key, key, key, apply_regional_coefficient=eligible
            ),
        )
    with pytest.raises(ValueError, match="select explainer='permutation'"):
        analyze(model, frame, months, registry=registry)
    result = analyze(
        model,
        frame,
        months,
        registry=registry,
        explainer="permutation",
        n_permutations=16,
    )
    assert result["permanent"]["Alpha"]["audit"]["drivers"] == 20
    assert len(result["permanent"]["Alpha"]["impacts"]) == 20
    assert set(f"extra_{i}" for i in range(6)) <= set(result["bridge"].driver)


def test_team_order_does_not_change_samples(model, complete_frame, months):
    """Stable team seeds make sampling independent of outer-axis order."""
    other = complete_frame.copy()
    other["lvl7_mapped_management_unit_nm"] = "Beta"
    other["mdm_employee_rk_hash"] += "-beta"
    canonical = model["adapt_production"](pd.concat([complete_frame, other]))
    a, b = model["build_comparison_states"](canonical, *months)["states"]
    forward = model["permanent_games"](
        a, b, explainer="permutation", n_permutations=8
    )
    a, b = deepcopy(a), deepcopy(b)
    for state in (a, b):
        state["_teams"] = state["_teams"][::-1]
        state["team_mix"] = state["team_mix"][::-1]
    backward = model["permanent_games"](
        a, b, explainer="permutation", n_permutations=8
    )
    for team in forward:
        np.testing.assert_array_equal(
            forward[team]["impacts"], backward[team]["impacts"]
        )
        assert (
            forward[team]["audit"]["random_seed"]
            == backward[team]["audit"]["random_seed"]
        )


@pytest.mark.parametrize(
    "base,worked,expected_salary,expected_time",
    [
        (110000, 0.625, 6250, 0),
        (100000, 0.75, 0, 12500),
        (110000, 0.75, 6875, 13125),
    ],
)
def test_sampled_salary_time_independence(
    model, months, base, worked, expected_salary, expected_time
):
    """Isolated two-way effects remain exact under antithetic sampling."""
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0].date()), "p", base=100000, worked=0.625),
            payroll_row(str(months[1].date()), "p", base=base, worked=worked),
        ]
    )
    result = analyze(
        model, frame, months, explainer="permutation", n_permutations=2
    )
    impacts = result["bridge"].set_index("driver").impact
    assert impacts["underlying_salary_level"] == pytest.approx(
        expected_salary, abs=1e-6
    )
    assert impacts["worked_hours"] == pytest.approx(expected_time, abs=1e-6)


def test_sampled_signed_zero_parent(model, months):
    """Propagate signed children without dividing by a zero parent change."""
    frame = pd.DataFrame(
        [
            payroll_row(
                str(months[0].date()),
                "p",
                sum_pay_bl_reg_base=-150,
                sum_pay_prazdnik_reg_base=50,
            ),
            payroll_row(
                str(months[1].date()),
                "p",
                sum_pay_bl_reg_base=-200,
                sum_pay_prazdnik_reg_base=100,
            ),
        ]
    )
    result = analyze(
        model, frame, months, explainer="permutation", n_permutations=2
    )
    impacts = result["bridge"].set_index("driver").impact
    assert impacts["sick_leave_pay"] == pytest.approx(-50)
    assert impacts["holiday_pay"] == pytest.approx(50)
    assert result["bridge"].share_of_net_change.isna().all()
