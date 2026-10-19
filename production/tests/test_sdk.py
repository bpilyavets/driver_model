"""Analyst declaration, public API and counterfactual extension acceptance."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import payroll_row

from labor_cost import (
    AnalysisRequest,
    DatasetSnapshot,
    InputSpec,
    ModelSpec,
    WorkforceStructure,
    analyze,
    compile_model,
    cost,
    field,
    labor_cost_model,
)


def request(months, **options):
    """Use small sampling budgets for integration, not accuracy tests."""
    return AnalysisRequest(
        str(months[0]),
        str(months[1]),
        explainer="permutation",
        n_permutations=4,
        **options,
    )


def test_compiled_model_parity(notebook_model, complete_frame, months):
    """The declarative built-in has the same exact bridge as the PoC."""
    compiled = compile_model(labor_cost_model())
    restored = compile_model(ModelSpec.from_dict(compiled.spec.to_dict()))
    assert restored.fingerprint == compiled.fingerprint
    assert restored.inspect()["formula"] == compiled.inspect()["formula"]
    original = notebook_model["analyze_labor_cost"](
        complete_frame, *months, source_cost_column="actual_cost"
    )
    result = analyze(
        compiled,
        DatasetSnapshot("Your payroll", "v1", complete_frame),
        AnalysisRequest(*map(str, months), source_cost_column="actual_cost"),
    )
    for name in ("bridge", "driver_bridge", "hierarchy", "team_summary"):
        pd.testing.assert_frame_equal(original[name], result.table(name))
    for state, period in zip(result.states, months):
        expected = complete_frame.loc[
            pd.to_datetime(complete_frame.period).eq(period), "actual_cost"
        ].sum()
        assert cost(state, compiled) == pytest.approx(expected)


def test_input_integrity_and_team_views(public_result, complete_frame, months):
    """Views never redefine the population or mutate calculation results."""
    dataset = DatasetSnapshot("payroll", "v1", complete_frame)
    copied = dataset.frame
    copied.iloc[0, 0] = "changed"
    pd.testing.assert_frame_equal(dataset.frame, complete_frame)
    bridge = public_result.table("bridge")
    view = public_result.select_teams(["Alpha"])
    assert view.run_id == public_result.run_id
    assert set(view.local_bridges) == {"Alpha"}
    assert view.organization_scope.driver.tolist() == [
        "workforce_scale",
        "team_mix",
    ]
    pd.testing.assert_frame_equal(bridge, public_result.table("bridge"))
    with pytest.raises(ValueError, match="Unknown teams"):
        public_result.select_teams(["missing"])
    with pytest.raises(ValueError, match="sequence"):
        public_result.select_teams("Alpha")


def test_additive_payment_extension(months):
    """A declared signed payment reaches source, cost and bridges together."""
    base = labor_cost_model()
    bonus = InputSpec(
        "retention_pay",
        "retention_amount",
        "Retention pay",
        "payment",
        "currency",
    )
    spec = replace(
        base,
        version="2.0.0",
        inputs=base.inputs + (bonus,),
        formula=base.formula + field("retention_pay"),
    )
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0]), "a", retention_amount=-10),
            payroll_row(str(months[1]), "a", retention_amount=30),
        ]
    )
    frame["actual_cost"] += frame.retention_amount
    result = analyze(
        spec,
        DatasetSnapshot("test", "1", frame),
        request(months, source_cost_column="actual_cost"),
    )
    impacts = result.table("driver_bridge").set_index("driver").impact
    assert impacts["retention_pay"] == pytest.approx(40)
    assert result.table("validation").passed.all()
    reconstructed = analyze(
        spec, DatasetSnapshot("test", "1", frame), request(months)
    )
    np.testing.assert_allclose(
        reconstructed.table("canonical_data").source_cost, frame.actual_cost
    )
    assert "reconstructed declared payroll" in set(
        reconstructed.table("canonical_data").reference_basis
    )


@pytest.mark.parametrize("explainer", ["exact", "permutation"])
def test_aligned_product_extension(months, explainer):
    """A multiplicative input becomes its own player without fake raises."""
    base = labor_cost_model()
    factor = InputSpec(
        "base_multiplier",
        "pay_factor",
        "Base pay multiplier",
        "aligned",
        "dimensionless",
        "positive",
    )
    earned = (
        field("regional_coefficient")
        * field("base_salary")
        * field("worked_fraction")
    )
    spec = replace(
        base,
        version="2.1.0",
        inputs=base.inputs + (factor,),
        formula=base.formula - earned + earned * field("base_multiplier"),
    )
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0]), "a", pay_factor=1.0),
            payroll_row(str(months[1]), "a", pay_factor=1.2),
        ]
    )
    frame["actual_cost"] *= frame.pay_factor
    result = analyze(
        spec,
        DatasetSnapshot("test", "1", frame),
        AnalysisRequest(
            *map(str, months),
            source_cost_column="actual_cost",
            explainer=explainer,
            n_permutations=4,
        ),
    )
    impacts = result.table("driver_bridge").set_index("driver").impact
    assert impacts.base_multiplier == pytest.approx(20)
    assert impacts.underlying_salary_level == pytest.approx(0, abs=1e-8)
    assert impacts.worked_hours == pytest.approx(0, abs=1e-8)
    assert result.table("validation").passed.all()


def test_aligned_joint_information_and_transfer(months):
    """Preserve pairing and carry only the same employee's values."""
    base = labor_cost_model()
    factor = InputSpec(
        "base_multiplier",
        "factor",
        "Multiplier",
        "aligned",
        "dimensionless",
        "positive",
    )
    earned = field("regional_coefficient") * field("base_salary")
    spec = replace(
        base,
        version="3",
        inputs=base.inputs + (factor,),
        formula=base.formula + earned * field("base_multiplier"),
    )
    frame = pd.DataFrame(
        [
            payroll_row(str(months[0]), "a", team="Alpha", base=100, factor=2),
            payroll_row(str(months[0]), "b", team="Beta", base=200, factor=1),
            payroll_row(str(months[1]), "a", team="Beta", base=100, factor=2),
            payroll_row(str(months[1]), "b", team="Alpha", base=200, factor=1),
            payroll_row(str(months[1]), "c", team="Alpha", base=150, factor=3),
        ]
    )
    frame.actual_cost += frame.base_stavka * frame.factor
    result = analyze(
        spec,
        DatasetSnapshot("test", "1", frame),
        request(months, source_cost_column="actual_cost"),
    )
    impacts = result.table("driver_bridge").set_index("driver").impact
    assert impacts.base_multiplier == pytest.approx(0, abs=1e-8)
    assert impacts.underlying_salary_level == pytest.approx(0, abs=1e-8)
    assert result.table("validation").passed.all()
    assert "base_multiplier" in set(
        result.table("diagnostics/aligned_inputs").driver
    )


def test_composite_composition_dimensions(complete_frame, months):
    """Declared joint grade dimensions use one complete conditional matrix."""
    frame = complete_frame.assign(position=["one", "two"] * 5)
    model = replace(
        labor_cost_model(),
        version="2",
        structure=replace(
            WorkforceStructure(), grade_columns=("grade", "position")
        ),
    )
    result = analyze(
        model,
        DatasetSnapshot("test", "1", frame),
        request(months, source_cost_column="actual_cost"),
    )
    assert result.table("validation").passed.all()
    assert '["G1", "one"]' in set(result.table("canonical_data").grade)


def test_compile_rejects_unsupported_semantics():
    """Reject invalid joint representation, units, bindings and hierarchy."""
    base = labor_cost_model()
    with pytest.raises(ValueError, match="separately aggregated"):
        compile_model(
            replace(
                base,
                formula=base.formula
                + field("bonus_kpi") * field("worked_fraction"),
            )
        )
    with pytest.raises(ValueError, match="currency units"):
        compile_model(
            replace(base, formula=field("base_salary") * field("base_salary"))
        )
    with pytest.raises(ValueError, match="unbound"):
        compile_model(replace(base, formula=base.formula + field("unknown")))
    with pytest.raises(ValueError, match="nonlinear parents"):
        compile_model(
            replace(
                base,
                structure=replace(
                    base.structure, propagation="divide_by_parent_change"
                ),
            )
        )
    distributions = base.structure.distributions
    changed = replace(distributions[-1], given=("team", "permanent"))
    with pytest.raises(ValueError, match="conditioning"):
        compile_model(
            replace(
                base,
                structure=replace(
                    base.structure,
                    distributions=distributions[:-1] + (changed,),
                ),
            )
        )


def test_standalone_import_without_notebook_or_shap(tmp_path):
    """Installed costing imports neither SHAP nor frontend modules."""
    script = """
import sys
from labor_cost import cost, default_model
model = default_model()
assert model.spec.model_id == 'labor-cost'
assert 'shap' not in sys.modules
assert 'matplotlib' not in sys.modules
assert 'IPython' not in sys.modules
assert 'streamlit' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_coupled_source_and_immutable_declarations():
    """Reuse one driver for one source and freeze supplied collections."""
    base = labor_cost_model()
    declarations = list(base.inputs)
    spec = replace(base, inputs=declarations)
    identity = spec.fingerprint
    declarations.clear()
    assert spec.fingerprint == identity
    duplicate = InputSpec(
        "salary_copy",
        "base_stavka",
        "Copied salary",
        "aligned",
        "currency",
        "positive",
    )
    with pytest.raises(ValueError, match="sharing a source"):
        compile_model(replace(base, inputs=base.inputs + (duplicate,)))
    with pytest.raises(ValueError, match="nonfinite coefficient"):
        compile_model(replace(base, formula=base.formula * 1e308 * 1e308))


def test_custom_state_requires_matching_model(public_result):
    """Prevent standalone costing from silently choosing a different model."""
    from copy import deepcopy

    state = deepcopy(public_result.states[0])
    state["_model_fingerprint"] = "another-model"
    with pytest.raises(ValueError, match="compiled model"):
        cost(state)
    with pytest.raises(ValueError, match="fingerprints differ"):
        cost(state, labor_cost_model())
