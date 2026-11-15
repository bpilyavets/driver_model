"""Independent aggregate identities and alternative-model acceptance."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labor_cost import (
    AnalysisRequest,
    Catalog,
    DatasetSnapshot,
    Factor,
    FieldSpec,
    Game,
    ModelGraph,
    ModelSpec,
    RowCost,
    SourceSchema,
    Switch,
    analyze,
    build_state,
    compile_model,
    cost,
    export_run,
    field,
    labor_cost_model,
    load_run,
    local_game,
    organization_game,
    outer_factors,
    publish_model,
    render_formula,
    render_model,
    report,
    sum_over,
    value,
)
from labor_cost.graph_cost import Tensor

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
spec = importlib.util.spec_from_file_location(
    "comparison_models", EXAMPLES / "comparison_models.py"
)
examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(examples)


def aggregate_model(*, basis="actual", fraction=False, zero="exposure_mean"):
    """Declare a standalone model unrelated to built-in payroll schemas."""
    time = "dimensionless" if fraction else "time"
    normalized = "1/exposure" if fraction else "time/exposure"
    factors = outer_factors() + (
        Factor("mix", "distribution", dimensions=("cell",)),
        Factor(
            "rate",
            "weighted_mean",
            dimensions=("cell",),
            source="r",
            weight="h",
            basis=basis,
            zero_weight=zero,
            fallbacks=((),),
        ),
        Factor(
            "time",
            "mean",
            dimensions=("cell",),
            source="h",
            basis=basis,
            fallbacks=((),),
        ),
    )
    unit = Game(
        "unit",
        sum_over(value("mix") * value("rate") * value("time"), "cell"),
        (Switch("mix"), Switch("rate"), Switch("time")),
    )
    return ModelSpec(
        "aggregate-test",
        "1",
        graph=ModelGraph(
            SourceSchema(exposure="e", dimensions={"cell": "cell"}),
            (),
            (
                FieldSpec(
                    "r",
                    "r",
                    units="currency" if fraction else "currency/time",
                    constraint="positive",
                ),
                FieldSpec(
                    "h",
                    "h" if basis == "actual" else "normalized_h",
                    units=time if basis == "actual" else normalized,
                    constraint="nonnegative",
                ),
                FieldSpec(
                    "actual_h", "h", units=time, constraint="nonnegative"
                ),
            ),
            (RowCost("all", field("r") * field("actual_h")),),
            factors,
            (unit,),
            "unit",
            organization_game("unit"),
            local_game("unit"),
        ),
    )


def frame():
    """Return a two-period hours reassignment with fixed employee rates."""
    result = pd.DataFrame(
        [
            (period, person, "staff", "A", "G1", rate, hours, exposure)
            for period, hours_pair in (
                ("2026-01-01", (10.0, 20.0)),
                ("2026-08-01", (20.0, 10.0)),
            )
            for person, rate, hours, exposure in zip(
                ("a", "b"), (100.0, 200.0), hours_pair, (0.5, 1.0)
            )
        ],
        columns=[
            "period",
            "employee_id",
            "employment_type",
            "team",
            "cell",
            "r",
            "h",
            "e",
        ],
    )
    return prepare(result)


def prepare(rows):
    """Rebuild independent source totals and normalized input columns."""
    rows = rows.copy()
    rows["actual"] = rows.r * rows.h
    rows["normalized_h"] = rows.h / rows.e
    return rows


def run(model, rows, **options):
    """Analyze independent actual payroll with explicit comparison settings."""
    return analyze(
        model,
        DatasetSnapshot("test", "1", rows),
        AnalysisRequest(
            "2026-01", "2026-08", source_cost_column="actual", **options
        ),
    )


@pytest.mark.parametrize("basis", ["actual", "per_exposure"])
@pytest.mark.parametrize("fraction", [False, True])
@pytest.mark.parametrize("method", ["exact", "permutation"])
def test_identity_and_reassignment(basis, fraction, method):
    """Preserve paired payroll while assigning changed hour mix to rate."""
    model = aggregate_model(basis=basis, fraction=fraction)
    rows = frame()
    original = rows.copy(deep=True)
    result = run(model, rows, explainer=method, n_permutations=8)
    assert result.table("validation").passed.all()
    bridge = result.table("driver_bridge").set_index("driver").impact
    assert bridge["unit::rate"] == pytest.approx(-1000)
    assert bridge["unit::time"] == pytest.approx(0)
    evidence = result.table("diagnostics/aggregate_factors")
    for period, group in evidence.groupby("period"):
        values = group.set_index("factor").value
        source = rows.loc[pd.to_datetime(rows.period).eq(period)]
        assert source.e.sum() * values.rate * values.time == pytest.approx(
            source.actual.sum()
        )
    assert cost(
        build_state(model, DatasetSnapshot("t", "1", rows), "2026-01"), model
    ) == pytest.approx(5000)
    pd.testing.assert_frame_equal(rows, original)
    reverse = analyze(
        model,
        DatasetSnapshot("test", "1", rows),
        AnalysisRequest("2026-08", "2026-01"),
    )
    np.testing.assert_allclose(
        reverse.table("driver_bridge").impact,
        -result.table("driver_bridge").impact,
        atol=1e-6,
    )


@pytest.mark.parametrize("both", [False, True])
def test_zero_hours_and_provenance(both):
    """Complete only the undefined rate, without borrowing worked hours."""
    rows = frame()
    mask = (
        np.ones(len(rows), dtype=bool)
        if both
        else rows.period.eq("2026-01-01")
    )
    rows.loc[mask, "h"] = 0
    result = run(aggregate_model(), prepare(rows))
    assumptions = result.table("assumptions")
    assert set(assumptions.method) == {"zero_weight_exposure_mean"}
    evidence = result.table("diagnostics/aggregate_factors")
    selected = evidence.loc[evidence.method.eq("zero_weight_exposure_mean")]
    assert (selected.denominator == 0).all()
    np.testing.assert_allclose(selected.value, 500 / 3)
    assert set(evidence.completion_method) == {"observed"}
    assert (
        evidence.loc[
            evidence.factor.eq("time")
            & evidence.period.eq(pd.Timestamp("2026-01-01")),
            "value",
        ]
        == 0
    ).all()
    with pytest.raises(ValueError, match="zero total weight"):
        run(aggregate_model(zero="error"), prepare(rows))


def test_tiny_weights_are_not_zero():
    """Positive workloads never activate zero-workload completion."""
    rows = frame()
    rows["h"] *= 1e-200
    result = run(aggregate_model(), prepare(rows))
    assert result.table("assumptions").empty
    rates = result.table("diagnostics/aggregate_factors").query(
        "factor == 'rate'"
    )
    assert (rates.denominator > 0).all()
    assert rates.value.iloc[0] == pytest.approx(500 / 3)


@pytest.mark.parametrize("change", ["rate", "hours"])
def test_single_input_changes(change):
    """Uniform rate or time changes have the intended independent driver."""
    rows = frame()
    rows.loc[rows.period.eq("2026-08-01"), "h"] = [10.0, 20.0]
    target = "r" if change == "rate" else "h"
    rows.loc[rows.period.eq("2026-08-01"), target] *= 1.2
    result = run(aggregate_model(), prepare(rows))
    bridge = result.table("driver_bridge").set_index("driver").impact
    assert bridge["unit::" + ("rate" if change == "rate" else "time")] == (
        pytest.approx(1000)
    )
    assert bridge["unit::" + ("time" if change == "rate" else "rate")] == (
        pytest.approx(0)
    )


@pytest.mark.parametrize(
    "edit,match",
    [
        ({"weight": None}, "declared field"),
        ({"weight": "absent"}, "declared field"),
        ({"weight": "r"}, "weight units"),
        ({"zero_weight": "zero"}, "zero_weight"),
        ({"source": "absent"}, "declared field"),
        ({"given": ("cell",)}, "repeated dimensions"),
        ({"dimensions": ("unknown",)}, "unmapped dimension"),
        ({"formula": field("r")}, "use source"),
        ({"kind": "mean"}, "weight options"),
    ],
)
def test_invalid_factors(edit, match):
    """Reject unsupported declarations before state construction."""
    model = aggregate_model()
    factors = tuple(
        replace(f, **edit) if f.name == "rate" else f
        for f in model.graph.factors
    )
    with pytest.raises(ValueError, match=match):
        compile_model(
            replace(model, graph=replace(model.graph, factors=factors))
        )


def test_invalid_weights_and_references():
    """Enforce weight declarations and supplied-state constraints."""
    model = aggregate_model()
    fields = tuple(
        replace(f, constraint="signed") if f.name == "h" else f
        for f in model.graph.fields
    )
    with pytest.raises(ValueError, match="weight must be nonnegative"):
        compile_model(
            replace(model, graph=replace(model.graph, fields=fields))
        )
    rows = frame()
    rows.loc[0, "h"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        run(model, prepare(rows))
    state = build_state(model, DatasetSnapshot("t", "1", frame()), "2026-01")
    state["teams"]["A"]["populations"]["all"]["time"] = Tensor(
        ("cell",), (("G1",),), np.array([-1.0])
    )
    with pytest.raises(ValueError, match="nonnegative"):
        cost(state, model)


def test_sparse_and_absent_support():
    """Keep sparse donors local and whole-team completion distinguishable."""
    rows = frame()
    rows.loc[rows.period.eq("2026-08-01"), "cell"] = "G2"
    result = run(aggregate_model(), rows)
    assert not result.table("diagnostics/fallbacks").empty
    assert "sparse_fallback" in set(
        result.table("diagnostics/aggregate_factors").method
    )
    rows.loc[rows.period.eq("2026-08-01"), "team"] = "B"
    model = aggregate_model()
    result = run(model, rows)
    evidence = result.table("diagnostics/aggregate_factors")
    copied = evidence.loc[evidence.completion_method.eq("carry_observed")]
    assert not copied.empty and copied.numerator.notna().all()
    with pytest.raises(ValueError, match="require a reference"):
        run(model, rows, missing_economics="require_reference")
    references = {
        (s["period"], team): {"team_state": s["teams"][team]}
        for s in result.states
        for team, exposure in s["exposure_by_team"].items()
        if exposure == 0
    }
    explicit = run(
        model,
        rows,
        references=references,
        missing_economics="require_reference",
    )
    records = explicit.table("diagnostics/aggregate_factors")
    selected = records.loc[
        records.completion_method.eq("explicit_team_reference")
    ]
    assert selected.numerator.isna().all()


def comparison_frame():
    """Use the guide's fields with a fixed coefficient and signed bonus."""
    rows = frame().rename(
        columns={
            "r": "permanent_rate",
            "h": "permanent_hours",
            "cell": "grade",
        }
    )
    rows["employment_type"] = "permanent"
    rows["reg_coef"] = 1.2
    rows["permanent_bonus"] = -30.0
    rows["actual"] = (rows.permanent_rate * rows.permanent_hours - 30) * 1.2
    return rows


def test_three_models_and_portability(tmp_path):
    """All specifications preserve payroll while producing distinct bridges."""
    rows = comparison_frame()
    results = {}
    for name, model in examples.comparison_models().items():
        result = run(model, rows, exposure_column="e")
        results[name] = result
        restored = ModelSpec.from_dict(model.to_dict())
        assert restored.fingerprint == model.fingerprint
        assert result.table("validation").passed.all()
        export_run(result, tmp_path / name)
        loaded = load_run(tmp_path / name)
        assert loaded.states is None
        pd.testing.assert_frame_equal(
            loaded.table("diagnostics/aggregate_factors"),
            result.table("diagnostics/aggregate_factors"),
        )
        charts = report(loaded, show=False)
        assert "organization" in charts.figures
    aligned = results["aligned"].table("driver_bridge").set_index("driver")
    effective = results["effective"].table("driver_bridge").set_index("driver")
    assert aligned.loc["permanent::permanent_rate", "impact"] == (
        pytest.approx(0, abs=1e-6)
    )
    assert effective.loc["permanent::effective_rate", "impact"] == (
        pytest.approx(-1200)
    )
    assert all(
        r.manifest["dataset"]["selected_fingerprint"]
        == results["aligned"].manifest["dataset"]["selected_fingerprint"]
        for r in results.values()
    )
    model = examples.comparison_models()["effective"]
    publish_model(
        model, tmp_path / "model.json", validated_result=results["effective"]
    )
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": model.model_id,
                        "version": model.version,
                        "path": "model.json",
                        "fingerprint": model.fingerprint,
                    }
                ],
                "datasets": [],
            }
        )
    )
    assert (
        Catalog(tmp_path / "catalog.json")
        .model(model.model_id, model.version)
        .fingerprint
        == model.fingerprint
    )


def test_previews_and_old_fingerprints():
    """Show aggregate definitions without changing existing identities."""
    expected = (
        "8c1c5816b8465efa77ba324fa20cfaacea3ad80f896daebe56c07e1a09c5e990",
        "cbdb8d7123a97113dc03eac507be17d20ecffa23f59fc71bbaffafa555f874ef",
    )
    for version, fingerprint in enumerate(expected, 1):
        model = labor_cost_model(schema_version=version)
        assert model.fingerprint == fingerprint
        assert ModelSpec.from_dict(model.to_dict()).fingerprint == fingerprint
    model = aggregate_model()
    preview = render_model(model)
    assert r"\begin{cases}" in preview.latex
    assert "time/exposure" in preview.markdown
    entry = next(
        e
        for e in preview.legend
        if e["identifier"] == "rate" and e["namespace"] == "factors"
    )
    assert entry["weight"] == "h"
    assert (
        "Aggregate factor: rate"
        in render_formula(value("rate"), model=model).markdown
    )


@pytest.mark.parametrize(
    "scenario",
    ["replacement", "transfer", "types", "cells", "negative", "zero"],
)
def test_production_variants(scenario):
    """Reconcile each specification under workforce and payroll changes."""
    rows = comparison_frame()
    later = rows.period.eq("2026-08-01")
    if scenario == "replacement":
        rows.loc[later, "employee_id"] = ["c", "d"]
    elif scenario == "transfer":
        rows.loc[later & rows.employee_id.eq("a"), "team"] = "B"
    elif scenario == "types":
        temp = rows.iloc[[0, 2]].copy()
        temp["employment_type"] = "temporary"
        temp["temporary_rate"] = 25.0
        temp["temporary_hours"] = 4.0
        temp["overtime_rate"] = 30.0
        temp["overtime_hours"] = 2.0
        temp["actual"] = 160.0
        rows = pd.concat([rows, temp], ignore_index=True)
    elif scenario == "cells":
        rows.loc[rows.employee_id.eq("b"), "reg_coef"] = 1.5
        rows.loc[later & rows.employee_id.eq("a"), "grade"] = "G2"
        rows["actual"] = rows.reg_coef * (
            rows.permanent_rate * rows.permanent_hours + rows.permanent_bonus
        )
    elif scenario == "negative":
        rows["permanent_bonus"] = -10000.0
        rows["actual"] = rows.reg_coef * (
            rows.permanent_rate * rows.permanent_hours + rows.permanent_bonus
        )
    else:
        rows.loc[~later, "permanent_hours"] = 0.0
        rows["actual"] = rows.reg_coef * (
            rows.permanent_rate * rows.permanent_hours + rows.permanent_bonus
        )
    for model in examples.comparison_models().values():
        result = run(model, rows)
        assert result.table("validation").passed.all()
        for period, state in zip(("2026-01-01", "2026-08-01"), result.states):
            assert cost(state, model) == pytest.approx(
                rows.loc[rows.period.eq(period), "actual"].sum()
            )
        if model.model_id.endswith("effective"):
            canonical = result.table("canonical_data")
            evidence = result.table("diagnostics/aggregate_factors")
            for period, period_rows in canonical.groupby("period"):
                permanent = period_rows.loc[
                    period_rows.employment_type.eq("permanent")
                ]
                for (team, band, grade), cell in permanent.groupby(
                    ["team", "coefficient_band", "grade"]
                ):
                    selected = evidence.loc[
                        evidence.period.eq(period)
                        & evidence.team.eq(team)
                        & evidence.coordinates.map(
                            lambda x: (
                                x
                                == {
                                    "coefficient_band": band,
                                    "grade": grade,
                                }
                            )
                        )
                    ].set_index("factor")
                    reconstructed = (
                        cell.exposure.sum()
                        * selected.loc["effective_rate", "value"]
                        * selected.loc["hours_per_exposure", "value"]
                    )
                    assert reconstructed == pytest.approx(
                        (cell.permanent_rate * cell.permanent_hours).sum()
                    )


def test_basis_equivalence_with_zero_hours():
    """Actual and normalized declarations give identical counterfactuals."""
    rows = frame()
    rows.loc[rows.period.eq("2026-01-01"), "h"] = 0
    rows = prepare(rows)
    actual = run(aggregate_model(), rows)
    normalized = run(aggregate_model(basis="per_exposure"), rows)
    np.testing.assert_allclose(
        actual.table("driver_bridge").impact,
        normalized.table("driver_bridge").impact,
    )


def test_weight_population_and_reference_axes():
    """Reject scoped weights and typed reference tensors with wrong axes."""
    from labor_cost import Population

    model = aggregate_model()
    fields = tuple(
        replace(f, population="other") if f.name == "h" else f
        for f in model.graph.fields
    )
    graph = replace(
        model.graph,
        fields=fields,
        populations=(Population("other", {"employment_type": "staff"}),),
    )
    with pytest.raises(ValueError, match="weight population"):
        compile_model(replace(model, graph=graph))
    state = build_state(model, DatasetSnapshot("t", "1", frame()), "2026-01")
    state["teams"]["A"]["populations"]["all"]["rate"] = Tensor.scalar(100.0)
    with pytest.raises(ValueError, match="incorrect named axes"):
        cost(state, model)


def test_aggregate_cost_without_optional_imports():
    """Build and cost aggregates without optional imports."""
    import subprocess
    import sys

    payload = json.dumps(
        {
            "model": aggregate_model().to_dict(),
            "rows": frame().to_dict("records"),
        }
    )
    code = """
import json
import sys
import pandas as pd
from labor_cost import ModelSpec, DatasetSnapshot, build_state, cost
payload = json.load(sys.stdin)
model = ModelSpec.from_dict(payload["model"])
dataset = DatasetSnapshot("test", "1", pd.DataFrame(payload["rows"]))
state = build_state(model, dataset, "2026-01")
assert abs(cost(state, model) - 5000) < 1e-6
for name in ("IPython", "shap", "plotly", "nbformat", "jupyterlab"):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
