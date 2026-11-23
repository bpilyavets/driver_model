"""Accounting regressions for crossed team and line dimensions."""

import json
from dataclasses import replace
from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from labor_cost import (
    AnalysisRequest,
    DatasetSnapshot,
    Factor,
    FieldSpec,
    Game,
    ModelGraph,
    ModelSpec,
    OrganizationalHierarchy,
    RowCost,
    SourceSchema,
    Switch,
    analyze,
    build_state,
    cost,
    export_run,
    field,
    load_run,
    local_game,
    organization_game,
    outer_factors,
    prepare_input,
    report,
    value,
)


def pay_model():
    """Return a signed one-player unit model for hand-calculable games."""
    return ModelSpec(
        "pay",
        "1",
        graph=ModelGraph(
            source=SourceSchema(),
            populations=(),
            fields=(FieldSpec("pay", "actual"),),
            row_costs=(RowCost("all", field("pay")),),
            factors=outer_factors()
            + (Factor("pay_unit", "payment", formula=field("pay")),),
            games=(Game("unit", value("pay_unit"), (Switch("pay_unit"),)),),
            team_game="unit",
            organization=organization_game("unit"),
            local=local_game("unit"),
            hierarchy=OrganizationalHierarchy("lane"),
        ),
    )


def source(counts, rates):
    """Build two periods from team-major, line-minor cell counts/rates."""
    rows = []
    for month, (numbers, values) in enumerate(zip(counts, rates), 1):
        for cell, (count, amount) in enumerate(zip(numbers, values)):
            for employee in range(count):
                rows.append(
                    (
                        f"2025-0{month}-01",
                        f"{cell}/{employee}",
                        "P",
                        ("A", "B")[cell // 2],
                        ("X", "Y")[cell % 2],
                        amount,
                    )
                )
    return DatasetSnapshot(
        "test",
        "1",
        pd.DataFrame(
            rows,
            columns=[
                "period",
                "employee_id",
                "employment_type",
                "team",
                "lane",
                "actual",
            ],
        ),
    )


def settings(order, **kwargs):
    """Select a full-organization comparison with independent source cost."""
    return AnalysisRequest(
        "2025-01",
        "2025-02",
        source_cost_column="actual",
        hierarchy=order,
        **kwargs,
    )


def exhaustive(vector_a, vector_b, evaluate):
    """Compute a small unreduced Shapley game by permutation enumeration."""
    result = np.zeros(len(vector_a))
    orders = list(permutations(range(len(vector_a))))
    for ordering in orders:
        state = list(vector_a)
        previous = evaluate(*state)
        for i in ordering:
            state[i] = vector_b[i]
            current = evaluate(*state)
            result[i] += current - previous
            previous = current
    return result / len(orders)


@pytest.mark.parametrize("order", OrganizationalHierarchy("lane").orders)
def test_orders_and_full_parent_game(order, tmp_path):
    """Check observed endpoints, full root game and both portable views."""
    dataset = source(
        ((10, 30, 30, 30), (10, 30, 70, 90)), ((10, 10, -1, 3), (13, 13, 1, 4))
    )
    model = pay_model()
    run = analyze(model, dataset, settings(order))
    assert run.table("validation").passed.all()
    root = run.table("node_summary").query("depth == 0").iloc[0]
    assert (root.period_A_cost, root.period_B_cost) == pytest.approx(
        (460, 950)
    )
    state = build_state(model, dataset, "2025-02", hierarchy=order)
    assert cost(state, model) == pytest.approx(950)
    hierarchy = run.table("hierarchy")
    parents = run.table("node_summary").query("depth == 1")
    inputs = []
    for period in "AB":
        exposure = parents["exposure_" + period].to_numpy()
        cost_values = parents["period_" + period + "_cost"].to_numpy()
        inputs.append(
            (
                exposure.sum(),
                exposure / exposure.sum(),
                *(cost_values / exposure),
            )
        )
    expected = exhaustive(*inputs, lambda n, p, *u: n * np.dot(p, u))
    top = hierarchy.loc[
        hierarchy.view.eq("organization") & hierarchy.level.eq(0)
    ]
    assert top.impact.tolist() == pytest.approx(expected.tolist())
    node = parents.iloc[0].node_path
    view = run.select_node(node)
    if order[0] == "team":
        assert view[
            "organization_contributions"
        ].impact.sum() == pytest.approx(130)
        assert view["local_bridge"].impact.sum() == pytest.approx(120)
    if len(order) == 2:
        for parent in parents.itertuples():
            child_rows = run.table("node_summary")
            child_rows = child_rows.loc[
                child_rows.parent_path.map(lambda p: p == parent.node_path)
            ]
            inputs = []
            for period in "AB":
                e = child_rows["exposure_" + period].to_numpy()
                amounts = child_rows["period_" + period + "_cost"].to_numpy()
                inputs.append((e / e.sum(), *(amounts / e)))
            split = exhaustive(*inputs, lambda p, *u: np.dot(p, u))
            unit_delta = np.diff([np.dot(v[0], v[1:]) for v in inputs])[0]
            descendants = hierarchy.loc[
                hierarchy.view.eq("organization")
                & hierarchy.parent_path.map(lambda p: p == parent.node_path)
            ]
            if unit_delta:
                parent_impact = run.select_node(parent.node_path)[
                    "organization_contributions"
                ].impact.sum()
                assert descendants.impact.tolist() == pytest.approx(
                    split * parent_impact / unit_delta
                )
    restored = load_run(export_run(run, tmp_path / "run"))
    pd.testing.assert_frame_equal(
        restored.select_node(node)["local_bridge"], view["local_bridge"]
    )
    book = report(restored, nodes=[node], show=False)
    assert "node_summary" in book.tables
    manifest_path = tmp_path / "run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["settings"]["hierarchy"] = (
        ["team"] if order != ("team",) else ["line"]
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hierarchy"):
        load_run(manifest_path.parent)


def test_direction_changes_only_mix_explanation():
    """Unchanged line rates and team-conditional mixes isolate the order."""
    sample = source(
        ((60, 20, 5, 15), (15, 5, 20, 60)), ((100, 0, 100, 0),) * 2
    )
    for order in (("team", "line"), ("line", "team")):
        run = analyze(pay_model(), sample, settings(order))
        drivers = run.table("driver_bridge").set_index("driver").impact
        assert drivers[order[0] + "_mix"] == pytest.approx(-3000)
        assert drivers[order[1] + "_mix_within_" + order[0]] == pytest.approx(
            0
        )


def test_sparse_reference_and_conflicting_assignments():
    """Whole-parent references restore strict runs without adding exposure."""
    data = source(((1, 1, 1, 1), (0, 0, 2, 1)), ((-5, 6, 3, 4),) * 2)
    model, order = pay_model(), ("team", "line")
    run = analyze(model, data, settings(order))
    assert "carry_node_mix" in set(run.table("assumptions").method)
    with pytest.raises(ValueError, match="reference"):
        analyze(
            model, data, settings(order, missing_economics="require_reference")
        )
    parent = (("team", "A"),)
    references = {("2025-02", parent): {"mix": {"X": 0.8, "Y": 0.2}}}
    for path, node in run.states[0]["nodes"].items():
        if len(path) == 2 and path[:1] == parent:
            references["2025-02", path] = {
                "team_state": run.states[0]["teams"][node["key"]]
            }
    strict = analyze(
        model,
        data,
        settings(
            order, missing_economics="require_reference", references=references
        ),
    )
    assert strict.select_node(parent)["node_summary"].iloc[0].exposure_B == 0
    assert "explicit_node_mix" in set(strict.table("assumptions").method)
    extra = data.frame.iloc[[0]].assign(
        employment_type="Q", lane="Y", actual=np.nan
    )
    bad = DatasetSnapshot("bad", "1", pd.concat([data.frame, extra]))
    for action in (analyze, prepare_input):
        with pytest.raises(ValueError, match="one line"):
            action(model, bad, settings(order))


def test_sample_reuse_and_zero_parent():
    """Compare leaf seeds across orders and preserve offsetting children."""
    data = source(((1, 1, 1, 1),) * 2, ((0, 0, 0, 0), (4, -4, 0, 0)))
    model = pay_model()
    seeds = []
    for order in (("team", "line"), ("line", "team")):
        request = settings(order, explainer="permutation", n_permutations=2)
        prepared = prepare_input(model, data, request)
        with pytest.raises(ValueError, match="hierarchy"):
            analyze(model, prepared, replace(request, hierarchy=("line",)))
        run = analyze(model, prepared, request)
        audit = run.table("coalition_audit")
        sampled = audit.loc[audit.method.eq("permutation")]
        assert len(sampled) == 4
        seeds.append(
            dict(zip(zip(sampled.team, sampled.line), sampled.random_seed))
        )
        if order[0] == "team":
            bridge = run.select_node((("team", "A"),))[
                "organization_contributions"
            ]
            assert sorted(bridge.impact[bridge.impact.abs() > 1e-8]) == [-4, 4]
    assert seeds[0] == seeds[1]


def test_period_correct_references_and_offsetting_source_errors():
    """Require node-level evidence even when team and organization agree."""
    data = source(((1, 1, 1, 1),) * 2, ((2, 2, 2, 2),) * 2)
    frame = data.frame
    frame["reference"] = frame.actual
    frame.loc[[0, 1], "reference"] = [1, 3]
    request = replace(
        settings(("team", "line")), source_cost_column="reference"
    )
    with pytest.raises(AssertionError, match="source_cost"):
        analyze(pay_model(), DatasetSnapshot("bad", "1", frame), request)


def test_template_transfers_exposure_and_team_compatibility(
    complete_frame, months
):
    """Preserve paired employee inputs, transfers and the team-only bridge."""
    from labor_cost import labor_cost_model

    frame = complete_frame.copy()
    employee = "mdm_employee_rk_hash"
    names = sorted(frame[employee].unique())
    frame["lane"] = frame[employee].map(
        dict(zip(names, ("XY" * len(names))[: len(names)]))
    )
    first = frame.period.eq(months[0])
    frame.loc[first, "lane"] = frame.loc[first, "lane"].map(
        {"X": "Y", "Y": "X"}
    )
    dataset = DatasetSnapshot("transfer", "1", frame)
    request = AnalysisRequest(
        *map(str, months),
        source_cost_column="actual_cost",
        explainer="permutation",
        n_permutations=4,
    )
    base = labor_cost_model(schema_version=2)
    model = labor_cost_model(schema_version=2, line_source="lane")
    old, new = [analyze(m, dataset, request) for m in (base, model)]
    for owner in (
        "driver_bridge",
        *[k for k in old.tables if k.startswith("local_bridges/")],
    ):
        left, right = [
            r.table(owner).groupby("driver").impact.sum() for r in (old, new)
        ]
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-6)
    assert new.select_teams(
        [new.table("team_summary").team.iloc[0]]
    ).local_bridges
    order = ("line", "team")
    run = analyze(model, dataset, replace(request, hierarchy=order))
    matching = run.table("diagnostics/employee_alignment")
    for row in matching.itertuples():
        actual = frame.loc[
            (frame[employee] == row.employee_id)
            & pd.to_datetime(frame.period).eq(row.source_period)
        ]
        assert set(actual.lane) == {row.source_line}
    allocation = run.table("diagnostics/exposure_allocation")
    np.testing.assert_allclose(
        allocation.exposure, 1 / allocation.employment_type_count
    )


def test_model_approved_choices_and_audited_missing_lines():
    """Reject undeclared orders and audit removable missing identifiers."""
    model = pay_model()
    limited = replace(
        model,
        graph=replace(
            model.graph,
            hierarchy=OrganizationalHierarchy("lane", (("line",),), ("line",)),
        ),
    )
    data = source(((25, 25, 25, 25),) * 2, ((1, 2, 3, 4),) * 2)
    frame = data.frame
    frame.loc[0, "lane"] = None
    bad = DatasetSnapshot("one-missing", "1", frame)
    with pytest.raises(ValueError, match="Unsupported hierarchy"):
        build_state(limited, data, "2025-01", hierarchy=("team",))
    with pytest.raises(ValueError, match="line"):
        analyze(limited, bad, settings(None))
    prepared = prepare_input(limited, bad, settings(None))
    assert len(prepared.frame) == len(frame) - 1
    result = analyze(limited, prepared, settings(None))
    assert result.manifest["settings"]["hierarchy"] == ["line"]
    assert len(result.table("input_quality/excluded_rows")) == 1


def test_many_teams_and_empty_organization_endpoint():
    """Large fan-out uses exact term games and a missing month adds no pay."""
    sample = source(((1, 1, 1, 1),) * 2, ((1, 1, 1, 1),) * 2)
    first = sample.frame.iloc[[0]]
    frame = pd.concat(
        [
            first.assign(
                team=f"Team-{i}",
                employee_id=f"Employee-{i}",
                actual=i - 9,
                lane="Single line",
            )
            for i in range(17)
        ],
        ignore_index=True,
    )
    data = DatasetSnapshot("many", "1", frame)
    run = analyze(pay_model(), data, settings(("line", "team")))
    audit = run.table("coalition_audit")
    assert audit.drivers.max() == 3
    assert audit.method.eq("exact").all()
    assert run.table("node_summary").exposure_B.eq(0).all()
    assert run.table("bridge").impact.sum() == pytest.approx(
        -frame.actual.sum()
    )
    assert () in set(run.table("assumptions").node_path.dropna())


def test_hierarchy_notebook_fresh_kernel(tmp_path):
    """Execute the worked notebook and keep its source outputs empty."""
    from pathlib import Path

    import nbformat
    from test_quality import _execute_notebook

    example = Path(__file__).resolve().parents[1] / "examples"
    notebook = nbformat.read(example / "organizational_hierarchy.ipynb", 4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
    _execute_notebook(notebook, tmp_path)
