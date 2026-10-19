"""Acceptance of genuinely different declared states, costs and games."""

import importlib.util
from dataclasses import replace
from pathlib import Path

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
    Population,
    RowCost,
    SourceSchema,
    Switch,
    analyze,
    child,
    compile_model,
    cost,
    export_run,
    field,
    labor_cost_model,
    load_run,
    local_game,
    organization_game,
    outer_factors,
    select,
    sum_over,
    value,
)

spec = importlib.util.spec_from_file_location(
    "author_models",
    Path(__file__).resolve().parents[1] / "examples" / "author_models.py",
)
authors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(authors)


@pytest.fixture
def hourly_frame():
    """Provide changing joint composition, signed pay and cross-team moves."""
    rows = [
        ("2025-03-01", "a", "Alpha", "G1", "R1", 10, 8, 2),
        ("2025-03-01", "b", "Alpha", "G2", "R2", 20, 4, -10),
        ("2025-03-01", "c", "Beta", "G1", "R2", 15, 6, 3),
        ("2025-04-01", "a", "Beta", "G2", "R1", 11, 9, -5),
        ("2025-04-01", "b", "Alpha", "G1", "R1", 20, 4, 0),
        ("2025-04-01", "d", "Alpha", "G2", "R2", 25, 5, 8),
    ]
    frame = pd.DataFrame(
        rows,
        columns=(
            "period",
            "employee_id",
            "team",
            "grade",
            "region",
            "hourly_rate",
            "paid_hours",
            "allowance",
        ),
    )
    frame["employment_type"] = "agency"
    frame["actual"] = frame.hourly_rate * frame.paid_hours + frame.allowance
    return frame


def run_model(model, frame, mode="exact", **options):
    """Execute small independent fixtures with reproducible settings."""
    return analyze(
        model,
        DatasetSnapshot("fixture", "1", frame),
        AnalysisRequest(
            "2025-03",
            "2025-04",
            source_cost_column="actual",
            explainer=mode,
            n_permutations=8,
            **options,
        ),
    )


@pytest.mark.parametrize("mode", ["exact", "permutation"])
@pytest.mark.parametrize("grade_first", [True, False])
def test_arbitrary_formula_conditioning_and_pairing(
    hourly_frame, mode, grade_first
):
    """A model without any PoC field/branch executes both factorizations."""
    model = authors.hourly_model(grade_first=grade_first)
    original = hourly_frame.copy(deep=True)
    result = run_model(model, hourly_frame, mode)
    assert result.table("validation").passed.all()
    for state, period in zip(result.states, ("2025-03-01", "2025-04-01")):
        assert cost(state, model) == pytest.approx(
            hourly_frame.loc[hourly_frame.period.eq(period), "actual"].sum()
        )
    pd.testing.assert_frame_equal(hourly_frame, original)
    assert result.select_teams(["Alpha"]).run_id == result.run_id
    restored = ModelSpec.from_dict(model.to_dict())
    assert restored.fingerprint == model.fingerprint
    assert compile_model(restored).inspect()["schema_version"] == 2


def test_factorization_changes_game_not_endpoints(hourly_frame):
    """Reparameterizing joint composition changes valid hybrid costs."""
    rows = []
    for period, counts in (
        ("2025-03-01", (1, 2, 1, 4)),
        ("2025-04-01", (3, 1, 2, 2)),
    ):
        for cell, count in enumerate(counts):
            for i in range(count):
                rows.append(
                    {
                        "period": period,
                        "employee_id": f"{cell}-{i}",
                        "team": "A",
                        "employment_type": "agency",
                        "grade": f"G{cell // 2}",
                        "region": f"R{cell % 2}",
                        "hourly_rate": (10, 50, 20, 90)[cell],
                        "paid_hours": 1,
                        "allowance": 0,
                        "actual": (10, 50, 20, 90)[cell],
                    }
                )
    frame = pd.DataFrame(rows)
    left = run_model(authors.hourly_model(), frame)
    right = run_model(authors.hourly_model(grade_first=False), frame)
    assert left.table("bridge").impact.sum() == pytest.approx(
        right.table("bridge").impact.sum()
    )
    a = left.table("driver_bridge").set_index("driver").impact
    b = right.table("driver_bridge").set_index("driver").impact
    assert not np.allclose(a, b)


@pytest.mark.parametrize("nested,grouped", [(True, False), (False, True)])
def test_explicit_games_and_grouped_switches(hourly_frame, nested, grouped):
    """Analysts change boundaries and atomic ownership by declaration."""
    result = run_model(
        authors.hourly_model(nested=nested, grouped=grouped), hourly_frame
    )
    assert result.table("validation").passed.all()
    drivers = result.table("driver_bridge").driver.tolist()
    if grouped:
        assert "unit::earned_pay_inputs" in drivers
        assert "unit::rates" not in drivers
    else:
        assert "earnings::rates" in drivers
        assert (~result.table("hierarchy").leaf).any()


def scoped_model():
    """Declare arbitrary employment populations with different formulas."""
    mixture = value("kind_mix")
    a, b = child("internal"), child("external")
    return ModelSpec(
        "scoped",
        "1",
        graph=ModelGraph(
            SourceSchema(),
            (
                Population("internal", {"employment_type": "staff"}),
                Population("external", {"employment_type": "vendor"}),
            ),
            (
                FieldSpec("staff_pay", "staff_amount", "internal"),
                FieldSpec("vendor_pay", "invoice_amount", "external"),
            ),
            (
                RowCost("internal", field("staff_pay")),
                RowCost("external", field("vendor_pay")),
            ),
            outer_factors()
            + (
                Factor(
                    "kind_mix",
                    "population_mix",
                    dimensions=("kind",),
                    members=("internal", "external"),
                ),
                Factor(
                    "staff_rate",
                    "payment",
                    "internal",
                    formula=field("staff_pay"),
                ),
                Factor(
                    "vendor_rate",
                    "payment",
                    "external",
                    formula=field("vendor_pay"),
                ),
            ),
            (
                Game(
                    "internal",
                    value("staff_rate"),
                    (Switch("staff_rate"),),
                    "internal",
                ),
                Game(
                    "external",
                    value("vendor_rate"),
                    (Switch("vendor_rate"),),
                    "external",
                ),
                Game(
                    "unit",
                    select(mixture, "kind", "internal") * a
                    + select(mixture, "kind", "external") * b,
                    (
                        Switch("kind_mix"),
                        Switch("internal_cost", child="internal"),
                        Switch("external_cost", child="external"),
                    ),
                ),
            ),
            "unit",
            organization_game("unit"),
            local_game("unit"),
        ),
    )


def test_scoped_applicability_and_absent_branches():
    """Require applicable pay, without demanding unused input columns."""
    frame = pd.DataFrame(
        {
            "period": ["2025-03-01", "2025-04-01"],
            "employee_id": ["a", "a"],
            "team": ["A", "A"],
            "employment_type": ["vendor", "vendor"],
            "invoice_amount": [-20.0, 30.0],
            "actual": [-20.0, 30.0],
        }
    )
    model = scoped_model()
    result = run_model(model, frame)
    assert result.table("bridge").impact.sum() == pytest.approx(50)
    assert (
        not result.table("driver_bridge").driver.str.contains("internal").any()
    )
    frame.loc[1, "invoice_amount"] = np.nan
    with pytest.raises(ValueError, match="applicable input"):
        run_model(model, frame)


def test_alternative_aggregate_expression(hourly_frame):
    """Direct team-exposure aggregation replaces organizational N/mix."""
    model = authors.hourly_model()
    organization = Game(
        "organization",
        sum_over(value("team_exposure") * child("unit"), "team"),
        (
            Switch("team_exposures", ("team_exposure",)),
            Switch("team_cost", child="unit"),
        ),
    )
    model = replace(
        model, graph=replace(model.graph, organization=organization)
    )
    result = run_model(model, hourly_frame)
    assert result.table("validation").passed.all()
    assert "organization::team_exposures" in set(
        result.table("driver_bridge").driver
    )
    assert "team_mix" not in set(result.table("driver_bridge").driver)


def test_zero_parent_with_offsetting_children():
    """Keep offsetting signed children when their parent does not change."""
    frame = pd.DataFrame(
        {
            "period": ["2025-03-01", "2025-04-01"],
            "employee_id": ["a", "a"],
            "employment_type": ["agency", "agency"],
            "team": ["A", "A"],
            "grade": ["G", "G"],
            "region": ["R", "R"],
            "hourly_rate": [10.0, 20.0],
            "paid_hours": [2.0, 1.0],
            "allowance": [0.0, 0.0],
            "actual": [20.0, 20.0],
        }
    )
    result = run_model(authors.hourly_model(nested=True), frame)
    effects = result.table("driver_bridge").set_index("driver").impact
    assert effects["earnings::rates"] == pytest.approx(15)
    assert effects["earnings::hours"] == pytest.approx(-15)
    assert result.table("validation").passed.all()


def test_compile_rejections():
    """Reject invalid units, ownership, conditioning and nonlinear children."""
    model = authors.hourly_model()
    game = model.graph.games[0]
    with pytest.raises(ValueError, match="multiple atomic owners"):
        compile_model(
            replace(
                model,
                graph=replace(
                    model.graph,
                    games=(
                        replace(
                            game,
                            switches=game.switches
                            + (Switch("duplicate", ("rates",)),),
                        ),
                    ),
                ),
            )
        )
    factors = list(model.graph.factors)
    index = next(i for i, f in enumerate(factors) if f.name == "first_mix")
    factors[index] = replace(factors[index], given=("region",))
    with pytest.raises(ValueError, match="cyclic"):
        compile_model(
            replace(model, graph=replace(model.graph, factors=tuple(factors)))
        )
    nested = authors.hourly_model(nested=True)
    parent = nested.graph.games[-1]
    with pytest.raises(ValueError, match="nonlinear"):
        compile_model(
            replace(
                nested,
                graph=replace(
                    nested.graph,
                    games=nested.graph.games[:-1]
                    + (
                        replace(
                            parent,
                            formula=child("earnings") * child("earnings")
                            + value("allowances"),
                        ),
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="incompatible"):
        compile_model(
            replace(
                model,
                graph=replace(
                    model.graph,
                    games=(
                        replace(game, formula=game.formula + value("rates")),
                    ),
                ),
            )
        )


def test_invalid_source_declarations_and_composite_values(hourly_frame):
    """Reject ignored legacy options and overwritten or missing dimensions."""
    model = authors.hourly_model()
    for option in (
        {"exposure_column": "fte"},
        {"missing_economics": "require_reference"},
    ):
        with pytest.raises(ValueError, match="inside ModelGraph"):
            compile_model(replace(model, **option))
    source = model.graph.source
    with pytest.raises(ValueError, match="overwrite canonical keys"):
        compile_model(
            replace(
                model,
                graph=replace(
                    model.graph,
                    source=replace(source, dimensions={"team": "grade"}),
                ),
            )
        )
    composite = replace(
        model,
        graph=replace(
            model.graph,
            source=replace(
                source,
                dimensions={
                    "grade": ("grade", "position"),
                    "region": "region",
                },
            ),
        ),
    )
    frame = hourly_frame.assign(position="engineer")
    frame.loc[frame.index[0], "position"] = None
    with pytest.raises(ValueError, match="missing applicable dimension"):
        run_model(composite, frame)


def test_v2_bundle_roundtrip(hourly_frame, tmp_path):
    """Preserve arbitrary scopes, metadata and numerical tables."""
    result = run_model(authors.hourly_model(), hourly_frame, "permutation")
    restored = load_run(export_run(result, tmp_path / "run"))
    assert (
        restored.manifest["compiled_graph"]
        == result.manifest["compiled_graph"]
    )
    for name in result.tables:
        pd.testing.assert_frame_equal(restored.table(name), result.table(name))
    assert restored.states is None


@pytest.mark.parametrize("mode", ["exact", "permutation"])
def test_v2_template_numerical_parity(
    notebook_model, complete_frame, months, mode
):
    """The native v2 template matches the read-only notebook's actual game."""
    expected = notebook_model["analyze_labor_cost"](
        complete_frame,
        *months,
        source_cost_column="actual_cost",
        explainer=mode,
        n_permutations=8,
    )
    result = analyze(
        labor_cost_model(schema_version=2),
        DatasetSnapshot("payroll", "1", complete_frame),
        AnalysisRequest(
            *map(str, months),
            source_cost_column="actual_cost",
            explainer=mode,
            n_permutations=8,
        ),
    )
    for key in ("driver_bridge",):
        left = expected[key].set_index("driver").impact.sort_index()
        right = result.table(key).set_index("driver").impact.sort_index()
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-6)
    for team in expected["local_bridges"]:
        left = (
            expected["local_bridges"][team]
            .set_index("driver")
            .impact.sort_index()
        )
        right = (
            result.table("local_bridges/" + team)
            .set_index("driver")
            .impact.sort_index()
        )
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-6)


@pytest.mark.parametrize(
    "scenario", ["salary", "time", "transfer", "exposure", "signed", "absent"]
)
def test_native_semantic_regressions(notebook_model, scenario):
    """Exercise identity, signed amounts and completion in the new core."""
    from conftest import payroll_row

    a, b = "2025-03-01", "2025-04-01"
    rows = [payroll_row(a, "a", worked=0.5), payroll_row(b, "a", worked=0.5)]
    if scenario == "salary":
        rows[1] = payroll_row(b, "a", base=150, worked=0.5)
    elif scenario == "time":
        rows[1] = payroll_row(b, "a", worked=0.8)
    elif scenario == "transfer":
        rows = [
            payroll_row(a, "a", team="A", base=100),
            payroll_row(a, "b", team="B", base=150),
            payroll_row(b, "a", team="B", base=110),
            payroll_row(b, "b", team="A", base=150),
        ]
    elif scenario == "exposure":
        rows += [payroll_row(b, "a", contract="ГПД", rate=-20)]
    elif scenario == "signed":
        rows = [
            payroll_row(a, "a", sum_pay_compacted=-300),
            payroll_row(b, "a", sum_pay_compacted=-500),
        ]
    elif scenario == "absent":
        rows = [
            payroll_row(a, "a", team="A"),
            payroll_row(b, "b", team="B", contract="ГПД", rate=-20),
        ]
    frame = pd.DataFrame(rows)
    settings = dict(
        source_cost_column="actual_cost",
        explainer="permutation",
        n_permutations=8,
    )
    expected = notebook_model["analyze_labor_cost"](frame, a, b, **settings)
    actual = analyze(
        labor_cost_model(schema_version=2),
        DatasetSnapshot("x", "1", frame),
        AnalysisRequest(a, b, **settings),
    )
    left = expected["driver_bridge"].set_index("driver").impact.sort_index()
    right = (
        actual.table("driver_bridge").set_index("driver").impact.sort_index()
    )
    pd.testing.assert_index_equal(left.index, right.index)
    np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-6)
    assert actual.table("validation").passed.all()


def test_declared_fallback_and_reference_policies(hourly_frame):
    """Missing economics is explicit, while sparse cells stay within scope."""
    model = authors.hourly_model()
    strict = replace(
        model,
        graph=replace(model.graph, missing_economics="require_reference"),
    )
    frame = hourly_frame.copy()
    frame.loc[
        frame.period.eq("2025-04-01") & frame.team.eq("Beta"), "team"
    ] = "Gamma"
    with pytest.raises(ValueError, match="reference"):
        run_model(strict, frame)
    completed = run_model(model, frame)
    references = {}
    for state in completed.states:
        for team, exposure in state["exposure_by_team"].items():
            if exposure == 0:
                references[(state["period"], team)] = {
                    "team_state": state["teams"][team]
                }
    explicit = run_model(strict, frame, references=references)
    assert explicit.table("validation").passed.all()
    assert set(explicit.table("assumptions").method) == {
        "explicit_team_reference"
    }
    factors = tuple(
        replace(f, fallbacks=()) if f.kind == "membership" else f
        for f in model.graph.factors
    )
    with pytest.raises(ValueError, match="unsupported cell"):
        run_model(
            replace(model, graph=replace(model.graph, factors=factors)),
            hourly_frame,
        )


def test_supplied_exposure_rng_and_input_integrity(hourly_frame):
    """Preserve explicit exposures, observed rate/time and caller RNG state."""
    frame = hourly_frame.assign(fte=[0.3, 0.7, 0.5, 0.2, 0.8, 0.6])
    before = frame.copy(deep=True)
    np.random.seed(725)
    state = np.random.get_state()
    result = run_model(
        authors.hourly_model(), frame, "permutation", exposure_column="fte"
    )
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    pd.testing.assert_frame_equal(frame, before)
    np.testing.assert_array_equal(
        result.table("canonical_data").exposure, frame.fte
    )
    normalized = result.table("canonical_data").per_exposure__allowances
    np.testing.assert_allclose(normalized, frame.allowance / frame.fte)
    assert result.table("validation").passed.all()


def test_grain_partition_and_shared_state_rejections(hourly_frame):
    """Reject invalid source grain and ambiguous declaration semantics."""
    with pytest.raises(ValueError, match="Duplicate"):
        run_model(
            authors.hourly_model(),
            pd.concat([hourly_frame, hourly_frame.iloc[:1]]),
        )
    dated = hourly_frame.copy()
    dated.loc[0, "period"] = "2025-03-02"
    with pytest.raises(ValueError, match="month starts"):
        run_model(authors.hourly_model(), dated)
    nested = authors.hourly_model(nested=True)
    parent = nested.graph.games[-1]
    parent = replace(
        parent,
        formula=parent.formula * value("first_mix"),
        switches=parent.switches + (Switch("first_mix"),),
    )
    with pytest.raises(ValueError):
        compile_model(
            replace(
                nested,
                graph=replace(
                    nested.graph, games=nested.graph.games[:-1] + (parent,)
                ),
            )
        )


def test_large_leaf_guard_and_permutation(hourly_frame):
    """The exact safeguard remains explicit for extensible leaf games."""
    model = authors.hourly_model()
    graph = model.graph
    fields, factors = list(graph.fields), list(graph.factors)
    game = graph.games[0]
    formula = game.formula
    switches = list(game.switches)
    frame = hourly_frame.copy()
    for i in range(11):
        name = f"supplement_{i}"
        fields.append(FieldSpec(name, name))
        factors.append(Factor(name, "payment", formula=field(name)))
        switches.append(Switch(name))
        formula = formula + value(name)
        frame[name] = 0.0
    extended = replace(
        model,
        graph=replace(
            graph,
            fields=tuple(fields),
            factors=tuple(factors),
            games=(replace(game, formula=formula, switches=tuple(switches)),),
        ),
    )
    with pytest.raises(ValueError, match="15 players"):
        run_model(extended, frame)
    assert (
        run_model(extended, frame, "permutation")
        .table("validation")
        .passed.all()
    )


def test_v1_bundle_reader(tmp_path):
    """Read a genuine fixture exported by the previous 0.1.0 wheel."""
    import zipfile

    path = Path(__file__).parent / "fixtures" / "schema1_run.zip"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tmp_path)
    restored = load_run(tmp_path)
    assert restored.manifest["engine_version"] == "0.1.0"
    assert restored.manifest["bundle_version"] == 1
    assert (
        restored.manifest["model_fingerprint"]
        == labor_cost_model().fingerprint
    )
    assert restored.table("driver_bridge").impact.sum() == pytest.approx(20)
    assert restored.table("validation").passed.all()


def test_observed_only_costing_and_v2_catalog(hourly_frame, tmp_path):
    """Build an observed state and resolve a published graph."""
    import json

    from labor_cost import Catalog, build_state, publish_model

    model = authors.hourly_model()
    snapshot = DatasetSnapshot("hourly", "1", hourly_frame)
    observed = build_state(
        model, snapshot, "2025-03", source_cost_column="actual"
    )
    assert cost(observed, model) == pytest.approx(
        hourly_frame.loc[hourly_frame.period.eq("2025-03-01"), "actual"].sum()
    )
    result = run_model(model, hourly_frame, "permutation")
    publish_model(model, tmp_path / "model.json", validated_result=result)
    hourly_frame.to_parquet(tmp_path / "data.parquet", index=True)
    config = {
        "models": [
            {
                "id": model.model_id,
                "version": model.version,
                "fingerprint": model.fingerprint,
                "path": "model.json",
            }
        ],
        "datasets": [
            {
                "id": "hourly",
                "version": "1",
                "fingerprint": snapshot.fingerprint,
                "path": "data.parquet",
            }
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(config))
    compiled = Catalog(path).model(model.model_id, model.version)
    assert compiled.spec == model
    pd.testing.assert_frame_equal(
        Catalog(path).dataset("hourly", "1").frame, hourly_frame
    )


def test_cached_aligned_costs_use_shared_formula(hourly_frame):
    """Compare cached/direct rate, hour and membership combinations."""
    from itertools import product

    from labor_cost.graph import aligned_mean
    from labor_cost.graph_cost import evaluate

    model = authors.hourly_model()
    result = run_model(model, hourly_frame, "permutation")
    a, b = [s["teams"]["Alpha"]["populations"]["all"] for s in result.states]
    formula = aligned_mean(value("rates") * value("hours"), "members")
    cache = {}
    for choices in product((0, 1), repeat=3):
        state = dict(a)
        for key, endpoint in zip(("rates", "hours", "members"), choices):
            state[key] = (a, b)[endpoint][key]
        direct = evaluate(formula, state, {}).data
        cached = evaluate(formula, state, {}, cache).data
        np.testing.assert_array_equal(direct, cached)
    assert len(cache) == 8
    assert all("cache" not in key for state in result.states for key in state)


def test_lifecycle_cli(hourly_frame, tmp_path):
    """Run the executable analyst/business lifecycle using supplied data."""
    import subprocess
    import sys

    source = tmp_path / "input.parquet"
    hourly_frame.to_parquet(source)
    example = (
        Path(__file__).resolve().parents[1] / "examples" / "model_lifecycle.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(example),
            str(source),
            str(tmp_path / "catalog"),
            "--period-a",
            "2025-03",
            "--period-b",
            "2025-04",
            "--explainer",
            "exact",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        load_run(tmp_path / "catalog" / "business-run")
        .table("validation")
        .passed.all()
    )


def test_effective_inputs_after_whole_team_copy():
    """Diagnostics show copied economics instead of unmatched live vectors."""
    from conftest import payroll_row

    frame = pd.DataFrame(
        [
            payroll_row("2025-03-01", "a", team="A", base=100),
            payroll_row("2025-04-01", "a", team="B", base=150),
        ]
    )
    result = analyze(
        labor_cost_model(schema_version=2),
        DatasetSnapshot("x", "1", frame),
        AnalysisRequest(
            "2025-03",
            "2025-04",
            source_cost_column="actual_cost",
            explainer="permutation",
            n_permutations=2,
        ),
    )
    inputs = result.table("diagnostics/aligned_inputs")
    copied = inputs.loc[
        inputs.team.eq("A")
        & inputs.period.eq(pd.Timestamp("2025-04-01"))
        & inputs.driver.eq("underlying_salary_level")
    ]
    assert copied.value.tolist() == [100]
    assert copied.source_period.tolist() == [pd.Timestamp("2025-03-01")]
    assert copied.completion_method.tolist() == ["carry_observed"]
    profiles = result.table("diagnostics/membership_profiles")
    assert set(profiles.employee_id) == {"a"}


def test_explicit_per_exposure_amount_basis():
    """Normalized inputs reconstruct actual payroll with declared units."""
    graph = ModelGraph(
        SourceSchema(exposure="allocation"),
        (),
        (
            FieldSpec("unit_pay", "unit_pay", units="currency/exposure"),
            FieldSpec(
                "allocation",
                "allocation",
                units="exposure",
                constraint="positive",
            ),
        ),
        (RowCost("all", field("unit_pay") * field("allocation")),),
        outer_factors()
        + (
            Factor(
                "pay",
                "payment",
                formula=field("unit_pay"),
                basis="per_exposure",
            ),
        ),
        (Game("unit", value("pay"), (Switch("pay"),)),),
        "unit",
        organization_game("unit"),
        local_game("unit"),
    )
    frame = pd.DataFrame(
        {
            "period": ["2025-03-01", "2025-04-01"],
            "employee_id": ["a", "a"],
            "employment_type": ["staff", "staff"],
            "team": ["A", "A"],
            "unit_pay": [100.0, 150.0],
            "allocation": [0.5, 0.4],
            "actual": [50.0, 60.0],
        }
    )
    result = run_model(ModelSpec("normalized", "1", graph=graph), frame)
    assert result.table("validation").passed.all()
    np.testing.assert_array_equal(
        result.table("canonical_data").per_exposure__pay, [100.0, 150.0]
    )
