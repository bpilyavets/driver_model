"""Table-only Plotly reports preserve accounting scopes and stored values."""

import importlib
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from labor_cost import (
    AnalysisRequest,
    AnalysisResult,
    DatasetSnapshot,
    analyze,
    export_run,
    labor_cost_model,
    load_run,
    plot_driver_by_team,
    plot_organization_waterfall,
    plot_team_waterfall,
    report,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stored_result():
    """Store offsetting organization and different local allocations."""
    records = []
    for driver, label, team, impact in (
        ("scale", "Scale", None, 0.),
        ("x", "Payment", "Alpha", 30.),
        ("x", "Payment", "Beta", -30.),
        ("y", "Payment", "Alpha", -20.),
        ("y", "Payment", "Beta", 20.),
        ("zero", "Unchanged", "Alpha", 0.),
    ):
        records.append(dict(
            driver=driver, label=label, team=team, impact=impact,
            period_A_cost=-100., period_B_cost=-100., view="organization",
            leaf=True, units="currency",
        ))
    bridge = pd.DataFrame(records)
    drivers = bridge.drop_duplicates("driver").copy()
    drivers["team"] = None
    drivers["impact"] = 0.
    tables = {
        "bridge": bridge, "driver_bridge": drivers,
        "team_summary": pd.DataFrame({
            "team": ["Alpha", "Beta"], "period_A_cost": [-40., -60.],
            "period_B_cost": [-30., -70.], "cost_change": [10., -10.],
        }),
        "validation": pd.DataFrame([{"check": "stored", "passed": True}]),
        "assumptions": pd.DataFrame([{
            "method": "carry_observed", "reference": {"values": [1.]},
        }]),
        "coalition_audit": pd.DataFrame({
            "method": ["exact", "permutation"],
            "n_permutations": [None, 8.], "random_seed": [None, 999.],
        }),
    }
    for team, start, end, amounts in (
        ("Alpha", -40., -30., [4., 6.]),
        ("Beta", -60., -70., [-3., -7.]),
    ):
        local = bridge.loc[bridge.team.eq(team)].iloc[:2].copy()
        local["impact"] = amounts
        local["period_A_cost"], local["period_B_cost"] = start, end
        local["view"] = "local"
        tables["local_bridges/" + team] = local
    return AnalysisResult({
        "run_id": "stored", "periods": ["2026-08", "2026-09"],
        "settings": {"source_cost_column": "actual", "explainer": "exact"},
        "model": {"model_id": "fixture", "version": "1"},
    }, tables, states=None)


def test_signed_scopes_selection_and_labels(stored_result):
    """Keep offsetting bars, original order, local values and stable IDs."""
    charts = report(stored_result, teams=["Alpha"], driver="x", show=False)
    org = charts.figures["organization"]
    assert list(org.data[0].y) == [-100., 0., 0., 0., 0., -100.]
    assert list(org.data[0].measure) == [
        "absolute", "relative", "relative", "relative", "relative",
        "absolute",
    ]
    assert org.layout.meta["driver_ids"] == ["", "scale", "x", "y", "zero", ""]
    assert "Payment [x]" in org.layout.xaxis.ticktext
    assert "Payment [y]" in org.layout.xaxis.ticktext
    assert list(charts.figures["local/Alpha"].data[0].y) == [
        -40., 4., 6., -30.,
    ]
    by_team = charts.figures["driver_by_team"]
    assert list(by_team.data[0].x) == [30.]
    assert by_team.layout.meta["organization_total"] == 0.
    assert by_team.layout.meta["selected_subtotal"] == 30.
    assert "Selected subtotal" in by_team.layout.title.text
    assert charts.tables["team_summary"].team.tolist() == ["Alpha"]
    whole = plot_driver_by_team(stored_result, "x")
    assert list(whole.data[0].x) == [30., -30.]
    assert "Selected subtotal" not in whole.layout.title.text


@pytest.mark.parametrize("teams", [None, []])
def test_default_selection_and_missing_evidence(stored_result, teams):
    """Empty selection adds no local charts and missing audits stay unknown."""
    stored_result.tables.pop("coalition_audit")
    stored_result.tables.pop("validation")
    stored_result.tables.pop("assumptions")
    charts = report(stored_result, teams=teams, driver="x", show=False)
    assert list(charts.figures) == ["organization", "driver_by_team"]
    assert len(charts.tables["team_summary"]) == 2
    for name in ("estimator_summary", "assumption_summary"):
        assert charts.tables[name].method.tolist() == ["Unavailable"]
    assert charts.tables["validation_summary"].recorded_status.tolist() == [
        "Unavailable",
    ]


def test_recorded_evidence_and_deep_copies(stored_result):
    """Use actual methods and preserve nested cells, tables and manifest."""
    before = deepcopy(stored_result)
    charts = report(stored_result, show=False)
    assert charts.tables["estimator_summary"].method.tolist() == [
        "exact", "permutation",
    ]
    assert charts.metadata["source_cost_basis"] == (
        "Independent source column: actual"
    )
    charts.tables["assumptions"].iloc[0]["reference"]["values"].append(2.)
    charts.tables["driver_bridge"].loc[:, "impact"] = 77.
    charts.metadata["periods"].append("changed")
    for name, frame in before.tables.items():
        pd.testing.assert_frame_equal(frame, stored_result.tables[name])
    assert before.manifest == stored_result.manifest
    stored_result.tables["validation"].loc[0, "passed"] = False
    charts = report(stored_result, show=False)
    assert charts.tables["validation_summary"].recorded_status.tolist() == [
        "Failed",
    ]


@pytest.mark.parametrize("driver, message", [
    ("missing", "leaf driver ID"), ("parent", "leaf driver ID"),
    ("scale", "organization-wide"), ("Payment", "leaf driver ID"),
])
def test_invalid_driver_selection(stored_result, driver, message):
    """IDs, labels, parents and global effects are not interchangeable."""
    with pytest.raises(ValueError, match=message):
        plot_driver_by_team(stored_result, driver)


@pytest.mark.parametrize("teams", [["Unknown"], "Alpha"])
def test_invalid_team_selection(stored_result, teams):
    """Reject accidental string selectors and absent team names."""
    with pytest.raises(ValueError):
        report(stored_result, teams=teams, show=False)


@pytest.mark.parametrize("column, value", [
    ("impact", np.nan), ("impact", np.inf), ("impact", "30"),
    ("impact", 31.), ("leaf", False), ("view", "local"),
    ("units", "currency/exposure"), ("period_B_cost", -99.),
])
def test_invalid_chart_data(stored_result, column, value):
    """Refuse corrupted, non-leaf or inconsistent data instead of balancing."""
    frame = stored_result.tables["bridge"]
    if isinstance(value, str):
        frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value
    with pytest.raises(ValueError):
        plot_organization_waterfall(stored_result)


def test_swapped_driver_totals_and_local_scope(stored_result):
    """Total efficiency alone must not authorize incorrect chart groupings."""
    stored_result.tables["driver_bridge"].loc[1, "impact"] = 1.
    stored_result.tables["driver_bridge"].loc[3, "impact"] = -1.
    with pytest.raises(ValueError, match="Driver totals"):
        plot_organization_waterfall(stored_result)
    stored_result.tables["driver_bridge"]["impact"] = 0.
    stored_result.tables["local_bridges/Alpha"]["team"] = "Beta"
    with pytest.raises(ValueError, match="local team scope"):
        plot_team_waterfall(stored_result, "Alpha")


def test_missing_columns_and_legacy_summary(stored_result):
    """Adapt legacy summary names internally and report missing contracts."""
    stored_result.tables["team_summary"] = stored_result.tables[
        "team_summary"
    ].rename(columns={
        "period_A_cost": "period_a_cost", "period_B_cost": "period_b_cost",
        "cost_change": "local_change",
    })
    assert report(stored_result, show=False).tables[
        "team_summary"
    ].cost_change.tolist() == [10., -10.]
    stored_result.tables["driver_bridge"].drop(columns="units", inplace=True)
    with pytest.raises(ValueError, match="missing columns"):
        report(stored_result, show=False)


def test_reporting_calls_no_engine(stored_result, monkeypatch):
    """Fail if reporting attempts compilation, costing or attribution."""
    def fail(*args, **kwargs):
        """Trap any accidental engine call during presentation."""
        raise AssertionError("Calculation during reporting")

    for module, names in {
        "labor_cost": ["analyze", "build_state", "cost", "compile_model"],
        "labor_cost.graph_api": [
            "analyze_graph", "graph_cost", "build_states",
        ],
        "labor_cost.graph_attribution": ["decompose_graph", "_run"],
    }.items():
        imported = importlib.import_module(module)
        for name in names:
            monkeypatch.setattr(imported, name, fail)
    assert report(stored_result, teams=["Alpha"], driver="x", show=False)


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("mode", ["exact", "permutation"])
def test_actual_results_and_roundtrip(complete_frame, months, tmp_path,
                                      schema, mode):
    """Match real template results and restored bundles for both estimators."""
    result = analyze(
        labor_cost_model(schema_version=schema),
        DatasetSnapshot("reporting", "1", complete_frame),
        AnalysisRequest(*map(str, months), explainer=mode, n_permutations=2,
                        source_cost_column="actual_cost"),
    )
    teams = result.table("team_summary").team.tolist()
    driver = result.table("bridge").loc[
        lambda x: x.team.notna(), "driver"
    ].iloc[0]
    charts = report(result, teams=teams, driver=driver, show=False)
    np.testing.assert_array_equal(
        charts.figures["organization"].data[0].y[1:-1],
        result.table("driver_bridge").impact,
    )
    for team in teams:
        np.testing.assert_array_equal(
            charts.figures["local/" + team].data[0].y[1:-1],
            result.table("local_bridges/" + team).impact,
        )
    restored = load_run(export_run(result, tmp_path / "run"))
    assert restored.states is None
    copied = report(restored, teams=teams, driver=driver, show=False)
    for name in charts.figures:
        assert charts.figures[name].to_json() == copied.figures[name].to_json()


def test_schema_one_bundle(tmp_path):
    """Render the genuine older container with no calculation states."""
    with zipfile.ZipFile(ROOT / "tests/fixtures/schema1_run.zip") as archive:
        archive.extractall(tmp_path)
    manifest = next(tmp_path.rglob("manifest.json"))
    result = load_run(manifest.parent)
    charts = report(result, teams=["Fixture team"], show=False)
    assert charts.figures["organization"].data[0].type == "waterfall"
    assert "period_A_cost" in charts.tables["team_summary"]


def test_optional_import_isolation():
    """Import and compile without optional plotting/notebook dependencies."""
    script = '''
import sys
from importlib.abc import MetaPathFinder

class Block(MetaPathFinder):
    """Reject optional imports even when installed in the test environment."""

    def find_spec(self, fullname, path=None, target=None):
        """Fail on forbidden presentation modules."""
        if fullname.split('.')[0] in {'plotly', 'IPython', 'nbformat', 'shap'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, Block())
import labor_cost
labor_cost.compile_model(labor_cost.labor_cost_model(schema_version=2))
from labor_cost.reporting import _plotly
try:
    _plotly()
except ImportError as error:
    assert 'labor-cost-model[reporting]' in str(error)
else:
    raise AssertionError('Missing dependency was not detected')
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def test_show_does_not_change_renderers(stored_result, monkeypatch):
    """Delegate renderer choice and escape externally supplied labels."""
    import IPython.display
    import plotly.graph_objects as go

    output, renderers = [], []

    def capture(figure, renderer=None):
        """Capture display requests without opening a frontend."""
        renderers.append(renderer)

    monkeypatch.setattr(go.Figure, "show", capture)
    monkeypatch.setattr(IPython.display, "display", output.append)
    previous = pio.renderers.default
    stored_result.manifest["run_id"] = "<script>alert(1)</script>"
    charts = report(stored_result, teams=["Alpha"], driver="x",
                    renderer="notebook")
    assert pio.renderers.default == previous
    assert renderers == ["notebook"] * 3
    assert "<script>alert" not in output[0].data
    assert "&lt;script&gt;" in output[0].data
    assert "NotebookReport" in repr(charts)


def test_html_export_and_requirements(stored_result, tmp_path):
    """Export an offline figure and keep deployment extras synchronized."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    from packaging.requirements import Requirement

    figure = plot_organization_waterfall(stored_result)
    path = tmp_path / "organization.html"
    figure.write_html(path, include_plotlyjs=True)
    text = path.read_text()
    assert "Plotly.newPlot" in text
    assert '<script src="https://cdn.plot.ly' not in text
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    recipe = ROOT / "runtime/requirements-reporting.txt"
    lines = recipe.read_text().splitlines()
    actual = {str(Requirement(x)) for x in lines
              if x and not x.startswith(("#", "-r"))}
    assert actual == {str(Requirement(x)) for x in extras["reporting"]}
    assert set(extras["reporting"]).issubset(extras["notebook"])
    assert not any("plotly" in x for x in project["dependencies"])


def test_report_in_fresh_kernel(tmp_path):
    """Display a restored result through the actual Plotly notebook MIME."""
    from test_quality import _execute_notebook

    with zipfile.ZipFile(ROOT / "tests/fixtures/schema1_run.zip") as archive:
        archive.extractall(tmp_path / "run")
    path = next((tmp_path / "run").rglob("manifest.json")).parent
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
        "from labor_cost import load_run, report\n"
        f"result = load_run({str(path)!r})\n"
        "charts = report(result, teams=['Fixture team'], "
        "renderer='plotly_mimetype')\n"
        "assert result.states is None\n"
    )])
    _execute_notebook(notebook, tmp_path)
    figures = [o for o in notebook.cells[0].outputs
               if "application/vnd.plotly.v1+json" in o.get("data", {})]
    assert len(figures) == 2


@pytest.mark.parametrize("mode", ["exact", "permutation"])
def test_custom_nested_model_with_team_entry_exit(mode):
    """Render arbitrary nested drivers and complete team entry/exit."""
    from test_graph import authors

    rows = pd.DataFrame([
        ("2025-03-01", "a", "Alpha", 10., 8., -100.),
        ("2025-03-01", "b", "Beta", 10., 1., 0.),
        ("2025-04-01", "a", "Alpha", 12., 8., -100.),
        ("2025-04-01", "c", "Gamma", 5., 2., 0.),
    ], columns=["period", "employee_id", "team", "hourly_rate",
                "paid_hours", "allowance"])
    rows["grade"], rows["region"], rows["employment_type"] = "G1", "R1", "temp"
    rows["actual"] = rows.hourly_rate * rows.paid_hours + rows.allowance
    result = analyze(
        authors.hourly_model(nested=True),
        DatasetSnapshot("nested", "1", rows),
        AnalysisRequest("2025-03", "2025-04", source_cost_column="actual",
                        explainer=mode, n_permutations=2),
    )
    charts = report(result, teams=["Alpha", "Beta", "Gamma"],
                    driver="earnings::rates", show=False)
    assert charts.figures["local/Beta"].data[0].y[-1] == 0.
    assert charts.figures["local/Gamma"].data[0].y[0] == 0.
    org = charts.figures["organization"]
    assert org.data[0].y[0] == pytest.approx(-10.)
    assert org.data[0].y[-1] == pytest.approx(6.)
    assert "unit::base_earnings" not in org.layout.meta["driver_ids"]
    assert "earnings::rates" in org.layout.meta["driver_ids"]


def test_unrounded_chart_values(stored_result):
    """Keep precise impacts even though chart text uses display rounding."""
    amounts = [
        0., 30.123456789, -30.123456789, -20.123456789, 20.123456789, 0.,
    ]
    stored_result.tables["bridge"]["impact"] = amounts
    figure = plot_driver_by_team(stored_result, "x")
    assert list(figure.data[0].x) == amounts[1:3]
    assert list(figure.data[0].text) == ["+30.12", "-30.12"]


@pytest.mark.parametrize("column, dtype", [
    ("view", "string"), ("leaf", "boolean"), ("units", "string"),
])
def test_nullable_scope_is_not_implicitly_valid(stored_result, column, dtype):
    """Missing nullable flags cannot pass pandas' skip-NA reductions."""
    frame = stored_result.tables["bridge"]
    frame[column] = frame[column].astype(dtype)
    frame.loc[1, column] = pd.NA
    with pytest.raises(ValueError, match="currency leaves"):
        plot_organization_waterfall(stored_result)
