"""Protect aggregation of organizational impacts and report selection."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import assert_equal_states, payroll_row
from test_analysis import analyze


@pytest.fixture(scope="module")
def team_analysis(model, months):
    """Change scale, team shares and signed rates with different local effects.
    """
    frame = pd.DataFrame(
        [
            payroll_row(
                str(months[0].date()), "a", contract="ГПД", rate=-100,
            ),
            payroll_row(
                str(months[0].date()), "b", team="Beta",
                contract="ГПД", rate=-200,
            ),
            *[
                payroll_row(
                    str(months[1].date()), employee,
                    contract="ГПД", rate=-60,
                )
                for employee in ("a", "c", "d")
            ],
            payroll_row(
                str(months[1].date()), "b", team="Beta",
                contract="ГПД", rate=-230,
            ),
        ]
    )
    return analyze(model, frame, months)


def test_organizational_summary_not_local_sum(team_analysis):
    """Preserve organizational multipliers when scale and team mix interact."""
    result = team_analysis
    summary = result["driver_bridge"].set_index("driver")
    assert summary.loc["temp_reimbursement", "impact"] == pytest.approx(
        265 / 6
    )
    local_total = sum(
        bridge.loc[bridge.driver.eq("temp_reimbursement"), "impact"].sum()
        for bridge in result["local_bridges"].values()
    )
    assert local_total == pytest.approx(50)
    assert summary.impact.sum() == pytest.approx(-110)
    assert (summary.period_A_cost == -300).all()
    assert (summary.period_B_cost == -410).all()
    np.testing.assert_allclose(
        summary.share_of_net_change, summary.impact / -110
    )
    assert summary.team.isna().all()
    assert summary.view.eq("organization").all()
    assert summary.units.eq("currency").all()
    assert summary.leaf.all()
    assert list(summary.index) == [
        "workforce_scale",
        "team_mix",
        "contract_type_mix",
        "temp_reimbursement",
    ]
    for driver, rows in result["bridge"].groupby("driver"):
        assert summary.loc[driver, "impact"] == pytest.approx(
            rows.impact.sum()
        )


def test_cancelling_driver_keeps_signed_team_detail(
    model, months, monkeypatch,
):
    """Keep zero totals and undefined shares when team impacts offset."""
    frame = pd.DataFrame(
        [
            payroll_row(
                str(month.date()), team, team=team, contract="ГПД", rate=rate,
            )
            for month, rates in zip(months, ((100, 200), (120, 180)))
            for team, rate in zip(("Alpha", "Beta"), rates)
        ]
    )
    result = analyze(model, frame, months)
    summary = result["driver_bridge"].set_index("driver")
    assert summary.loc["temp_reimbursement", "impact"] == pytest.approx(0)
    assert summary.share_of_net_change.isna().all()
    shown = {}
    monkeypatch.setitem(
        model, "show", lambda title, value: shown.update({title: value}),
    )
    model["report"](
        result, organization_view="drivers", driver="temp_reimbursement",
        plots=False,
    )
    detail = shown["Temp reimbursement — ORGANIZATIONAL contributions by team"]
    np.testing.assert_allclose(detail.impact, [20, -20])
    total = shown["Selected driver total — organizational currency"]
    assert total.team_contribution_total == pytest.approx(0)
    assert total.aggregated_driver_impact == pytest.approx(0)


def test_aggregation_uses_registry_without_recomputation(
    model, team_analysis, monkeypatch,
):
    """Honor registry metadata and preserve the source analytical results."""
    before = deepcopy(team_analysis)
    registry = tuple(
        replace(spec, label="Updated " + spec.label)
        for spec in reversed(model["DRIVERS"])
    )

    def unexpected_calculation(*args, **kwargs):
        """Fail if reporting invokes any analytical computation."""
        pytest.fail("Reporting must reuse the calculated impacts")

    for name in ("cost", "exact_game", "permutation_game", "decompose"):
        monkeypatch.setitem(model, name, unexpected_calculation)
    summary = model["aggregate_driver_bridge"](
        team_analysis["bridge"], registry,
    )
    assert list(summary.driver) == list(
        reversed(before["driver_bridge"].driver)
    )
    assert summary.label.str.startswith("Updated ").all()
    monkeypatch.setitem(model, "show", lambda title, value: None)
    model["report"](
        team_analysis, organization_view="drivers",
        driver="temp_reimbursement", plots=False,
    )
    for key in ("bridge", "driver_bridge", "hierarchy", "coalition_audit"):
        pd.testing.assert_frame_equal(team_analysis[key], before[key])
    for team, bridge in before["local_bridges"].items():
        pd.testing.assert_frame_equal(
            team_analysis["local_bridges"][team], bridge
        )
    for actual, original in zip(team_analysis["states"], before["states"]):
        assert_equal_states(actual, original)


def test_reject_local_rows_and_parents(model, team_analysis):
    """Prevent aggregation from counting parents or local views."""
    for rows in (
        team_analysis["local_bridges"]["Alpha"],
        team_analysis["hierarchy"].query("view == 'organization'"),
    ):
        with pytest.raises(ValueError, match="organizational leaf"):
            model["aggregate_driver_bridge"](rows)


def test_reconciliation_detects_misallocated_summary(model, team_analysis):
    """A correct grand total must not hide wrong individual driver totals."""
    corrupted = dict(team_analysis)
    corrupted["driver_bridge"] = team_analysis["driver_bridge"].copy()
    corrupted["driver_bridge"].loc[0, "impact"] += 1
    corrupted["driver_bridge"].loc[1, "impact"] -= 1
    with pytest.raises(AssertionError, match="Organizational driver total"):
        model["reconcile_games"](*team_analysis["states"], corrupted)


@pytest.mark.parametrize("view", ["teams", "drivers"])
def test_report_view_and_waterfall(model, team_analysis, monkeypatch, view):
    """Select tables, exposure grain and waterfall rows for each view."""
    shown, plots = {}, []
    monkeypatch.setitem(
        model, "show", lambda title, value: shown.update({title: value}),
    )
    monkeypatch.setitem(
        model, "waterfall",
        lambda bridge, start, end, title: plots.append((bridge, start, end)),
    )
    options = {} if view == "teams" else {"organization_view": view}
    model["report"](team_analysis, **options)
    assert len(plots) == 1
    plotted, start, end = plots[0]
    assert (start, end) == (-300, -410)
    assert "Attribution methods" in shown
    assert any("reference basis" in title for title in shown)
    assert "Organizational endpoints" in shown
    counts = shown["Employee-month counts and exposure"]
    if view == "drivers":
        pd.testing.assert_frame_equal(
            plotted, team_analysis["driver_bridge"]
        )
        table = shown["Organizational driver bridge — summed across teams"]
        pd.testing.assert_series_equal(table.driver, plotted.driver)
        assert "Team comparison — two distinct views" not in shown
        assert "Organizational leaf bridge" not in shown
        assert counts.index.names == ["period", "contract_type"]
    else:
        assert (plotted.level == 1).all()
        assert "Team comparison — two distinct views" in shown
        assert "Organizational leaf bridge" in shown
        assert counts.index.names == ["period", "team", "contract_type"]


def test_driver_plot_labels_and_separate_local_report(
    model, team_analysis, monkeypatch,
):
    """Plot driver labels without team prefixes and preserve local reports."""
    bridge = team_analysis["driver_bridge"]
    labels = model["waterfall"](bridge, -300, -410, "Organizational drivers")
    assert labels[1:-1] == [
        f"{row.label} ({row.impact:+,.0f})" for row in bridge.itertuples()
    ]
    shown = {}
    monkeypatch.setitem(
        model, "show", lambda title, value: shown.update({title: value}),
    )
    model["report"](team_analysis, team="Alpha", plots=False)
    local_title = next(title for title in shown if "LOCAL Alpha" in title)
    np.testing.assert_allclose(
        shown[local_title].impact,
        team_analysis["local_bridges"]["Alpha"].impact,
    )


@pytest.mark.parametrize(
    "driver",
    [
        "missing", "workforce_scale", "team_mix", "permanent_unit_cost",
        "underlying_salary_level",
    ],
)
def test_invalid_driver_selection(model, team_analysis, driver):
    """Reject unknown, inactive, parent and global selectors before display."""
    with pytest.raises(ValueError, match="economics leaf"):
        model["report"](
            team_analysis, organization_view="drivers", driver=driver,
            plots=False,
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"organization_view": "pooled"}, "organization_view"),
        ({"driver": "temp_reimbursement"}, "requires"),
        (
            {
                "organization_view": "drivers",
                "driver": "temp_reimbursement",
                "team": "Alpha",
            },
            "requires",
        ),
    ],
)
def test_invalid_report_selection(model, team_analysis, options, message):
    """Reject incompatible report and driver selector combinations."""
    with pytest.raises(ValueError, match=message):
        model["report"](team_analysis, plots=False, **options)
