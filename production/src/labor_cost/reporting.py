"""Plotly notebook reports built exclusively from stored result tables."""

import html
import textwrap
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import AnalysisResult

_INCREASE = "#D97706"
_DECREASE = "#2563EB"
_TOTAL = "#475569"
_EXTRA = "Install labor-cost-model[reporting] in the notebook environment."
_ENDPOINTS = ["period_A_cost", "period_B_cost"]


def _plotly():
    """Load the optional plotting dependency only when building figures."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(_EXTRA) from None
    return go


def _table(result, name, *, optional=False):
    """Copy tables, including mutable object cells, without reading states."""
    if name not in result.tables:
        if optional:
            return pd.DataFrame()
        raise ValueError(f"Reporting requires result table {name!r}.")
    frame = result.table(name)
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].map(deepcopy)
    return frame


def _columns(frame, names, context):
    """Require unambiguous table columns before validating their contents."""
    if not frame.columns.is_unique:
        raise ValueError(f"{context}: duplicate column names.")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}.")


def _finite(frame, names, context):
    """Reject missing, nonnumeric and nonfinite chart amounts."""
    _columns(frame, names, context)
    for name in names:
        values = frame[name]
        if (
            not pd.api.types.is_numeric_dtype(values)
            or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)
            or values.isna().any()
            or not np.isfinite(values.to_numpy(dtype=float)).all()
        ):
            raise ValueError(f"{context}: {name} must contain finite numbers.")


def _equal(actual, expected, context):
    """Check presentation arithmetic with the analytical tolerances."""
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-6):
        raise ValueError(f"{context}: inconsistent amounts/reconciliation.")


def _bridge(frame, scope, context, *, team=None):
    """Validate a complete, signed currency leaf bridge and its endpoints."""
    _columns(frame, ["driver", "label", "view", "leaf", "units", "team"],
             context)
    if frame.empty:
        raise ValueError(f"{context}: a complete bridge must not be empty.")
    _finite(frame, ["impact", *_ENDPOINTS], context)
    if frame[["view", "leaf", "units"]].isna().any().any() or not (
        frame.view.eq(scope).all() and frame.leaf.eq(True).all()
        and frame.units.eq("currency").all()
    ):
        raise ValueError(f"{context}: expected {scope} currency leaves only.")
    for column in ("driver", "label"):
        if not frame[column].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        ).all():
            raise ValueError(f"{context}: {column} needs nonempty strings.")
    if team is not None and not frame.team.eq(team).all():
        raise ValueError(f"{context}: mixed or incorrect local team scope.")
    endpoints = frame[_ENDPOINTS].iloc[0].to_numpy(dtype=float)
    _equal(frame[_ENDPOINTS].to_numpy(dtype=float), endpoints, context)
    _equal(frame.impact.sum(), endpoints[1] - endpoints[0], context)
    return endpoints


class _ReportData:
    """Adapt the portable v1/v2 table contract, without calculation imports."""

    def __init__(self, result):
        """Validate organization bridges and normalize local team endpoints."""
        if not isinstance(result, AnalysisResult):
            raise TypeError("Pass an AnalysisResult, including from load_run.")
        self.result = result
        self.manifest = result.metadata()
        self.bridge = _table(result, "bridge")
        self.drivers = _table(result, "driver_bridge")
        self.endpoints = _bridge(self.bridge, "organization", "bridge")
        _equal(
            _bridge(self.drivers, "organization", "driver_bridge"),
            self.endpoints, "Organizational endpoints",
        )
        if self.drivers.driver.duplicated().any():
            raise ValueError("driver_bridge: driver IDs must be unique.")
        if self.drivers.team.notna().any():
            raise ValueError("driver_bridge: expected organization totals.")
        summed = self.bridge.groupby("driver", sort=False).impact.sum()
        if set(summed.index) != set(self.drivers.driver):
            raise ValueError("driver_bridge: IDs differ from bridge leaves.")
        _equal(summed.reindex(self.drivers.driver), self.drivers.impact,
               "Driver totals versus organizational leaves")
        summary = _table(result, "team_summary")
        summary = summary.rename(columns={
            "period_a_cost": "period_A_cost",
            "period_b_cost": "period_B_cost", "local_change": "cost_change",
        })
        _columns(summary, ["team", *_ENDPOINTS, "cost_change"], "team_summary")
        if (
            summary.empty or summary.team.duplicated().any()
            or not summary.team.map(
                lambda x: isinstance(x, str) and bool(x)
            ).all()
        ):
            raise ValueError("team_summary: expected unique named teams.")
        _finite(summary, [*_ENDPOINTS, "cost_change"], "team_summary")
        _equal(summary.cost_change,
               summary.period_B_cost - summary.period_A_cost, "Team changes")
        _equal(summary[_ENDPOINTS].sum().to_numpy(dtype=float),
               self.endpoints, "Team totals versus organizational endpoints")
        unknown = set(self.bridge.team.dropna()) - set(summary.team)
        if unknown:
            raise ValueError("bridge: team IDs absent from team_summary.")
        self.summary = summary

    def teams(self, teams):
        """Normalize a presentation selector; empty means unrestricted."""
        if teams is None:
            return ()
        if isinstance(teams, str):
            raise ValueError("Pass teams as a sequence, not a string.")
        selected = tuple(dict.fromkeys(teams))
        if any(t not in set(self.summary.team) for t in selected):
            raise ValueError("Unknown team; choose from team_summary.team.")
        return selected

    def local(self, team):
        """Validate the independent local bridge against local endpoints."""
        self.teams([team])
        frame = _table(self.result, "local_bridges/" + team)
        endpoints = _bridge(frame, "local", f"Local {team}", team=team)
        expected = self.summary.loc[self.summary.team.eq(team), _ENDPOINTS]
        _equal(endpoints, expected.iloc[0], f"Local {team} endpoints")
        return frame

    def driver_teams(self, driver, teams):
        """Sum stored organizational leaves by team, never local effects."""
        if driver not in set(self.drivers.driver):
            raise ValueError(
                "Unknown/inactive or parent driver. Choose a leaf driver ID "
                "from result.table('driver_bridge').driver."
            )
        rows = self.bridge.loc[self.bridge.driver.eq(driver)]
        if rows.team.isna().any():
            raise ValueError(
                f"Driver {driver!r} has organization-wide contributions "
                "without a team allocation. Use the organization waterfall."
            )
        selected = self.teams(teams)
        names = list(selected) if selected else self.summary.team.tolist()
        amounts = rows.groupby("team", sort=False).impact.sum()
        frame = pd.DataFrame({
            "team": names, "driver": driver,
            "impact": amounts.reindex(names, fill_value=0.0).to_numpy(),
        })
        total = float(rows.impact.sum())
        return frame, total, bool(selected)


def _label(value):
    """Escape externally supplied labels for Plotly's HTML text fields."""
    return html.escape(str(value), quote=True)


def _currency(value):
    """Require a plain nonempty unit label without interpreting currency."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("currency_label must be a nonempty string.")
    return value


def _periods(data):
    """Use recorded comparison periods or explicit unavailable labels."""
    periods = data.manifest.get("periods", [])
    if len(periods) != 2:
        return ["Period A", "Period B"]
    return ["A: " + str(periods[0])[:10], "B: " + str(periods[1])[:10]]


def _waterfall(data, frame, title, currency_label):
    """Create absolute endpoints and relative leaves without rounding data."""
    go = _plotly()
    currency_label = _currency(currency_label)
    counts = Counter(frame.label)
    labels = [
        f"{row.label} [{row.driver}]" if counts[row.label] > 1 else row.label
        for row in frame.itertuples()
    ]
    endpoints = frame[_ENDPOINTS].iloc[0].to_numpy(dtype=float).tolist()
    periods = _periods(data)
    labels = [periods[0], *labels, periods[1]]
    ids = ["", *frame.driver.tolist(), ""]
    amounts = [endpoints[0], *frame.impact.tolist(), endpoints[1]]
    positions = list(range(len(amounts)))
    figure = go.Figure(go.Waterfall(
        x=positions, y=amounts,
        measure=["absolute", *(["relative"] * len(frame)), "absolute"],
        customdata=[[_label(label), _label(driver)]
                    for label, driver in zip(labels, ids)],
        text=[f"{amount:+,.2f}" if 0 < i < len(amounts) - 1
              else f"{amount:,.2f}" for i, amount in enumerate(amounts)],
        textposition="outside", cliponaxis=False,
        hovertemplate=(
            "%{customdata[0]}<br>ID: %{customdata[1]}<br>"
            + _label(currency_label) + ": %{y:,.6f}<extra></extra>"
        ),
        increasing={"marker": {"color": _INCREASE}},
        decreasing={"marker": {"color": _DECREASE}},
        totals={"marker": {"color": _TOTAL}},
        connector={"line": {"color": "#CBD5E1"}},
    ))
    figure.update_layout(
        title={"text": _label(title)}, template="plotly_white",
        showlegend=False, height=600, margin={"b": 170, "t": 90},
        yaxis={"title": _label(currency_label), "zeroline": True},
        xaxis={
            "tickmode": "array", "tickvals": positions,
            "ticktext": ["<br>".join(_label(s) for s in textwrap.wrap(x, 22))
                         for x in labels],
            "tickangle": -30 if len(frame) > 8 else 0,
        },
        meta={"run_id": data.manifest.get("run_id"), "driver_ids": ids,
              "period_A_cost": endpoints[0], "period_B_cost": endpoints[1]},
    )
    return figure


def _driver_plot(data, driver, teams, currency_label):
    """Plot signed team contributions and explicitly label any subtotal."""
    go = _plotly()
    currency_label = _currency(currency_label)
    frame, total, restricted = data.driver_teams(driver, teams)
    label = data.drivers.loc[data.drivers.driver.eq(driver), "label"].iloc[0]
    subtitle = f"Organizational driver total: {total:,.2f} {currency_label}"
    subtotal = float(frame.impact.sum())
    if restricted:
        subtitle += f" · Selected subtotal: {subtotal:,.2f} {currency_label}"
    figure = go.Figure(go.Bar(
        x=frame.impact.tolist(), y=list(range(len(frame))), orientation="h",
        marker_color=[_INCREASE if x > 0 else _DECREASE if x < 0 else _TOTAL
                      for x in frame.impact],
        customdata=[[_label(t), _label(driver)] for t in frame.team],
        text=[f"{x:+,.2f}" for x in frame.impact], textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{customdata[0]}<br>ID: %{customdata[1]}<br>"
            "Organizational contribution: %{x:,.6f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        title={"text": "Organizational contributions by team: "
               + _label(label) + "<br><sup>" + _label(subtitle) + "</sup>"},
        template="plotly_white", height=max(350, 38 * len(frame) + 150),
        margin={"l": 160, "r": 90, "t": 100}, showlegend=False,
        xaxis={"title": _label(currency_label), "zeroline": True},
        yaxis={"tickmode": "array", "tickvals": list(range(len(frame))),
               "ticktext": [_label(t) for t in frame.team],
               "autorange": "reversed"},
        meta={"run_id": data.manifest.get("run_id"), "driver": driver,
              "organization_total": total, "selected_subtotal": subtotal,
              "selected_teams": frame.team.tolist(), "restricted": restricted},
    )
    return figure, frame


def plot_organization_waterfall(result, *, currency_label="Currency"):
    """Return a full-organization Plotly Figure from stored driver leaves.

    Parameters
    ----------
    result : AnalysisResult
        In-memory analysis or restored run; internal states are unused.
    currency_label : str
        Display unit only. No conversion or currency inference occurs.

    Returns
    -------
    plotly.graph_objects.Figure
        Editable figure; not displayed automatically.
    """
    data = _ReportData(result)
    return _waterfall(data, data.drivers, "Organizational driver waterfall",
                      currency_label)


def plot_team_waterfall(result, team, *, currency_label="Currency"):
    """Return a team's independent local cost waterfall.

    Parameters
    ----------
    result : AnalysisResult
        Analysis with portable organizational and local tables.
    team : str
        Existing team name, never an input filter.
    currency_label : str
        Display unit without conversion.

    Returns
    -------
    plotly.graph_objects.Figure
        Local endpoints and local allocations; no organizational mixing.
    """
    data = _ReportData(result)
    return _waterfall(data, data.local(team),
                      f"Local team cost change: {team}", currency_label)


def plot_driver_by_team(result, driver, *, teams=None,
                        currency_label="Currency"):
    """Return signed organizational contributions for one leaf driver.

    Parameters
    ----------
    result : AnalysisResult
        Analysis containing full organizational leaf contributions.
    driver : str
        Leaf driver ID with a team allocation, not a display label.
    teams : sequence of str, optional
        None/empty displays every team; a selection shows its subtotal.
    currency_label : str
        Display unit without conversion.

    Returns
    -------
    plotly.graph_objects.Figure
        Horizontal bars without percentage normalization.
    """
    return _driver_plot(_ReportData(result), driver, teams, currency_label)[0]


def _summaries(result):
    """Summarize recorded evidence, never infer missing checks or methods."""
    raw = {name: _table(result, name, optional=True) for name in (
        "validation", "assumptions", "coalition_audit",
    )}
    validation = raw["validation"]
    statuses = []
    if not validation.empty and "passed" in validation:
        statuses = validation.passed.map(
            lambda x: "Passed" if isinstance(x, (bool, np.bool_)) and x
            else "Failed" if isinstance(x, (bool, np.bool_))
            else "Unavailable"
        ).tolist()
    validation_summary = pd.DataFrame(
        Counter(statuses).items(), columns=["recorded_status", "checks"],
    ) if statuses else pd.DataFrame(
        [{"recorded_status": "Unavailable", "checks": None}]
    )
    assumptions = raw["assumptions"]
    if "assumptions" not in result.tables:
        assumption_summary = pd.DataFrame([{"method": "Unavailable"}])
    elif assumptions.empty:
        assumption_summary = pd.DataFrame([{"method": "None recorded"}])
    elif "method" in assumptions:
        assumption_summary = assumptions.groupby(
            "method", sort=False, dropna=False,
        ).size().rename("records").reset_index()
    else:
        assumption_summary = pd.DataFrame([{
            "method": "Unavailable", "records": len(assumptions),
        }])
    audit = raw["coalition_audit"]
    estimates = []
    if not audit.empty and "method" in audit:
        for method, rows in audit.groupby("method", sort=False, dropna=False):
            record = {"method": method if pd.notna(method) else "Unavailable",
                      "games": len(rows)}
            for column in ("n_permutations", "evaluation_budget",
                           "model_evaluations", "coalitions"):
                values = (
                    rows[column].dropna().unique() if column in rows else []
                )
                record[column] = ", ".join(map(str, values)) or "Unavailable"
            estimates.append(record)
    return {
        **raw, "validation_summary": validation_summary,
        "assumption_summary": assumption_summary,
        "estimator_summary": pd.DataFrame(estimates or [
            {"method": "Unavailable", "games": None},
        ]),
    }


@dataclass(repr=False)
class NotebookReport:
    """Editable figures, copied tables and presentation metadata.

    Figure keys are ``organization``, ``local/<team>`` and optionally
    ``driver_by_team``. Audit tables are retained alongside compact summaries.
    This object has no calculation states and does not modify its source run.
    """

    figures: dict
    tables: dict
    metadata: dict

    def __repr__(self):
        """Give a compact terminal representation without redisplaying."""
        return f"NotebookReport(figures={list(self.figures)!r})"

    def show(self, renderer=None):
        """Display stored report content without changing global renderers."""
        try:
            from IPython.display import HTML, display
        except ImportError:
            raise ImportError(_EXTRA) from None
        heading = "<h3>Labor-cost analysis</h3><dl>" + "".join(
            f"<dt>{_label(k)}</dt><dd>{_label(v)}</dd>"
            for k, v in self.metadata.items()
        ) + "</dl>"
        display(HTML(heading))
        self.figures["organization"].show(renderer=renderer)
        for name in ("driver_bridge", "team_summary", "validation_summary",
                     "assumption_summary", "estimator_summary"):
            self._display_table(name, display, HTML)
        for name, figure in self.figures.items():
            if name != "organization":
                figure.show(renderer=renderer)
                self._display_table(name, display, HTML)

    def _display_table(self, name, display, html_type):
        """Escape table cells and format numbers only in rendered output."""
        frame = self.tables[name]
        if name == "driver_bridge" or name.startswith("local/"):
            frame = frame[["driver", "label", "impact"]]
        display(html_type(
            "<h4>" + _label(name.replace("_", " ")) + "</h4>"
            + frame.to_html(index=False, escape=True,
                            float_format="{:,.2f}".format)
        ))


def report(result, *, teams=None, driver=None, currency_label="Currency",
           show=True, renderer=None):
    """Build and optionally display a report from existing analysis tables.

    Parameters
    ----------
    result : AnalysisResult
        An analyzed or restored run; no calculations are executed.
    teams : sequence of str, optional
        Selected local waterfalls and summary/breakdown teams. None/empty
        adds no local waterfalls and leaves other team views unrestricted.
        The organizational waterfall always covers the full organization.
    driver : str, optional
        Add organizational contributions by team for this leaf driver ID.
    currency_label : str
        Presentation unit; defaults to Currency, with no conversion.
    show : bool
        Display immediately by default; False only builds the report.
    renderer : str, optional
        Plotly renderer for this display call, without changing defaults.

    Returns
    -------
    NotebookReport
        Reusable figures, independent table copies and report metadata.
    """
    data = _ReportData(result)
    selected = data.teams(teams)
    figures = {"organization": _waterfall(
        data, data.drivers, "Organizational driver waterfall", currency_label,
    )}
    summary = data.summary
    if selected:
        summary = summary.set_index("team").loc[list(selected)].reset_index()
    tables = {"driver_bridge": data.drivers, "team_summary": summary,
              **_summaries(result)}
    for team in selected:
        frame = data.local(team)
        key = "local/" + team
        figures[key] = _waterfall(data, frame,
                                  f"Local team cost change: {team}",
                                  currency_label)
        tables[key] = frame
    if driver is not None:
        figures["driver_by_team"], tables["driver_by_team"] = _driver_plot(
            data, driver, selected, currency_label,
        )
    settings = data.manifest.get("settings", {})
    source = settings.get("source_cost_column")
    basis = (f"Independent source column: {source}" if source else
             "Reconstructed from model row formulas")
    if "source_cost_column" not in settings:
        basis = "Unavailable"
    metadata = {
        "run_id": data.manifest.get("run_id", "Unavailable"),
        "periods": data.manifest.get("periods", "Unavailable"),
        "model": {k: data.manifest.get("model", {}).get(k, "Unavailable")
                  for k in ("model_id", "version")},
        "dataset": {k: data.manifest.get("dataset", {}).get(k, "Unavailable")
                    for k in ("id", "version")},
        "currency_label": currency_label, "selected_teams": selected,
        "source_cost_basis": basis,
        "exposure": data.manifest.get("exposure_config", "Unavailable"),
        "interpretation": "Accounting attribution, not a causal estimate",
        "validation_scope": "Recorded checks only; not a complete audit. "
                            "Existing architecture gaps remain open.",
    }
    output = NotebookReport(figures, tables, metadata)
    if show:
        output.show(renderer=renderer)
    return output
