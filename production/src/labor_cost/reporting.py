"""Plotly notebook reports built exclusively from stored result tables."""

import html
import textwrap
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import AnalysisResult

_INCREASE = "#DC2626"
_DECREASE = "#16A34A"
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
        if "node_summary" in result.tables:
            self.nodes = _table(result, "node_summary")
            _finite(self.nodes, [*_ENDPOINTS, "cost_change"], "node_summary")
            if self.nodes.node_path.duplicated().any():
                raise ValueError("Duplicate organizational node paths")
            top = self.nodes.loc[self.nodes.depth.eq(1)]
            _equal(top[_ENDPOINTS].sum().to_numpy(), self.endpoints,
                   "Node totals versus organizational endpoints")
            self.summary = top
            return
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


def _color(amount):
    """Color cost changes by sign, with neutral totals handled separately."""
    return _INCREASE if amount > 0 else _DECREASE if amount < 0 else _TOTAL


def _top_n(top_n):
    """Validate a presentation-only limit without coercing booleans."""
    if top_n is not None and (
        isinstance(top_n, (bool, np.bool_))
        or not isinstance(top_n, (int, np.integer)) or top_n <= 0
    ):
        raise ValueError("top_n must be a positive integer or None.")


def _chart_rows(frame, *, top_n=None, ranked=False):
    """Copy driver totals and retain complete membership of condensed bars."""
    _top_n(top_n)
    rows = []
    for driver, group in frame.groupby("driver", sort=False):
        if group.label.nunique() != 1:
            raise ValueError(f"Driver {driver!r} has inconsistent labels.")
        amount = float(group.impact.sum())
        label = group.label.iloc[0]
        rows.append({
            "driver": driver, "label": label, "impact": amount,
            "kind": "driver", "member_count": 1,
            "positive_impact": max(0., amount),
            "negative_impact": min(0., amount),
            "members": [{"driver": driver, "label": label,
                         "impact": amount}],
        })
    if ranked or top_n is not None:
        rows.sort(key=lambda row: -abs(row["impact"]))
    if top_n is not None and len(rows) > top_n:
        rest = rows[top_n:]
        rows = rows[:top_n] + [{
            "driver": None, "label": f"Other ({len(rest)} drivers)",
            "impact": sum(row["impact"] for row in rest), "kind": "other",
            "member_count": len(rest),
            "positive_impact": sum(row["positive_impact"] for row in rest),
            "negative_impact": sum(row["negative_impact"] for row in rest),
            "members": [member for row in rest for member in row["members"]],
        }]
    result = pd.DataFrame(rows)
    _equal(result.impact.sum(), frame.impact.sum(), "Condensed driver totals")
    return result


def _chart_labels(frame):
    """Disambiguate duplicate driver labels and synthetic Other groups."""
    counts = Counter(frame.label)
    return [
        f"{row.label} [{row.driver}]"
        if row.kind == "driver" and (counts[row.label] > 1
                                     or row.label.startswith("Other"))
        else row.label for row in frame.itertuples()
    ]


def _hover_data(frame, currency_label):
    """Bind hover values to stored impacts, never cumulative coordinates."""
    custom, templates = [], []
    for label, row in zip(_chart_labels(frame), frame.itertuples()):
        custom.append([_label(label), _label(row.driver or ""), row.impact,
                       row.kind, row.member_count, row.positive_impact,
                       row.negative_impact])
        template = "%{customdata[0]}<br>"
        if row.kind == "driver":
            template += "ID: %{customdata[1]}<br>"
        template += ("Impact: " + ("+" if row.impact >= 0 else "")
                     + "%{customdata[2]:,.6f} " + _label(currency_label))
        if row.kind == "other":
            template += (
                "<br>Grouped drivers: %{customdata[4]}"
                "<br>Positive impacts: +%{customdata[5]:,.6f} "
                + _label(currency_label)
                + "<br>Negative impacts: %{customdata[6]:,.6f} "
                + _label(currency_label)
            )
        templates.append(template + "<extra></extra>")
    return custom, templates


def _chart_meta(data, frame, endpoints, chart, top_n):
    """Retain endpoints and grouped-driver evidence in exported figures."""
    return {
        "run_id": data.manifest.get("run_id"), "chart": chart,
        "top_n": None if top_n is None else int(top_n),
        "driver_ids": frame.driver.tolist(),
        "period_A_cost": endpoints[0], "period_B_cost": endpoints[1],
        "chart_data": frame.to_dict("records"),
    }


def _waterfall(data, frame, title, currency_label, *, top_n=None):
    """Create absolute endpoints and full or condensed relative impacts."""
    go = _plotly()
    currency_label = _currency(currency_label)
    endpoints = frame[_ENDPOINTS].iloc[0].to_numpy(dtype=float).tolist()
    frame = _chart_rows(frame, top_n=top_n)
    periods = _periods(data)
    labels = [periods[0], *_chart_labels(frame), periods[1]]
    amounts = [endpoints[0], *frame.impact.tolist(), endpoints[1]]
    custom, templates = _hover_data(frame, currency_label)
    endpoint_hover = ("%{customdata[0]}<br>Period total: "
                      "%{customdata[2]:,.6f} " + _label(currency_label)
                      + "<extra></extra>")
    custom = [[_label(periods[0]), "", endpoints[0], "endpoint", 0, 0, 0],
              *custom,
              [_label(periods[1]), "", endpoints[1], "endpoint", 0, 0, 0]]
    positions = list(range(len(amounts)))
    meta = _chart_meta(data, frame, endpoints, "waterfall", top_n)
    meta["driver_ids"] = ["", *frame.driver.tolist(), ""]
    figure = go.Figure(go.Waterfall(
        x=positions, y=amounts,
        measure=["absolute", *(["relative"] * len(frame)), "absolute"],
        customdata=custom,
        text=[f"{amount:+,.2f}" if 0 < i < len(amounts) - 1
              else f"{amount:,.2f}" for i, amount in enumerate(amounts)],
        textposition="outside", cliponaxis=False,
        hovertemplate=[endpoint_hover, *templates, endpoint_hover],
        hoverlabel={"bgcolor": [_TOTAL, *map(_color, frame.impact), _TOTAL]},
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
        }, meta=meta,
    )
    return figure


def _impacts(data, frame, title, currency_label, *, top_n=None):
    """Separate endpoint indicators from the scale of ranked driver bars."""
    go = _plotly()
    currency_label = _currency(currency_label)
    endpoints = frame[_ENDPOINTS].iloc[0].to_numpy(dtype=float).tolist()
    frame = _chart_rows(frame, top_n=top_n, ranked=True)
    labels = _chart_labels(frame)
    wrapped = [textwrap.wrap(label, 34) for label in labels]
    custom, templates = _hover_data(frame, currency_label)
    height = max(550, sum(max(40, 19 * len(x)) for x in wrapped) + 260)
    chart_top = 1 - 140 / (height - 130)
    bound = max(abs(frame.impact)) * 1.35 or 1.
    figure = go.Figure(go.Bar(
        x=frame.impact.tolist(), y=list(range(len(frame))), orientation="h",
        marker_color=[_color(amount) for amount in frame.impact],
        customdata=custom, hovertemplate=templates,
        text=[f"{amount:+,.2f}" for amount in frame.impact],
        textposition="outside", cliponaxis=False,
    ))
    net_change = endpoints[1] - endpoints[0]
    for i, (label, amount) in enumerate(zip(
        [*_periods(data), "Net change"], [*endpoints, net_change],
    )):
        figure.add_trace(go.Indicator(
            mode="number", value=amount,
            title={"text": _label(label) + "<br><span style='font-size:12px'>"
                   + _label(currency_label) + "</span>",
                   "font": {"size": 14}},
            number={"valueformat": ",.2f",
                    "prefix": "+" if i == 2 and amount >= 0 else "",
                    "font": {"size": min(26, 230 / len(f"{amount:,.2f}")),
                             "color": _color(amount) if i == 2 else _TOTAL}},
            domain={"x": [i / 3, (i + 1) / 3 - .035],
                    "y": [chart_top + 24 / (height - 130), 1]},
        ))
    figure.update_layout(
        title={"text": _label(title)}, template="plotly_white",
        showlegend=False, height=height,
        margin={"l": 260, "r": 110, "t": 80, "b": 50},
        xaxis={"title": "Driver impact · " + _label(currency_label),
               "range": [-bound, bound], "zeroline": True,
               "zerolinecolor": _TOTAL, "zerolinewidth": 1.5},
        yaxis={"domain": [0, chart_top], "autorange": "reversed",
               "tickmode": "array", "tickvals": list(range(len(frame))),
               "ticktext": ["<br>".join(map(_label, lines))
                            for lines in wrapped], "automargin": True},
        meta=_chart_meta(data, frame, endpoints, "impacts", top_n),
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


def plot_organization_waterfall(result, *, currency_label="Currency",
                                top_n=None):
    """Return a full-organization Plotly Figure from stored driver leaves.

    Parameters
    ----------
    result : AnalysisResult
        In-memory analysis or restored run; internal states are unused.
    currency_label : str
        Display unit only. No conversion or currency inference occurs.

    top_n : int, optional
        Largest absolute drivers to retain, with the rest summed as Other.
        None preserves the complete waterfall in its original order.

    Returns
    -------
    plotly.graph_objects.Figure
        Editable figure; not displayed automatically.
    """
    data = _ReportData(result)
    return _waterfall(data, data.drivers, "Organizational driver waterfall",
                      currency_label, top_n=top_n)


def plot_team_waterfall(result, team, *, currency_label="Currency",
                        top_n=None):
    """Return a team's independent local cost waterfall.

    Parameters
    ----------
    result : AnalysisResult
        Analysis with portable organizational and local tables.
    team : str
        Existing team name, never an input filter.
    currency_label : str
        Display unit without conversion.

    top_n : int, optional
        Largest absolute drivers to retain, with the rest summed as Other.
        None preserves the complete waterfall in its original order.

    Returns
    -------
    plotly.graph_objects.Figure
        Local endpoints and local allocations; no organizational mixing.
    """
    data = _ReportData(result)
    return _waterfall(data, data.local(team),
                      f"Local team cost change: {team}", currency_label,
                      top_n=top_n)


def plot_organization_impacts(result, *, currency_label="Currency",
                              top_n=None):
    """Return organization totals above ranked signed driver impacts.

    Parameters
    ----------
    result : AnalysisResult
        Stored organizational contributions, including loaded run bundles.
    currency_label : str
        Display unit without conversion.
    top_n : int, optional
        Show this many largest absolute impacts plus Other. None shows all.

    Returns
    -------
    plotly.graph_objects.Figure
        One exportable figure; no display or attribution is performed.
    """
    data = _ReportData(result)
    return _impacts(data, data.drivers, "Organizational cost drivers",
                    currency_label, top_n=top_n)


def plot_team_impacts(result, team, *, currency_label="Currency", top_n=None):
    """Return local team totals above ranked local driver impacts.

    Parameters
    ----------
    result : AnalysisResult
        Stored results with independent local bridges.
    team : str
        Existing team name, never a recalculated organizational subset.
    currency_label : str
        Display unit without conversion.
    top_n : int, optional
        Show this many largest absolute impacts plus Other. None shows all.

    Returns
    -------
    plotly.graph_objects.Figure
        One exportable figure using this team's local allocations.
    """
    data = _ReportData(result)
    return _impacts(data, data.local(team), f"Local cost drivers: {team}",
                    currency_label, top_n=top_n)


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
        summary = (
            "node_summary" if "node_summary" in self.tables else "team_summary"
        )
        for name in ("driver_bridge", summary, "validation_summary",
                     "assumption_summary", "estimator_summary"):
            self._display_table(name, display, HTML)
        for name in self.tables:
            if name == "driver_by_node" or name.startswith(
                "organizational_contributions/"
            ):
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


def report(
    result,
    *,
    teams=None,
    nodes=None,
    driver=None,
    currency_label="Currency",
    show=True,
    renderer=None,
    chart="impacts",
    top_n=None,
):
    """Build and optionally display a report from existing analysis tables.

    Parameters
    ----------
    result : AnalysisResult
        An analyzed or restored run; no calculations are executed.
    nodes : sequence of node paths, optional
        Show each selected node's organizational contributions and local
        payroll bridge. A path is an ordered tuple of dimension/value pairs.
    teams : sequence of str, optional
        Selected local charts and summary/breakdown teams. None/empty
        adds no local charts and leaves other team views unrestricted.
        The organizational chart always covers the full organization.
    driver : str, optional
        Add organizational contributions by team for this leaf driver ID.
    currency_label : str
        Presentation unit; defaults to Currency, with no conversion.
    chart : {"impacts", "waterfall"}
        Default impacts separates period totals from ranked driver bars.
    top_n : int, optional
        Show this many largest absolute drivers plus Other in each scope.
        None shows all drivers; full waterfalls retain their original order.
    show : bool
        Display immediately by default; False only builds the report.
    renderer : str, optional
        Plotly renderer for this display call, without changing defaults.

    Returns
    -------
    NotebookReport
        Reusable figures, independent table copies and report metadata.
    """
    if chart not in ("impacts", "waterfall"):
        raise ValueError("chart must be 'impacts' or 'waterfall'.")
    _top_n(top_n)
    data = _ReportData(result)
    hierarchical = "node_summary" in result.tables
    if nodes is not None and not hierarchical:
        raise ValueError("Node selection requires a hierarchy-enabled run")
    if hierarchical and teams and "team_summary" not in result.tables:
        raise ValueError("Select dimension-qualified nodes for this order")
    selected = data.teams(teams)
    plot = _impacts if chart == "impacts" else _waterfall
    figures = {"organization": plot(
        data, data.drivers, "Organizational cost drivers", currency_label,
        top_n=top_n,
    )}
    summary = data.nodes if hierarchical else data.summary
    if selected and not hierarchical:
        summary = summary.set_index("team").loc[list(selected)].reset_index()
    summary_name = "node_summary" if hierarchical else "team_summary"
    tables = {"driver_bridge": data.drivers, summary_name: summary,
              **_summaries(result)}
    for team in selected:
        frame = data.local(team)
        key = "local/" + team
        figures[key] = plot(data, frame,
                            f"Local team cost change: {team}",
                            currency_label, top_n=top_n)
        tables[key] = frame
    for node in nodes or ():
        path = tuple(tuple(part) for part in node)
        selection = result.select_node(path)
        if not path:
            continue
        key = repr(path)
        tables["organizational_contributions/" + key] = selection[
            "organization_contributions"
        ]
        local_frame = selection["local_bridge"]
        local_endpoints = _bridge(local_frame, "local", f"Local {path}")
        _equal(local_endpoints, selection["node_summary"][_ENDPOINTS].iloc[0],
               "Local bridge versus observed node endpoints")
        figures["local/" + key] = plot(
            data, local_frame, f"Local payroll change: {path}", currency_label,
            top_n=top_n,
        )
        tables["local/" + key] = local_frame
    for key, figure in figures.items():
        tables["chart_data/" + key] = pd.DataFrame(
            deepcopy(figure.layout.meta["chart_data"])
        )
    if (
        driver is not None
        and hierarchical
        and "team_summary" not in result.tables
    ):
        scoped = data.bridge.loc[data.bridge.driver.eq(driver)]
        if scoped.empty:
            raise ValueError("Unknown organizational driver")
        tables["driver_by_node"] = scoped.copy()
    elif driver is not None:
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
        **({"hierarchy": settings["hierarchy"]} if hierarchical else {}),
        "chart": chart, "top_n": top_n,
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
