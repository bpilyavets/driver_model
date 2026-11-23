# Plotly result reporting

For a runnable example, open the
[reporting showcase notebook](../examples/reporting_showcase.ipynb).
It uses the model and nine mock rows from `poc_extented_model_test.py`,
then demonstrates reports, individual figures, styling and saved runs.

`report` consumes an existing `AnalysisResult` from `analyze` or `load_run`.
It reads portable tables and metadata only, including runs whose `states`
are `None`. It never compiles a model, constructs states, evaluates costs
or reruns attribution. Formula previews remain a separate authoring feature.

## Install

From `production/`, install into your selected Python environment:

```bash
python -m pip install -e '.[reporting]'
```

The `reporting` extra contains Plotly 6.x, IPython and nbformat. The
`notebook` extra includes those dependencies plus JupyterLab. Ordinary
core imports and calculations require none of these presentation packages.
Plotly is imported on first figure creation; IPython only on display.

For hosted GitLab deployment use
[`runtime/requirements-reporting.txt`](../runtime/requirements-reporting.txt)
with your company's existing environment procedure, then load the
package through the [GitLab notebook](../examples/gitlab_load.ipynb).
Use `lc.report(...)` through its `lc = CODE.package` handle.
That requirements file extends the core deployment requirements; it does
not install a Jupyter server. Restart the kernel after dependency changes;
loading the package source itself needs no restart.

## Build and display a report

Given an existing result with a team named Operations:

```python
from labor_cost import report

charts = report(result, teams=["Operations"], currency_label="RUB")
```

`currency_label` is display text only, defaulting to `Currency`. It does
not infer or convert currency. `report` displays by default and returns
a `NotebookReport`; `show=False` builds it without display. Reuse it with
`charts.show(renderer=...)`, edit its figures, or inspect copied tables.
The object has a compact text representation, so returning it as the last
cell expression does not redisplay all its charts.

The report contains organizational totals above ranked driver bars and a
complete driver table,
team summary, compact recorded validation/assumption/estimator summaries,
and local impact charts for the selected teams. Its header identifies the
run, periods, model/dataset versions, exposure assumption and source-cost
basis. Actual estimator methods come from `coalition_audit`, rather than
assuming every game uses the requested estimator. The unabridged audit
table retains effective seeds, counts and timings for inspection.

None or an empty `teams` sequence adds no local charts and leaves
the team summary and optional driver breakdown unrestricted. A nonempty
sequence restricts those team views. The organizational chart always
explains the full organization; a selected subtotal is never presented as
a newly calculated selected-population bridge.

To add a horizontal team comparison, pass a **driver ID** from the stored
organizational leaves that have a team allocation:

```python
available = result.table("bridge").loc[
    lambda rows: rows.team.notna(), ["driver", "label"]
].drop_duplicates()
driver_id = available.iloc[0]["driver"]  # Choose from this result.
charts = report(result, driver=driver_id, show=False)
```

Driver bars are organizational contributions. Local waterfalls explain
each team's own cost change through a different interaction allocation.
The driver chart displays the full organizational driver total and, if
teams were selected, their subtotal. Unknown/inactive IDs, parent nodes,
labels supplied instead of IDs, and drivers without a complete team
allocation produce an actionable error. A known team with no recorded
contribution to this otherwise present driver receives a zero bar.

## Figures and tables

Standalone functions return Plotly `Figure` objects without displaying:

```python
from labor_cost import (
    plot_organization_impacts, plot_team_impacts,
    plot_organization_waterfall, plot_team_waterfall, plot_driver_by_team,
)

organization = plot_organization_impacts(result)
compact = plot_organization_waterfall(result, top_n=5)
local = plot_team_impacts(result, "Operations")
comparison = plot_driver_by_team(result, driver_id, teams=["Operations"])
```

All figure functions accept `currency_label`. A report's `figures` dictionary uses
keys `organization`, `local/<team>` and optionally `driver_by_team`.
Its `tables` dictionary contains `driver_bridge`, normalized `team_summary`,
`local/<team>` and optionally `driver_by_team`, plus `validation`,
`assumptions`, `coalition_audit` and their compact `validation_summary`,
`assumption_summary`, `estimator_summary` counterparts. Mutating these
tables or report metadata does not mutate the source result.

The default `chart="impacts"` places Period A, Period B and net change above
horizontal driver bars. Bars are ranked by absolute impact, with original
order breaking ties. Their linear axis is centered on zero and independent
of the period totals, so small drivers remain comparable even when payroll
is much larger. The indicators and bars form one exportable Plotly figure.

```python
charts = report(result, chart="impacts", show=False)  # Default; all drivers.
full = report(result, chart="waterfall", show=False)
compact = report(result, chart="waterfall", top_n=5, show=False)
```

`top_n` also works with impact charts and every standalone organization/local
chart function. It must be a positive integer or None; booleans are rejected.
With a limit, the largest absolute driver impacts appear in descending order,
followed by **Other**. Other is the signed sum of the remaining drivers. It
is retained when those effects cancel to zero and omitted when no drivers
remain. Its tooltip shows the grouped count, net effect, positive total and
negative total. Real drivers named Other retain their IDs in their labels.
Each local chart selects its own largest local impacts; organization charts
use organizational driver totals. Organization and local allocations remain
separate; these are presentation groupings, not newly estimated drivers.

Inspect the exact plotted amounts and Other membership using
`charts.tables["chart_data/organization"]` or
`charts.tables["chart_data/local/Operations"]`. Their columns include
`driver`, `label`, `impact`, `kind`, `member_count`, `positive_impact`,
`negative_impact` and `members` (original driver IDs, labels and amounts).
Standalone figures retain the same evidence in `figure.layout.meta` under
`chart_data`. The existing `driver_bridge` and local report tables remain
complete, uncondensed copies. Editing any of these copies does not change
the analysis result. View settings and grouping evidence belong to the
report/figure; analytical run bundles are unchanged.

Full waterfalls keep period-A absolute cost, relative driver effects and
period-B absolute cost. With `top_n=None`, they preserve declaration order.
No balancing bars or residual corrections are introduced. Negative endpoints,
zero impacts and zero net changes are supported in both layouts.

Across all charts, **red means a cost increase**, **green means a cost
decrease**, and slate denotes period totals or zero effects. Signed numeric
labels accompany the colors. Waterfall tooltips show the selected driver's
stored impact explicitly, not the running total; endpoint tooltips show
period totals. Figure values retain numerical precision; labels show two
decimals and hover amounts six. No net-driver percentages are calculated.

## Renderers and HTML export

For JupyterLab or VS Code:

```python
charts.show(renderer="plotly_mimetype")
```

For classic Jupyter Notebook, use `renderer="notebook"`, which embeds
Plotly.js. A `renderer` argument applies only to that call; the library
does not change global Plotly settings. Omit it to use your configured
renderer. `notebook_connected` requires CDN access, whereas the classic
`notebook` renderer can run offline. The `browser` renderer is intended
for a locally running kernel, so it is unsuitable for your hosted setup.
See [Plotly's renderer documentation](https://plotly.com/python/renderers/).

If the hosted frontend cannot render interactive outputs, save a
self-contained figure and open it in a browser on your computer:

```python
charts.figures["organization"].write_html(
    "organization.html", include_plotlyjs=True,
)
```

This explicitly creates an HTML file with its chart data and bundled
JavaScript; no hosting service is required. It does not export raw payroll
tables. Self-contained export is documented by
[Plotly](https://plotly.com/python/interactive-html-export/). PNG/PDF export
and widgets are outside this implementation.

## Contract and validation boundaries

Both v1/v2 result renderings and schema-one/two run bundles are supported.
The adapter normalizes `period_a_cost`/`period_b_cost`/`local_change` to
`period_A_cost`/`period_B_cost`/`cost_change` in the report's copied team
summary. Source result schemas and exported bundle formats do not change.

Before plotting, the adapter checks required columns, finite amounts,
currency units, leaf scope, consistent endpoints, driver totals against
organizational leaves, and team endpoints against organization totals.
Requested local bridges also reconcile to their team endpoints. Numerical
tolerances remain `rtol=1e-10`, `atol=1e-6`; there is no intermediate rounding,
rescaling or residual correction. Malformed/inconsistent bridges fail clearly.

Recorded failed validations remain visible in the report even if the
chart's own arithmetic reconciles. Missing audit/status data is marked
`Unavailable`, never inferred as passed or exact. `None recorded` for an
existing empty assumptions table differs from an absent table. The report
does not claim complete validation or restore missing coverage/support
diagnostics; all [architecture gaps](architecture-gaps.md) remain open.
Generic inner-game cell impacts have different units and are not charted
as organizational or local currency contributions.


## Team and line nodes

For a hierarchy-enabled run, pass
`report(result, nodes=[(("team", "Alpha"),)], show=False)` to build the
organizational report plus the selected team's local payroll chart and
organizational contribution table. A team–line pair takes a two-element
path in the run's order. Use `result.table("node_summary")` to see all
paths and observed endpoints. `result.select_node(path)` also exposes
the values directly. Team-only reports retain their team selectors.
