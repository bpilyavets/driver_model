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
with your company's existing environment procedure, then install the
package through the [GitLab notebook](../examples/gitlab_install.ipynb).
That requirements file extends the core deployment requirements; it does
not install a Jupyter server. Restart the kernel after environment changes.

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

The report contains a full organizational waterfall and driver table,
team summary, compact recorded validation/assumption/estimator summaries,
and local waterfalls for the selected teams. Its header identifies the
run, periods, model/dataset versions, exposure assumption and source-cost
basis. Actual estimator methods come from `coalition_audit`, rather than
assuming every game uses the requested estimator. The unabridged audit
table retains effective seeds, counts and timings for inspection.

None or an empty `teams` sequence adds no local waterfalls and leaves
the team summary and optional driver breakdown unrestricted. A nonempty
sequence restricts those team views. The organizational waterfall always
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
    plot_organization_waterfall, plot_team_waterfall, plot_driver_by_team,
)

organization = plot_organization_waterfall(result)
local = plot_team_waterfall(result, "Operations")
comparison = plot_driver_by_team(result, driver_id, teams=["Operations"])
```

All three accept `currency_label`. A report's `figures` dictionary uses
keys `organization`, `local/<team>` and optionally `driver_by_team`.
Its `tables` dictionary contains `driver_bridge`, normalized `team_summary`,
`local/<team>` and optionally `driver_by_team`, plus `validation`,
`assumptions`, `coalition_audit` and their compact `validation_summary`,
`assumption_summary`, `estimator_summary` counterparts. Mutating these
tables or report metadata does not mutate the source result.

Waterfalls use period-A absolute cost, existing relative leaf effects and
period-B absolute cost. No parents, balancing bars or omitted small effects
are introduced. Existing order and active zero effects are retained. Colors
distinguish increases (orange), decreases (blue) and endpoints (slate),
without assigning favorable/unfavorable meanings. Signed negative endpoints
and zero net changes are supported. Figure values retain numerical precision;
display labels show two decimals, hover amounts six. Drivers with duplicate
labels are disambiguated by ID. No net-driver percentages are calculated.

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
