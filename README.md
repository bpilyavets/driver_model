# Labor-cost decomposition PoC

Open [labor_cost_decomposition.ipynb](labor_cost_decomposition.ipynb) for the executed,
standalone demonstration. Analytical requirements are in
[the methodology](docs/labor-cost-decomposition.md); agent instructions are in `AGENTS.MD`.

The notebook builds workforce states from employee-month Pandas DataFrames. It explains
organization-wide cost changes through workforce scale, team mix and nested team economics,
then supplies separate local team bridges. Local bridges reconcile each team's own cost
change; organizational team contributions allocate scale interactions differently.
**Do not combine rows from these two views.**

## Run

Use Python 3.10 or a compatible environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 labor_cost_decomposition.ipynb
```

On Debian/Ubuntu, virtual-environment creation requires `python3-venv`. For interactive use,
select that environment in a Jupyter-compatible editor, or install `jupyterlab` and run
`python -m jupyter lab`. Section 14 prints measured execution time and the final validation
review. Exact permanent games enumerate 8,192 coalitions each; the extension enumerates
16,384. The limit is 14 drivers per game, intended for small team sets (up to about 10).

The main example generates deterministic **SYNTHETIC** production-shaped data for three
teams, with different trends, team transfers, composition changes and sparse cells. It
needs no input file. Source payroll is calculated independently from employee records.
The supplementary `costs_poc.csv` adapter runs only if the fixture exists.

## Production inputs and payroll

After executing the function definitions, use the production DataFrame adapter:

```python
frame = adapt_production(production_df, source_cost_column=None)
policy = make_policy(frame)  # Supply the selected A/B observations.
a, fallback_a, coverage_a = build_state(frame, PERIOD_A, policy)
b, fallback_b, coverage_b = build_state(frame, PERIOD_B, policy)
result = decompose(a, b, dataset="Production — reconstructed permanent reference")
report(frame, a, b, result)
```

The complete production mapping is displayed in notebook section 3 and methodology section
21. `motivation_type` maps ТК/ГПД/ПКЦ; temporary and outsourced payroll use `gpd_pay` and
`pkc_pay`. Each employee-month has one team, and sharing bonuses stay with that recorded
team. Production exposure is one; `vyrabotka_percent=0.8` means 80%, with values above one
allowed. There is no additional availability multiplier in the production employee formula.

Salary, sick leave, vacation, holiday, night shift, sharing and quarterly pay are pre-regional
inputs and receive the coefficient once. KPI, overtime and other bundled pay are unadjusted.
KPI is already realized; quarterly bonuses enter the recorded month. Salary/time aggregation
uses salary-weighted worked-time factors to preserve payroll exactly.

An optional independent `source_cost_column` covers every employee-month. Otherwise permanent
payroll is reconstructed and labeled accordingly; reconciliation then tests aggregation,
not independent payroll completeness. Missing applicable production amounts are errors.

For explicitly identified eligible amounts that already contain regional adjustment, call
`normalize_regional_inputs(frame, already_regionalized=(...))` once before building states.
It preserves actual source costs. Coefficient bands (`region_1.15`, etc.) do not identify
physical geography; fixed regional policy is an assumption, separate from band composition.

## Missing support and interpretation

Sparse fallback stays inside the same team and period: cell → grade → region → team permanent
population. Missing regions use that team's marginal grade distribution. Unknown salaries
are never replaced by zero. Missing teams or contract categories require explicit references
through `references[(period, team)]`; methodology section 16 documents the reference shapes.
Coverage, every fallback value, and unsupported counterfactual exposure are reported.

The **Within-team/region/grade salary level** driver includes changes in cell membership,
including team transfers, not only individual raises. Salary cell audits reconcile to SHAP
and show observed/imputed support in both reporting currencies. Matched-ID diagnostics show
salary changes and distinguish transfers from organization sample entries/exits.

The legacy fixture uses one mock team and its narrower recorded payroll. Its pre-regional
base salary, sick leave and holiday values reconstruct **494,716.700 in January** and
**625,148.125 in February**. Legacy ГПХ maps to temporary only in this adapter. Other components
are explicitly unavailable in the fixture. Five matched permanent IDs have unchanged base
salary; use the mock's salary cell audit to understand composition and fallback effects.

## Extensibility and planning

Section 13 adds a regionally adjusted transport allowance with one source-data/payroll
addition and one registry entry. State builders, costing, exact SHAP, propagation, reporting,
diagnostics and waterfalls discover it automatically. Night-shift pay is now standard payroll.

`permanent_cost(state)`, `team_unit_cost(state)` and `cost(workforce_state)` work without
historical comparison or SHAP. Future planning could vary workforce size, team shares and
within-team contract/region/grade mixes under staffing and service constraints. No optimizer
or production infrastructure is implemented.
