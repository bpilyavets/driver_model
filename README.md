# Labor-cost analysis — bring your own DataFrame

Open [labor_cost_decomposition.ipynb](labor_cost_decomposition.ipynb), load your
employee-month DataFrame, and edit **Your data and settings**:

```python
input_df = production_df
period_a = "2026-01"
period_b = "2026-02"
source_cost_column = None
```

Then **Run All**. The notebook shows the organizational bridge, team summary,
validation and missing-economics assumptions. Set `selected_team` to a team name
for its local bridge; enable `show_detailed_diagnostics` for salary/fallback audits.
There is no synthetic data generation or CSV loading. Ingestion is yours to supply.
Without configured input, the notebook displays setup guidance rather than running
an example. All implementation functions remain inside collapsible notebook cells.

## Setup

Use Python 3.10 or a compatible environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name labor-cost-poc
```

Select that environment in your Jupyter-compatible editor. On Debian/Ubuntu,
creating a virtual environment requires `python3-venv`. JupyterLab is optional;
install it separately if your editor does not provide a notebook interface.

## Input rules

The complete mapping appears near the top of the notebook and in
[methodology section 21](docs/labor-cost-decomposition.md#21-production-schema-signed-values-and-notebook-workflow).
Keep the production names, including `mdm_employee_rk_hash`,
`lvl7_mapped_management_unit_nm` and `motivation_type` (ТК/ГПД/ПКЦ).
Temporary and outsourced payroll use `gpd_pay` and `pkc_pay`.

- One employee-month row, with one recorded team; month-start periods.
- Payment components and payroll totals may be negative. Applicable values must
  still be numeric, finite and present. Inapplicable values may be null.
- `base_stavka` and `reg_coef` must be positive. `vyrabotka_percent` is nonnegative:
  0.8 means 80%, and values above one are allowed.
- Eligible payment amounts are pre-regional. Salary, sick leave, vacation, holiday,
  night shift, sharing and quarterly pay receive the coefficient once. KPI,
  overtime and other bundled pay are unadjusted.
- Production exposure is one. Do not add another availability multiplier.
  `vyrabotka_percent` is actual/planned hours and multiplies base salary only;
  it is not workforce exposure and does not scale sick leave or other payments.

If you have an independent employee-month total, set `source_cost_column` to its
column name. Otherwise permanent payroll is reconstructed and labeled accordingly;
its reconciliation verifies aggregation, not independent payroll completeness.
Negative independent totals must still be explained by their components.

## Missing contracts and teams

The default `missing_economics = "carry_observed"` does the following:

| Situation | Handling |
|---|---|
| Contract category observed in one month | Use that same team's observed rate in both months. |
| Permanent category observed in one month | Copy its complete permanent state. |
| Whole team appears or disappears | Copy its unit economics, retaining zero workforce share in the absent month. |
| Category inactive in both states | Skip its economics; do not invent a rate or permanent profile. |

Copied economics have zero rate-change impact **by assumption**. The notebook
reports the missing period, donor period and affected economics. Explicit references
can override missing economics; `"require_reference"` restores strict handling of
one-sided absence. Neither policy fills missing pay for existing employees.
Employees permanent in only one month carry their observed salary/time inputs;
this does not create employment in the missing month. Transfers use the same
person's history, not the other team's average. Complete-copy category/team
assumptions still imply zero branch economics change. References must contain
aligned employee vectors and sparse cell profiles; old salary/time-matrix-only
states must be rebuilt. Reference formats are documented in [the methodology](docs/labor-cost-decomposition.md#16-sparse-support-and-automatic-missing-economics-completion).

Sparse cells within an observed permanent population still use same-period,
same-team fallback: cell → grade → region → team permanent population. Coefficient
bands are not identified geography. Regional-policy changes remain outside scope.

## Reading and reusing results

```python
analysis["bridge"]          # Organizational leaf impacts.
analysis["local_bridges"]   # Local bridges indexed by team name.
analysis["team_summary"]    # Team endpoints and the two reporting views.
analysis["assumptions"]     # Inactive, copied or referenced economics.
analysis["validation"]      # Detailed pass/error records.
analysis["diagnostics"]     # Salary/time/membership and matched-ID audits.
```

Organizational and local bridges allocate scale interactions differently: do not
add their rows together. Signed payroll explains the recorded month, including
reversals. Base salary changes and the proportion of planned hours worked are
switched separately using matched employee inputs. A raise cannot itself
create a time impact. A third driver, Within-cell workforce mix — base pay, explains changing
membership; grade and membership effects may offset. Salary changes include
promotions without establishing their cause. Other payments keep their existing
per-cell definitions.

For programmatic use after loading the notebook's implementation cells:

```python
analysis = analyze_labor_cost(
    production_df,
    "2026-01",
    "2026-02",
    source_cost_column=None,
)
report(analysis)
```

Use `help(analyze_labor_cost)` or other function docstrings for contracts and units.

## Choose the permanent-driver explainer

The input cell and API accept:

```python
explainer = "exact"       # Default; or "permutation" for sampling.
n_permutations = 256      # Forward/reverse cycles; permutation mode only.
random_seed = 0           # Nonnegative integer for reproducibility.
```

Pass the same keywords to `analyze_labor_cost(...)` or `decompose(...)`.
Small organizational/team games and propagation multipliers always remain exact.
The selected method applies to permanent games, which can grow with your registry.

Exact mode enumerates 16,384 coalitions for the standard 14 drivers. Its 15-driver
limit is a PoC safeguard, not a SHAP hard limit; above it, choose permutation mode
explicitly. There is no silent method change. Permutation mode supports 20 or more
drivers and uses a budget of `n_permutations * (2*d + 1)` per team. Sampling
settings are validated in both modes but used only in permutation mode.

Results show the requested configuration and the actual method, budget, seed,
evaluation counts and runtime per game. A stable team-derived seed makes runs
reproducible regardless of team iteration order, without changing NumPy's caller
random state. Sampled cell diagnostics use the same allocation as the bridge.
Numerical reconciliation does **not** establish individual-impact accuracy;
sampled extrema describe evaluated states only.

The separate [accuracy assessment](docs/explainer-accuracy.md) contains measured
tradeoffs, assumptions and reproduction instructions. No benchmark or accuracy
estimate runs in the notebook. `cost(state)` stays independent of comparison and
SHAP, with economics required for positive-share categories. No optimizer or
ingestion infrastructure is included.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Tests load the notebook's tagged implementation cells as the single source of
production logic. Small test fixtures, signed/counterfactual cases and a new-driver
extension live in `tests/`; they do not run in the user workflow. The suite checks
PEP-8 style and function/class docstrings as well as analytical reconciliation.

Fresh-kernel tests require local Jupyter socket access. To run only non-kernel
checks in a restricted environment, use `python -m pytest -m "not kernel"`.
Kernel tests execute a temporary copy both unconfigured and with a test DataFrame;
the delivered notebook remains free of stored payroll outputs. Cost tolerances
are `rtol=1e-10`, `atol=1e-6`, without intermediate rounding.
