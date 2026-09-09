# Repository purpose

This repository contains analytical tooling for:

1. explaining changes in labor costs through driver decomposition; and
2. eventually supporting workforce planning / optimization.

The current implementation is a proof of concept. Prefer analytical correctness, transparency, and ease of iteration over production-grade abstractions.

For the labor-cost decomposition methodology and business semantics, read:

`docs/labor-cost-decomposition.md`

before modifying decomposition, workforce-state, counterfactual, or cost-model logic.

# Core architecture

Keep these concerns separated:

```text
raw Pandas DataFrame
        ↓
workforce state builder
        ↓
cost engine
        ↓
attribution / Shapley engine
```

The cost engine must not depend on SHAP.

It should be possible to evaluate:

```python
cost(workforce_state)
```

independently of any historical decomposition.

This is important because the same cost engine may later be reused for plan-oriented optimization.

# Analytical principles

* The cost engine is the source of truth.
* Attribution explains changes in the cost engine; attribution logic must not redefine payroll logic.
* Every reported decomposition must reconcile to the corresponding modeled cost difference within numerical tolerance.
* SHAP features represent business drivers / interventions, not arbitrary DataFrame columns.
* Structurally inseparable variables must be switched together.
* Vector-valued or matrix-valued distributions must be switched as complete drivers when their elements are constrained to sum to one.
* Counterfactual states must have coherent business semantics.
* Do not manufacture economically named variables that are not identifiable from the available data.
* Unknown or residual concepts must be labeled conservatively and transparently.
* Never silently fill unsupported counterfactual cells with zero.
* Make modeling assumptions and fallback logic visible.

# Engineering principles

This is currently a PoC.

Prefer:

* clear functions;
* simple dataclasses/configuration where useful;
* analyst-readable Python;
* a well-structured Jupyter notebook if that is the simplest deliverable;
* explicit validation;
* declarative driver configuration.

Avoid unless clearly necessary:

* complex class hierarchies;
* dependency-injection frameworks;
* generic workflow engines;
* APIs/web services;
* persistence abstractions;
* distributed computation;
* elaborate caching;
* premature packaging.

# Driver extensibility

Adding, removing, or renaming a business driver must be low-friction.

Driver definitions should be centralized in one registry/configuration mechanism wherever practical.

Adding a driver should not require unrelated edits across the cost engine, SHAP wrapper, reporting, validation, plotting, and labels.

Whenever changing driver architecture, verify this property explicitly.

# Data assumptions

Input data may be assumed to be available as Pandas DataFrames.

The source data are stored at **individual employee-month level**, but the decomposition model is **not employee-level**. Employee records are inputs used to construct aggregated period-level workforce states such as workforce exposure, contract mix, regional mix, grade composition within region, and region × grade salary levels.

Do not run SHAP directly over employees or interpret the sum of employee-level attributions as workforce-composition effects unless explicitly requested.

Data acquisition from source systems is outside the scope of this repository unless explicitly requested.

# Validation requirements

At minimum validate:

1. modeled period-A cost against the corresponding source cost;
2. modeled period-B cost against the corresponding source cost;
3. top-level attribution reconciliation;
4. permanent-cost attribution reconciliation;
5. flattened hierarchical reconciliation;
6. validity of probability distributions;
7. absence of unexpected NaNs / unsupported states;
8. explicit reporting of fallback usage.

Invalid model states should fail loudly rather than silently producing plausible-looking attribution.

# Future planning support

When reasonable without complicating the PoC, design workforce states and cost functions so that future code can use:

```text
decision variables
        ↓
workforce state
        ↓
cost(state)
```

for constrained optimization.

Do not implement an optimizer unless explicitly requested.

# Team-aware payroll and attribution requirements

* Source grain remains one employee-month with exactly one recorded team. Sharing bonuses
  belong to that team. Employee IDs and team transfers are diagnostics, never SHAP players.
* Use the organizational hierarchy `N * sum(team_share * team_unit_cost)`, four-player
  nested team-unit games, and separate permanent games. Local team bridges first split
  local scale and unit economics; never combine their rows with organizational impacts.
* Switch complete team, contract and region vectors and complete conditional-grade
  matrices atomically. Align region/grade axes within each team across A/B.
* Keep sparse fallback within the same team and period: cell → grade → region → team
  permanent population. Missing teams/categories require explicit references; never
  silently pool other teams, carry other-period economics or fill unknown salary with zero.
* Propagate with auxiliary 0→1 multipliers M (organization/team), K (team/permanent), and
  L (local/team). Assert their analytical values and parent replacement at every level.
  Never divide by a parent's observed change. Preserve zero-change offsetting-child tests.
* The reduced three-player organizational term games are exact by Shapley additivity and
  dummy invariance. Retain a full-outer-enumeration equivalence test. Exact permanent games
  currently have 13 drivers; the supported PoC maximum is 14, including the extension.
* Production mappings and the full employee payroll formula are requirements in section 21
  and section 6 of the methodology. Production contracts are ТК/ГПД/ПКЦ; only the legacy
  adapter accepts ГПХ. Use gpd_pay/pkc_pay for the corresponding contract payroll.
* Production exposure is one. vyrabotka_percent is a nonnegative fraction, possibly above
  one. Do not add a second availability multiplier to employee-month payments. Canonical
  non-unit exposure requires unit-normalized monetary inputs.
* Regional coefficients apply once to salary, sick leave, vacation, holiday, night shift,
  sharing and quarterly pay. KPI, overtime and other bundled pay are unadjusted. Declare
  eligibility and payroll contributions in the registry. Normalize explicitly identified
  already-regionalized values before aggregation/fallback and preserve actual source_cost.
* Coefficient bands are not identified geography. Fixed policy is an explicit assumption;
  reject inconsistent supplied A/B schedules. The source coefficient alone cannot reveal
  whether a change was policy or a move between bands.
* Keep the salary label Within-team/region/grade salary level and salary-weighted hours.
  Preserve salary cell-impact/provenance audits, net/absolute support summaries and matched
  ID diagnostics that distinguish team transfers from organization sample entries/exits.
* Night shift is now standard. Demonstrate a new transport allowance through one source
  payroll addition and one registration, with no special cases in unrelated pipeline code.
* Label reconstructed permanent references separately from independent actual/generated
  payroll. Retain the legacy mock totals 494,716.700 and 625,148.125. Production inputs are
  DataFrames; CSV parsing is optional and confined to the legacy adapter.
* Execute the notebook from fresh kernels with and without the CSV. Save outputs and show
  reconciliation, invalid-input/counterfactual checks and runtime. Use rtol=1e-10 and
  atol=1e-6 for costs without rounding intermediate calculations.
