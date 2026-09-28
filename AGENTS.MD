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

The source data are stored at **individual employee-month level**, but the decomposition model is **not employee-level**. Employee records are inputs used to construct aggregated period-level workforce states such as workforce exposure, contract mix, regional mix, grade composition within region, and employee-aligned salary/time inputs with sparse within-cell membership profiles.

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

* Keep all production calculation functions inside the notebook. Its normal flow
  must be a DataFrame input/settings cell, analysis call, results and optional
  diagnostics. Hide implementation cells where the viewer supports it. Do not
  add a separate production module, synthetic generator, automatic CSV loading
  or demonstration outputs to this user workflow.
* Employee-month records have exactly one team. Sharing bonuses belong to that
  team. Employee IDs align salary/time observations, including transfers, but
  are never Shapley players.
* Preserve the organizational N * sum(team_share * team_unit_cost) hierarchy,
  nested team-unit and permanent games, and separate local team bridges. Never
  combine organizational and local impacts. Switch complete probability vectors
  and conditional-grade matrices atomically.
* Missing contract economics default to the same team's observed opposite-period
  rate; missing permanent economics copy the full permanent state. An absent
  whole team copies observed unit economics, never observed workforce exposure.
  Record every assumption and donor period. Explicit references override missing
  economics only. Retain require_reference as an optional strict policy.
* Categories with zero share in both completed endpoints are inactive. Do not
  demand rates, salary policies or permanent matrices/games for these categories.
  Unknown economics are None, not zero rates. Cost may skip a zero-share unknown;
  a positive-share unknown must fail. Keep completion outside cost(state).
* Sparse cells within an observed permanent population still use only that team
  and period: cell → grade → region → team permanent population. Do not pool other
  teams or borrow another period to fill individual sparse cells automatically.
* Accept signed finite monetary components, reimbursements, references, source
  costs, unit costs and totals. Never clip legitimate negative values. Missing
  applicable amounts remain errors, not an absent-population fallback case.
* Base salary and regional coefficients stay strictly positive. Worked time
  stays nonnegative (values above one allowed); exposure and probability bounds
  remain unchanged. Production exposure is one; no extra availability multiplier
  belongs in its employee-month payroll formula.
* Follow the production mappings and formula in the methodology. Apply regional
  adjustment once to salary, sick/vacation/holiday/night pay and sharing/quarterly
  bonuses. KPI, overtime and other bundled pay are unadjusted. Signed normalization
  precedes aggregation/fallback and preserves actual source_cost.
* Fixed coefficient policy remains an assumption. Reject inconsistent A/B policy
  schedules; coefficient bands alone cannot distinguish geography or policy moves.
* Propagate with auxiliary M, K and L multipliers, preserving analytical checks
  and parent reconciliation. Never divide by parent change. Skip inactive games.
  Keep small upper games and multipliers exact. Permanent games explicitly select
  exact or permutation; exact remains the default with its 15-driver safeguard.
  Never switch silently. Permutation mode must support larger registered sets.
* Use employee-aligned salary and actual/planned-hours vectors. Never reconstruct
  time as a salary-weighted ratio or multiply independent salary/time means.
  Register the complete normalized within-cell membership profiles as one extra
  base-pay driver. Keep sparse integer-index/weight arrays, not a dense tensor.
* Label the drivers Base salary changes, Proportion of planned hours worked and
  Within-cell workforce mix — base pay. Salary changes include promotions without
  identifying their cause. Supplemental payment matrices retain their existing
  interpretation; the new membership player affects base pay only.
* Match each employee's permanent inputs across the organization, including team,
  region and grade transfers. Missing permanent observations carry that person's
  observed salary/time, never workers or exposure. Record matching/carry provenance.
  Same-person transfer history is allowed; pooled cross-team fallback is not.
* Keep standalone costing independent of comparison. Complete-copy rules for an
  absent category/team still override branch economics changes by assumption.
  References need the revised aligned profile/vector schema; reject legacy
  aggregate-only references rather than invent pairing.
* Reuse the shared salary formula for eight cached comparison scenarios and the
  salary/time/membership cell audits. Caches stay local to a synchronous game;
  never retain them in returned states. Verify cached/direct equality.
* Use shap.PermutationExplainer with one zero background and a one-vector input.
  One forward/reverse cycle costs a budget of 2*d+1 in pinned SHAP. Default to
  256 cycles and seed 0; validate settings, derive stable per-team seeds and
  preserve the caller's NumPy RNG state. Reconciliation is not sampling accuracy.
* Approximate permanent games explain cell-cost vectors; summed cell impacts are
  the driver impacts. Approximate cell audits reuse those same samples. Never
  compare sampled impacts with independently exact diagnostics or rescale them.
  Keep exact-mode audits and exact_game compatibility intact.
* Report actual methods, evaluation budgets/counts, seeds and runtime. Validate
  every evaluated counterfactual; require exhaustive coverage only in exact mode.
  Sampled extrema are extrema of evaluated states, not all possible states.
* Keep empirical accuracy benchmarks and results outside the notebook. External
  experiments may exploit validated component additivity for a 20-driver exact
  reference; do not introduce that shortcut as another production mode. Time the
  real validated engine separately from any benchmark-only lookup acceleration.
* Preserve cell-impact/provenance audits, support shares and matched-ID diagnostics
  with salary and worked-fraction changes. Pass comparison periods explicitly;
  never hard-code January/February in functions.
* Label reconstructed references separately from independent source costs. Adding
  a driver must stay localized to source payroll and one registration. Verify
  this in external tests instead of running a demo in the user notebook.

# Readability and verification

* Document every named function/class with a meaningful docstring. Substantial
  public functions need parameters, returned values/units, failure conditions and
  analytical assumptions; concise helpers need clear summaries. Use NumPy-style
  sections where appropriate and follow PEP 257 conventions.
* Comply with PEP 8: descriptive snake_case, four spaces, grouped imports, normal
  whitespace and 79-character code lines. Avoid packed semicolon statements and
  excessively dense expressions. Wrap prose at 72 characters where practical.
* External tests must load tagged notebook implementation cells without copying
  production logic. Keep small test fixtures and extension checks outside the
  user-facing notebook. Run automated style/docstring checks and meaningful
  analytical, signed-payment, missing-economics and counterfactual tests,
  including zero time impact for salary-only changes and zero salary impact
  for time-only changes.
* Test fresh-kernel execution both unconfigured and with a production-schema test
  DataFrame injected into the designated input cell. Keep executed test outputs
  temporary; deliver a notebook without stored payroll data/results.
* Use rtol=1e-10 and atol=1e-6 without intermediate rounding. Validate source/team/
  contract costs, all active games, probabilities, multipliers, both bridges and
  explicit assumptions. Preserve no-input guidance and caller DataFrame integrity.
