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
