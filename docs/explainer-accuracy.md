# Exact versus permutation attribution: measured accuracy

The default remains exact. For larger permanent games, **256 forward/reverse
cycles is a useful starting point, not a precision guarantee**. In the tested
20-driver payroll case, median aggregate allocation error was **1.22%**, with
**1.80% at the 95th percentile** across 20 seeds. At 1,024 cycles these fell to
**0.48% and 0.90%**. Individual driver errors can be much larger: the grade-mix
row's 95th-percentile relative error was **13.0%** at 256 cycles, and the small
net within-cell membership impact was especially unstable.

These are deterministic test-fixture results, **not an estimate validated on
production payroll**, and not a guarantee for future formulas. No experiment,
error estimate or repeated-seed benchmark is embedded in the notebook.

## What the percentages mean

The main allocation-error measure is:

$$
E=\frac{\sum_j|\widehat{\phi}_j-\phi_j|}{\sum_j|\phi_j|}.
$$

Here the reference is exact permanent-driver attribution. The denominator is
**gross absolute attribution**, not the net payroll change. Thus 1.80% is not
“98.20% accurate”, and does not mean each row is within 1.80% of its true impact.

Median and P95 are empirical quantiles across seeds 0–19, not confidence bounds
or a worst-case guarantee. A driver is “material” if its exact absolute impact
is at least 1% of gross absolute attribution. Relative errors are undefined for
near-zero exact impacts; net-change percentages are not used for error scoring.

All sampled games and both propagated bridges reconciled within the notebook's
`rtol=1e-10`, `atol=1e-6` tolerances. Reconciliation does not prove that the
allocation between drivers is exact.

## Payroll results

| Fixture | Cycles | Median allocation error | P95 allocation error |
|---|---:|---:|---:|
| Existing payroll fixture (14) | 64 | 2.85% | 5.40% |
| Existing payroll fixture (14) | 256 | 1.49% | 2.79% |
| Existing payroll fixture (14) | 1,024 | 0.79% | 1.13% |
| Payroll stress fixture (14) | 64 | 4.57% | 8.36% |
| Payroll stress fixture (14) | 256 | 2.35% | 3.55% |
| Payroll stress fixture (14) | 1,024 | 0.96% | 1.85% |
| Extended payroll stress fixture (20) | 64 | 1.78% | 4.55% |
| Extended payroll stress fixture (20) | 256 | 1.22% | 1.80% |
| Extended payroll stress fixture (20) | 1,024 | 0.48% | 0.90% |

The existing fixture has sparse cells and several unchanged payment drivers.
The stress fixture deliberately changes region/grade composition, membership,
salary, worked fractions and all nine payments, with signed corrections. The
20-driver fixture adds six registered payments to the same stress scenario.
Its lower normalized error than the 14-driver stress case is not evidence that
more drivers improve accuracy: the added payments change the denominator and
interaction mix, and the sampled orders also change.

The set of the three largest absolute impacts was unchanged in every payroll
run, with no boundary ties. This does not establish identical ordering within
that set. At 64 cycles, the 14-driver stress case had five sign errors in its
material within-cell membership row across 20 runs. No material-driver sign
errors occurred at 256 or 1,024 cycles in the payroll fixtures.

### Individual impacts in the 20-driver case

The following are **organizational-currency permanent-driver impacts**, not
employee-level effects. Display rounding is applied only in this report.

| Driver | Exact impact | P95 absolute error, 256 cycles | P95 relative error, 256 | P95 absolute error, 1,024 cycles | P95 relative error, 1,024 |
|---|---:|---:|---:|---:|---:|
| Regional mix | 353,194.83 | 4,128.21 | 1.2% | 1,957.74 | 0.6% |
| Grade mix | -95,658.86 | 12,429.79 | 13.0% | 5,030.25 | 5.3% |
| Base salary changes | 53,715.12 | 1,076.28 | 2.0% | 546.95 | 1.0% |
| Worked fraction | 161,804.65 | 9,527.09 | 5.9% | 4,967.27 | 3.1% |
| Within-cell workforce mix | 10,329.15 | 12,185.87 | 118.0% | 4,867.40 | 47.1% |

The within-cell membership effect is below this case's 1% materiality threshold,
but it is shown because it exposes a practical risk: its P95 relative error is
118% at 256 cycles and still 47% at 1,024. A low overall error can coexist with a
poor estimate of a small net impact formed from offsetting effects. Do not use
that row alone for a decision without stronger validation.

All 20 drivers, every budget and both currency views are available in
[driver_errors.csv](../benchmarks/results/driver_errors.csv), including median,
P95 and maximum absolute errors, relative errors and material sign errors.

### Organizational versus local propagation

The 20-driver fixture's modeled organizational endpoints are 676,250.00 and
2,343,964.00. The permanent-unit endpoints are 52,656.25 and 205,196.40. Only
permanent impacts are approximate; the upper games and multipliers remain exact.

For the tested team, the organizational multiplier M*K is 9.0253427128 and the
local multiplier L*K is 9.0606060606. They multiply the same sampled permanent
impacts and their errors. Consequently the normalized errors agree across unit,
organizational and local views, while absolute currency errors differ. These
metrics score the permanent leaves, not the gross attribution of the complete
organization bridge, which also contains exact upper-level contributions.

| Cycles | P95 largest driver error: unit currency | Organization currency | Local currency |
|---|---:|---:|---:|
| 64 | 3,076.17 | 27,763.51 | 27,871.99 |
| 256 | 1,377.21 | 12,429.79 | 12,478.35 |
| 1,024 | 587.16 | 5,299.30 | 5,320.00 |

## Why there is no universal accuracy percentage

SHAP's forward/reverse permutation estimator is exact for effects involving at
most two input switches. Higher-order interactions are sampled. Our salary
term can involve five switches (region, grade, membership, salary and time),
while a payment term can involve three. Adding payment drivers does not by
itself increase that interaction order. See the
[SHAP permutation documentation](https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html).

Controlled examples emphasize the role of interactions. Cubic and quintic cases
use `600 * product(switches)`. The zero-net case uses a cubic interaction minus
`600 * switch_0`, so the endpoint change is exactly zero while attribution is
nonzero. Percent error divided by net change would be undefined.

| Controlled game | P95 allocation error: 64 cycles | 256 cycles | 1,024 cycles |
|---|---:|---:|---:|
| cubic | 10.47% | 6.60% | 2.77% |
| quintic | 17.19% | 10.90% | 4.74% |
| zero net cubic | 7.85% | 4.95% | 2.08% |

Even these small sampled games visited every possible coalition, yet remained
approximate: visiting coalitions is not the same as assigning exact Shapley
weights to their marginals. The quintic game's five exact impacts are tied;
“top three” membership is not a unique reference there.

Monte Carlo uncertainty typically falls approximately with the square root of
the cycle count: four times the sampling budget roughly halves sampling noise,
not necessarily the realized error in a particular run. The controlled and
payroll results demonstrate why a fixed seed ensures reproducibility, not
accuracy. A low-budget 20-driver result can be quite usable for large impacts,
but precision requirements must be checked on the intended payroll and formula.

## Runtime and evaluation counts

Payroll timings below are the median of three warmed runs of the **real notebook
permutation evaluator**, with full counterfactual validation and vector cell
outputs. They exclude the benchmark-only lookup accelerator, ingestion, upper
games, diagnostics and plotting. The normal eight salary-scenario cache remains
part of the real evaluator, as it is in ordinary notebook use.

| Fixture | 64 cycles | 256 cycles | 1,024 cycles | Full exact game |
|---|---:|---:|---:|---:|
| Existing payroll fixture (14) | 0.72 s | 2.88 s | 11.55 s | 6.18 s |
| Payroll stress fixture (14) | 0.48 s | 1.90 s | 7.55 s | 4.07 s |
| Extended payroll stress fixture (20) | 0.73 s | 2.92 s | 11.77 s | Not enumerated |

At 14 drivers, 1,024 cycles were slower than full exact enumeration on these
fixtures; approximation is not automatically a speed improvement. At 20 drivers,
full enumeration would require 1,048,576 coalitions, but it was not timed here.
The 15-driver cap is this PoC's safeguard, not a SHAP mathematical restriction.
See [SHAP's exact-explainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.ExactExplainer.html).

| 20-driver cycles | Sampling evaluation budget | Model rows including endpoint checks | Median distinct coalitions |
|---|---:|---:|---:|
| 64 | 2,624 | 2,626 | 2,135 |
| 256 | 10,496 | 10,498 | 7,790 |
| 1,024 | 41,984 | 41,986 | 27,718 |

The first single-feature permutation call took **3.80 s**,
including first-use SHAP/Numba setup; it is separate from the warmed timings.
The complete benchmark took **231.0 s**. These are local measurements,
not production latency predictions. Environment: Python 3.10.12, SHAP
0.48.0, NumPy 1.26.4, Pandas 2.3.3, Linux/WSL2.

## Exact references, reproducibility and recommendation

For both 14-driver cases, component references were checked against full exact
enumeration. For 20 drivers, the current additive payroll formula permits an
exact reference assembled from the five-player salary game and fifteen
three-player payment games: **32 + 15×8 = 152 component coalitions**. Their
Shapley values add to the full game's exact values by additivity and dummy-player
invariance. All component tables were generated by the notebook's shared cost
engine, then checked against that engine on endpoints and 100 deterministic
mixed states. Every sampled payroll run also passed source and hierarchy
reconciliation checks.

This is an important limitation of the “20 drivers requires approximation” claim:
our particular additive formula permits specialized exact computation. The
20-driver reference/table construction took about
**0.083 s** here, including its checks.
That benchmark-specific shortcut is not a third notebook explainer mode and
would need renewed validation if future components introduced other interactions.

To make the 360 seed/budget experiments inexpensive, accuracy sweeps used those
validated component lookup tables. For each payroll case and budget, the first
three sampled seeds were checked against the real evaluator. Lookup runtimes
are excluded from the production-path timing table above. No parallel payroll
test suite was running during those timing measurements.

Reproduce from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python benchmarks/explainer_accuracy.py
```

This loads the notebook's tagged implementation cells and external test fixtures;
it does not execute or alter its user input workflow. Default outputs are
[summary.csv](../benchmarks/results/summary.csv),
[driver_errors.csv](../benchmarks/results/driver_errors.csv) and
[metadata.json](../benchmarks/results/metadata.json). Use `--output-dir` to save
elsewhere. The default is 20 seeds and budgets 64, 256 and 1,024. Report values
are rounded for display only; CSV results retain full precision.

**Recommendation:** keep exact mode for the current small model when runtime is
acceptable. Start 20-driver exploratory work at 256 cycles, and use 1,024 when
allocation precision matters more than runtime. Neither setting guarantees
accurate small, cancelling effects. Validate decision-critical rows separately
against an exact reference where feasible and against repeated seeds outside
the notebook before attaching a production accuracy target.
