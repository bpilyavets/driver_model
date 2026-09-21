# Labor Cost Decomposition Methodology

## 1. Objective

The tool explains the change in total labor cost between two observed periods:

$$
\Delta C=C_B-C_A
$$

by attributing the change to economically interpretable business drivers.

The primary output is a monetary bridge:

$$
\Delta C
=
\sum_k \phi_k
$$

where \(\phi_k\) is the Shapley contribution of driver \(k\).

The decomposition must reconcile to the modeled cost change within numerical tolerance.

The tool is initially **fact-oriented**:

> Which factors influenced labor cost between periods A and B, and by how much?

The architecture should leave open a future **plan-oriented** use case:

> What workforce composition should we target under operational constraints?

---

# 2. Why Shapley decomposition

Labor-cost drivers interact multiplicatively and through workforce composition.

Sequential substitution is therefore order-dependent.

Shapley attribution solves this by defining each driver's contribution as its average marginal effect across possible orders in which period-A drivers can be replaced by period-B drivers.

For a set of drivers \(F\), the contribution of driver \(i\) is conceptually:

$$
\phi_i
=
\sum_{S\subseteq F\setminus\{i\}}
w(S)
\left[
v(S\cup\{i\})-v(S)
\right].
$$

Here \(v(S)\) is the modeled cost of a counterfactual workforce where drivers in \(S\) use period-B state and the remaining drivers use period-A state.

The difficult part is therefore not the Shapley formula itself but defining economically coherent \(v(S)\).

---

# 3. Counterfactual semantics

A SHAP feature should represent a **business driver**, not necessarily a single raw variable.

For example:

0 = use period-A regional composition
1 = use period-B regional composition

If regional composition is a vector, the entire vector switches together.
Similarly, a conditional grade-distribution matrix switches as a single driver.
Counterfactual states do not need to have occurred historically.

They do need to be:

* mathematically valid;
* compatible with payroll rules;
* interpretable as business counterfactuals.

For example:

> February regional footprint under January grade composition within each region

is a valid counterfactual even if that exact workforce never existed.
By contrast, switching a grade while retaining a grade coefficient deterministically belonging to another grade is structurally invalid.
Such variables must be grouped into one driver or derived inside the cost engine.
---

# 4. Level of source data vs level of decomposition

The raw analytical dataset is stored at the **individual employee-month level**.
This granularity is necessary because employee-level records contain the information required to reconstruct:

* employment type;
* workforce exposure;
* region;
* grade;
* base salary;
* regional salary coefficient;
* actual/planned hours fraction;
* bonus/KPI variables;
* overtime;
* holiday pay;
* and other payroll components.

However, the Shapley decomposition itself is **not performed at employee level**.

Employee-level observations are first aggregated into a **period-level workforce state** containing economically meaningful business objects such as:

$$
N,\quad
P(ContractType),\quad
P(R),\quad
P(G\mid R),\quad
w_{i\mid t,r,g},\quad B_i,\quad h_i,\quad
\ldots
$$

The attribution game switches complete workforce-state business objects between
periods A and B. Salary and time inputs retain employee pairing; employees are
not individual players. Exposure-weighted membership profiles determine their
contribution within cells.

The intended pipeline is therefore:

```text
employee-month records
        ↓
aggregate / derive workforce state
        ↓
period-A state vs period-B state
        ↓
counterfactual cost engine
        ↓
Shapley decomposition
```

This distinction is methodologically important.

For example, the business concept:

> grade composition effect

is **not** equal to the sum of grade changes made by individual employees.

Grade composition may change because:

* existing employees are promoted or demoted;
* higher-grade employees leave;
* lower-grade employees are hired;
* different grades enter/leave at different rates.

Therefore composition effects must be defined through changes in the aggregate workforce distribution rather than through employee-level attribution.

Employee identity may still be used for:

* constructing period-level states;
* reconciliation;
* diagnostics;
* identifying joiners/leavers;
* validating payroll logic.

But employee identity is not itself a SHAP player in the primary workforce-cost decomposition.

---

# 5. Workforce hierarchy and the two reporting views

Teams are operationally distinct. Each employee-month has exactly one recorded team;
sharing bonuses are charged to that team. Team names identify branches exactly: upstream
renames/reorganizations need an explicit crosswalk if they should represent the same team.

```text
Total labor cost
├── Workforce scale
├── Team mix
└── Team economics (one branch for each team)
    ├── Contract type mix
    ├── Temp reimbursement
    ├── Outsource reimbursement
    └── Permanent unit cost
        ├── Regional coefficient-band mix
        ├── Grade composition within region
        ├── Within-cell workforce mix — base pay
        ├── Base salary changes
        ├── Proportion of planned hours worked
        └── Nine registered payment drivers
```

For team t, define blended cost per exposure:

$$
u_t=p_{P,t}c_{P,t}+p_{T,t}r_{T,t}+p_{O,t}r_{O,t}.
$$

The organization costs:

$$
C=N\sum_tq_tu_t,\qquad q_t=N_t/N.
$$

N is organizational workforce exposure; q is a normalized team-share vector. Contract
shares are conditional on team and sum to one. Neither vector's elements are separate players.

The **organizational bridge** has scale, team mix, and one economics player per team.
Its propagated team leaves reconcile to that team's organizational economics parent,
which need not equal that team's own observed cost change.

The **local team bridge** first explains C_t=N_t u_t using two players, local scale and
unit economics, then reuses the same nested team/permanent games. It reconciles to that
team's cost change. Sum local team endpoint changes to recover the organization change,
but never combine organizational and local bridge rows: they allocate scale interactions
differently. A one-team organization's bridge equals its local bridge, with zero team-mix impact.

---

# 6. Permanent workforce model

The current employee-month payroll formula is:

```python
cost = reg_coef * (
    base_stavka * vyrabotka_percent
    + sum_pay_bl_reg_base
    + sum_pay_otpusk_reg_base
    + sum_pay_prazdnik_reg_base
    + sum_pay_oklad_night_reg_base
) + bonus_kpi_comp + bonus_overtime_comp + reg_coef * (
    bonus_sharing_comp_reg_base + bonus_quarterly_comp_reg_base
) + sum_pay_compacted
```

Vacation and holiday pay are distinct. KPI is a realized payout, not a bonus base to be
multiplied by another achievement factor. Quarterly bonuses are recognized in the recorded
month, without amortization. Sharing is a distinct payment to the employee's recorded team,
not a transfer to the team receiving help. Other bundled pay is an explicit source component,
not a residual inserted to force reconciliation.

The permanent cost engine must stay extensible: payroll contributions and their regional
eligibility belong in the driver registry, not in SHAP or reporting code. There are 14 current
permanent drivers: five composition/salary/time drivers and nine separate payment drivers.

---

# 7. Employee-aligned base salary and worked fraction

`vyrabotka_percent` is actual hours divided by planned hours. For example,
100/160 is 0.625. It is not workforce exposure: a person can work zero hours
and still be employed and receive sick-leave or vacation payments. Production
exposure remains one per employee-month. The fraction multiplies base salary
only; do not multiply other payments by it. Values above one are permitted.

For each team-period, keep an aligned employee axis, pre-regional salary vector
B, worked-fraction vector h, and sparse cell membership profiles w. Each profile
contains employee indices and normalized exposure weights. At observed cells:

$$
w_{i\mid t,r,g}=\frac{e_i}{\sum_{j\in(t,r,g)}e_j},\qquad
S_{t,r,g}=\sum_i w_{i\mid t,r,g}B_i h_i.
$$

The permanent formula is:

$$
c_{P,t}=\sum_{r,g}p_{r|t}p_{g|r,t}
\left[R_r(S_{t,r,g}+Sick+Vacation+Holiday+Night+Sharing+Quarterly)
+KPI+Overtime+Other\right].
$$

This reproduces the employee formula exactly, with salary and time paired by
identity. Do not replace it with a product of independent averages or with a
salary-weighted time factor. The former loses association; the latter can create
a time impact from a salary increase even when nobody's worked fraction changed.

The three separate complete business switches are:

- `underlying_salary_level`: **Base salary changes**, now an employee vector.
- `worked_hours`: **Proportion of planned hours worked**, now an employee vector.
- `within_cell_workforce_mix`: **Within-cell workforce mix — base pay**, a complete
  collection of normalized sparse profiles.

The membership driver affects base pay only. Supplemental payments retain their
per-cell average definitions, which can still include changing membership.

For one fixed employee and regional coefficient one, salary 100,000→110,000 and
worked fraction 0.625→0.75 give a 20,000 increase. The isolated two-driver Shapley
split is 6,875 salary and 13,125 time. If the fraction stays 0.625, salary gets
6,250 and time gets zero. The full hierarchy allocates additional composition
interactions through its specified games, not a separate employee bridge.

Match permanent employees by ID across the organization before constructing
team comparisons. Transfers use that person's actual other-period salary/time,
not another team's average. If a person is permanent in only one period,
including a contract transition, carry their observed salary/time values to
the missing endpoint. This assumption never adds employees or exposure.
Both employee values are positive/nonnegative as before; membership weights
are normalized and nonnegative. Joined/leaving profile IDs cannot create an
inferred raise or time change under this default.

The ratio does not distinguish changes in actual hours from changes in planned
hours. Base-salary changes include promotion-related increases, decreases and
other observed changes; the data do not identify their cause.

Canonical salary and eligible payments are pre-regional currency per exposure.
Signed payments stay signed. Apply the destination coefficient once to salary,
sick/vacation/holiday/night pay and sharing/quarterly bonuses; leave KPI, overtime
and other bundled pay unadjusted. Eligibility remains registry metadata.

`normalize_regional_inputs(frame, already_regionalized=(...))` divides identified
eligible inputs by their source coefficient before building vectors/profiles or
fallbacks. Apply it once, preserving an already-correct source payroll total.

Production coefficient bands are not identified geography. Fixed coefficient
policy remains an assumption; differing A/B coefficient schedules fail explicitly.

---

# 8. Grade and salary identification

Grades determine permitted salary ranges, but employees can occupy different positions within those ranges.

The available dataset contains:

* grade;
* pre-regional base salary (or regionalized source salary normalized as above);
* region;
* regional coefficient.

It does not currently contain:

* official grade-range median;
* CR / compa-ratio;
* a unique grade coefficient.

Therefore it is not possible to identify a pure:

```text
grade coefficient
```

or:

```text
CR effect
```

from the available data.

Do not infer one using arbitrary salary normalization.

Instead use observed base-salary changes and explicit workforce-composition
components. A promotion does not establish what portion of a raise it caused.

---

# 9. Team-conditional, region-first composition

For permanent workers in team t, deliberately parameterize:

$$
P(R,G\mid T=t,Permanent)=p_{r|t}p_{g|r,t}.
$$

The regional share vector is one driver. The complete conditional-grade matrix is another;
each region row sums to one, including fallback rows for absent regions. Team membership
is handled by the outer team-share vector and nested team economics, not by splitting
regional/grade probabilities into independently switchable scalar shares.

Within each team/region/grade, a further membership profile specifies who
occupies the cell. It is separate from the cell's share of the workforce.
The entire collection switches atomically. This keeps salary/time changes
separate from composition even when employees enter, leave or move cells.

Grade mix and within-cell membership can offset. A person moving grades without
a salary change must not produce a base-salary-change impact. These are
hierarchical accounting counterfactuals, not causal estimates of promotion pay.
Keep provenance and matched-employee diagnostics beside the bridge.

---

# 10. Salary composition identity

Within a team:

$$
\bar S_t=\sum_{r,g}p_{r|t}p_{g|r,t}R_r
\sum_i w_{i\mid t,r,g}B_i h_i.
$$

Composition weights count exposure. The payroll contribution is the profile's
weighted mean of paired salary × worked fraction. This shared formula must be
used by standalone costing, exact attribution, cached salary scenarios and
cell-impact audits. Attribution must not redefine payroll.

---

# 11. Workforce states and the DataFrame entry point

The notebook contains the complete implementation and one user input/settings cell.
It neither generates demonstration data nor loads a CSV. The usual entry point is:

```python
analysis = analyze_labor_cost(
    input_df,
    period_a,
    period_b,
    source_cost_column=None,
    references=None,
    missing_economics="carry_observed",
)
```

It selects the requested months, copies/adapts the production DataFrame, builds
comparison states, performs exact attribution, reconciles results and gathers
diagnostics. All dates are explicit; no function assumes January/February.
The caller's frame is unchanged. Without an independent source-cost column,
permanent reference payroll is reconstructed and labeled accordingly.

The result contains `bridge`, `local_bridges`, `hierarchy`, `permanent`, `upper`,
`coalition_audit`, `states`, `periods`, `frame`, `team_summary`, `validation`,
`assumptions`, `diagnostics` and `elapsed_seconds`.

States retain the original structure: organizational `workforce_scale`, complete
`team_mix`, `_teams` axis, and a dictionary of nested team economics. Each team
holds complete `contract_type_mix`, reimbursement rates and its permanent state.
Permanent states hold fixed region/grade axes and coefficients, `_employees`,
registered composition/payment matrices, employee salary/time vectors and
`within_cell_workforce_mix`. Its keys are `(region_index, grade_index)`; values
are `{"indices": integer_array, "weights": probability_array}`. Every aligned cell
has a supported profile, even when its observed exposure is zero.

Unavailable economics are `None`, not zero unit rates. An observed absent whole
team may have a `None` team state. Zero-share unknowns contribute zero cost;
positive-share unknowns fail. The comparison builder completes the economics
needed by A/B counterfactuals before running any games.

Interfaces remain separate:

- `make_policy(frame)` aligns axes; teams with no permanent observations need no
  permanent policy unless an explicit reference supplies one.
- `build_state(frame, period, policy, ...)` returns an observed state, fallbacks
  and coverage. Missing economics remain explicit and observed costs are usable.
- `build_comparison_states(frame, period_a, period_b, ...)` completes the pair
  according to section 16 and returns states, periods, fallbacks, coverage and
  an assumption table. This comparison-aware step owns automatic copying.
- `permanent_cost(state)`, `team_unit_cost(state)` and `cost(state)` evaluate signed
  costs without historical comparison, automatic completion or SHAP.
- `decompose(a, b, ...)` explains completed states; `reconcile(...)` and
  `diagnostics(...)` use the actual state periods, not notebook-wide defaults.
- `report(analysis, team=None)` shows organizational results or one local team.
  `show_diagnostics(analysis, details=False, team=None)` controls diagnostic detail.

Employees are aligned inside state construction; no employee-level SHAP players
or separate employee attribution bridge are introduced.

---

# 12. Driver registry

Business-driver definitions should be centralized.

The exact implementation is flexible, but adding a driver should ideally require declaring:

* name;
* label;
* level / parent;
* state value builder or state key;
* payroll cost contribution, where applicable;
* whether that contribution receives the regional coefficient;
* optionally validation/fallback behavior.

The generic SHAP engine should discover the applicable drivers from this registry rather than duplicating hard-coded driver lists throughout the codebase.

---

# 13. SHAP switch game

For a decomposition between states \(A\) and \(B\), expose one binary switch feature per business driver:

```text
0 -> use state A for this driver
1 -> use state B for this driver
```

Example:

```text
[0, 1, 0, 1]
```

means some drivers use period B while others remain at period A.

A generic wrapper should construct the corresponding hybrid state and run it through the cost engine.

For small driver sets, exact Shapley enumeration is preferred.

---

# 14. Exact hierarchical games

Run a 14-player permanent-unit game for each team with active permanent economics,
then a team-unit game with up to four players (contract mix, temporary rate,
outsourcing rate, permanent unit). Skip inactive category economics.
The organization has scale, whole team mix and one economics scalar per team.
A local team has a two-player outer game (local scale, unit economics).

Use `shap.ExactExplainer` with a single zero background and explain the one vector,
allowing 2**d evaluations. Validate all binary coalitions, including probability objects
and finite costs. The PoC has an explicit 15-driver limit per exact game; no silent
switch to approximate attribution. Standard games have 16,384 coalitions; an
added 15th driver has 32,768. External regression tests verify this without
adding demonstrations to the user notebook.
Precompute the eight membership/salary/time matrices with the shared cost helper.
Reuse them only inside the current game/audit; returned states carry no cache.
Validate cached versus direct costing and every constructed coalition.

For organizational cost, exploit exact Shapley additivity:

$$
C=\sum_tNq_tu_t.
$$

Compute one three-player game per term, with scale, **whole** team-share vector and
that team's economics. Other teams' economics are dummy players for the term. Sum scale
and mix impacts across terms and retain team economics impacts. This equals the full
outer game, verified by full enumeration on the three-team fixture. Term evaluation
must use the same cost-engine contributions, not redefine costing inside SHAP.

This preserves the declared hierarchy's allocation of interactions. It is not equivalent
to an unrestricted flat-player Shapley game, nor to placing scale beside the four team
unit drivers in an alternative five-player local game.

---

# 15. Interaction-consistent propagation

For each team, obtain M by running its organizational term game with unit economics
endpoints replaced by 0 and 1. Obtain K from its team-unit game with permanent cost 0→1.
Obtain L from its local two-player game with unit economics 0→1. Other drivers keep
normal A/B endpoints. Check independently:

$$
M_t=\frac{N_Aq_{t,A}}3+\frac{N_Aq_{t,B}+N_Bq_{t,A}}6+\frac{N_Bq_{t,B}}3,
\quad K_t=\frac{p_{P,t,A}+p_{P,t,B}}2,
\quad L_t=\frac{N_{t,A}+N_{t,B}}2.
$$

For team-unit impacts phi and permanent-unit impacts psi:

- Organizational team-unit children: M_t phi; permanent leaves: M_t K_t psi.
- Local team-unit children: L_t phi; permanent leaves: L_t K_t psi.

Replace each parent by its children, with reconciliation assertions at each boundary.
Never divide by a parent's observed change. Offsetting children are meaningful even when
permanent-unit or team-unit change is exactly zero. Auxiliary zero/one scalar endpoints
are used to measure sensitivity, not to impute unknown employee salary cells.

---

# 16. Sparse support and automatic missing-economics completion

Distinguish empty populations from missing payment values for actual employees.
Presence is determined from employee records/exposure, never from payment sign or
whether a rate equals zero. Missing applicable values remain input errors.

## Sparse cells in an observed permanent population

Fallbacks remain within the same period and team:

1. observed region × grade population;
2. that team's corresponding grade population;
3. that team's corresponding region population;
4. that team's permanent population.

Payments recompute their exposure-weighted means. Salary fallback selects the
complete normalized employee profile from the chosen population. An absent
region uses that team-period's marginal grade distribution. Never fill unknown
salary/payment cells with zero or automatically borrow another team's pay.
Disabling needed fallback must fail explicitly.

Payment fallbacks are currency per exposure, pre-regional when eligible.
Salary profiles contain dimensionless probabilities; employee salary inputs are
pre-regional currency per exposure and worked fractions are dimensionless.
Signed payments remain signed through normalization, pooling and destination
adjustment. Counts of zero indicate absence, not zero unit economics.

Salary-profile fallback is distinct from identity matching: populations stay
within team and period, but a matched employee's own salary/time history may come
from their other team. Missing individual values are carried only when there is
no permanent observation in that period; missing applicable payroll is an error.

## Missing categories and teams

The default `missing_economics="carry_observed"` policy is:

| Situation | Completion |
|---|---|
| Category observed in both periods | Preserve both observed economics. |
| Temporary/outsourced category observed in one period | Copy that same team's observed-period reimbursement into the missing period. |
| Permanent category observed in one period | Copy its complete permanent state, including profiles, employee vectors, distributions and payment matrices. |
| Category share zero in both completed endpoint states | Inactive; no rate or reference is required and no economics player is run. |
| Whole team absent in one period | Copy the observed team's complete unit economics, retaining its observed zero team exposure/share. |

Copying means zero economics change **by assumption**, not an observed unchanged
rate. Whole-team appearances/disappearances are allocated through organization
scale/team mix and local team scale when economics are copied. The hierarchy's
interaction allocation remains unchanged. Complete-copy assumptions still give
zero branch economics change when some employees are observed elsewhere; this
qualifies the interpretation of individual-input drivers for absent branches.

Explicit business references take precedence over copying missing economics;
they never replace an actually observed category. Reference keys are `(period, team)`:

```python
references = {
    (period, team): {
        "temporary": ...,   # signed all-in rate if missing
        "outsourced": ...,  # signed all-in rate if missing
        "permanent": ...,   # complete aligned permanent state if missing
        "team_state": ...,  # complete unit economics for an absent team
    }
}
```

Supply only applicable entries. Permanent references must use the revised state
schema: `_employees`, paired positive/nonnegative salary/time vectors, normalized
membership profiles and all registered matrices. The employee axis must equal
the sorted union of IDs in the team's comparison/profile states. Use revised
state builders to create reference states, then supply aligned business values.
Legacy salary/time-matrix-only references fail with a rebuild instruction; no
employee pairing can be recovered from two averages. The optional `"require_reference"` policy rejects
one-sided missing economics without explicit references, but still skips inactive
categories. Whole-team references include their counterfactual contract mix;
category references do not overwrite observed shares. If such a reference gives a
category positive counterfactual share, its economics must be completed even if
its actual organizational exposure is zero.

Unavailable values remain `None`. A planning state assigning positive share to an
unavailable category or team must supply economics. Cost engines never silently
carry history or interpret unknown rates as zero. Permanent branches inactive in
both states need neither salary profiles/vectors nor permanent games/audits.

## Provenance

The compact assumptions table records team, contract category, missing period,
source period, method, affected economics and interpretation. Distinguish inactive,
explicit-reference and carried-observation cases. Copy actual economics only; do
not manufacture employees or copy exposure into an absent period.

Detailed value diagnostics contain `period`, `team`, `region`, `grade`, `variable`,
`fallback_level`, `fallback_value`, `source_period` and, for employee vectors,
`employee_id`. Profile provenance rows store normalization mass 1, not a salary
or worker count; `membership_profiles` exposes the actual weights. Preserve all reference/copy
records, including later explicit refinements of an automatically copied team;
cell audits use the final effective value/provenance. Coverage and maximum
unsupported counterfactual exposure remain available. Support shares are not
uncertainty intervals.

---

# 17. Validation and external regression tests

Use rtol=1e-10 and atol=1e-6 for costs without intermediate rounding. Every actual
analysis performs source/model reconciliation by period/team/contract, exact-game
efficiency checks, analytical M/K/L checks, parent replacement checks and both
flattened-view reconciliations. Inactive branches have no artificial games.
Every enumerated counterfactual has valid probabilities, aligned policies and
finite costs, including when costs are negative.

Keep regression fixtures outside the user-facing notebook. Tests execute tagged
implementation cells from the notebook, not a duplicated implementation or a
separate production module. Cover:

- signed components, reimbursements, references, unit costs and negative endpoints;
- positive coefficients/base salary, nonnegative worked time, valid exposure and
  normalized complete probability objects;
- category appearance/disappearance in either direction, entire team entry/exit,
  copied complete permanent states and explicit-reference precedence;
- categories absent in both periods, including all-temporary/outsourced datasets;
- observed zero/negative payments distinguished from absent populations;
- strict-mode failures, invalid/missing values and unavailable fallback support;
- same-team fallback isolation, salary/time reconciliation, signed regional
  normalization and pure regional/team-mix changes;
- identical/reversed comparisons, one-team local equivalence, complete outer-game
  equivalence and unchanged-driver zero impacts;
- zero-change parents with offsetting signed children and automatic extension
  through one payment registration plus independently calculated source payroll;
- arbitrary months, period-correct diagnostics and unchanged caller frames;
- salary-only/time-only independence, unequal raises, joint interaction arithmetic,
  and changed salary/time association with unchanged ordinary means;
- employee entry/exit, contract transitions, promotions and cross-team matching;
- sparse profile normalization, invalid indices, legacy-reference migration,
  cached/direct salary equality and the 15-driver extension limit.

Run fresh-kernel workflow tests with no configured input and with a small
production-shaped test DataFrame injected into the input cell. Execute plots and
diagnostics against negative endpoints. Executed fixture outputs remain temporary;
do not embed example or real payroll results in the delivered notebook.

Require PEP-8 style checks on notebook code/tests and docstrings on all named
functions/classes. Show a compact validation/assumption summary in the normal
workflow; detailed reconciliation, salary and support tables remain optional.

---

# 18. Reporting and salary diagnostics

Return complete hierarchy and separate flattened organizational/local bridge
frames. Include dataset, view, team, driver/path, parent, level, units, A/B costs,
impact and signed share of the relevant net change. Shares are undefined near
zero net change. Only leaves belong in flattened bridges.

Default output is an organizational overview waterfall, leaf table, team summary,
source-reference basis and compact validation/assumption summary. A selected team
gets its local bridge and waterfall. Detailed salary/fallback diagnostics are
opt-in. Tables and plots must handle negative A/B costs and signed effects.

Cell audits separately reconcile salary, worked fraction and within-cell base-pay
membership to their full-game players. They share the registered salary costing
formula and use M*K / L*K multipliers for organizational/local currency. By
Shapley additivity and dummy-player invariance, additive payment players can be
omitted from these marginal calculations; assertions compare against the full
14-player game. Cells and employees are not additional players.

Descriptive mean salary/time columns explain cell populations but are not the
switched driver values. Retain coverage, fallback provenance and net/absolute
support shares. Skip these audits for inactive permanent branches.

`employee_alignment` records matching/carry decisions and donor periods/teams.
`employee_inputs` records final effective vectors, including references and whole
copied states; `membership_profiles` exposes their weights. Category references
and copied economics are qualified by the assumptions/fallback tables. Matched
records include salary/time changes, grades, region bands, transfers and sample
entries/exits. They do not establish hiring dates or causes of salary changes.
Organizational and local bridge rows must never be combined.

---

# 19. Interpretation

The bridge is an accounting / analytical attribution, not necessarily a causal decomposition.

For example:

```text
grade composition within region
```

means:

> effect associated with replacing the period-A conditional grade distribution with the period-B conditional grade distribution under Shapley counterfactual averaging.

It does not establish that grade changes causally produced all downstream payroll changes.

The label **Base salary changes** denotes observed input changes, including
promotions; it does not establish discretionary raises or CR. Unchanged salary
or worked-fraction vectors must have zero impacts, regardless of how their
membership weights change. Explicit hypothetical references can supply changes
that are assumptions rather than observations.

Use labels consistent with what is actually observed and identified.

---

# 20. Future optimization

A future planner can construct a workforce state directly and call `cost(state)`. Decision
variables can include workforce size, team shares, and within-team contract, regional and
conditional-grade distributions. Explicit employee salary/time vectors, cell
membership profiles, payment and reimbursement assumptions complete the planned
economics, with references for unsupported categories.

Business constraints could impose team staffing minima, service capacity, regional labor
availability, contract limits, grade/skill coverage, quality/SLA requirements and hiring
limits. Sharing-pay components do not establish cross-team capacity effects; those would
need a separate planning model. Historical builders, costing, attribution and any future
optimizer remain separate. Do not implement optimization in this PoC.

---


# 21. Production schema, signed values and notebook workflow

`adapt_production(frame, *, source_cost_column=None, registry=...)` copies a Pandas
DataFrame. Production exposure is one per employee-month. `vyrabotka_percent` is a
nonnegative factor (0.8 = 80%; values above one are allowed), not an integer
percentage or an extra availability weight. Canonical non-unit exposure remains
supported with monetary inputs per exposure and source totals including exposure.

| Production column | Canonical field |
|---|---|
| period | period |
| mdm_employee_rk_hash | employee_id |
| lvl7_mapped_management_unit_nm | team |
| motivation_type | contract_type: ТК permanent, ГПД temporary, ПКЦ outsourced |
| grade | grade |
| reg_coef | regional_coefficient; derive coefficient band |
| base_stavka | base_salary |
| vyrabotka_percent | worked_fraction |
| sum_pay_bl_reg_base | sick_leave_pay |
| sum_pay_otpusk_reg_base | vacation_pay |
| sum_pay_prazdnik_reg_base | holiday_pay |
| sum_pay_oklad_night_reg_base | night_shift_pay |
| bonus_kpi_comp | bonus_kpi |
| bonus_overtime_comp | overtime_pay |
| bonus_sharing_comp_reg_base | sharing_bonus |
| bonus_quarterly_comp_reg_base | quarterly_bonus |
| sum_pay_compacted | other_pay |
| gpd_pay | temporary reimbursement/source reference |
| pkc_pay | outsourced reimbursement/source reference |

Required identifiers are nonempty strings, periods are timezone-naive month starts,
and employee-month pairs are unique. Keep the schema columns even if a contract
category is absent; inapplicable row values may be null. Applicable monetary values
must be finite and nonmissing but **may be negative**. This includes all payment
components, contract reimbursements, source totals, monetary references and derived
costs. Never clip, take absolute values or replace negative amounts with zero.

Base salary and regional coefficients remain strictly positive; worked time is
nonnegative; exposure and composition retain their quantity constraints. Signed
payments do not create negative aggregation weights; base salary and
worked fraction keep their separate constraints. Interpretation is recorded
monthly net payroll, including reversals; a negative source total still needs explaining components.

Without `source_cost_column`, permanent payroll is reconstructed using section 6
and other contract references use gpd_pay/pkc_pay. An independent source column
covers all employee-months and is never altered by regional normalization.
For a newly registered payment beyond the known production formula, supply its
canonical source column and an independent source total including the addition;
one registry entry handles state construction, costing and attribution.

The notebook has one input/settings cell, no synthetic generator and no automatic
legacy CSV adapter. All calculation functions remain in collapsible notebook
sections. With no data selected it gives setup guidance; with data it runs the
analysis directly. Detailed docstrings explain function purpose, inputs, units,
returns and errors. Use PEP-8 formatting: descriptive snake_case, grouped imports,
four-space indentation and 79-character code lines; wrap prose at 72 where practical.
The notebook is delivered without stored payroll outputs. See the README for the
copy-and-run workflow and external test commands.
