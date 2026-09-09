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
* worked hours;
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
\mu_{GR},\quad
\ldots
$$

The attribution game operates on changes in these aggregate workforce-state drivers between periods A and B.

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
        ├── Within-team/region/grade salary level
        ├── Worked hours (salary-weighted)
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
eligibility belong in the driver registry, not in SHAP or reporting code. There are 13 current
permanent drivers: four composition/salary/time drivers and nine separate payment drivers.

---

# 7. Base salary and component-level regional adjustment

Canonical salary and eligible payment amounts are pre-regional currency per exposure.
Salary, sick leave, vacation, holiday, night shift, sharing and quarterly bonuses receive
the destination region's coefficient once. KPI, overtime and other bundled pay do not.

$$
c_{P,t}=\sum_{r,g}p_{r|t}p_{g|r,t}
\left[R_r(\mu_{t,r,g}h^*_{t,r,g}+Sick+Vacation+Holiday+Night+Sharing+Quarterly)
+KPI+Overtime+Other\right].
$$

Within each team × region × grade cell:

$$
\mu=\frac{\sum_i e_iB_i}{\sum_i e_i},\qquad
h^*=\frac{\sum_i e_iB_ih_i}{\sum_i e_iB_i}.
$$

This exactly preserves the employee-level salary/time product. The worked-hours driver
can include changing salary/time association within a cell and is not a pure causal hours
effect. Ordinary salary and time means miss the legacy mock salary totals by 40.50 and
91.6666666667 respectively (modeled minus actual: negative in both periods).

Declare `apply_regional_coefficient` for every registered cost contribution, including
salary's salary × time contribution. Sharing a multiplier does not merge payment drivers.

`normalize_regional_inputs(frame, already_regionalized=(...))` divides explicitly identified
eligible columns by their **source-region** coefficient before aggregation/fallback. It
never infers their basis, adjusts an ineligible component, or changes `source_cost`.
Apply it once; production and the confirmed mock already use pre-regional inputs directly.

Production has no geographical region identifier: `region_{reg_coef}` is a coefficient
band, not a known geographical area. Movement between bands is a composition change under
this parameterization. These fields alone cannot distinguish a policy change from band
movement. Fixed coefficient policy is therefore an explicit assumption. Supplied A/B
states with differing coefficient schedules for aligned regions fail explicitly.

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

Instead use observable workforce-composition and within-cell salary components.

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

The `underlying_salary_level` matrix is labeled **Within-team/region/grade salary level**:

$$
\mu_{t,r,g}=E_e[BaseSalary\mid T=t,R=r,G=g,Permanent].
$$

The expectation is weighted by exposure. It includes within-cell pay revisions, differently
paid employees entering/leaving a cell or team, position within the grade salary range,
and unsupported-cell fallback changes. Transfers can change it even if matched employees
receive no raises. It is neither a pure raise effect, CR nor a grade coefficient.

Keep cell provenance and matched-employee diagnostics beside the attribution; do not
manufacture unidentified economic variables from grade and salary alone.

---

# 10. Salary composition identity

Within a team, the salary component is:

$$
\bar S_t=\sum_{r,g}p_{r|t}p_{g|r,t}R_r\mu_{t,r,g}h^*_{t,r,g}.
$$

```python
joint_mix = permanent_state['regional_mix'][:, None] * permanent_state['grade_mix_within_region']
```

The shared cell-cost engine multiplies this composition by registered payroll contributions.
It supplies both summed permanent cost and the cell contributions used in the salary audit.
Changing payroll logic must not require editing the attribution formula.

---

# 11. Workforce states and interfaces

```python
state = {
    'workforce_scale': ...,
    'team_mix': ...,                   # normalized vector aligned with _teams
    '_teams': ('Team A', 'Team B'),
    'teams': {
        'Team A': {
            'contract_type_mix': ...,
            'temp_reimbursement': ...,
            'outsource_reimbursement': ...,
            'permanent': {
                '_policy': {
                    'regions': ...,
                    'grades': ...,
                    'regional_coefficients': ...,
                },
                'regional_mix': ...,
                'grade_mix_within_region': ...,
                'underlying_salary_level': ...,
                'worked_hours': ...,
                # registered payment matrices
            },
        },
        # other teams
    },
}
```

Use `make_policy(frame)` on the selected A/B observations to align team axes and each
team's region/grade axes. `build_state(frame, period, policy, registry=..., references=...)`
returns `(state, fallbacks, coverage)`. Cost functions accept states only:

- `permanent_cost(permanent_state)` gives all-in permanent cost per exposure.
- `team_unit_cost(team_state)` gives blended team cost per exposure.
- `cost(workforce_state)` gives organizational total cost.
- `decompose(a, b, registry=..., dataset=...)` gives permanent/upper games, multipliers,
  complete hierarchy, organizational/local flattened bridges and coalition diagnostics.
- `diagnostics(...)` adds source-dependent salary, fallback, coverage and matched-ID audits.

The cost functions do not depend on SHAP or historical comparison. States contain no
inferred employee-level counterfactual payroll records.

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

For a decomposition between states \(A\) and \(B\), expose one synthetic binary feature per business driver:

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

Run a 13-player permanent-unit game independently for each team, then a four-player
team-unit game (contract mix, temporary rate, outsourcing rate, permanent unit).
The organization has scale, whole team mix and one economics scalar per team.
A local team has a two-player outer game (local scale, unit economics).

Use `shap.ExactExplainer` with a single zero background and explain the one vector,
allowing 2**d evaluations. Validate all binary coalitions, including probability objects
and finite costs. The PoC has an explicit 14-driver limit per exact game; no silent
switch to approximate attribution. The extension example has 16,384 coalitions.

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

# 16. Sparse support and explicit references

Fallbacks stay within the same period and team:

1. observed region × grade population;
2. that team's corresponding grade population;
3. that team's corresponding region population;
4. that team's permanent population.

Recompute the original exposure-weighted mean or salary-weighted ratio on the selected
population. An absent region uses the same team-period marginal grade distribution.
Fallbacks are configurable; no support means failure. Do not pool another team's pay or
silently borrow the other period. Never fill unknown salary/payment cells with zero.
Zero coverage/count records indicate observed absence, not zero unit economics.

Eligible salary/payment fallbacks are pre-regional currency per exposure; normalize
already-adjusted values before pooling, then apply the destination coefficient in costing.
Unadjusted payments are currency per exposure; probabilities/hours are dimensionless.

Diagnostics include `period`, `team`, `region`, `grade`, `variable`, `fallback_level`,
`fallback_value`. Display observed counts/exposure, all imputations and maximum unsupported
counterfactual permanent-composition share and exposure. Support diagnostics are not
statistical uncertainty intervals.

Missing teams/contract categories require explicit business references, including when
the missing category's observed share is zero:

```python
references = {
    (period, team): {
        'temporary': ...,   # scalar all-in reimbursement per exposure, if absent
        'outsourced': ...,  # scalar all-in reimbursement per exposure, if absent
        'permanent': ...,   # complete aligned permanent state, if absent
        # For an absent whole team, supply instead:
        'team_state': ...,  # full contract mix, rates and permanent state
    }
}
```

Only the applicable entries are needed. Validate reference states and record every
reference value. An absent team still has zero observed team share. References complete
unit economics only; they do not manufacture observed workers. If a team has no permanent
observations in either period, supply its aligned permanent policy through
`make_policy(frame, permanent_policies={team: policy})` and the required state references.

---

# 17. Required validation

Use cost tolerances rtol=1e-10 and atol=1e-6, without intermediate rounding.

- Reconcile each period's modeled organization/team/contract costs to source references,
  distinguishing independently supplied/generated costs from reconstructed payroll.
- Reconcile organizational, team-unit and permanent-unit SHAP sums to their endpoint changes.
- Reconcile every propagated parent and both flattened views; sum local endpoint changes
  to the organizational change. Check M/K/L against analytical values.
- Validate finite nonnegative probabilities, normalized team/contract/region vectors and
  every conditional-grade row, aligned axes and finite costs for all enumerated coalitions.
- Check reduced organizational games against full enumeration, one-team local equivalence,
  identical and reversed comparisons, and unchanged-driver zero impacts.
- Check pure team and region composition changes; unchanged payments must receive zero
  impact while regional mix captures changes across all eligible payments.
- Check coefficients once using a hand calculation; equivalent pre-regional and explicitly
  normalized source values must yield equal states, fallback values, costs and impacts.
- Test zero permanent change with offsetting sick/holiday payments and zero team-unit
  change with offsetting reimbursement effects; propagation must not divide by changes.
- Exercise team-local sparse/absent-region fallback, reference-required failures, valid
  explicit references and absent support. Reject duplicates, missing identifiers, unknown
  categories, inconsistent coefficients and invalid applicable payroll values.
- Reconcile and automatically report the registered transport extension, including all
  16,384 coalitions per extended permanent game.

Execute and save fresh-kernel output with the legacy CSV, and execute from a directory
without the CSV. Show a compact pass/error table and elapsed time.

---

# 18. Reporting and salary diagnostics

Return the complete hierarchy and separate flattened organizational/local bridge tables.
Include dataset, view, team, driver/path, parent, level, units, A/B costs, impact, and signed
share of the relevant net change. Shares are undefined for near-zero net changes. Only
leaves belong in flattened bridges. Display endpoint costs and reconciliation errors.

Show organization and local waterfalls with A/B endpoints. Identify dataset, periods,
team/contract employee-month counts, exposure and reference basis beside the results.
Synthetic, legacy mock and registered extension outputs must remain distinguishable.

Salary audits include team × region × grade counts, pre-regional means, observed/fallback
provenance, and contributions in permanent-unit, organizational and local currency. Use
the existing salary player's coalition weights and M*K / L*K multipliers. Shared cell-cost
contributions provide matrix-valued marginals, avoiding separate engine calls for each cell.
Assert cell additivity and reconciliation to salary SHAP. Cells are not additional players.
Report net and absolute-impact shares by observed/imputed support, undefined at zero denominators.

Matched permanent IDs report individual base-salary changes, grade/region changes and team
transfers separately. Team sample entries/exits distinguish employees present elsewhere
in the organization's other-period sample from organization sample entries/exits. Permanent
sample entries/exits remain separate from organization membership changes. These observations
are not extra bridge impacts, causal raise estimates or proof of hiring/termination dates.

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

Similarly:

```text
underlying salary level
```

is deliberately broader than CR because CR cannot currently be identified.

Use labels consistent with what is actually observed and identified.

---

# 20. Future optimization

A future planner can construct a workforce state directly and call `cost(state)`. Decision
variables can include workforce size, team shares, and within-team contract, regional and
conditional-grade distributions. Explicit salary, payment and reimbursement assumptions
complete the planned economics, with references for unsupported categories.

Business constraints could impose team staffing minima, service capacity, regional labor
availability, contract limits, grade/skill coverage, quality/SLA requirements and hiring
limits. Sharing-pay components do not establish cross-team capacity effects; those would
need a separate planning model. Historical builders, costing, attribution and any future
optimizer remain separate. Do not implement optimization in this PoC.

---


# 21. Production and legacy adapters

`adapt_production(frame, *, source_cost_column=None)` takes a Pandas DataFrame. Production
exposure is one per employee-month row. `vyrabotka_percent` is a finite nonnegative factor
(0.8 = 80%; values above one are allowed), not an integer percentage or an extra availability
weight. Apply the employee formula directly. Canonical non-unit exposure remains supported
only with monetary inputs expressed per exposure and source totals multiplied by exposure.

| Production column | Canonical field |
|---|---|
| period | period |
| mdm_employee_rk_hash | employee_id |
| lvl7_mapped_management_unit_nm | team |
| motivation_type | contract_type: ТК permanent, ГПД temporary, ПКЦ outsourced |
| grade | grade |
| reg_coef | regional_coefficient; derive region coefficient band |
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
| gpd_pay | temporary contract_reimbursement and source reference |
| pkc_pay | outsourced contract_reimbursement and source reference |

Required identifiers are nonempty strings; periods are datetime month starts and employee-month
pairs are unique. Permanent base salary/coefficient must be positive. Applicable payment/rate
values must be finite and nonnegative; missing applicable amounts are errors. Inapplicable
fields may be null. This PoC does not interpret signed payroll corrections automatically.

Without `source_cost_column`, permanent source references are reconstructed from the given
formula; non-permanent references use gpd_pay/pkc_pay. Label that basis visibly. If supplied,
the independent source-cost column covers every employee-month and is not modified by
normalization. A reconstructed reference verifies aggregation, not independent payroll completeness.

The optional `costs_poc.csv` uses a separate adapter: legacy ГПХ maps to temporary, one mock
team, exposure one, utilization as worked fraction and pre-regional base/sick/holiday amounts.
Missing supplemental entries mean no recorded payment in this fixture. Other new components
and KPI/overtime are explicitly unavailable and zero for that limited reconstruction only.
Escaped line endings and decimal commas are handled only here. Its reconstructed totals remain
494,716.700 for January and 625,148.125 for February; it does not represent multiple teams.

Night shift is standard payroll. The extension example is a regionally eligible transport
allowance, requiring one source-data/payroll addition and one driver registration. The cost,
state, SHAP, propagation, reporting and diagnostic functions must remain unchanged.
