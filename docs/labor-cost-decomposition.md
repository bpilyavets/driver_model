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

```text
0 = use period-A regional composition
1 = use period-B regional composition
```

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

# 4. Workforce hierarchy

The workforce contains three employment types:

1. permanent;
2. temporary;
3. outsourced.

Their cost structures differ materially.

Use a hierarchical decomposition.

## Top level

Model total labor cost approximately as:

$$
C=
N
\left[
p_Pc_P
+
p_Tr_T
+
p_Or_O
\right]
$$

where:

* \(N\) = total workforce exposure;
* \(p_P,p_T,p_O\) = permanent/temp/outsource shares;
* \(c_P\) = average all-in permanent unit cost;
* \(r_T\) = average temporary reimbursement;
* \(r_O\) = average outsourcing reimbursement.

Initially the top-level business drivers are:

```text
workforce_scale
contract_type_mix
permanent_unit_cost
temp_reimbursement
outsource_reimbursement
```

`contract_type_mix` is one vector-valued driver:

$$
(p_P,p_T,p_O)
$$

and must always satisfy:

$$
p_P+p_T+p_O=1.
$$

Do not expose the three shares as independently switchable SHAP features.

---

# 5. Permanent workforce model

Permanent employee economics are decomposed further.

The exact payroll formula will evolve, but currently contains concepts such as:

* base salary;
* regional adjustment;
* worked-hours/exposure;
* grade composition;
* bonus base;
* KPI achievement;
* overtime;
* holiday pay;
* other payroll components.

The permanent cost engine must be extensible rather than hard-coded around the current illustrative formula.

---

# 6. Regional salary rule

Region has a deterministic business-rule coefficient.

For employee \(i\):

$$
ObservedSalary_i
=
PreRegionalSalary_i
\times R_{r(i)}
$$

where \(R_r\) is the official coefficient for region \(r\).

Therefore:

$$
PreRegionalSalary_i
=
\frac{ObservedSalary_i}{R_{r(i)}}.
$$

Regional coefficient values belong inside the payroll/cost logic.

If coefficient policy itself does not change between periods, regional coefficients are not a changing SHAP driver.

The relevant changing driver is the distribution of workforce across regions.

---

# 7. Grade and salary identification

Grades determine permitted salary ranges, but employees can occupy different positions within those ranges.

The available dataset contains:

* grade;
* observed base salary;
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

# 8. Region-first composition parameterization

Represent the joint region × grade distribution as:

$$
P(R,G)=P(R)P(G\mid R).
$$

This parameterization is intentional.

It corresponds to the useful business hierarchy:

```text
Where are employees located?
            ↓
What grade mix exists within each region?
            ↓
What is the underlying salary level within a region × grade cell?
```

## Regional composition

$$
p_r=P(R=r)
$$

is a vector whose elements sum to one.

This is one SHAP driver.

## Grade composition within region

$$
p_{g|r}=P(G=g\mid R=r)
$$

is a matrix.

Each region row must sum to one.

The complete matrix is initially one SHAP driver:

```text
grade_mix_within_region
```

Do not expose individual grade shares as independent features.

## Underlying salary level

For each employee remove the deterministic regional multiplier:

$$
B_i=
\frac{ObservedSalary_i}{R_{r(i)}}.
$$

Then define:

$$
\mu_{gr}
=
E[B_i\mid G=g,R=r].
$$

This is a region × grade matrix.

Call this driver something such as:

```text
underlying_salary_level
```

or:

```text
within_grade_region_salary
```

Do not label it as CR or pure grade effect.

It contains unidentifiable within-cell effects such as:

* position inside grade salary range;
* within-grade salary revisions;
* hiring/attrition of differently paid people within the same cell;
* other salary variation not explained by observed region/grade composition.

---

# 9. Salary composition identity

A simplified average base-salary component for permanent staff is:

$$
\bar S
=
\sum_r
p_r R_r
\sum_g
p_{g|r}\mu_{gr}.
$$

Equivalently, construct the joint workforce distribution:

$$
p_{rg}=p_rp_{g|r}.
$$

Then:

$$
\bar S
=
\sum_{r,g}
p_{rg}
R_r
\mu_{gr}.
$$

In NumPy:

```python
joint_mix = (
    region_mix[:, None]
    * grade_given_region
)

actual_salary = (
    underlying_salary
    * regional_coeff[:, None]
)

average_salary = (
    joint_mix
    * actual_salary
).sum()
```

The three relevant changing components are therefore initially:

```text
regional_mix
grade_mix_within_region
underlying_salary_level
```

The fixed regional coefficient remains a deterministic parameter inside the function.

---

# 10. Workforce states

Historical data should first be transformed into a period-level workforce state.

Conceptually:

```python
state = {
    "workforce_scale": ...,
    "contract_type_mix": ...,

    "permanent": {
        "region_mix": ...,
        "grade_mix_within_region": ...,
        "underlying_salary_level": ...,
        "worked_hours": ...,
        "bonus_base": ...,
        "kpi_achievement": ...,
        "overtime": ...,
        "holiday_pay": ...,
        ...
    },

    "temp_reimbursement": ...,
    "outsource_reimbursement": ...,
}
```

The exact implementation may differ.

Important properties:

* the cost engine consumes a workforce state rather than raw historical data;
* state entries can be scalars, vectors, or matrices;
* state construction and cost calculation remain separate.

---

# 11. Driver registry

Business-driver definitions should be centralized.

The exact implementation is flexible, but adding a driver should ideally require declaring:

* name;
* label;
* level / parent;
* state value builder or state key;
* optionally validation/fallback behavior.

The generic SHAP engine should discover the applicable drivers from this registry rather than duplicating hard-coded driver lists throughout the codebase.

---

# 12. SHAP switch game

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

# 13. Hierarchical decomposition

Run two logical games.

## Level 1: total cost

Explain total labor cost using:

```text
workforce_scale
contract_type_mix
permanent_unit_cost
temp_reimbursement
outsource_reimbursement
```

## Level 2: permanent unit cost

Explain \(c_P\) using its internal business drivers:

```text
regional_mix
grade_mix_within_region
underlying_salary_level
worked_hours
bonus/KPI
overtime
holiday_pay
...
```

The output of Level 2 is initially in permanent-unit-cost currency.

Those impacts must be propagated back into total-cost currency without arbitrarily multiplying them by period-A or period-B permanent headcount.

---

# 14. Permanent exposure multiplier

The top-level model is linear in permanent unit cost:

$$
C=N[p_Pc_P+\ldots].
$$

Therefore all inner permanent-unit-cost Shapley contributions can be propagated through a common Shapley-weighted permanent exposure multiplier.

A robust way to obtain the multiplier is to run the top-level game with:

```text
permanent_unit_cost_A = 0
permanent_unit_cost_B = 1
```

while other top-level drivers retain their normal A/B states.

The SHAP impact assigned to `permanent_unit_cost` is then the effective exposure multiplier \(M\).

For inner permanent driver \(j\):

$$
\Phi_j^{Total}
=
M
\phi_j^{Permanent}.
$$

Then:

$$
\sum_j\Phi_j^{Total}
=
\Phi_{PermanentUnitCost}^{Top}.
$$

This allows the parent `permanent_unit_cost` line to be replaced by its detailed children while preserving reconciliation.

---

# 15. Sparse and unsupported cells

Historical workforce data may not populate every region × grade cell in both periods.

For example, February may contain Grade 7 in Region A while January does not.

Then:

$$
\mu_{A,7}^{January}
$$

is not directly observed.

Do not fill such cells with zero.

Implement an explicit, visible fallback policy suitable for the PoC.

A possible hierarchy is:

1. observed region × grade mean;
2. corresponding grade mean;
3. corresponding region mean;
4. global permanent mean;
5. user-provided business reference, where available.

The exact policy is configurable.

Every fallback use must be captured in a diagnostics table with at least:

```text
period
region
grade
variable
fallback_level
fallback_value
```

Unsupported categories and potentially dubious counterfactuals should be visible to the analyst.

---

# 16. Required validation

## Cost reconciliation

$$
ModeledCost_A\approx SourceCost_A
$$

$$
ModeledCost_B\approx SourceCost_B.
$$

## Top-level attribution

$$
\sum_k\phi_k^{Top}
\approx
C_B-C_A.
$$

## Permanent attribution

$$
\sum_j\phi_j^{Permanent}
\approx
c_{P,B}-c_{P,A}.
$$

## Flattened hierarchical bridge

After replacing the permanent parent with its children:

$$
\sum_l\Phi_l^{Final}
\approx
C_B-C_A.
$$

## Distribution validity

Check:

$$
\sum_c P(Contract=c)=1
$$

$$
\sum_rP(R=r)=1
$$

and for every region:

$$
\sum_gP(G=g\mid R=r)=1.
$$

Do not proceed silently with invalid distributions.

---

# 17. Reporting

The main result should be a table approximately containing:

```text
driver
parent_driver
level
impact
share_of_net_change
```

Example:

```text
Workforce scale
Contract type mix
Permanent: regional mix
Permanent: grade composition within region
Permanent: underlying salary level
Permanent: worked hours
Temp reimbursement
Outsource reimbursement
...
```

Also report:

```text
period_A_total
period_B_total
total_change
reconciliation_error
```

A simple waterfall chart is useful but secondary to numerical correctness.

---

# 18. Interpretation

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

# 19. Future optimization

The future planning problem may look like:

$$
\min_x Cost(x)
$$

subject to constraints involving:

* service capacity;
* regional labor availability;
* contract-type limits;
* grade availability;
* quality/SLA requirements;
* hiring limits;
* operational constraints.

Therefore the historical-state builder, cost engine, attribution engine, and future optimizer should remain conceptually distinct.

The current PoC does not need to implement optimization.
