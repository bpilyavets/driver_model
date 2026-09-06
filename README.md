# Labor-cost decomposition PoC

Open **[labor_cost_decomposition.ipynb](labor_cost_decomposition.ipynb)** for the executed
demonstration and its assumptions, diagnostics, bridge, waterfall, and validation results.
Analytical requirements are in [the methodology](docs/labor-cost-decomposition.md).

Validated on Python 3.10. From this directory, using that Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 labor_cost_decomposition.ipynb
```

On Debian/Ubuntu, virtual-environment creation requires the `python3-venv` package.

For interactive use, open the notebook in your Jupyter/IPython-compatible editor and select
the environment above. Alternatively install `jupyterlab` in that environment and run
`python -m jupyter lab`.

The main demonstration creates deterministic synthetic Pandas DataFrames and needs no input
file. Production callers supply DataFrames, not CSV paths. A supplementary adapter uses the
provided `costs_poc.csv` when present; the notebook documents its narrower driver set and
reconstructed payroll reference. The confirmed mock salary, sick-leave pay, and holiday pay
are **pre-regional**. Its reconstructed totals are **494,716.700 for January** and
**625,148.125 for February**.

The canonical DataFrame accepts `base_salary` directly. Salary, sick leave, and holiday pay
receive the regional coefficient through `apply_regional_coefficient` in the driver registry;
bonus/KPI and overtime are explicitly unadjusted in this example. If source amounts already
include the coefficient, call `normalize_regional_inputs` once with those eligible columns
before building states. Normalization preserves actual `source_cost` and precedes sparse-cell
fallbacks. Regional coefficients remain fixed A/B payroll rules, separate from changing
regional workforce composition.

The source grain is employee-month; business-driver attribution runs on aggregate workforce
states. The cost engine works without historical comparison or SHAP. Section 13 demonstrates
adding a regionally eligible night-shift premium with one source-data/payroll change and one
driver registration.

Each result identifies its dataset, A/B employee counts, permanent counts and exposure.
The main example is **SYNTHETIC**, with an explicit 4.5% February salary uplift and independently
drawn monthly employee characteristics. Use the notebook's **MOCK Jan–Feb results** link to
jump to the supplied fixture; section 13 returns to **SYNTHETIC + night-shift premium**.

The **Within-region/grade salary level** driver includes within-cell hiring/attrition effects,
not only raises. Its cell audit shows observed/fallback salary means and monetary contributions,
with a support summary. A separate matched-employee diagnostic reports salary changes and
permanent-sample entries/exits without treating those records as SHAP players. For the mock,
five permanent employees appear in both months and none changes base salary; two offsetting
cell contributions yield approximately **+11.36** total currency of salary impact.
