"""Measure the validated graph runtime on reproducible synthetic payroll.

This is an external measurement fixture, never part of a business client.
Use a fresh process per estimator to include its first-use initialization.
"""

import argparse
import json
import platform
from time import perf_counter

import pandas as pd

from labor_cost import (
    AnalysisRequest,
    DatasetSnapshot,
    analyze,
    labor_cost_model,
)

ADJUSTED = (
    "sum_pay_bl_reg_base",
    "sum_pay_otpusk_reg_base",
    "sum_pay_prazdnik_reg_base",
    "sum_pay_oklad_night_reg_base",
    "bonus_sharing_comp_reg_base",
    "bonus_quarterly_comp_reg_base",
)
UNADJUSTED = (
    "bonus_kpi_comp",
    "bonus_overtime_comp",
    "sum_pay_compacted",
)


def fixture():
    """Return four teams, two months and signed independently costed rows."""
    rows = []
    for month, period in enumerate(("2025-03-01", "2025-04-01")):
        for team in range(4):
            for person in range(30):
                if month == 1 and team == 3 and person == 29:
                    continue
                destination = 1 if month and team == person == 0 else team
                employee = f"employee-{team}-{person}"
                if month and team == 0 and person == 23:
                    employee = "new-employee"
                contract = (
                    "ТК" if person < 24 else "ГПД" if person < 28 else "ПКЦ"
                )
                coefficient = 1 + 0.2 * ((person + month) % 2)
                base = 1000 + 50 * person + 10 * month * (person % 4)
                worked = 0.5 + 0.1 * ((person + month) % 6)
                payments = {
                    name: (person + 1) * (index + 1) * 0.2
                    + month * (-1 if index % 2 else 1)
                    for index, name in enumerate(ADJUSTED + UNADJUSTED)
                }
                rate = 400 + 20 * person + 25 * month
                row = {
                    "period": period,
                    "mdm_employee_rk_hash": employee,
                    "lvl7_mapped_management_unit_nm": f"Team-{destination}",
                    "motivation_type": contract,
                    "grade": f"G{(person + month) % 3}",
                    "reg_coef": coefficient,
                    "base_stavka": base,
                    "vyrabotka_percent": worked,
                    "gpd_pay": rate,
                    "pkc_pay": -rate if person == 29 else rate,
                    **payments,
                }
                if contract == "ТК":
                    actual = coefficient * (
                        base * worked + sum(payments[x] for x in ADJUSTED)
                    ) + sum(payments[x] for x in UNADJUSTED)
                else:
                    actual = row["gpd_pay" if contract == "ГПД" else "pkc_pay"]
                rows.append({**row, "actual_cost": actual})
                if person < 3 + month:
                    rows.append(
                        {
                            **row,
                            "motivation_type": "ГПД",
                            "gpd_pay": -25.0 + 10 * month,
                            "actual_cost": -25.0 + 10 * month,
                        }
                    )
    return pd.DataFrame(rows)


def main():
    """Print workload, wall time and actual estimator metadata as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explainer", choices=("exact", "permutation"), default="exact"
    )
    args = parser.parse_args()
    frame = fixture()
    snapshot = DatasetSnapshot("synthetic-runtime", "1", frame)
    request = AnalysisRequest(
        "2025-03",
        "2025-04",
        source_cost_column="actual_cost",
        explainer=args.explainer,
        n_permutations=256,
        random_seed=0,
    )
    started = perf_counter()
    result = analyze(labor_cost_model(schema_version=2), snapshot, request)
    elapsed = perf_counter() - started
    print(
        json.dumps(
            {
                "seconds": elapsed,
                "measurement": "validated analyze, first use, no profiler",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "rows": len(frame),
                "employees": frame.mdm_employee_rk_hash.nunique(),
                "teams": frame.lvl7_mapped_management_unit_nm.nunique(),
                "settings": result.manifest["settings"],
                "dependencies": result.manifest["dependencies"],
                "dataset": result.manifest["dataset"],
                "validation_passed": bool(
                    result.table("validation").passed.all()
                ),
                "audit": json.loads(
                    result.table("coalition_audit").to_json(orient="records")
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
