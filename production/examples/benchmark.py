"""Measure the full validated workflow on a supplied dataset."""

import argparse
import json
import tracemalloc
from pathlib import Path
from time import perf_counter

import pandas as pd

from labor_cost import (
    AnalysisRequest,
    DatasetSnapshot,
    analyze,
    labor_cost_model,
)


def main():
    """Measure the supplied workload without inferring production sizing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--period-a", required=True)
    parser.add_argument("--period-b", required=True)
    parser.add_argument("--source-cost-column")
    parser.add_argument(
        "--explainer", choices=["exact", "permutation"], default="exact"
    )
    parser.add_argument("--cycles", type=int, default=256)
    args = parser.parse_args()
    frame = pd.read_parquet(args.dataset)
    snapshot = DatasetSnapshot("benchmark", "input", frame)
    request = AnalysisRequest(
        args.period_a,
        args.period_b,
        source_cost_column=args.source_cost_column,
        explainer=args.explainer,
        n_permutations=args.cycles,
    )
    tracemalloc.start()
    started = perf_counter()
    result = analyze(labor_cost_model(), snapshot, request)
    elapsed = perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    selected = result.table("source_snapshot")
    print(
        json.dumps(
            {
                "seconds": elapsed,
                "python_allocation_peak_bytes": peak,
                "allocation_caveat": "Measures Python allocations only",
                "rows": len(selected),
                "teams": selected.lvl7_mapped_management_unit_nm.nunique(),
                "employees": selected.mdm_employee_rk_hash.nunique(),
                "settings": result.manifest["settings"],
                "dependencies": result.manifest["dependencies"],
                "dataset_fingerprint": result.manifest["dataset"],
                "validation_passed": bool(
                    result.table("validation").passed.all()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
