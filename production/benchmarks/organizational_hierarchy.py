"""Measure one validated hierarchy in a fresh process (Linux peak RSS)."""

import argparse
import json
import resource
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from graph_runtime import fixture

from labor_cost import (
    AnalysisRequest,
    DatasetSnapshot,
    analyze,
    export_run,
    labor_cost_model,
)


def main():
    """Measure state building, attribution, peak memory and portable output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--order",
        choices=["team", "line", "team,line", "line,team"],
        required=True,
    )
    parser.add_argument("--teams", type=int, default=4)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument(
        "--explainer", choices=["exact", "permutation"], default="exact"
    )
    parser.add_argument("--cycles", type=int, default=256)
    args = parser.parse_args()
    if args.teams < 2:
        parser.error("--teams must be at least two")
    frame = fixture(args.teams)
    person = frame.mdm_employee_rk_hash.str.split("-").str[-1]
    frame["line"] = person.map(
        lambda p: "X" if p.isdigit() and int(p) % 2 else "Y"
    )
    team = "lvl7_mapped_management_unit_nm"
    if args.sparse:
        frame = frame.loc[
            ~(
                frame.period.eq("2025-04-01")
                & frame[team].eq("Team-1")
                & frame.line.eq("X")
            )
        ]
    snapshot = DatasetSnapshot("hierarchy-runtime", "1", frame)
    order = tuple(args.order.split(","))
    req = AnalysisRequest(
        "2025-03",
        "2025-04",
        hierarchy=order,
        source_cost_column="actual_cost",
        explainer=args.explainer,
        n_permutations=args.cycles,
    )
    model = labor_cost_model(schema_version=2, line_source="line")
    start = perf_counter()
    result = analyze(model, snapshot, req)
    seconds = perf_counter() - start
    audit = result.table("coalition_audit")
    with TemporaryDirectory() as directory:
        path = export_run(result, Path(directory) / "run")
        size = sum(file.stat().st_size for file in path.iterdir())
    print(
        json.dumps(
            {
                **vars(args),
                "rows": len(frame),
                "active_pairs": frame.groupby([team, "line"]).ngroups,
                "seconds": seconds,
                **result.manifest["timings"],
                "permanent_games": int(audit.game.eq("permanent").sum()),
                "peak_rss_mib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024,
                "bundle_bytes": size,
                "validation": bool(result.table("validation").passed.all()),
            }
        )
    )


if __name__ == "__main__":
    main()
