"""Author, validate, publish, resolve and analyze supplied fixed-grain data."""

import argparse
import json
from pathlib import Path

import pandas as pd
from author_models import hourly_model

from labor_cost import (
    AnalysisRequest,
    Catalog,
    DatasetSnapshot,
    analyze,
    compile_model,
    export_run,
    publish_model,
)


def main():
    """Run the analyst/business lifecycle on an existing Parquet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--period-a", required=True)
    parser.add_argument("--period-b", required=True)
    parser.add_argument("--source-cost-column", default="actual")
    parser.add_argument("--dataset-version", default="1")
    parser.add_argument(
        "--explainer", choices=("exact", "permutation"), default="exact"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new catalog directory")
    model = compile_model(hourly_model())
    print(json.dumps(model.inspect(), indent=2))
    snapshot = DatasetSnapshot(
        "hourly-payroll", args.dataset_version, pd.read_parquet(args.dataset)
    )
    request = AnalysisRequest(
        args.period_a,
        args.period_b,
        source_cost_column=args.source_cost_column,
        explainer=args.explainer,
    )
    validated = analyze(model, snapshot, request)
    args.output.mkdir(parents=True)
    publish_model(
        model, args.output / "model.json", validated_result=validated
    )
    snapshot.frame.to_parquet(args.output / "dataset.parquet", index=True)
    export_run(validated, args.output / "validation-run")
    config = {
        "models": [
            {
                "id": model.spec.model_id,
                "version": model.spec.version,
                "path": "model.json",
                "fingerprint": model.fingerprint,
            }
        ],
        "datasets": [
            {
                "id": snapshot.dataset_id,
                "version": snapshot.version,
                "path": "dataset.parquet",
                "fingerprint": snapshot.fingerprint,
            }
        ],
    }
    index = args.output / "catalog.json"
    index.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    catalog = Catalog(index)
    business_result = analyze(
        catalog.model(model.spec.model_id, model.spec.version),
        catalog.dataset(snapshot.dataset_id, snapshot.version),
        request,
    )
    export_run(business_result, args.output / "business-run")
    print(f"Published catalog: {index}")
    print(f"Validated business run: {business_result.run_id}")


if __name__ == "__main__":
    main()
