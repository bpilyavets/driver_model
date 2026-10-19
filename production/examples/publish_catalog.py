"""Publish a supplied dataset and model after fixture validation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from labor_cost import (
    AnalysisRequest,
    DatasetSnapshot,
    analyze,
    export_run,
    labor_cost_model,
    publish_model,
)


def main():
    """Validate supplied Parquet data and create a new catalog directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--period-a", required=True)
    parser.add_argument("--period-b", required=True)
    parser.add_argument("--dataset-id", default="payroll")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--source-cost-column")
    parser.add_argument("--exposure-column")
    parser.add_argument(
        "--explainer", choices=["exact", "permutation"], default="exact"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    frame = pd.read_parquet(args.dataset)
    snapshot = DatasetSnapshot(args.dataset_id, args.dataset_version, frame)
    model = labor_cost_model()
    result = analyze(
        model,
        snapshot,
        AnalysisRequest(
            args.period_a,
            args.period_b,
            source_cost_column=args.source_cost_column,
            exposure_column=args.exposure_column,
            explainer=args.explainer,
        ),
    )
    args.output.mkdir(parents=True)
    publish_model(model, args.output / "model.json", validated_result=result)
    frame.to_parquet(args.output / "dataset.parquet", index=True)
    export_run(result, args.output / "validation-run")
    config = {
        "models": [
            {
                "id": model.model_id,
                "version": model.version,
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
    path = args.output / "catalog.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
