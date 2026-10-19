"""Portable artifact integrity and catalog publication acceptance."""

import json
from copy import deepcopy

import pandas as pd
import pytest

from labor_cost import (
    Catalog,
    DatasetSnapshot,
    export_run,
    labor_cost_model,
    load_run,
    publish_model,
)


def test_bundle_roundtrip(public_result, tmp_path):
    """Preserve tables, signed values, dtypes and nested provenance."""
    path = export_run(public_result, tmp_path / "run")
    restored = load_run(path)
    assert restored.run_id == public_result.run_id
    assert restored.states is None
    assert restored.manifest["model"] == public_result.manifest["model"]
    assert restored.manifest["settings"] == public_result.manifest["settings"]
    for name in public_result.tables:
        pd.testing.assert_frame_equal(
            public_result.tables[name], restored.tables[name]
        )
    assert restored.select_teams(["Alpha"]).run_id == public_result.run_id
    with pytest.raises(FileExistsError):
        export_run(public_result, path)


def test_bundle_tamper_detection(public_result, tmp_path):
    """Reject changed table files and a modified in-memory source snapshot."""
    path = export_run(public_result, tmp_path / "run")
    manifest = json.loads((path / "manifest.json").read_text())
    target = path / manifest["tables"]["bridge"]["file"]
    with target.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_run(path)
    changed = deepcopy(public_result)
    changed.tables["source_snapshot"].iloc[0, 0] = "changed"
    with pytest.raises(ValueError, match="snapshot changed"):
        export_run(changed, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".run-*"))


def test_catalog_publication(public_result, complete_frame, tmp_path):
    """Publish validated declarations and resolve pinned dataset/model IDs."""
    path = publish_model(
        labor_cost_model(),
        tmp_path / "model.json",
        validated_result=public_result,
    )
    snapshot = DatasetSnapshot("payroll", "v1", complete_frame)
    complete_frame.to_parquet(tmp_path / "dataset.parquet", index=True)
    config = {
        "models": [
            {
                "id": "labor-cost",
                "version": "1.0.0",
                "path": path.name,
                "fingerprint": public_result.manifest["model_fingerprint"],
            }
        ],
        "datasets": [
            {
                "id": "payroll",
                "version": "v1",
                "path": "dataset.parquet",
                "fingerprint": snapshot.fingerprint,
            }
        ],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(config))
    catalog = Catalog(catalog_path)
    assert catalog.model("labor-cost", "1.0.0").spec == labor_cost_model()
    pd.testing.assert_frame_equal(
        catalog.dataset("payroll", "v1").frame, complete_frame
    )
    with pytest.raises(FileExistsError):
        publish_model(labor_cost_model(), path, validated_result=public_result)
    with pytest.raises(KeyError):
        catalog.model("labor-cost", "latest")
    config["datasets"][0]["fingerprint"] = "changed"
    catalog_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Dataset changed"):
        catalog.dataset("payroll", "v1")
