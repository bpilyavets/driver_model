"""Configuration-backed published models and predefined dataset snapshots."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .compiler import CompiledModel, compile_model
from .contracts import DatasetSnapshot
from .spec import ModelSpec


def publish_model(model, destination, *, validated_result):
    """Publish a model JSON after a matching fixture analysis has passed.

    Publication requires a successful analysis of this exact specification.
    Run the analytical regression suite before releasing engine extensions.
    Existing files are never overwritten; use a new version/path instead.
    """
    compiled = (
        model if isinstance(model, CompiledModel) else compile_model(model)
    )
    if validated_result.manifest["model_fingerprint"] != compiled.fingerprint:
        raise ValueError("Validation result belongs to a different model")
    checks = validated_result.tables["validation"]
    if checks.empty or not checks.passed.all():
        raise ValueError("Publication requires passing analytical validation")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "spec": compiled.spec.to_dict(),
                "fingerprint": compiled.fingerprint,
                "validated_run": validated_result.run_id,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
    return destination


@dataclass(frozen=True)
class Catalog:
    """Resolve explicitly versioned model JSON and dataset Parquet entries.

    Configuration has ``models`` and ``datasets`` lists. Each entry has
    id, version, path and fingerprint. Paths are relative to the catalog.
    Loading a model never imports analyst Python; it validates declarations.
    """

    path: Path

    def _entries(self, kind):
        """Read entries and reject ambiguous version identifiers."""
        config = json.loads(Path(self.path).read_text("utf-8"))
        entries = config.get(kind, [])
        keys = [(e["id"], e["version"]) for e in entries]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate {kind} catalog versions")
        return entries

    def list_models(self):
        """Return available published model IDs and explicit versions."""
        return self._entries("models")

    def list_datasets(self):
        """Return available dataset IDs and immutable versions."""
        return self._entries("datasets")

    def _resolve(self, kind, identifier, version):
        """Resolve an explicit version without selecting latest silently."""
        for entry in self._entries(kind):
            if (entry["id"], entry["version"]) == (identifier, version):
                path = Path(self.path).resolve().parent / entry["path"]
                return entry, path
        raise KeyError(f"Unknown {kind} entry: {identifier}@{version}")

    def model(self, identifier, version):
        """Compile a published model after checking its identity."""
        entry, path = self._resolve("models", identifier, version)
        published = json.loads(path.read_text("utf-8"))
        compiled = compile_model(ModelSpec.from_dict(published["spec"]))
        if (compiled.spec.model_id, compiled.spec.version) != (
            identifier,
            version,
        ):
            raise ValueError("Published model identity differs from catalog")
        if not (
            compiled.fingerprint
            == entry["fingerprint"]
            == published["fingerprint"]
        ):
            raise ValueError("Published model fingerprint mismatch")
        return compiled

    def dataset(self, identifier, version):
        """Read an immutable dataset and verify its declared content hash."""
        entry, path = self._resolve("datasets", identifier, version)
        snapshot = DatasetSnapshot(identifier, version, pd.read_parquet(path))
        if snapshot.fingerprint != entry["fingerprint"]:
            raise ValueError("Dataset changed without a catalog revision")
        return snapshot
