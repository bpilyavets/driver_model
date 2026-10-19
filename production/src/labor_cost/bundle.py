"""Portable, checked run bundles for dashboards without calculation imports."""

import json
import shutil
import tempfile
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pandas as pd

from .contracts import (
    BUNDLE_VERSION,
    AnalysisResult,
    decode_value,
    frame_fingerprint,
    json_value,
)
from .spec import ModelSpec


def _hash_file(path):
    """Compute an artifact checksum without loading a whole table into RAM."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_table(table):
    """Encode heterogeneous object columns explicitly for Parquet transport."""
    encoded = table.copy(deep=True)
    object_columns = []
    for name in table.columns:
        if table[name].dtype == object:
            object_columns.append(name)
            encoded[name] = table[name].map(
                lambda x: json.dumps(
                    json_value(x), ensure_ascii=False, allow_nan=False
                )
            )
    return encoded, object_columns


def export_run(result, destination):
    """Write a new run directory atomically, without overwriting prior runs.

    Parameters
    ----------
    result : AnalysisResult
        Validated analysis with source/canonical tables and provenance.
    destination : path-like
        A new directory. Existing destinations raise FileExistsError.

    Returns
    -------
    pathlib.Path
        Bundle path containing a JSON manifest and Parquet tables.

    Notes
    -----
    Nested object cells use documented JSON encoding. Internal Python
    states are not serialized. Temporary output is removed on failure.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".run-", dir=destination.parent))
    try:
        manifest = deepcopy(result.manifest)
        if (
            frame_fingerprint(result.tables["source_snapshot"])
            != (manifest["dataset"]["selected_fingerprint"])
        ):
            raise ValueError("Source snapshot changed after analysis")
        manifest["tables"] = {}
        for index, (name, table) in enumerate(sorted(result.tables.items())):
            filename = f"table_{index:04d}.parquet"
            portable, object_columns = _portable_table(table)
            path = temporary / filename
            portable.to_parquet(path, engine="pyarrow", index=True)
            manifest["tables"][name] = {
                "file": filename,
                "sha256": _hash_file(path),
                "rows": len(table),
                "columns": list(table.columns),
                "dtypes": {k: str(v) for k, v in table.dtypes.items()},
                "json_columns": object_columns,
            }
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        # Recheck to avoid replacing an existing empty directory.
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination


def load_run(source):
    """Read a supported bundle, verifying checksums and dataset identity.

    Return portable tables and metadata with ``states=None``. Reject
    unsupported schemas, missing/changed files and paths outside the bundle.
    Loading never executes model code or recomputes attribution.
    """
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text("utf-8"))
    if manifest.get("bundle_version") not in {1, BUNDLE_VERSION}:
        raise ValueError("Unsupported run bundle version")
    spec = ModelSpec.from_dict(manifest["model"])
    if spec.fingerprint != manifest["model_fingerprint"]:
        raise ValueError("Model specification fingerprint mismatch")
    tables = {}
    for name, entry in manifest["tables"].items():
        path = (source / entry["file"]).resolve()
        if path.parent != source or path.suffix != ".parquet":
            raise ValueError("Table path must stay inside the run bundle")
        if _hash_file(path) != entry["sha256"]:
            raise ValueError(f"Table checksum mismatch: {name}")
        table = pd.read_parquet(path, engine="pyarrow")
        if (
            len(table) != entry["rows"]
            or list(table.columns) != entry["columns"]
        ):
            raise ValueError(f"Table schema mismatch: {name}")
        for column in entry["json_columns"]:
            table[column] = pd.Series(
                [decode_value(json.loads(x)) for x in table[column]],
                index=table.index,
                dtype=object,
            )
        tables[name] = table
    if (
        frame_fingerprint(tables["source_snapshot"])
        != (manifest["dataset"]["selected_fingerprint"])
    ):
        raise ValueError("Dataset snapshot fingerprint mismatch")
    return AnalysisResult(manifest, tables)
