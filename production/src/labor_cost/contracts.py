"""Stable data, request and result contracts for all frontends."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
import pandas as pd

ENGINE_VERSION = "0.2.0"
BUNDLE_VERSION = 2


def frame_fingerprint(frame):
    """Hash ordered rows, index, column names and snapshot dtypes."""
    schema = json.dumps(
        {
            "columns": list(frame.columns),
            "dtypes": [str(d) for d in frame.dtypes],
            "index_names": list(frame.index.names),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = sha256(schema.encode())
    digest.update(
        pd.util.hash_pandas_object(frame, index=True)
        .to_numpy(dtype="uint64")
        .tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True, init=False)
class DatasetSnapshot:
    """A named dataset revision owning a defensive copy of source rows."""

    dataset_id: str
    version: str
    fingerprint: str
    _frame: pd.DataFrame = field(repr=False, compare=False)

    def __init__(self, dataset_id, version, frame):
        """Freeze a caller-supplied DataFrame without changing its contents."""
        if not all(
            isinstance(x, str) and x.strip() for x in (dataset_id, version)
        ):
            raise ValueError("Dataset ID and version must be nonempty strings")
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("DatasetSnapshot needs a pandas DataFrame")
        if not frame.columns.is_unique:
            raise ValueError("Dataset column names must be unique")
        copied = frame.copy(deep=True)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "fingerprint", frame_fingerprint(copied))
        object.__setattr__(self, "_frame", copied)

    @property
    def frame(self):
        """Return a copy; consumers cannot mutate the retained snapshot."""
        return self._frame.copy(deep=True)


@dataclass(frozen=True)
class AnalysisRequest:
    """Periods and explicit estimator/settings overrides for one run.

    Model and dataset versions are supplied as immutable inputs to
    ``analyze``. Team selectors belong to result views, never this request.
    None uses the model's declared exposure and completion policies.
    """

    period_a: str
    period_b: str
    source_cost_column: str | None = None
    exposure_column: str | None = None
    missing_economics: str | None = None
    references: dict | None = field(default=None, repr=False, compare=False)
    explainer: str = "exact"
    n_permutations: int = 256
    random_seed: int = 0


@dataclass(frozen=True)
class ResultView:
    """Existing contributions with separate organizational/local scopes."""

    run_id: str
    teams: tuple[str, ...]
    organization_contributions: pd.DataFrame
    organization_scope: pd.DataFrame
    local_bridges: dict[str, pd.DataFrame]
    team_summary: pd.DataFrame


@dataclass
class AnalysisResult:
    """Versioned tables and metadata, independent of calculation internals.

    ``states`` are optional in-memory objects for analyst inspection and
    standalone costing. Exported runs contain portable tables and metadata,
    not Python state objects or executable code.
    """

    manifest: dict
    tables: dict[str, pd.DataFrame]
    states: tuple | None = field(default=None, repr=False)

    @property
    def run_id(self):
        """Return this run's immutable identifier."""
        return self.manifest["run_id"]

    def table(self, name):
        """Return a defensive copy of one named output or dataset table."""
        return self.tables[name].copy(deep=True)

    def select_teams(self, teams):
        """Select existing results without recomputing or renormalizing.

        Selected organizational contributions are not a bridge of the
        selected population. Global scale/mix rows are returned separately;
        each local bridge retains its own endpoints and allocations.
        """
        if isinstance(teams, str):
            raise ValueError("Pass a sequence of team names, not a string")
        selected = tuple(dict.fromkeys(teams))
        summary = self.tables["team_summary"]
        unknown = set(selected) - set(summary.team)
        if unknown:
            raise ValueError(f"Unknown teams: {sorted(unknown)}")
        bridge = self.tables["bridge"]
        return ResultView(
            self.run_id,
            selected,
            bridge.loc[bridge.team.isin(selected)].copy(),
            bridge.loc[bridge.team.isna()].copy(),
            {t: self.table("local_bridges/" + t) for t in selected},
            summary.loc[summary.team.isin(selected)].copy(),
        )

    def metadata(self):
        """Return a copy of run metadata for dashboard consumers."""
        return deepcopy(self.manifest)


def json_value(value):
    """Encode metadata and reference states without pickle or live objects.

    Tuple keys, arrays and timestamps retain type information so reference
    assumptions can be reconstructed independently from exported tables.
    Nonfinite optional metadata becomes null; numeric model validation is
    performed before serialization.
    """
    from .graph_cost import Profiles, Tensor

    if isinstance(value, Tensor):
        return {
            "type": "tensor",
            "dims": json_value(value.dims),
            "axes": json_value(value.axes),
            "data": json_value(value.data),
        }
    if isinstance(value, Profiles):
        return {
            "type": "profiles",
            "dims": json_value(value.dims),
            "axes": json_value(value.axes),
            "cells": json_value(value.cells),
            "employee_count": value.employee_count,
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "value": value.tolist(),
        }
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return {"type": "timestamp", "value": str(pd.Timestamp(value))}
    if isinstance(value, tuple):
        return {"type": "tuple", "value": [json_value(x) for x in value]}
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value):
            return {k: json_value(v) for k, v in value.items()}
        return {
            "type": "mapping",
            "value": [
                [json_value(k), json_value(v)] for k, v in value.items()
            ],
        }
    if isinstance(value, list):
        return [json_value(x) for x in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported metadata value: {type(value).__name__}")


def decode_value(value):
    """Restore values encoded by ``json_value`` without executing code."""
    if isinstance(value, list):
        return [decode_value(x) for x in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("type")
    if kind in {"tensor", "profiles"}:
        from .graph_cost import Profiles, Tensor

        data = {k: decode_value(v) for k, v in value.items() if k != "type"}
        return Tensor(**data) if kind == "tensor" else Profiles(**data)
    if kind == "ndarray" and set(value) == {"type", "dtype", "value"}:
        return np.asarray(value["value"], dtype=value["dtype"])
    if kind == "timestamp" and set(value) == {"type", "value"}:
        return pd.Timestamp(value["value"])
    if kind == "tuple" and set(value) == {"type", "value"}:
        return tuple(decode_value(x) for x in value["value"])
    if kind == "mapping" and set(value) == {"type", "value"}:
        return {decode_value(k): decode_value(v) for k, v in value["value"]}
    return {k: decode_value(v) for k, v in value.items()}
