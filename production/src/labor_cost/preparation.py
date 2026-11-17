"""Explicit, audited source exclusions before state construction."""

import json
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .compiler import CompiledModel, compile_model
from .contracts import AnalysisRequest, DatasetSnapshot
from .engine.inputs import month_start
from .graph_states import adapt_graph, population_masks


class InputPreparationError(ValueError):
    """An exclusion-limit failure retaining the complete proposed audit."""

    def __init__(self, message, audit, preparation):
        """Retain evidence even though no prepared dataset is returned."""
        super().__init__(message)
        self._audit = deepcopy(audit)
        self._preparation = deepcopy(preparation)

    @property
    def audit(self):
        """Return independent copies of rejected-input audit tables."""
        return deepcopy(self._audit)

    @property
    def preparation(self):
        """Return the attempted preparation policy and exclusion counts."""
        return deepcopy(self._preparation)


@dataclass(frozen=True, init=False)
class PreparedInput(DatasetSnapshot):
    """A retained snapshot carrying input-quality evidence into analysis.

    Create through ``prepare_input``. ``frame`` and ``audit`` return copies.
    The snapshot fingerprint describes retained rows; ``preparation`` also
    identifies the original snapshot, model, periods and validation policy.
    """

    _audit: dict = field(repr=False, compare=False)
    _preparation: dict = field(repr=False, compare=False)

    def __init__(self, dataset, retained, audit, preparation):
        """Own retained source rows and their preparation evidence."""
        super().__init__(dataset.dataset_id, dataset.version, retained)
        object.__setattr__(self, "_audit", deepcopy(audit))
        object.__setattr__(self, "_preparation", deepcopy(preparation))

    @property
    def audit(self):
        """Return defensive copies of all input-quality tables."""
        return deepcopy(self._audit)

    @property
    def preparation(self):
        """Return policy, source identity, scope and exclusion counts."""
        return deepcopy(self._preparation)

    def _validate_context(self, compiled, request, *, standalone=False):
        """Prevent reuse under different preparation assumptions."""
        context = _context(compiled, request)
        original = self._preparation
        for key, value in context.items():
            if key == "periods" and standalone:
                matches = set(value).issubset(original[key])
            else:
                matches = value == original[key]
            if not matches:
                raise ValueError(
                    f"Prepared input {key} differs; prepare the raw snapshot "
                    "again for these settings"
                )


def _context(compiled, request):
    """Capture settings that determine which rows are valid or selected."""
    return {
        "model_fingerprint": compiled.fingerprint,
        "periods": sorted(
            {
                str(month_start(request.period_a)),
                str(month_start(request.period_b)),
            }
        ),
        "source_cost_column": request.source_cost_column,
        "exposure_column": (
            request.exposure_column or compiled.plan.graph.source.exposure
        ),
    }


def _blank(values):
    """Identify null and blank categorical inputs without coercing IDs."""
    return values.isna() | values.map(
        lambda x: isinstance(x, str) and not x.strip()
    )


def _categories(graph, data):
    """Map categories as in graph adaptation, including joint dimensions."""
    schema = graph.source
    categories = pd.DataFrame(
        {
            "employee_id": data[schema.employee],
            "employment_type": data[schema.employment_type],
            "team": data[schema.team],
        }
    )
    for name, source in schema.dimensions:
        columns = (
            list(source) if isinstance(source, (list, tuple)) else [source]
        )
        if not set(columns).issubset(data.columns):
            continue
        if isinstance(source, (list, tuple)):
            components = data[columns]
            missing = components.apply(_blank).any(axis=1)
            categories[name] = None
            categories.loc[~missing, name] = components.loc[~missing].apply(
                lambda row: json.dumps(row.tolist(), ensure_ascii=False),
                axis=1,
            )
        else:
            categories[name] = data[source]
    return categories


def _validate_grain(schema, data, dates):
    """Reject structural conflicts even if one conflicting row is defective."""
    keys = pd.DataFrame(
        {
            "period": dates.to_numpy(),
            "employee": data[schema.employee],
            "type": data[schema.employment_type],
            "team": data[schema.team],
        }
    )
    valid = keys[["employee", "type", "team"]].apply(
        lambda column: column.map(
            lambda x: isinstance(x, str) and bool(x.strip())
        )
    )
    keyed = keys.loc[valid.employee & valid.type]
    if keyed.duplicated(["period", "employee", "type"]).any():
        raise ValueError("Duplicate employee/employment-type/month rows")
    assigned = keys.loc[valid.employee & valid.team]
    if (assigned.groupby(["period", "employee"]).team.nunique() > 1).any():
        raise ValueError("Each employee-month must have exactly one team")


def prepare_input(model, dataset, request, *, missing_team=None):
    """Prepare comparison rows with a hard one-percent exclusion limit.

    Parameters
    ----------
    model : ModelSpec or CompiledModel
        Declared source mappings, populations and input constraints.
    dataset : DatasetSnapshot
        Original CSV or DWH snapshot, never modified in place.
    request : AnalysisRequest
        Selects comparison months, exposure and independent source costs.
    missing_team : str, optional
        Explicit replacement for null/blank teams, e.g. ``"Unassigned"``.
        Without this setting, such rows are invalid and count as exclusions.

    Returns
    -------
    PreparedInput
        A DatasetSnapshot accepted by ``analyze`` and ``build_state``.
        Analysis automatically retains its audit tables and provenance.
        A count/reason summary is always printed, without employee records.

    Raises
    ------
    InputPreparationError
        More than 1% of selected rows would be excluded. ``audit`` and
        ``preparation`` remain available on the exception for inspection.
    ValueError
        Invalid schema, periods, unknown population categories, duplicate
        keys or inconsistent team assignments; these are not auto-repaired.

    Notes
    -----
    Each excluded row counts once across both selected months. Exactly 1%
    is permitted. Outside months and team replacements do not count as
    exclusions. Applicable invalid inputs are never filled with zero.
    Default exposure is assigned after exclusions on the retained workforce.
    """
    if not isinstance(dataset, DatasetSnapshot):
        raise ValueError("Expected a versioned DatasetSnapshot")
    if isinstance(dataset, PreparedInput):
        raise ValueError("Prepare the original snapshot, not PreparedInput")
    if not isinstance(request, AnalysisRequest):
        raise ValueError("Expected AnalysisRequest")
    if missing_team is not None and (
        not isinstance(missing_team, str) or not missing_team.strip()
    ):
        raise ValueError("missing_team must be a nonempty string or None")
    compiled = (
        model if isinstance(model, CompiledModel) else compile_model(model)
    )
    graph = compiled.plan.graph
    schema = graph.source
    context = _context(compiled, request)
    raw = dataset.frame
    required = {
        schema.period,
        schema.employee,
        schema.employment_type,
        schema.team,
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source key columns: {sorted(missing)}")
    reserved = {"audit_source_row", "audit_reasons", "audit_assigned_team"}
    if reserved.intersection(raw.columns):
        raise ValueError("Source columns conflict with reserved audit columns")
    dates = pd.to_datetime(raw[schema.period], errors="raise")
    if dates.isna().any():
        raise ValueError("Missing periods must be corrected before selection")
    if dates.dt.tz is not None:
        raise ValueError("Source periods must be timezone-naive")
    selected = dates.dt.to_period("M").isin(
        pd.to_datetime(context["periods"]).to_period("M")
    )
    positions = np.flatnonzero(selected.to_numpy())
    original = raw.loc[selected].copy()
    if original.empty:
        raise ValueError("No rows in the requested comparison months")
    selected_dates = dates.loc[selected]
    if (
        not selected_dates.dt.is_month_start.all()
        or not selected_dates.eq(selected_dates.dt.normalize()).all()
    ):
        raise ValueError(
            "Periods must be month starts without time components"
        )
    # Positional processing also supports duplicate caller index labels.
    data = original.reset_index(drop=True)
    team_missing = _blank(data[schema.team])
    assigned_mask = (
        team_missing if missing_team is not None else team_missing & False
    )
    assigned = original.iloc[np.flatnonzero(assigned_mask)].copy()
    assigned.insert(0, "audit_source_row", positions[assigned_mask])
    assigned["audit_assigned_team"] = missing_team
    if missing_team is not None and team_missing.any():
        data[schema.team] = data[schema.team].astype(object)
        data.loc[team_missing, schema.team] = missing_team
    _validate_grain(schema, data, selected_dates)
    reasons = [[] for _ in range(len(data))]

    def flag(mask, reason):
        """Retain every reason while counting each excluded row only once."""
        for row in np.flatnonzero(np.asarray(mask)):
            reasons[row].append(reason)

    for column in (schema.employee, schema.employment_type, schema.team):
        valid = data[column].map(
            lambda x: isinstance(x, str) and bool(x.strip())
        )
        flag(~valid, f"{column}: missing or invalid identifier")
    categories = _categories(graph, data)
    unresolved_population = np.zeros(len(data), dtype=bool)
    for population in graph.populations:
        possible = pd.Series(True, index=data.index)
        missing_category = pd.Series(False, index=data.index)
        for name, values in population.where:
            if name not in categories:
                raise ValueError(
                    f"population {population.name}: missing {name}"
                )
            blank = _blank(categories[name])
            possible &= categories[name].isin(values) | blank
            missing_category |= blank
        unresolved_population |= (possible & missing_category).to_numpy()
        for name, _ in population.where:
            flag(
                possible & _blank(categories[name]),
                f"{name}: missing category",
            )
    masks = population_masks(graph, categories)
    covered = np.zeros(len(data), dtype=bool)
    for component in graph.row_costs:
        covered |= masks[component.population].to_numpy()
    valid_type = data[schema.employment_type].map(
        lambda x: isinstance(x, str) and bool(x.strip())
    )
    unknown = ~covered & ~unresolved_population & valid_type.to_numpy()
    if unknown.any():
        raise ValueError(
            f"{int(unknown.sum())} rows have unrecognized population "
            "categories; correct the population/source mapping"
        )
    for item in graph.fields:
        active = masks[item.population]
        if not active.any():
            continue
        if item.source not in data:
            raise ValueError(
                f"field {item.name}: missing source {item.source}"
            )
        values = pd.to_numeric(data[item.source], errors="coerce").astype(
            float
        )
        finite = np.isfinite(values)
        flag(active & ~finite, f"{item.name}: missing/nonfinite input")
        if item.constraint == "positive":
            flag(
                active & finite & values.le(0),
                f"{item.name}: expected positive value",
            )
        elif item.constraint == "nonnegative":
            flag(
                active & finite & values.lt(0),
                f"{item.name}: expected nonnegative value",
            )
    for population, dimension in sorted(
        {
            (factor.population, dim)
            for factor in graph.factors
            if factor.kind != "population_mix"
            for dim in factor.dimensions + factor.given
            if dim != "team"
        }
    ):
        active = masks[population]
        if not active.any():
            continue
        if dimension not in categories:
            raise ValueError(f"{population}: missing dimension {dimension}")
        flag(
            active & _blank(categories[dimension]),
            f"{population}/{dimension}: missing category",
        )
    for setting in ("exposure_column", "source_cost_column"):
        column = context[setting]
        if column is None:
            continue
        if column not in data:
            raise ValueError(f"Missing {setting}: {column}")
        values = pd.to_numeric(data[column], errors="coerce").astype(float)
        invalid = ~np.isfinite(values)
        if setting == "exposure_column":
            invalid |= values.le(0) | values.gt(1)
        flag(invalid, f"{setting}: invalid input")
    rejected = np.array([bool(items) for items in reasons])
    excluded = original.iloc[np.flatnonzero(rejected)].copy()
    excluded.insert(0, "audit_source_row", positions[rejected])
    excluded["audit_reasons"] = [
        "; ".join(dict.fromkeys(reasons[i])) for i in np.flatnonzero(rejected)
    ]
    retained = data.iloc[np.flatnonzero(~rejected)].copy()
    retained.index = original.index[~rejected]
    counts = (
        pd.DataFrame(
            {
                "period": selected_dates.to_numpy(),
                "employment_type": data[schema.employment_type].to_numpy(),
                "input_rows": 1,
                "excluded_rows": rejected.astype(int),
                "retained_rows": (~rejected).astype(int),
                "team_assignments": assigned_mask.to_numpy().astype(int),
            }
        )
        .groupby(["period", "employment_type"], dropna=False)
        .sum()
        .reset_index()
    )
    reason_counts = (
        pd.Series(
            [reason for items in reasons for reason in dict.fromkeys(items)],
            dtype="object",
        )
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="affected_rows")
    )
    audit = {
        "summary": counts,
        "reasons": reason_counts,
        "excluded_rows": excluded,
        "team_assignments": assigned,
        "source_snapshot": original,
    }
    preparation = {
        **context,
        "raw_fingerprint": dataset.fingerprint,
        "input_rows": len(data),
        "excluded_rows": len(excluded),
        "retained_rows": len(retained),
        "drop_fraction": len(excluded) / len(data),
        "max_drop_fraction": 0.01,
        "missing_team": missing_team,
        "team_assignments": len(assigned),
        "exposure_assignment": "after_exclusions",
    }
    print("Input preparation (reason counts can overlap):")
    print(counts.to_string(index=False))
    if not reason_counts.empty:
        print(reason_counts.to_string(index=False))
    print(
        f"Excluded {len(excluded)} of {len(data)} selected rows "
        f"({preparation['drop_fraction']:.3%}); limit 1%. "
        f"Retained {len(retained)}; team assignments: {len(assigned)}."
    )
    print(
        "Inspect .audit for source rows and reasons. Exposure is assigned "
        "after exclusions; analysis covers retained records only."
    )
    # Integer arithmetic makes the exactly-one-percent boundary unambiguous.
    if len(excluded) * 100 > len(data):
        raise InputPreparationError(
            f"Excluded {len(excluded)} of {len(data)} selected rows "
            f"({preparation['drop_fraction']:.3%}), exceeding the 1% limit; "
            "analysis stopped. Inspect this exception's .audit.",
            audit,
            preparation,
        )
    adapt_graph(graph, retained.reset_index(drop=True), request)
    return PreparedInput(dataset, retained, audit, preparation)
