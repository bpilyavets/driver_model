"""Adapt fixed-grain records and build comparison-aware generic states."""

from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd

from .engine.inputs import month_start
from .expressions import monomials
from .graph_cost import Profiles, Tensor, validate_factors


def row_evaluate(expression, frame):
    """Evaluate declared actual payroll before exposure normalization."""
    result = np.zeros(len(frame))
    for coefficient, names in monomials(expression):
        term = np.full(len(frame), coefficient)
        for name in names:
            term = term * frame[name].to_numpy(dtype=float)
        result += term
    if not np.isfinite(result).all():
        raise ValueError("Row payroll contains nonfinite values")
    return result


def population_masks(graph, frame):
    """Select declared populations from canonical categorical columns."""
    masks = {"all": pd.Series(True, index=frame.index)}
    for population in graph.populations:
        mask = masks["all"].copy()
        for name, values in population.where:
            if name not in frame:
                raise ValueError(
                    f"population {population.name}: missing {name}"
                )
            mask &= frame[name].isin(values)
        masks[population.name] = mask
    return masks


def adapt_graph(graph, frame, request):
    """Copy selected source records, validate their grain and retain bases.

    Applicable monetary values are signed. Exposure is assigned once and
    source/row payroll remains unnormalized; state builders convert actual
    amounts only at declared payment/profile boundaries.
    """
    schema = graph.source
    selected = frame.copy(deep=True)
    mapping = {
        "period": schema.period,
        "employee_id": schema.employee,
        "employment_type": schema.employment_type,
        "team": schema.team,
    }
    missing = set(mapping.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source key columns: {sorted(missing)}")
    canonical = pd.DataFrame({k: frame[v] for k, v in mapping.items()})
    canonical["period"] = pd.to_datetime(canonical.period)
    if (
        canonical.period.dt.tz is not None
        or not canonical.period.dt.is_month_start.all()
    ):
        raise ValueError("Periods must be timezone-naive month starts")
    if not (canonical.period == canonical.period.dt.normalize()).all():
        raise ValueError(
            "Periods must be month starts without time components"
        )
    for name in ("employee_id", "employment_type", "team"):
        if (
            not canonical[name]
            .map(lambda x: isinstance(x, str) and bool(x.strip()))
            .all()
        ):
            raise ValueError(f"{name}: identifiers must be nonempty strings")
    keys = ["period", "employee_id", "employment_type"]
    if canonical.duplicated(keys).any():
        raise ValueError("Duplicate employee/employment-type/month rows")
    if (canonical.groupby(keys[:2]).team.nunique() != 1).any():
        raise ValueError("Each employee-month must have exactly one team")
    for name, source in schema.dimensions:
        if isinstance(source, (list, tuple)):
            import json

            if set(source).issubset(selected.columns):
                components = selected[list(source)]
                canonical[name] = components.apply(
                    lambda row: json.dumps(row.tolist(), ensure_ascii=False),
                    axis=1,
                )
                canonical.loc[components.isna().any(axis=1), name] = None
            continue
        if source in selected:
            canonical[name] = selected[source]
    masks = population_masks(graph, canonical)
    for item in graph.fields:
        active = masks[item.population]
        if not active.any():
            canonical[item.name] = np.nan
            continue
        if item.source not in frame:
            raise ValueError(
                f"field {item.name}: missing source {item.source}"
            )
        values = pd.to_numeric(frame.loc[active, item.source], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(
                f"field {item.name}: missing/nonfinite applicable input"
            )
        if item.constraint == "positive" and (values <= 0).any():
            raise ValueError(f"field {item.name}: expected positive values")
        if item.constraint == "nonnegative" and (values < 0).any():
            raise ValueError(f"field {item.name}: expected nonnegative values")
        canonical[item.name] = np.nan
        canonical.loc[active, item.name] = values
    counts = canonical.groupby(keys[:2]).employment_type.transform("nunique")
    exposure_column = request.exposure_column or schema.exposure
    if exposure_column is None:
        canonical["exposure"] = 1.0 / counts
    else:
        if exposure_column not in frame:
            raise ValueError(f"Missing exposure column {exposure_column}")
        exposure = pd.to_numeric(frame[exposure_column], errors="raise")
        if (
            not np.isfinite(exposure).all()
            or ((exposure <= 0) | (exposure > 1)).any()
        ):
            raise ValueError("Supplied exposure must be finite and in (0, 1]")
        canonical["exposure"] = exposure
    canonical["employment_type_count"] = counts
    canonical["base_pay_factor"] = 1 / canonical.exposure
    canonical["row_cost"] = 0.0
    coverage = np.zeros(len(canonical), dtype=int)
    for component in graph.row_costs:
        mask = masks[component.population]
        coverage += mask.to_numpy(dtype=int)
        canonical.loc[mask, "row_cost"] += row_evaluate(
            component.formula, canonical.loc[mask]
        )
    if (coverage == 0).any():
        raise ValueError("Row-cost populations do not cover every source row")
    if request.source_cost_column is None:
        canonical["source_cost"] = canonical.row_cost
        canonical["reference_basis"] = "reconstructed declared payroll"
    else:
        source = pd.to_numeric(
            frame[request.source_cost_column], errors="raise"
        )
        if not np.isfinite(source).all():
            raise ValueError("Independent source costs must be finite")
        canonical["source_cost"] = source
        canonical["reference_basis"] = "independent source cost"
    # Validate every partition before any counterfactual games are built.
    for factor in graph.factors:
        active = masks[factor.population]
        dimensions = factor.given + factor.dimensions
        if factor.kind != "population_mix":
            for name in dimensions:
                if active.any() and (
                    name not in canonical
                    or canonical.loc[active, name].isna().any()
                ):
                    raise ValueError(
                        f"{factor.name}: missing applicable dimension {name}"
                    )
        else:
            members = sum(masks[x].astype(int) for x in factor.members)
            if (members.loc[active] != 1).any():
                raise ValueError(
                    f"{factor.name}: partition must be disjoint and exhaustive"
                )
    canonical = canonical.reset_index(drop=True)
    return canonical, {
        "policy": "equal_split" if exposure_column is None else "supplied",
        "exposure_column": exposure_column,
        "source_grain": "employee/employment_type/month",
    }


def _sorted(values):
    """Sort categorical axes deterministically across data types."""
    return tuple(sorted(set(values), key=lambda x: (type(x).__name__, str(x))))


def _filter(frame, dimensions, cell):
    """Select a categorical cell without mutating its source frame."""
    mask = pd.Series(True, index=frame.index)
    for name, category in zip(dimensions, cell):
        mask &= frame[name].eq(category)
    return frame.loc[mask]


def _donor(frame, dimensions, cell, fallbacks, context, diagnostics):
    """Find same-team/period support using explicit ordered dimension pools."""
    labels = dict(zip(dimensions, cell))
    for number, pool in enumerate((dimensions,) + tuple(fallbacks)):
        donor = _filter(frame, pool, tuple(labels[x] for x in pool))
        if not donor.empty:
            if number:
                diagnostics.append(
                    {
                        **context,
                        "cell": labels,
                        "fallback_dimensions": pool,
                        "donor_rows": len(donor),
                        "source_period": context["period"],
                    }
                )
            return donor
    raise ValueError(
        f"{context}: unsupported cell {labels}; declare fallback or reference"
    )


def _alignment(graph, frame, masks, periods):
    """Align scoped employee/type observations across the full organization."""
    vectors, axes, records = {}, {}, []
    for factor in graph.factors:
        if factor.kind != "aligned":
            continue
        population = frame.loc[masks[factor.population]]
        ids = _sorted(zip(population.employee_id, population.employment_type))
        axes[factor.population] = ids
        observations = {
            (row.period, row.employee_id, row.employment_type): row
            for row in population.itertuples(index=False)
        }
        for index, period in enumerate(periods):
            values = []
            for employee, employment_type in ids:
                key = (period, employee, employment_type)
                actual = key in observations
                donor_period = period if actual else periods[1 - index]
                donor = observations[(donor_period, employee, employment_type)]
                numeric = getattr(donor, factor.source)
                values.append(numeric)
                records.append(
                    {
                        "period": period,
                        "population": factor.population,
                        "factor": factor.name,
                        "employee_id": employee,
                        "employment_type": employment_type,
                        "value": numeric,
                        "source_period": donor_period,
                        "source_team": donor.team,
                        "method": "observed"
                        if actual
                        else "same_person_carry",
                    }
                )
            vectors[(period, factor.name)] = Tensor(
                ("employee:" + factor.population,), (ids,), np.asarray(values)
            )
    return vectors, axes, records


def _build_factor(
    factor, rows, combined, axes, context, vectors, identities, diagnostics
):
    """Build one complete, typed state object from its declared population."""
    dims = factor.given + factor.dimensions
    if factor.kind == "aligned":
        return vectors[(context["period"], factor.name)]
    if factor.kind == "total":
        if not dims:
            return Tensor.scalar(rows.exposure.sum())
        amounts = [
            rows.loc[rows[dims[0]].eq(x)].exposure.sum() for x in axes[dims[0]]
        ]
        return Tensor(dims, tuple(axes[x] for x in dims), np.array(amounts))
    if factor.kind == "population_mix":
        values = []
        for member in factor.members:
            values.append(
                rows.loc[rows["__population_" + member]].exposure.sum()
            )
        return Tensor(
            factor.dimensions,
            (factor.members,),
            np.array(values) / rows.exposure.sum(),
        )
    if factor.kind == "distribution":
        shape = tuple(len(axes[x]) for x in dims)
        data = np.zeros(shape)
        context_cells = product(*(axes[x] for x in factor.given))
        for context_cell in context_cells:
            donor = _donor(
                rows,
                factor.given,
                context_cell,
                factor.fallbacks,
                context,
                diagnostics,
            )
            denominator = donor.exposure.sum()
            prefix = tuple(
                axes[x].index(v) for x, v in zip(factor.given, context_cell)
            )
            for cell in product(*(axes[x] for x in factor.dimensions)):
                subset = _filter(donor, factor.dimensions, cell)
                suffix = tuple(
                    axes[x].index(v) for x, v in zip(factor.dimensions, cell)
                )
                data[prefix + suffix] = subset.exposure.sum() / denominator
        return Tensor(dims, tuple(axes[x] for x in dims), data)
    cells = tuple(product(*(axes[x] for x in dims)))
    output = []
    for cell in cells:
        if factor.kind == "policy":
            donor = _filter(combined, dims, cell)
            values = donor[factor.source].dropna().unique()
            if len(values) != 1:
                raise ValueError(
                    f"{factor.name}: inconsistent or missing fixed policy"
                )
            output.append(float(values[0]))
            continue
        donor = _donor(
            rows, dims, cell, factor.fallbacks, context, diagnostics
        )
        weights = donor.exposure.to_numpy() / donor.exposure.sum()
        if factor.kind == "payment":
            amounts = row_evaluate(factor.formula, donor)
            if factor.basis == "actual":
                amounts = amounts / donor.exposure.to_numpy()
            output.append(float(np.dot(weights, amounts)))
        elif factor.kind == "membership":
            index = {
                key: i for i, key in enumerate(identities[factor.population])
            }
            indices = np.array(
                [
                    index[x]
                    for x in zip(donor.employee_id, donor.employment_type)
                ],
                dtype=int,
            )
            conversion = (
                1 / donor.exposure.to_numpy()
                if factor.basis == "actual"
                else np.ones(len(donor))
            )
            output.append((indices, weights, conversion))
        else:
            raise ValueError(f"Unknown factor kind {factor.kind}")
    labels = tuple(axes[x] for x in dims)
    if factor.kind == "membership":
        return Profiles(
            dims, labels, tuple(output), len(identities[factor.population])
        )
    return Tensor(
        dims, labels, np.array(output).reshape(tuple(map(len, labels)))
    )


def build_states(plan, frame, periods, request):
    """Build and complete A/B states before invoking the pure cost engine.

    References use ``(period, team) -> population -> factor dictionary``;
    ``team_state`` supplies a complete team economics state. They apply
    only where economics are absent and must match the typed state schema.
    """
    graph = plan.graph
    masks = population_masks(graph, frame)
    working = frame.copy(deep=True)
    for name, mask in masks.items():
        working["__population_" + name] = mask
    vectors, identities, alignment = _alignment(graph, working, masks, periods)
    teams = _sorted(working.team)
    states, fallbacks, assumptions = [], [], []
    population_names = tuple(
        dict.fromkeys(["all"] + [x.population for x in graph.games])
    )
    for period in periods:
        rows = working.loc[working.period.eq(period)]
        state = {
            "period": period,
            "teams": {},
            "factors": {},
            "exposure_by_team": {
                t: float(rows.loc[rows.team.eq(t)].exposure.sum())
                for t in teams
            },
        }
        for factor in graph.factors:
            if factor.scope == "organization":
                # A completely empty organization has no identified mix.
                if rows.empty and factor.kind == "distribution":
                    continue
                state["factors"][factor.name] = _build_factor(
                    factor,
                    rows,
                    working,
                    {"team": teams},
                    {"period": period, "team": None, "factor": factor.name},
                    vectors,
                    identities,
                    fallbacks,
                )
        for team in teams:
            team_rows = rows.loc[rows.team.eq(team)]
            combined_team = working.loc[working.team.eq(team)]
            team_state = {"populations": {}}
            for population in population_names:
                subset = team_rows.loc[team_rows["__population_" + population]]
                combined = combined_team.loc[
                    combined_team["__population_" + population]
                ]
                if subset.empty:
                    team_state["populations"][population] = None
                    continue
                factors = [
                    x
                    for x in graph.factors
                    if x.scope == "team" and x.population == population
                ]
                dimensions = {
                    d
                    for f in factors
                    if f.kind != "population_mix"
                    for d in f.given + f.dimensions
                }
                axes = {d: _sorted(combined[d]) for d in dimensions}
                values = {}
                for factor in factors:
                    context = {
                        "period": period,
                        "team": team,
                        "population": population,
                        "factor": factor.name,
                    }
                    values[factor.name] = _build_factor(
                        factor,
                        subset,
                        combined,
                        axes,
                        context,
                        vectors,
                        identities,
                        fallbacks,
                    )
                validate_factors(factors, values, graph.fields)
                team_state["populations"][population] = values
            state["teams"][team] = team_state
        states.append(state)
    references = {}
    for key, supplied in (request.references or {}).items():
        references[(month_start(key[0]), key[1])] = supplied
    missing = request.missing_economics or graph.missing_economics
    if missing not in {"carry_observed", "require_reference"}:
        raise ValueError("Unknown missing-economics policy")
    for index, state in enumerate(states):
        other = states[1 - index]
        for factor in graph.factors:
            if (
                factor.scope == "organization"
                and factor.name not in state["factors"]
            ):
                state["factors"][factor.name] = deepcopy(
                    other["factors"][factor.name]
                )
                assumptions.append(
                    {
                        "period": state["period"],
                        "team": None,
                        "population": "all",
                        "method": "carry_organization_mix",
                        "source_period": other["period"],
                    }
                )
        for team in teams:
            supplied = references.get((state["period"], team), {})
            target = state["teams"][team]
            donor = other["teams"][team]
            whole_absent = state["exposure_by_team"][team] == 0
            if whole_absent and "team_state" in supplied:
                state["teams"][team] = deepcopy(supplied["team_state"])
                assumptions.append(
                    {
                        "period": state["period"],
                        "team": team,
                        "population": "all",
                        "method": "explicit_team_reference",
                        "source_period": None,
                    }
                )
                continue
            for population in population_names:
                if target["populations"][population] is not None:
                    continue
                original = donor["populations"].get(population)
                explicit = supplied.get(population)
                if original is None and explicit is None:
                    continue
                if explicit is not None:
                    replacement, method, source_period = (
                        explicit,
                        "explicit_reference",
                        None,
                    )
                elif missing == "require_reference":
                    raise ValueError(
                        "Missing economics require a reference: "
                        f"{state['period']}, "
                        f"{team}, {population}"
                    )
                else:
                    replacement, method, source_period = (
                        original,
                        "carry_observed",
                        other["period"],
                    )
                target["populations"][population] = deepcopy(replacement)
                assumptions.append(
                    {
                        "period": state["period"],
                        "team": team,
                        "population": population,
                        "method": method,
                        "source_period": source_period,
                    }
                )
    for team in teams:
        for population in population_names:
            specs = [
                f
                for f in graph.factors
                if f.scope == "team" and f.population == population
            ]
            pair = [
                s["teams"][team]["populations"].get(population) for s in states
            ]
            for effective in pair:
                if effective is None:
                    continue
                if set(effective) != {f.name for f in specs}:
                    raise ValueError(
                        f"Incomplete reference: {team}/{population}"
                    )
                validate_factors(specs, effective, graph.fields)
            if any(x is None for x in pair):
                continue
            for factor in specs:
                left, right = [x[factor.name] for x in pair]
                if left.dims != right.dims or left.axes != right.axes:
                    raise ValueError(
                        f"Reference axes differ for {team}/{factor.name}"
                    )
                if factor.kind == "policy" and not np.array_equal(
                    left.data, right.data
                ):
                    raise ValueError(
                        f"Fixed policy differs across periods: {factor.name}"
                    )
    return tuple(states), {
        "fallbacks": pd.DataFrame(fallbacks),
        "employee_alignment": pd.DataFrame(alignment),
        "assumptions": pd.DataFrame(assumptions),
    }
