"""Public-result assembly for the common compiled graph runtime."""

from importlib.metadata import version
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from .contracts import (
    ENGINE_VERSION,
    AnalysisResult,
    frame_fingerprint,
    json_value,
)
from .engine.common import check
from .engine.inputs import month_start
from .graph_attribution import decompose_graph
from .graph_cost import Profiles, Tensor, graph_cost
from .graph_states import (
    adapt_graph,
    build_states,
    population_masks,
    row_evaluate,
)


def _state_tables(plan, states, diagnostic):
    """Expose typed state values and sparse memberships as portable tables."""
    values, profiles, aligned = [], [], []
    assumptions = diagnostic["assumptions"].to_dict("records")
    matching = {
        (
            x["period"],
            x["population"],
            x["factor"],
            x["employee_id"],
            x["employment_type"],
        ): x
        for x in diagnostic["employee_alignment"].to_dict("records")
    }
    specs = {f.name: f for f in plan.graph.factors}
    for state in states:
        scopes = [(None, "all", state["factors"])]
        for team, team_state in state["teams"].items():
            scopes.extend(
                (team, pop, factors)
                for pop, factors in team_state["populations"].items()
                if factors is not None
            )
        for team, population, factors in scopes:
            completion = _completion(
                state["period"], team, population, assumptions
            )
            employee_axis = next(
                (
                    x.axes[0]
                    for x in factors.values()
                    if isinstance(x, Tensor)
                    and x.dims
                    and x.dims[0].startswith("employee:")
                ),
                (),
            )
            for name, item in factors.items():
                context = {
                    "period": state["period"],
                    "team": team,
                    "population": population,
                    "factor": name,
                    **completion,
                }
                if specs[name].kind == "aligned":
                    for identity, amount in zip(item.axes[0], item.data):
                        employee, employment_type = identity
                        source = matching.get(
                            (
                                completion["source_period"],
                                population,
                                name,
                                employee,
                                employment_type,
                            ),
                            {},
                        )
                        explicit = completion["completion_method"].startswith(
                            "explicit"
                        )
                        aligned.append(
                            {
                                **context,
                                "driver": name,
                                "employee_id": employee,
                                "employment_type": employment_type,
                                "value": float(amount),
                                "source_period": None
                                if explicit
                                else source.get("source_period"),
                                "source_team": None
                                if explicit
                                else source.get("source_team"),
                                "method": "explicit_reference"
                                if explicit
                                else source.get("method"),
                            }
                        )
                if isinstance(item, Tensor):
                    for index in np.ndindex(item.data.shape):
                        coordinates = {
                            d: item.axes[j][i]
                            for j, (d, i) in enumerate(zip(item.dims, index))
                        }
                        values.append(
                            {
                                **context,
                                "coordinates": coordinates,
                                "value": float(item.data[index]),
                            }
                        )
                elif isinstance(item, Profiles):
                    shape = tuple(len(a) for a in item.axes)
                    for cell, contents in zip(np.ndindex(shape), item.cells):
                        coordinates = {
                            d: item.axes[j][i]
                            for j, (d, i) in enumerate(zip(item.dims, cell))
                        }
                        indices, weights, conversions = contents
                        for index, weight, conversion in zip(
                            indices, weights, conversions
                        ):
                            profiles.append(
                                {
                                    **context,
                                    "coordinates": coordinates,
                                    "employee_index": int(index),
                                    "employee_id": employee_axis[index][0],
                                    "employment_type": employee_axis[index][1],
                                    "weight": float(weight),
                                    "base_pay_factor": float(conversion),
                                }
                            )
    return {
        "diagnostics/state_factors": pd.DataFrame(values),
        "diagnostics/membership_profiles": pd.DataFrame(profiles),
        "diagnostics/aligned_inputs": pd.DataFrame(aligned),
    }


def _completion(period, team, population, assumptions):
    """Identify the completed economics supplying a scoped state's values."""
    for record in assumptions:
        if record["period"] != period or record["team"] != team:
            continue
        if (
            record["population"] == population
            or record["method"] == "explicit_team_reference"
        ):
            return {
                "completion_method": record["method"],
                "source_period": record["source_period"],
            }
    return {"completion_method": "observed", "source_period": period}


def analyze_graph(compiled, dataset, request):
    """Run a compiled graph and return reconciled, portable analysis data.

    The full organization is selected by month before adapting any data.
    All game/state diagnostics are derived from actual calculations; team
    selection remains an operation on the resulting tables.
    """
    started = perf_counter()
    graph = compiled.plan.graph
    periods = (month_start(request.period_a), month_start(request.period_b))
    frame = dataset.frame
    if graph.source.period not in frame:
        raise ValueError("Missing mapped period column")
    dates = pd.to_datetime(frame[graph.source.period])
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError(
            "Source periods must be nonmissing and timezone-naive"
        )
    selected = frame.loc[
        dates.dt.to_period("M").isin([p.to_period("M") for p in periods])
    ].copy()
    if selected.empty:
        raise ValueError("No source records for the selected periods")
    canonical, exposure = adapt_graph(graph, selected, request)
    states, diagnostic = build_states(
        compiled.plan, canonical, periods, request
    )
    for state in states:
        state["_model_fingerprint"] = compiled.fingerprint
        state["_graph_state"] = True
    validation = []
    for period, state in zip(periods, states):
        rows = canonical.loc[canonical.period.eq(period)]
        validation.append(
            check(
                f"{period} aggregate versus row payroll",
                graph_cost(compiled.plan, state),
                rows.row_cost.sum(),
            )
        )
        validation.append(
            check(
                f"{period} source payroll",
                rows.row_cost.sum(),
                rows.source_cost.sum(),
            )
        )
        for (team, kind), group in rows.groupby(["team", "employment_type"]):
            validation.append(
                check(
                    f"{period}/{team}/{kind} row/source",
                    group.row_cost.sum(),
                    group.source_cost.sum(),
                )
            )
    tables, games = decompose_graph(
        compiled.plan, states, request, dataset.dataset_id
    )
    for row in tables["team_summary"].itertuples(index=False):
        for period, value in zip(
            periods, (row.period_A_cost, row.period_B_cost)
        ):
            actual = canonical.loc[
                canonical.period.eq(period) & canonical.team.eq(row.team),
                "source_cost",
            ].sum()
            validation.append(
                check(f"{period}/{row.team} aggregate/source", value, actual)
            )
    validation.append(
        check(
            "Organization bridge",
            tables["bridge"].impact.sum(),
            graph_cost(compiled.plan, states[1])
            - graph_cost(compiled.plan, states[0]),
        )
    )
    validation.append(
        check(
            "Driver summary",
            tables["driver_bridge"].impact.sum(),
            tables["bridge"].impact.sum(),
        )
    )
    for row in tables["team_summary"].itertuples(index=False):
        validation.append(
            check(
                f"{row.team} local bridge",
                tables["local_bridges/" + row.team].impact.sum(),
                row.cost_change,
            )
        )
    if compiled.spec.graph is None:
        tables.update(
            _legacy_tables(compiled, states, games, dataset.dataset_id)
        )
    tables.update(_state_tables(compiled.plan, states, diagnostic))
    exported = canonical.copy(deep=True)
    masks = population_masks(graph, canonical)
    for factor in graph.factors:
        if factor.kind != "payment":
            continue
        active = masks[factor.population]
        normalized = row_evaluate(factor.formula, canonical.loc[active])
        if factor.basis == "actual":
            normalized = normalized / canonical.loc[active, "exposure"]
        exported["per_exposure__" + factor.name] = np.nan
        exported.loc[active, "per_exposure__" + factor.name] = normalized
    tables.update(
        {
            "source_snapshot": selected,
            "canonical_data": exported,
            "validation": pd.DataFrame(validation),
            "assumptions": diagnostic.pop("assumptions"),
        }
    )
    tables.update(
        {"diagnostics/" + name: data for name, data in diagnostic.items()}
    )
    tables["diagnostics/exposure_allocation"] = canonical[
        [
            "period",
            "employee_id",
            "employment_type",
            "team",
            "employment_type_count",
            "exposure",
            "base_pay_factor",
        ]
    ].copy()
    manifest = {
        "bundle_version": 2,
        "engine_version": ENGINE_VERSION,
        "run_id": str(uuid4()),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": compiled.spec.to_dict(),
        "model_fingerprint": compiled.fingerprint,
        "compiled_graph": compiled.plan.inspect(),
        "dataset": {
            "id": dataset.dataset_id,
            "version": dataset.version,
            "fingerprint": dataset.fingerprint,
            "selected_fingerprint": frame_fingerprint(selected),
        },
        "periods": [p.isoformat() for p in periods],
        "settings": {
            "source_cost_column": request.source_cost_column,
            "exposure_column": request.exposure_column
            or graph.source.exposure,
            "missing_economics": request.missing_economics
            or graph.missing_economics,
            "references": json_value(request.references),
            "explainer": request.explainer,
            "n_permutations": request.n_permutations,
            "random_seed": request.random_seed,
        },
        "exposure_config": exposure,
        "elapsed_seconds": perf_counter() - started,
        "dependencies": {
            n: version(n) for n in ("numpy", "pandas", "shap", "pyarrow")
        },
        "semantics": {
            "organization": "Full organization; team selection is a view",
            "local": "Separate local team bridge",
            "source_snapshot": "Unchanged actual source amounts",
            "canonical_data": "Mapped inputs and named per-exposure amounts",
            "state_factors": "Per-exposure payments; observed aligned inputs",
            "attribution": "Accounting attribution, not a causal estimate",
        },
    }
    return AnalysisResult(json_value(manifest), tables, states)


def _legacy_tables(compiled, states, results, dataset):
    """Render v1 table shapes from generic game results, without new games."""
    from .engine.attribution import assemble_bridges

    permanent = {
        t: r["permanent"] for t, r in results.items() if "permanent" in r
    }
    impacts = pd.Series(
        {"workforce_scale": 0.0, "team_mix": 0.0}, name="impact"
    )
    teams = {}
    for team, result in results.items():
        org, local, unit = (
            result["organization"],
            result["local"],
            result["team"],
        )
        for name in ("workforce_scale", "team_mix"):
            impacts.loc[name] += org["impacts"][name]
        impacts.loc["team::" + team] = org["impacts"]["team_economics"]
        teams[team] = {
            "unit": unit,
            "organization_term": org,
            "local": local,
            "M": org["multipliers"]["team_economics"],
            "L": local["multipliers"]["team_economics"],
            "K": unit["multipliers"].get("permanent_unit_cost", 0.0),
            "M_game": org["auxiliary"]["team_economics"],
            "L_game": local["auxiliary"]["team_economics"],
            "K_game": unit["auxiliary"].get("permanent_unit_cost"),
        }
    upper = {
        "impacts": impacts,
        "teams": teams,
        "base": graph_cost(compiled.plan, states[0]),
        "end": graph_cost(compiled.plan, states[1]),
    }
    rendered = assemble_bridges(
        {"_teams": tuple(teams)},
        {},
        permanent,
        upper,
        compiled.registry,
        dataset,
    )
    tables = {n: rendered[n] for n in ("bridge", "driver_bridge", "hierarchy")}
    tables.update(
        {"local_bridges/" + t: x for t, x in rendered["local_bridges"].items()}
    )
    summary = []
    for team, games in teams.items():
        local = games["local"]
        change = local["end"] - local["base"]
        summary.append(
            {
                "team": team,
                "period_a_cost": local["base"],
                "period_b_cost": local["end"],
                "local_change": change,
                "organization_economics_impact": impacts["team::" + team],
                "local_reconciliation_error": tables[
                    "local_bridges/" + team
                ].impact.sum()
                - change,
            }
        )
    tables["team_summary"] = pd.DataFrame(summary)
    return tables
