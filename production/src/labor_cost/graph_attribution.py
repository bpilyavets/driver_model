"""Generic hierarchical games over declared atomic state objects."""

from hashlib import sha256
from time import perf_counter

import numpy as np
import pandas as pd

from .engine.attribution import (
    permutation_game,
    switch_model,
    team_random_seed,
    validate_explainer_options,
)
from .engine.common import check
from .graph_compile import refs
from .graph_cost import (
    Tensor,
    evaluate,
    graph_cost,
    team_values,
    validate_factors,
)


def _exact(a, b, switches, engine):
    """Evaluate exact scalar/cell attribution with one shared coalition set."""
    import shap

    n = len(switches)
    if not 1 <= n <= 15:
        raise ValueError(
            "Exact games support 1–15 players; select permutation for leaves "
            "or simplify an upper game"
        )
    started = perf_counter()
    model, seen, extrema = switch_model(
        a, b, switches, engine, vector_output=True
    )
    background = np.zeros((1, n))
    explainer = shap.ExactExplainer(
        model,
        shap.maskers.Independent(background),
        feature_names=[s.name for s in switches],
    )
    explanation = explainer(np.ones((1, n)), max_evals=2**n, batch_size=2**n)
    impacts = np.asarray(explanation.values[0]).reshape(n, -1)
    base, end = model(background)[0], model(np.ones((1, n)))[0]
    check("Exact cell efficiency", impacts.sum(axis=0), end - base)
    check("Exact base", explanation.base_values[0], base)
    if len(seen) != 2**n:
        raise AssertionError("Exact game did not cover every coalition")
    return {
        "impacts": pd.Series(
            impacts.sum(axis=1), index=[s.name for s in switches]
        ),
        "cell_impacts": impacts,
        "base": float(base.sum()),
        "end": float(end.sum()),
        "audit": {
            "drivers": n,
            "method": "exact",
            "exhaustive": True,
            "n_permutations": None,
            "random_seed": None,
            "evaluation_budget": 2**n,
            "model_evaluations": model.evaluations,
            "coalitions": len(seen),
            "minimum_cost": extrema[0],
            "maximum_cost": extrema[1],
            "extrema_scope": "all binary coalitions",
            "seconds": perf_counter() - started,
            "passed": True,
        },
    }


def _seed(seed, game, team):
    """Use stable scoped seeds and preserve the original template's stream."""
    if game == "permanent":
        return team_random_seed(seed, team)
    payload = f"{seed}:{game}:{team}".encode()
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def _endpoint(game, factors, children):
    """Package whole objects under their explicit switch identities."""
    result = {}
    for switch in game.switches:
        if switch.child is not None:
            item = children[switch.child]
            result[switch.name] = None if item is None else float(item.data)
        else:
            result[switch.name] = {
                name: factors[name] for name in switch.factors
            }
    return result


def _engine(
    game,
    fixed,
    children,
    factors,
    cells=False,
    team_axis=None,
    team_index=None,
    fields=(),
):
    """Bind one scoped game without retaining comparison caches in states."""
    needed = set(refs(game.formula, "value"))
    specs = [x for x in factors if x.name in needed]
    cache = {}

    def cost(switched):
        """Validate whole objects and evaluate their cost contributions."""
        effective, child_values = dict(fixed), dict(children)
        for switch in game.switches:
            if switch.name not in switched:
                continue
            item = switched[switch.name]
            if switch.child is not None:
                if team_axis is not None:
                    values = np.zeros(len(team_axis))
                    values[team_index] = 0 if item is None else item
                    child_values[switch.child] = Tensor(
                        ("team",), (team_axis,), values
                    )
                else:
                    child_values[switch.child] = (
                        None if item is None else Tensor.scalar(item)
                    )
            else:
                effective.update(item)
        validate_factors(specs, effective, fields)
        expr = game.formula
        if cells and expr.op == "sum":
            result = evaluate(expr.args[0], effective, child_values, cache)
        else:
            result = evaluate(expr, effective, child_values, cache)
        if not hasattr(cost, "cell_coordinates"):
            cost.cell_coordinates = [
                {
                    d: result.axes[j][i]
                    for j, (d, i) in enumerate(zip(result.dims, index))
                }
                for index in np.ndindex(result.data.shape)
            ]
        return result.data.ravel()

    cost.aligned_cache = cache
    return cost


def _run(game, a, b, engine, request, team, leaf):
    """Run a selected leaf estimator, exact parents and exact multipliers."""
    switches = tuple(
        s
        for s in game.switches
        if not (
            s.child is not None and a[s.name] is None and b[s.name] is None
        )
    )
    if not switches:
        raise ValueError(f"game {game.name}: no active players")
    if leaf and request.explainer == "permutation":
        result = permutation_game(
            a,
            b,
            switches,
            engine,
            n_permutations=request.n_permutations,
            random_seed=_seed(request.random_seed, game.name, team),
        )
    else:
        result = _exact(a, b, switches, engine)
    result["audit"]["aligned_cache_entries"] = len(engine.aligned_cache)
    result["multipliers"] = {}
    result["auxiliary"] = {}
    for switch in switches:
        if switch.child is None:
            continue
        left, right = dict(a), dict(b)
        left[switch.name], right[switch.name] = 0.0, 1.0
        auxiliary = _exact(left, right, switches, engine)
        multiplier = float(auxiliary["impacts"][switch.name])
        delta = float(b[switch.name] or 0) - float(a[switch.name] or 0)
        check(
            f"{game.name}/{switch.name} parent replacement",
            result["impacts"][switch.name],
            multiplier * delta,
        )
        result["multipliers"][switch.name] = multiplier
        result["auxiliary"][switch.name] = auxiliary
    return result


def decompose_graph(plan, states, request, dataset):
    """Evaluate scoped games once and assemble both reconciled bridge views."""
    validate_explainer_options(
        request.explainer, request.n_permutations, request.random_seed
    )
    graph = plan.graph
    a, b = states
    teams = tuple(a["teams"])
    games = {g.name: g for g in graph.games}
    results, audits, cell_rows = {}, [], []
    endpoint_costs = (graph_cost(plan, a), graph_cost(plan, b))
    for team in teams:
        endpoints = [team_values(plan, s["teams"][team]) for s in states]
        results[team] = {}
        for name in plan.order:
            game = games[name]
            if all(c[name] is None for c in endpoints):
                continue
            factors_a, factors_b = [
                s["teams"][team]["populations"][game.population]
                for s in states
            ]
            if factors_a is None or factors_b is None:
                raise ValueError(f"game {name}: economics were not completed")
            left = _endpoint(game, factors_a, endpoints[0])
            right = _endpoint(game, factors_b, endpoints[1])
            leaf = not refs(game.formula, "child")
            engine = _engine(
                game,
                factors_a,
                endpoints[0],
                graph.factors,
                cells=leaf,
                fields=graph.fields,
            )
            game_result = _run(game, left, right, engine, request, team, leaf)
            results[team][name] = game_result
            _record_audits(audits, team, name, game_result)
            for i, switch_name in enumerate(game_result["impacts"].index):
                for cell, impact in enumerate(game_result["cell_impacts"][i]):
                    cell_rows.append(
                        {
                            "team": team,
                            "game": name,
                            "switch": switch_name,
                            "cell": cell,
                            "coordinates": engine.cell_coordinates[cell],
                            "impact_unit": float(impact),
                            "method": game_result["audit"]["method"],
                        }
                    )
        for view, game in (
            ("organization", graph.organization),
            ("local", graph.local),
        ):
            factors_a, factors_b = [s["factors"] for s in states]
            left = _endpoint(game, factors_a, endpoints[0])
            right = _endpoint(game, factors_b, endpoints[1])
            engine = _engine(
                game,
                factors_a,
                endpoints[0],
                graph.factors,
                team_axis=teams,
                team_index=teams.index(team),
                fields=graph.fields,
            )
            game_result = _run(game, left, right, engine, request, team, False)
            results[team][view] = game_result
            _record_audits(audits, team, view, game_result)
    hierarchy, summary = [], []
    for team in teams:
        org, local = results[team]["organization"], results[team]["local"]
        check(f"{team} A team cost", org["base"], local["base"])
        check(f"{team} B team cost", org["end"], local["end"])
        summary.append(
            {
                "team": team,
                "period_A_cost": local["base"],
                "period_B_cost": local["end"],
                "cost_change": local["end"] - local["base"],
                "exposure_A": a["exposure_by_team"][team],
                "exposure_B": b["exposure_by_team"][team],
            }
        )
        for view, root in (
            ("organization", graph.organization),
            ("local", graph.local),
        ):
            costs = (
                endpoint_costs
                if view == "organization"
                else (local["base"], local["end"])
            )
            _flatten(
                root,
                results[team][view],
                results[team],
                games,
                hierarchy,
                dataset,
                team,
                view,
                costs,
                1.0,
                (view, team),
                None,
            )
    table = pd.DataFrame(hierarchy)
    organization = table.loc[table.view.eq("organization") & table.leaf].copy()
    # Sum additive term impacts for complete organization factors.
    global_rows = organization.loc[organization.team.isna()]
    local_rows = organization.loc[organization.team.notna()]
    collapsed = []
    for _, rows in global_rows.groupby("driver", sort=False):
        row = rows.iloc[0].to_dict()
        row["impact"] = float(rows.impact.sum())
        row["path"] = ("organization", row["driver"])
        collapsed.append(row)
    organization = pd.concat(
        [pd.DataFrame(collapsed), local_rows], ignore_index=True
    )
    check(
        "Organization flattened efficiency",
        organization.impact.sum(),
        endpoint_costs[1] - endpoint_costs[0],
    )
    bridges = {}
    for team in teams:
        bridge = table.loc[
            table.view.eq("local") & table.team.eq(team) & table.leaf
        ].copy()
        local = results[team]["local"]
        check(
            f"{team} local flattened efficiency",
            bridge.impact.sum(),
            local["end"] - local["base"],
        )
        bridges[team] = bridge
    drivers = []
    for driver, rows in organization.groupby("driver", sort=False):
        row = rows.iloc[0].to_dict()
        row.update(driver=driver, team=None, impact=float(rows.impact.sum()))
        drivers.append(row)
    driver_bridge = pd.DataFrame(drivers)
    for frame in [organization, driver_bridge, table, *bridges.values()]:
        delta = frame.period_B_cost - frame.period_A_cost
        frame["share_of_net_change"] = frame.impact / delta.where(
            delta.abs() > 1e-6
        )
    return {
        "bridge": organization,
        "driver_bridge": driver_bridge,
        "hierarchy": table,
        "team_summary": pd.DataFrame(summary),
        "coalition_audit": pd.DataFrame(audits),
        "diagnostics/cell_impacts": pd.DataFrame(cell_rows),
        **{"local_bridges/" + team: frame for team, frame in bridges.items()},
    }, results


def _record_audits(rows, team, name, result):
    """Keep actual estimator and auxiliary-game metadata."""
    rows.append({"team": team, "game": name, **result["audit"]})
    for child, auxiliary in result["auxiliary"].items():
        rows.append(
            {
                "team": team,
                "game": name + "/multiplier/" + child,
                **auxiliary["audit"],
            }
        )


def _flatten(
    game,
    result,
    all_results,
    games,
    rows,
    dataset,
    team,
    view,
    costs,
    multiplier,
    path,
    parent,
):
    """Propagate child impacts without dividing by parent changes."""
    switches = {s.name: s for s in game.switches}
    for name, impact in result["impacts"].items():
        switch = switches[name]
        driver = getattr(switch, "driver_id", None) or game.name + "::" + name
        is_child = switch.child is not None
        current = path + (name,)
        global_row = view == "organization" and parent is None and not is_child
        rows.append(
            {
                "dataset": dataset,
                "view": view,
                "team": None if global_row else team,
                "driver": driver,
                "label": switch.label or name,
                "game": game.name,
                "population": game.population,
                "path": current,
                "parent": parent,
                "level": len(path) - 1,
                "leaf": not is_child,
                "impact": multiplier * float(impact),
                "period_A_cost": costs[0],
                "period_B_cost": costs[1],
                "units": "currency",
            }
        )
        if is_child:
            nested = all_results.get(switch.child)
            if nested is None:
                raise ValueError(f"Missing active child result {switch.child}")
            _flatten(
                games[switch.child],
                nested,
                all_results,
                games,
                rows,
                dataset,
                team,
                view,
                costs,
                multiplier * result["multipliers"][name],
                current,
                driver,
            )
