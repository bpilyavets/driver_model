"""Exact additive organizational games over declared node hierarchies."""

from time import perf_counter

import pandas as pd

from .engine.common import check
from .graph import Switch
from .graph_attribution import _exact, _flatten, economic_games
from .hierarchy import cost_term, node_units, reconcile_nodes


def term_game(inputs, index, audits, owner, term, game):
    """Explain one additive term and measure its unit-cost multiplier."""
    switches = tuple(Switch(name) for name in inputs[0])

    def evaluate(state):
        """Switch the whole mix and take this child's term contribution."""
        mix = state.get("mix")
        share = 1.0 if mix is None else mix[index]
        return [cost_term(state.get("scale", 1.0), share, state["unit"])]

    def run(values, name):
        """Record exhaustive coalition evidence for the actual term game."""
        result = _exact(*values, switches, evaluate)
        audits.append(
            {
                "node_path": owner,
                "term_path": term,
                "game": name,
                **result["audit"],
            }
        )
        return result["impacts"]

    impacts = run(inputs, game)
    auxiliary = [
        dict(endpoint, unit=float(i)) for i, endpoint in enumerate(inputs)
    ]
    multiplier = run(auxiliary, game + "/multiplier/unit")["unit"]
    return impacts, multiplier


def _node_games(plan, states, request, audits, cells, checks):
    """Run each economic, organizational and local game once."""
    graph = plan.graph
    nodes = [s["nodes"] for s in states]
    units = [node_units(plan, s) for s in states]
    games = {g.name: g for g in graph.games}
    summaries = []
    economics, internal, local = {}, {}, {}
    leaf_keys = {}
    for path in sorted(nodes[0]):
        observations = [node[path] for node in nodes]
        exposure = [n["exposure"] for n in observations]
        unit = [v[path] for v in units]
        amounts = [cost_term(n, 1.0, u) for n, u in zip(exposure, unit)]
        summaries.append(
            {
                "node_path": path,
                "parent_path": path[:-1] if path else None,
                **dict(path),
                "depth": len(path),
                "exposure_A": exposure[0],
                "exposure_B": exposure[1],
                "period_A_cost": amounts[0],
                "period_B_cost": amounts[1],
                "cost_change": amounts[1] - amounts[0],
            }
        )
        if "key" in observations[0]:
            key = observations[0]["key"]
            leaf_keys[key] = path
            _, economics[path] = economic_games(
                plan, states, request, key, games, audits, cells
            )
        else:
            internal[path] = []
            for i, child in enumerate(observations[0]["children"]):
                inputs = [
                    {
                        "mix": n["mix"],
                        "unit": values[child],
                        **({"scale": e} if not path else {}),
                    }
                    for n, values, e in zip(observations, units, exposure)
                ]
                impacts, multiplier = term_game(
                    inputs,
                    i,
                    audits,
                    path,
                    child,
                    "organization" if not path else "conditional_mix",
                )
                checks.append(
                    check(
                        f"{path}/{child} unit multiplier",
                        impacts["unit"],
                        multiplier * (units[1][child] - units[0][child]),
                    )
                )
                internal[path].append((child, impacts, multiplier))
        if path:
            inputs = [dict(scale=e, unit=u) for e, u in zip(exposure, unit)]
            local[path] = term_game(inputs, 0, audits, path, path, "local")
    summary = pd.DataFrame(summaries)
    endpoints = {
        r["node_path"]: (r["period_A_cost"], r["period_B_cost"])
        for r in summaries
    }
    return economics, internal, local, leaf_keys, summary, endpoints


def _render_hierarchy(
    graph, order, dataset, endpoints, economics, internal, local
):
    """Flatten shared unit games with each reporting owner's multipliers."""
    games = {g.name: g for g in graph.games}
    rows = []

    def append(owner, node, path, driver, label, impact, leaf, game):
        """Store absolute organizational or owner-local currency amounts."""
        a, b = endpoints[owner]
        rows.append(
            {
                "dataset": dataset,
                "view": "local" if owner else "organization",
                "owner_path": owner,
                "node_path": node,
                "team": dict(node).get("team"),
                "line": dict(node).get("line"),
                "path": path,
                "parent_path": path[:-1],
                "parent": path[-2] if len(path) > 1 else None,
                "level": len(path) - 1,
                "driver": driver,
                "label": label,
                "impact": impact,
                "leaf": leaf,
                "game": game,
                "population": "all",
                "period_A_cost": a,
                "period_B_cost": b,
                "units": "currency",
            }
        )

    def expand(owner, node, prefix, multiplier):
        """Replace an economics branch by its scaled immediate children."""
        if node in economics:
            first = len(rows)
            results = economics[node]
            root = graph.team_game
            _flatten(
                games[root],
                results[root],
                results,
                games,
                rows,
                dataset,
                dict(node).get("team"),
                "local" if owner else "organization",
                endpoints[owner],
                multiplier,
                prefix,
                prefix[-1],
            )
            for row in rows[first:]:
                row.update(
                    owner_path=owner,
                    node_path=node,
                    level=len(row["path"]) - 1,
                    line=dict(node).get("line"),
                    parent_path=row["path"][:-1],
                )
            return
        terms = internal[node]
        dimension = order[len(node)]
        mix_id = (
            dimension + "_mix"
            if not node
            else dimension + "_mix_within_" + order[0]
        )
        if not node:
            append(
                owner,
                node,
                prefix + ("workforce_scale",),
                "workforce_scale",
                "Workforce scale",
                sum(v["scale"] for _, v, _ in terms),
                True,
                "organization",
            )
        label = dimension.capitalize() + " mix"
        if node:
            label += " within " + order[0]
        append(
            owner,
            node,
            prefix + (mix_id,),
            mix_id,
            label,
            multiplier * sum(v["mix"] for _, v, _ in terms),
            True,
            "mix",
        )
        for child, impacts, weight in terms:
            path = prefix + (child[-1],)
            append(
                owner,
                child,
                path,
                dimension + "_economics",
                dimension.capitalize() + " economics",
                multiplier * impacts["unit"],
                False,
                "economics",
            )
            expand(owner, child, path, multiplier * weight)

    expand((), (), (), 1.0)
    for owner, (impacts, multiplier) in local.items():
        append(
            owner,
            owner,
            ("scale",),
            "workforce_scale",
            "Workforce scale",
            impacts["scale"],
            True,
            "local",
        )
        append(
            owner,
            owner,
            ("unit",),
            "unit_economics",
            "Unit economics",
            impacts["unit"],
            False,
            "local",
        )
        expand(owner, owner, ("unit",), multiplier)

    return pd.DataFrame(rows)


def hierarchy_tables(plan, states, request, dataset, canonical):
    """Calculate node bridges, observed payroll checks and portable audits."""
    started = perf_counter()
    order = states[0]["hierarchy"]
    audits, cells, checks = [], [], []
    economics, internal, local, leaf_keys, summary, endpoints = _node_games(
        plan, states, request, audits, cells, checks
    )
    reconcile_nodes(states, canonical, endpoints, checks)
    hierarchy = _render_hierarchy(
        plan.graph, order, dataset, endpoints, economics, internal, local
    )
    delta = hierarchy.period_B_cost - hierarchy.period_A_cost
    hierarchy["share_of_net_change"] = hierarchy.impact / delta.where(
        delta.abs() > 1e-6
    )
    for owner, group in hierarchy.groupby("owner_path", sort=False):
        child_sums = (
            group.groupby("parent_path", sort=False).impact.sum().to_dict()
        )
        for row in group.loc[~group.leaf].itertuples():
            checks.append(
                check(
                    f"{owner}/{row.path} replacement",
                    child_sums[row.path],
                    row.impact,
                )
            )
        a, b = endpoints[owner]
        checks.append(
            check(
                f"{owner} flattened bridge",
                group.loc[group.leaf].impact.sum(),
                b - a,
            )
        )
    leaves = hierarchy.loc[hierarchy.leaf]
    bridge = leaves.loc[leaves.view.eq("organization")].copy()
    drivers = bridge.drop_duplicates("driver").copy()
    drivers["impact"] = drivers.driver.map(
        bridge.groupby("driver").impact.sum()
    )
    drivers["share_of_net_change"] = drivers.impact / (
        drivers.period_B_cost - drivers.period_A_cost
    ).where(lambda x: x.abs() > 1e-6)
    for col in ("team", "line", "parent", "node_path", "parent_path", "path"):
        drivers[col] = None
    checks.append(
        check(
            "Organizational driver summary",
            drivers.impact.sum(),
            endpoints[()][1] - endpoints[()][0],
        )
    )
    for record in audits + cells:
        if "team" in record:
            path = leaf_keys[record["team"]]
            record.update(
                node_path=path,
                team=dict(path).get("team"),
                line=dict(path).get("line"),
            )
    tables = {
        "hierarchy": hierarchy,
        "node_summary": summary,
        "bridge": bridge,
        "diagnostics/node_mix": pd.DataFrame(
            [
                {
                    "period": state["period"],
                    "node_path": path,
                    "child_path": child,
                    "share": share,
                    "observed_exposure": state["nodes"][child]["exposure"],
                }
                for state in states
                for path, node in state["nodes"].items()
                if "children" in node
                for child, share in zip(node["children"], node["mix"])
            ]
        ),
        "driver_bridge": drivers,
        "local_bridges": leaves.loc[leaves.view.eq("local")].copy(),
        "coalition_audit": pd.DataFrame(audits),
        "diagnostics/cell_impacts": pd.DataFrame(cells),
    }
    if order == ("team",):
        tables["team_summary"] = summary.loc[summary.depth.eq(1)].copy()
        for row in tables["team_summary"].itertuples():
            tables["local_bridges/" + row.team] = (
                tables["local_bridges"]
                .loc[tables["local_bridges"].team.eq(row.team)]
                .copy()
            )
    return tables, checks, perf_counter() - started
