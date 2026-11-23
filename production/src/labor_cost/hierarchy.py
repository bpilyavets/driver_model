"""Organizational paths, comparison support and pure hierarchical costing."""

import json

import numpy as np

from .engine.common import check
from .engine.inputs import month_start
from .graph_cost import team_values


def leaf_paths(frame, order):
    """Index rows by a stable economic identity and retain display order."""
    paths, identities, keys = {}, {}, []
    for values in frame[list(order)].itertuples(index=False, name=None):
        if values not in identities:
            path = tuple(zip(order, values))
            key = (values[0] if order == ("team",) else
                   json.dumps(sorted(path), ensure_ascii=False))
            identities[values] = key
            paths[key] = path
        keys.append(identities[values])
    return keys, paths


def split_references(references, paths, periods):
    """Separate missing leaf economics from whole conditional mix tables."""
    lookup = {path: key for key, path in paths.items()}
    parents = {path[:i] for path in lookup for i in range(len(path))}
    leaves, mixes = {}, {}
    for (period, path), reference in (references or {}).items():
        period = month_start(period)
        if period not in periods:
            raise ValueError(f"Reference outside comparison: {period}")
        if path in lookup:
            if "node_state" in reference:
                if len(reference) != 1:
                    raise ValueError("Supply a complete node_state on its own")
                reference = {"team_state": reference["node_state"]}
            leaves[period, lookup[path]] = reference
        elif path in parents and set(reference) == {"mix"}:
            mixes[period, path] = reference["mix"]
        else:
            raise ValueError(f"Unknown hierarchy reference: {path!r}")
    return leaves, mixes


def probability(values, size):
    """Require a finite, complete probability vector on the child axis."""
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (size,)
        or not np.isfinite(array).all()
        or (array < 0).any()
        or not np.isclose(array.sum(), 1.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("Invalid organizational share vector")
    return tuple(array.tolist())


def attach_nodes(states, paths, order, mixes, policy, assumptions):
    """Construct exposure and conditional mixes for the union of paths."""
    children = {}
    for path in sorted(paths.values()):
        for i in range(len(path)):
            children.setdefault(path[:i], set()).add(path[: i + 1])
    for state in states:
        state["hierarchy"] = order
        nodes = state["nodes"] = {}
        for key, path in paths.items():
            nodes[path] = {
                "key": key,
                "exposure": state["exposure_by_team"][key],
            }
        for path in sorted(children, key=len, reverse=True):
            axis = tuple(sorted(children[path]))
            weights = [nodes[child]["exposure"] for child in axis]
            total = sum(weights)
            nodes[path] = {
                "children": axis,
                "exposure": total,
                "mix": tuple(w / total for w in weights) if total else None,
            }
    for index, state in enumerate(states):
        for path in children:
            node = state["nodes"][path]
            reference = mixes.get((state["period"], path))
            if node["mix"] is not None:
                if reference is not None:
                    raise ValueError(f"Reference for observed mix: {path!r}")
                continue
            donor = states[1 - index]
            labels = [child[-1][1] for child in node["children"]]
            if reference is not None:
                if set(reference) != set(labels):
                    raise ValueError(f"Incomplete mix reference at {path!r}")
                mix = [reference[label] for label in labels]
                method, source = "explicit_node_mix", None
            elif policy == "require_reference":
                raise ValueError(
                    f"Missing node mix requires a reference: {path}"
                )
            else:
                mix = donor["nodes"][path]["mix"]
                method, source = "carry_node_mix", donor["period"]
            node["mix"] = probability(mix, len(labels))
            assumptions.append(
                {
                    "period": state["period"],
                    "node_path": path,
                    "team": None,
                    "population": "all",
                    "method": method,
                    "source_period": source,
                    "value": dict(zip(labels, node["mix"])),
                }
            )


def cost_term(scale, share, unit):
    """Cost one additive term, retaining signed economics and zero mass."""
    if not np.isfinite(scale) or scale < 0 or not 0 <= share <= 1:
        raise ValueError("Invalid organizational scale or share")
    if unit is None:
        if share and scale:
            raise ValueError("Positive exposure needs unit economics")
        return 0.0
    amount = scale * share * unit
    if not np.isfinite(amount):
        raise ValueError("Nonfinite organizational cost")
    return amount


def node_units(plan, state):
    """Evaluate observed or completed node economics without attribution."""
    nodes = state["nodes"]
    units = {}
    for path in sorted(nodes, key=len, reverse=True):
        node = nodes[path]
        if "key" in node:
            factors = state["teams"][node["key"]]
            value = team_values(plan, factors)[plan.graph.team_game]
            units[path] = None if value is None else float(value.data)
        else:
            axis = node["children"]
            mix = probability(node["mix"], len(axis))
            units[path] = sum(
                cost_term(1.0, weight, units[child])
                for weight, child in zip(mix, axis)
            )
    return units


def reconcile_nodes(states, canonical, endpoints, checks):
    """Independently compare every node with actual rows and its children."""
    order = states[0]["hierarchy"]
    for idx, state in enumerate(states):
        selected = canonical.loc[canonical.period.eq(state["period"])]
        for depth in range(len(order) + 1):
            dims = list(order[:depth])
            totals = (
                selected.groupby(dims)[["row_cost", "source_cost"]].sum()
                if depth
                else selected[["row_cost", "source_cost"]].sum()
            )
            for path, amounts in endpoints.items():
                if len(path) != depth:
                    continue
                values = tuple(v for _, v in path)
                key = values[0] if depth == 1 else values
                observed = (
                    totals
                    if not depth
                    else totals.loc[key]
                    if key in totals.index
                    else {"row_cost": 0.0, "source_cost": 0.0}
                )
                for basis in ("row_cost", "source_cost"):
                    checks.append(
                        check(
                            f"{state['period']}/{path}/{basis}",
                            amounts[idx],
                            observed[basis],
                        )
                    )
                if "children" in state["nodes"][path]:
                    children = state["nodes"][path]["children"]
                    checks.append(
                        check(
                            f"{state['period']}/{path}/children",
                            sum(endpoints[c][idx] for c in children),
                            amounts[idx],
                        )
                    )
