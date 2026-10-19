"""Validate v2 graphs and produce inspectable execution dependencies."""

from dataclasses import dataclass

from .expressions import monomials
from .graph import Aggregate, ModelGraph

UNITS = {
    "dimensionless": (0, 0, 0),
    "currency": (1, 0, 0),
    "time": (0, 1, 0),
    "currency/time": (1, -1, 0),
    "exposure": (0, 0, 1),
    "currency/exposure": (1, 0, -1),
}
PER_EXPOSURE = (1, 0, -1)
EXPOSURE = (0, 0, 1)


def add_units(left, right):
    """Multiply units represented by base-dimension exponents."""
    return tuple(a + b for a, b in zip(left, right))


def row_units(expr, fields, path):
    """Check a row expression and return its physical units."""
    result = None
    for _, names in monomials(expr):
        units = (0, 0, 0)
        for name in names:
            if name not in fields:
                raise ValueError(f"{path}: unbound row field {name}")
            units = add_units(units, UNITS[fields[name].units])
        if result is not None and result != units:
            raise ValueError(f"{path}: incompatible row units")
        result = units
    return result or UNITS["currency"]


def refs(expr, kind):
    """Collect factor or child references without losing declaration order."""
    found = []
    if expr.op == kind:
        found.append(expr.args[0])
    for operand in expr.args:
        if isinstance(operand, Aggregate):
            found.extend(refs(operand, kind))
    if kind == "value" and expr.op == "aligned_mean":
        found.append(expr.args[1])
    return tuple(dict.fromkeys(found))


def child_terms(expr):
    """Expand child degrees to validate each nested scalar's affinity."""
    if expr.op == "child":
        return [{expr.args[0]: 1}]
    if expr.op in {"value", "constant"}:
        return [{}]
    if expr.op in {"sum", "select", "aligned_mean"}:
        return child_terms(expr.args[0])
    left, right = (child_terms(x) for x in expr.args)
    if expr.op == "add":
        return left + right
    terms = []
    for a in left:
        for b in right:
            term = dict(a)
            for name, degree in b.items():
                term[name] = term.get(name, 0) + degree
            terms.append(term)
    if len(terms) > 256:
        raise ValueError("Aggregate expression expands past 256 terms")
    return terms


def _unique(items, path):
    """Index named declarations, rejecting duplicates and unsafe names."""
    result = {}
    for item in items:
        if not isinstance(item.name, str) or not item.name.isidentifier():
            raise ValueError(f"{path}: names must be identifiers")
        if item.name in result:
            raise ValueError(f"{path}: duplicate name {item.name}")
        result[item.name] = item
    return result


@dataclass(frozen=True)
class GraphPlan:
    """Validated graph plus the dependency order for scalar team games."""

    graph: ModelGraph
    order: tuple

    def inspect(self):
        """Expose executable structure without a dataset or attribution."""
        return {
            "schema_version": 2,
            "source_grain": "employee/employment_type/month",
            "graph": self.graph.to_dict(),
            "game_order": list(self.order),
            "games": [
                {
                    "id": game.name,
                    "population": game.population,
                    "factors": list(refs(game.formula, "value")),
                    "children": list(refs(game.formula, "child")),
                    "switches": [s.name for s in game.switches],
                }
                for game in self.graph.games
            ],
        }


def compile_graph(graph):
    """Compile supported state and cost declarations into one graph plan.

    Parameters
    ----------
    graph : ModelGraph
        Closed, serializable declarations. Population coverage and source
        support are checked against data during state construction.

    Returns
    -------
    GraphPlan
        Typed graph and deterministic dependency order.

    Raises
    ------
    ValueError
        Unsupported units, dependencies, distributions, ownership or
        nonlinear nested costs, with the relevant declaration path.
    """
    if not isinstance(graph, ModelGraph):
        raise ValueError("Expected ModelGraph")
    populations = _unique(graph.populations, "populations")
    if "all" in populations:
        raise ValueError("Population 'all' is reserved for all source rows")
    fields = _unique(graph.fields, "fields")
    factors = _unique(graph.factors, "factors")
    games = _unique(graph.games, "games")
    if set(games) & {"organization", "local"}:
        raise ValueError(
            "games: organization and local are reserved root names"
        )
    reserved = {
        "period",
        "employee_id",
        "employment_type",
        "employment_type_count",
        "team",
        "exposure",
        "base_pay_factor",
        "source_cost",
        "row_cost",
        "reference_basis",
    }
    source_columns = (
        graph.source.employee,
        graph.source.employment_type,
        graph.source.period,
        graph.source.team,
    )
    if graph.source.exposure is not None:
        source_columns += (graph.source.exposure,)
    if not all(isinstance(x, str) and x.strip() for x in source_columns):
        raise ValueError("source: mappings need nonempty column names")
    dimensions = {"team", "employment_type"}
    pairs = graph.source.dimensions
    if len(dict(pairs)) != len(pairs):
        raise ValueError("source: duplicate dimension mappings")
    for name, source in pairs:
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name in reserved
        ):
            raise ValueError(
                "source: dimension names must not overwrite canonical keys"
            )
        columns = source if isinstance(source, tuple) else (source,)
        if not columns or not all(
            isinstance(x, str) and x.strip() for x in columns
        ):
            raise ValueError(f"source: dimension {name} needs source columns")
    dimensions.update(dict(pairs))
    for pop in populations.values():
        if not pop.where or any(k not in dimensions for k, _ in pop.where):
            raise ValueError(f"population {pop.name}: use mapped categories")
        if any(not values for _, values in pop.where):
            raise ValueError(f"population {pop.name}: empty predicate")
    allowed_populations = {"all", *populations}
    if set(fields) & (reserved | dimensions):
        raise ValueError("fields: names must not overwrite keys or dimensions")
    for item in fields.values():
        if not isinstance(item.source, str) or not item.source.strip():
            raise ValueError(f"field {item.name}: source column is required")
        if item.population not in allowed_populations:
            raise ValueError(f"field {item.name}: unknown population")
        if item.units not in UNITS or item.constraint not in {
            "signed",
            "positive",
            "nonnegative",
        }:
            raise ValueError(f"field {item.name}: invalid units/constraint")
    for row in graph.row_costs:
        if row.population not in allowed_populations:
            raise ValueError("row_costs: unknown population")
        if row_units(row.formula, fields, "row_costs") != UNITS["currency"]:
            raise ValueError("row_costs: actual payroll needs currency units")
    if not graph.row_costs:
        raise ValueError("row_costs: declare actual source-row payroll")
    for item in factors.values():
        path = f"factor {item.name}"
        if item.population not in allowed_populations:
            raise ValueError(f"{path}: unknown population")
        if item.scope not in {"team", "organization"}:
            raise ValueError(f"{path}: scope must be team or organization")
        if item.kind not in {
            "total",
            "distribution",
            "population_mix",
            "aligned",
            "membership",
            "payment",
            "policy",
        }:
            raise ValueError(f"{path}: unsupported state primitive")
        if item.basis not in {"actual", "per_exposure"}:
            raise ValueError(f"{path}: invalid amount basis")
        axes = item.given + item.dimensions
        if len(set(axes)) != len(axes):
            raise ValueError(f"{path}: cyclic or repeated dimensions")
        if item.kind == "population_mix":
            if len(item.dimensions) != 1 or item.given:
                raise ValueError(f"{path}: declare one partition axis")
            if not item.members or len(set(item.members)) != len(item.members):
                raise ValueError(f"{path}: declare unique population members")
            if set(item.members) - set(populations):
                raise ValueError(f"{path}: unknown partition member")
        elif set(axes) - dimensions:
            raise ValueError(f"{path}: unmapped dimension")
        if item.scope == "team" and "team" in axes:
            raise ValueError(f"{path}: team is the scope, not an inner axis")
        if item.scope == "organization" and (
            item.population != "all"
            or item.kind not in {"total", "distribution"}
            or set(axes) - {"team"}
        ):
            raise ValueError(f"{path}: organization factors use all teams")
        if item.kind == "distribution" and not item.dimensions:
            raise ValueError(f"{path}: distribution needs dimensions")
        if item.kind == "total" and (
            item.source not in {None, "exposure"} or set(axes) - {"team"}
        ):
            raise ValueError(f"{path}: totals sum workforce exposure")
        for fallback in item.fallbacks:
            support = item.given if item.kind == "distribution" else axes
            if set(fallback) - set(support):
                raise ValueError(f"{path}: fallback must drop cell dimensions")
        if item.kind in {"aligned", "policy"}:
            if item.source not in fields:
                raise ValueError(f"{path}: source must name a declared field")
            if fields[item.source].population not in {"all", item.population}:
                raise ValueError(f"{path}: field population is incompatible")
        if item.kind == "payment":
            expected = (
                UNITS["currency"] if item.basis == "actual" else PER_EXPOSURE
            )
            if (
                item.formula is None
                or row_units(item.formula, fields, path) != expected
            ):
                raise ValueError(
                    f"{path}: payment formula has wrong amount units"
                )
            for _, names in monomials(item.formula):
                if any(
                    fields[n].population not in {"all", item.population}
                    for n in names
                ):
                    raise ValueError(
                        f"{path}: row input population is incompatible"
                    )
    if graph.missing_economics not in {"carry_observed", "require_reference"}:
        raise ValueError("Unknown missing-economics policy")
    order, visiting = [], set()

    def visit(name):
        """Topologically order child games and reject dependency cycles."""
        if name not in games:
            raise ValueError(f"game {name}: unknown child")
        if name in visiting:
            raise ValueError(f"game {name}: cyclic game dependency")
        if name in order:
            return
        visiting.add(name)
        for target in refs(games[name].formula, "child"):
            visit(target)
        visiting.remove(name)
        order.append(name)

    visit(graph.team_game)
    if set(order) != set(games):
        raise ValueError("games: unused games must be removed or connected")
    parents = {}
    for name in order:
        game = games[name]
        if game.population not in allowed_populations:
            raise ValueError(f"game {name}: unknown population")
        _validate_game(game, factors, fields, games, PER_EXPOSURE)
        for target in refs(game.formula, "child"):
            if target in parents:
                raise ValueError(f"child {target}: ambiguous shared ownership")
            parents[target] = name
    ownership = {}
    for name in order:
        game = games[name]
        owned = set(refs(game.formula, "value"))
        for target in refs(game.formula, "child"):
            if owned.intersection(ownership[target]):
                raise ValueError(
                    f"{name}: ambiguous shared state in child games"
                )
            owned.update(ownership[target])
        ownership[name] = owned
    for game in (graph.organization, graph.local):
        _validate_game(game, factors, fields, games, UNITS["currency"], True)
        if refs(game.formula, "child") != (graph.team_game,):
            raise ValueError(f"game {game.name}: needs the team child family")
        if any(t != {graph.team_game: 1} for t in child_terms(game.formula)):
            raise ValueError(f"game {game.name}: require additive team costs")
    return GraphPlan(graph, tuple(order))


def _validate_game(game, factors, fields, games, expected, outer=False):
    """Check scalar result, units, complete switches and affine children."""
    path = f"game {game.name}"
    if any(
        any(n > 1 for n in term.values()) for term in child_terms(game.formula)
    ):
        raise ValueError(
            f"{path}: nonlinear child cost; use affine boundaries"
        )
    switches = _unique(game.switches, path + "/switches")
    claimed, child_claims = [], []
    for switch in switches.values():
        if switch.child is not None:
            if switch.factors:
                raise ValueError(f"{path}: a child switch cannot own factors")
            child_claims.append(switch.child)
        else:
            claimed.extend(switch.factors)
    if len(claimed) != len(set(claimed)):
        raise ValueError(f"{path}: a factor has multiple atomic owners")
    dependencies = set(refs(game.formula, "value"))
    if dependencies - set(factors):
        raise ValueError(
            f"{path}: unknown factors {dependencies - set(factors)}"
        )
    variable = {x for x in dependencies if factors[x].kind != "policy"}
    if (
        not variable.issubset(claimed)
        or set(claimed) - set(factors)
        or any(factors[x].kind == "policy" for x in claimed)
    ):
        raise ValueError(
            f"{path}: switches must cover complete varying factors"
        )
    children = refs(game.formula, "child")
    if len(child_claims) != len(set(child_claims)) or set(child_claims) != set(
        children
    ):
        raise ValueError(f"{path}: each child needs one explicit switch")
    for name in dependencies | set(claimed):
        item = factors[name]
        if not outer and item.population != game.population:
            raise ValueError(
                f"{path}: factor {name} has a different population"
            )
        if not outer and item.scope != "team":
            raise ValueError(
                f"{path}: team games cannot use organization state"
            )
    dims, units = _type(game.formula, factors, fields, games, outer)
    if dims or units != expected:
        raise ValueError(
            f"{path}: expected scalar {expected}, got {dims}, {units}"
        )
    # Conditional probability factors must form a directed factorization.
    edges, defined = {}, set()
    for name in dependencies:
        item = factors[name]
        if item.kind != "distribution":
            continue
        if defined.intersection(item.dimensions):
            raise ValueError(f"{path}: duplicated probability dimension")
        defined.update(item.dimensions)
        for dim in item.dimensions:
            edges[dim] = set(item.given)
    for dim in edges:
        seen, pending = set(), list(edges[dim])
        while pending:
            parent = pending.pop()
            if parent == dim:
                raise ValueError(f"{path}: cyclic distribution conditioning")
            if parent not in seen:
                seen.add(parent)
                pending.extend(edges.get(parent, ()))
        if seen - defined:
            raise ValueError(f"{path}: missing conditioning distribution")


def _type(expr, factors, fields, games, outer):
    """Infer named axes and units for each aggregate primitive."""
    op, args = expr.op, expr.args
    if op == "constant":
        from .graph import aggregate

        aggregate(args[0])
        return (), (0, 0, 0)
    if op == "child":
        if args[0] not in games:
            raise ValueError(f"Unknown child {args[0]}")
        return (("team",) if outer else ()), PER_EXPOSURE
    if op == "value":
        item = factors[args[0]]
        dims = item.given + item.dimensions
        if item.kind == "total":
            return dims, EXPOSURE
        if item.kind in {"distribution", "population_mix"}:
            return dims, (0, 0, 0)
        if item.kind == "payment":
            return dims, PER_EXPOSURE
        if item.kind == "membership":
            raise ValueError("Membership must be consumed by aligned_mean")
        units = UNITS[fields[item.source].units]
        if item.kind == "aligned":
            dims = ("employee:" + item.population,)
        return dims, units
    if op in {"add", "mul"}:
        (ld, lu), (rd, ru) = [
            _type(x, factors, fields, games, outer) for x in args
        ]
        if op == "add" and lu != ru:
            raise ValueError("Aggregate addition has incompatible units")
        return tuple(dict.fromkeys(ld + rd)), lu if op == "add" else add_units(
            lu, ru
        )
    dims, units = _type(args[0], factors, fields, games, outer)
    if op == "sum":
        if len(set(args[1])) != len(args[1]) or set(args[1]) - set(dims):
            raise ValueError("sum_over: missing or duplicate axis")
        if any(x.startswith("employee:") for x in args[1]):
            raise ValueError("Use aligned_mean to aggregate employee inputs")
        return tuple(x for x in dims if x not in args[1]), units
    if op == "select":
        if args[1] not in dims:
            raise ValueError("select: missing axis")
        return tuple(x for x in dims if x != args[1]), units
    if op == "aligned_mean":
        profile = factors.get(args[1])
        if profile is None or profile.kind != "membership":
            raise ValueError("aligned_mean needs a membership factor")
        if dims != ("employee:" + profile.population,):
            raise ValueError("aligned_mean requires jointly aligned inputs")
        expected = (
            UNITS["currency"] if profile.basis == "actual" else PER_EXPOSURE
        )
        if units != expected:
            raise ValueError(
                "aligned_mean input units differ from the profile basis"
            )
        return profile.dimensions, PER_EXPOSURE
    raise ValueError(f"Unsupported aggregate operation: {op}")
