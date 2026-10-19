"""Serializable declarations for population-scoped workforce cost graphs.

Row expressions use ``field``. Aggregate expressions use ``value`` and
``child``; reductions and aligned means preserve their named dimensions.
No declaration contains executable analyst Python.
"""

from dataclasses import asdict, dataclass

from .expressions import Expr


@dataclass(frozen=True)
class Aggregate:
    """Arithmetic over state factors and scalar child-game costs."""

    op: str
    args: tuple

    def __post_init__(self):
        """Freeze operands, including those restored from JSON."""
        args = tuple(self.args)
        if self.op == "sum" and len(args) == 2:
            args = (args[0], tuple(args[1]))
        object.__setattr__(self, "args", args)

    def __add__(self, other):
        """Add compatible aggregate quantities."""
        return Aggregate("add", (self, aggregate(other)))

    def __radd__(self, other):
        """Support Python's sum over expressions."""
        if isinstance(other, (int, float)) and other == 0:
            return self
        return aggregate(other) + self

    def __mul__(self, other):
        """Multiply quantities on compatible named axes."""
        return Aggregate("mul", (self, aggregate(other)))

    def __rmul__(self, other):
        """Support numeric coefficients."""
        return aggregate(other) * self

    def __neg__(self):
        """Negate signed costs without changing their meaning."""
        return self * -1

    def __sub__(self, other):
        """Subtract compatible quantities."""
        return self + -aggregate(other)

    def __rsub__(self, other):
        """Support subtraction from a numeric expression."""
        return aggregate(other) - self

    def to_dict(self):
        """Encode the expression without callbacks."""
        return {
            "op": self.op,
            "args": [
                x.to_dict() if isinstance(x, Aggregate) else x
                for x in self.args
            ],
        }

    @classmethod
    def from_dict(cls, data):
        """Decode only known expression operations."""
        if set(data) != {"op", "args"}:
            raise ValueError("Invalid aggregate expression")
        op, args = data["op"], data["args"]
        arities = {
            "constant": 1,
            "value": 1,
            "child": 1,
            "add": 2,
            "mul": 2,
            "sum": 2,
            "select": 3,
            "aligned_mean": 2,
        }
        if op not in arities or len(args) != arities[op]:
            raise ValueError(f"Unsupported aggregate operation: {op}")
        return cls(
            op,
            tuple(
                cls.from_dict(x) if isinstance(x, dict) else x for x in args
            ),
        )


def aggregate(value):
    """Coerce a finite scalar into aggregate arithmetic."""
    from .expressions import expression

    if isinstance(value, Aggregate):
        return value
    return Aggregate("constant", (expression(value).args[0],))


def value(name):
    """Refer to a declared state factor."""
    return Aggregate("value", (name,))


def child(name):
    """Refer to a declared child game's scalar cost."""
    return Aggregate("child", (name,))


def sum_over(expression, *dimensions):
    """Sum an aggregate expression over explicitly named axes."""
    return Aggregate("sum", (aggregate(expression), tuple(dimensions)))


def select(expression, dimension, category):
    """Select one category without splitting its distribution switch."""
    return Aggregate("select", (aggregate(expression), dimension, category))


def aligned_mean(expression, membership):
    """Aggregate employee products using one sparse membership collection.

    Conversion factors belong to membership and convert actual row payroll
    into currency per exposure. Inputs retain their observed values.
    """
    return Aggregate("aligned_mean", (aggregate(expression), membership))


@dataclass(frozen=True)
class SourceSchema:
    """Map the required row grain and optional categorical dimensions."""

    employee: str = "employee_id"
    employment_type: str = "employment_type"
    period: str = "period"
    team: str = "team"
    exposure: str | None = None
    dimensions: tuple = ()

    def __post_init__(self):
        """Freeze dimension name/source pairs."""
        pairs = self.dimensions
        if isinstance(pairs, dict):
            pairs = pairs.items()
        object.__setattr__(
            self,
            "dimensions",
            tuple(
                (name, tuple(source) if isinstance(source, list) else source)
                for name, source in pairs
            ),
        )


@dataclass(frozen=True)
class Population:
    """Select a population by categorical equality or membership.

    Predicates are ANDed. Values within a predicate are ORed. The implicit
    ``all`` population includes every source row.
    """

    name: str
    where: tuple = ()

    def __post_init__(self):
        """Normalize a mapping into immutable categorical predicates."""
        pairs = (
            self.where.items() if isinstance(self.where, dict) else self.where
        )
        normalized = []
        for key, values in pairs:
            if isinstance(values, str) or not isinstance(
                values, (tuple, list)
            ):
                values = (values,)
            normalized.append((key, tuple(values)))
        object.__setattr__(self, "where", tuple(normalized))


@dataclass(frozen=True)
class FieldSpec:
    """Bind a scoped row input with units and an applicable-value constraint.

    Units include currency, time, exposure and supported ratios. A field
    may appear repeatedly in formulas but is represented by one atomic aligned
    input in a game. Population restrictions also govern missing values.
    """

    name: str
    source: str
    population: str = "all"
    units: str = "currency"
    constraint: str = "signed"


@dataclass(frozen=True)
class RowCost:
    """An actual payroll expression applying to one population.

    Multiple components may overlap deliberately; their amounts add.
    Every observed row must be covered by at least one component.
    """

    population: str
    formula: Expr


@dataclass(frozen=True)
class Factor:
    """Describe a state object, independently of its attribution player.

    Kinds: total, distribution, population_mix, aligned, membership,
    payment, policy. Scope is team or organization. Distribution axes are
    ``given + dimensions``; each complete table is an atomic object.
    Sparse fallbacks are ordered subsets of the economic cell dimensions.
    An empty fallback sequence means unsupported cells must fail.
    """

    name: str
    kind: str
    population: str = "all"
    dimensions: tuple = ()
    given: tuple = ()
    source: str | None = None
    formula: Expr | None = None
    scope: str = "team"
    fallbacks: tuple = ()
    members: tuple = ()
    basis: str = "actual"

    def __post_init__(self):
        """Freeze dimension, partition and fallback declarations."""
        for name in ("dimensions", "given", "members"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self, "fallbacks", tuple(tuple(x) for x in self.fallbacks)
        )


@dataclass(frozen=True)
class Switch:
    """Switch complete factors together, or refine one child cost."""

    name: str
    factors: tuple = ()
    child: str | None = None
    label: str | None = None
    driver_id: str | None = None

    def __post_init__(self):
        """Freeze factors and default a simple switch to its own name."""
        factors = self.factors
        if not factors and self.child is None:
            factors = (self.name,)
        object.__setattr__(self, "factors", tuple(factors))


@dataclass(frozen=True)
class Game:
    """A scalar aggregate formula, complete switches and population scope.

    Nested child costs must enter affinely. Leaf games use the requested
    estimator; parents and their auxiliary multiplier games remain exact.
    """

    name: str
    formula: Aggregate
    switches: tuple
    population: str = "all"

    def __post_init__(self):
        """Freeze player order, which is part of model identity."""
        object.__setattr__(self, "switches", tuple(self.switches))


@dataclass(frozen=True)
class ModelGraph:
    """The complete v2 source, state, cost and attribution declarations.

    ``organization`` is homogeneous and affine in its team child family.
    ``local`` declares the local exposure/unit-cost game. Team children
    are evaluated with identical economics in both reporting views.
    """

    source: SourceSchema
    populations: tuple
    fields: tuple
    row_costs: tuple
    factors: tuple
    games: tuple
    team_game: str
    organization: Game
    local: Game
    missing_economics: str = "carry_observed"

    def __post_init__(self):
        """Freeze all declaration collections."""
        for name in ("populations", "fields", "row_costs", "factors", "games"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self):
        """Return the canonical declarative JSON payload."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Restore declarations using a closed grammar."""
        data = dict(data)
        data["source"] = SourceSchema(**data["source"])
        for key, constructor in (
            ("populations", Population),
            ("fields", FieldSpec),
        ):
            data[key] = tuple(constructor(**x) for x in data[key])
        data["row_costs"] = tuple(
            RowCost(x["population"], Expr.from_dict(x["formula"]))
            for x in data["row_costs"]
        )
        factors = []
        for item in data["factors"]:
            item = dict(item)
            if item["formula"] is not None:
                item["formula"] = Expr.from_dict(item["formula"])
            factors.append(Factor(**item))
        data["factors"] = tuple(factors)
        data["games"] = tuple(_read_game(x) for x in data["games"])
        for name in ("organization", "local"):
            data[name] = _read_game(data[name])
        return cls(**data)


def _read_game(data):
    """Decode a game including its switch and expression declarations."""
    data = dict(data)
    data["formula"] = Aggregate.from_dict(data["formula"])
    data["switches"] = tuple(Switch(**x) for x in data["switches"])
    return Game(**data)
