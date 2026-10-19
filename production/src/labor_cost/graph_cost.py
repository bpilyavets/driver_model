"""Pure named-axis cost evaluation, independent of history and SHAP."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tensor:
    """A finite numeric array whose axes have explicit category labels."""

    dims: tuple
    axes: tuple
    data: np.ndarray

    def __post_init__(self):
        """Validate axis identity and numeric shape at construction."""
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "axes", tuple(tuple(x) for x in self.axes))
        data = np.array(self.data, dtype=float, copy=True)
        if data.shape != tuple(len(x) for x in self.axes):
            raise ValueError("Tensor shape does not match declared axes")
        if len(self.dims) != len(set(self.dims)) or len(self.dims) != len(
            self.axes
        ):
            raise ValueError("Tensor has ambiguous axes")
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite aggregate state value")
        if any(len(set(labels)) != len(labels) for labels in self.axes):
            raise ValueError("Tensor category labels must be unique")
        data.setflags(write=False)
        object.__setattr__(self, "data", data)

    @classmethod
    def scalar(cls, value):
        """Wrap a finite scalar without categorical axes."""
        return cls((), (), np.asarray(value, dtype=float))

    def combine(self, other, operation):
        """Broadcast only along explicitly named, compatible axes."""
        dims = tuple(dict.fromkeys(self.dims + other.dims))
        axes = dict(zip(self.dims, self.axes))
        for dim, labels in zip(other.dims, other.axes):
            if dim in axes and axes[dim] != labels:
                raise ValueError(f"Incompatible categories on axis {dim}")
            axes[dim] = labels
        left, right = [x.broadcast(dims, axes) for x in (self, other)]
        return Tensor(
            dims, tuple(axes[x] for x in dims), operation(left, right)
        )

    def broadcast(self, dims, axes):
        """Align an array to a named union of axes without reweighting."""
        ordered = tuple(x for x in dims if x in self.dims)
        permutation = tuple(self.dims.index(x) for x in ordered)
        data = self.data.transpose(permutation) if permutation else self.data
        shape = tuple(len(axes[x]) if x in self.dims else 1 for x in dims)
        return data.reshape(shape)

    def sum(self, dims):
        """Reduce exactly the requested axes."""
        remaining = tuple(x for x in self.dims if x not in dims)
        axes = tuple(self.axes[self.dims.index(x)] for x in remaining)
        indices = tuple(self.dims.index(x) for x in dims)
        return Tensor(remaining, axes, self.data.sum(axis=indices))

    def select(self, dimension, category):
        """Select one category after whole-object switching."""
        axis = self.dims.index(dimension)
        try:
            index = self.axes[axis].index(category)
        except ValueError as exc:
            raise ValueError(
                f"Unknown {dimension} category {category}"
            ) from exc
        following = axis + 1
        return Tensor(
            self.dims[:axis] + self.dims[following:],
            self.axes[:axis] + self.axes[following:],
            np.take(self.data, index, axis=axis),
        )


@dataclass(frozen=True)
class Profiles:
    """Sparse employee memberships, weights and amount conversion factors."""

    dims: tuple
    axes: tuple
    cells: tuple
    employee_count: int

    def __post_init__(self):
        """Own immutable sparse arrays and validate the cell layout."""
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "axes", tuple(tuple(x) for x in self.axes))
        if len(self.dims) != len(self.axes) or len(set(self.dims)) != len(
            self.dims
        ):
            raise ValueError("Membership dimensions must be unique")
        if len(self.cells) != int(np.prod([len(x) for x in self.axes])):
            raise ValueError("Membership cell count differs from its axes")
        cells = []
        for cell in self.cells:
            arrays = tuple(np.array(x, copy=True) for x in cell)
            for array in arrays:
                array.setflags(write=False)
            cells.append(arrays)
        object.__setattr__(self, "cells", tuple(cells))

    def mean(self, values):
        """Evaluate paired row payroll before exposure-weighted aggregation."""
        if values.data.shape != (self.employee_count,):
            raise ValueError("Aligned input and membership axes differ")
        result = []
        for indices, weights, factors in self.cells:
            indices = np.asarray(indices)
            weights, factors = np.asarray(weights), np.asarray(factors)
            if (
                indices.dtype.kind not in "iu"
                or indices.ndim != 1
                or weights.shape != indices.shape
                or factors.shape != indices.shape
                or not len(indices)
                or (indices < 0).any()
                or (indices >= self.employee_count).any()
                or not np.isfinite(weights).all()
                or not np.isfinite(factors).all()
                or (weights < 0).any()
                or (factors <= 0).any()
                or not np.isclose(weights.sum(), 1, rtol=1e-10, atol=1e-12)
            ):
                raise ValueError("Invalid sparse membership profile")
            result.append(np.dot(weights * factors, values.data[indices]))
        return Tensor(
            self.dims,
            self.axes,
            np.asarray(result).reshape(tuple(len(x) for x in self.axes)),
        )


def evaluate(expr, factors, children, cache=None):
    """Evaluate one aggregate expression using fully supplied economics.

    Unknown scalar child economics may only be multiplied by zero weight.
    No state completion, source access or historical copying occurs here.
    """
    op, args = expr.op, expr.args
    key = None
    if cache is not None and op == "aligned_mean":
        from .graph_compile import refs

        key = (id(expr), tuple(id(factors[x]) for x in refs(expr, "value")))
        if key in cache:
            return cache[key]
    if op == "constant":
        return Tensor.scalar(args[0])
    if op == "value":
        return factors[args[0]]
    if op == "child":
        return children[args[0]]
    if op in {"add", "mul"}:
        left, right = [evaluate(x, factors, children, cache) for x in args]
        if left is None or right is None:
            known = right if left is None else left
            if op == "mul" and known is not None and np.all(known.data == 0):
                return Tensor(
                    known.dims, known.axes, np.zeros_like(known.data)
                )
            raise ValueError("Positive-share unknown economics")
        return left.combine(right, np.add if op == "add" else np.multiply)
    item = evaluate(args[0], factors, children, cache)
    if item is None:
        raise ValueError("Unknown economics used without a zero-share guard")
    if op == "sum":
        return item.sum(args[1])
    if op == "select":
        return item.select(args[1], args[2])
    if op == "aligned_mean":
        result = factors[args[1]].mean(item)
        if key is not None:
            cache[key] = result
        return result
    raise ValueError(f"Unsupported aggregate operation {op}")


def validate_factors(specs, values, fields=()):
    """Validate complete probability objects and quantity bounds."""
    field_map = {x.name: x for x in fields}
    for spec in specs:
        if spec.name not in values or values[spec.name] is None:
            raise ValueError(
                f"factor {spec.name}: missing applicable economics"
            )
        item = values[spec.name]
        if spec.kind == "membership":
            item.mean(
                Tensor(
                    ("employee",),
                    (tuple(range(item.employee_count)),),
                    np.ones(item.employee_count),
                )
            )
            continue
        expected_dims = spec.given + spec.dimensions
        if spec.kind == "aligned":
            expected_dims = ("employee:" + spec.population,)
        if item.dims != expected_dims:
            raise ValueError(f"factor {spec.name}: incorrect named axes")
        data = item.data
        if spec.kind in {"aligned", "policy"} and spec.source in field_map:
            constraint = field_map[spec.source].constraint
            if constraint == "positive" and (data <= 0).any():
                raise ValueError(
                    f"factor {spec.name}: values must be positive"
                )
            if constraint == "nonnegative" and (data < 0).any():
                raise ValueError(
                    f"factor {spec.name}: values must be nonnegative"
                )
        if not np.isfinite(data).all():
            raise ValueError(f"factor {spec.name}: nonfinite values")
        if (
            spec.kind in {"total", "distribution", "population_mix"}
            and (data < 0).any()
        ):
            raise ValueError(f"factor {spec.name}: negative quantity/share")
        if spec.kind in {"distribution", "population_mix"}:
            axes = tuple(item.dims.index(x) for x in spec.dimensions)
            if not np.allclose(data.sum(axis=axes), 1, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"factor {spec.name}: probabilities must sum to one"
                )


def team_values(plan, state, overrides=None):
    """Evaluate all scalar games in dependency order for one team."""
    values = {}
    by_name = {x.name: x for x in plan.graph.games}
    for name in plan.order:
        game = by_name[name]
        factors = state["populations"].get(game.population)
        if factors is None:
            values[name] = None
        elif overrides and name in overrides:
            values[name] = overrides[name]
        else:
            relevant = [x for x in plan.graph.factors if x.name in factors]
            validate_factors(relevant, factors, plan.graph.fields)
            values[name] = evaluate(game.formula, factors, values)
    return values


def graph_cost(plan, state):
    """Evaluate an organizational state without SHAP or completion."""
    graph = plan.graph
    costs = []
    for team in state["teams"]:
        value = team_values(plan, state["teams"][team])[graph.team_game]
        if value is None:
            if state["exposure_by_team"][team] > 0:
                raise ValueError("Positive team exposure needs unit economics")
            costs.append(0.0)
        else:
            costs.append(float(value.data))
    child_values = {
        graph.team_game: Tensor(
            ("team",), (tuple(state["teams"]),), np.array(costs)
        )
    }
    validate_factors(
        [x for x in graph.factors if x.scope == "organization"],
        state["factors"],
        graph.fields,
    )
    return float(
        evaluate(
            graph.organization.formula, state["factors"], child_values
        ).data
    )
