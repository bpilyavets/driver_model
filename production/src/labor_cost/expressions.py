"""Serializable arithmetic expressions; no Python evaluation or callbacks."""

import math
import numbers
from dataclasses import dataclass


@dataclass(frozen=True)
class Expr:
    """An immutable formula node over named row-level input fields."""

    op: str
    args: tuple

    def __post_init__(self):
        """Freeze child nodes so compiled formula identity cannot drift."""
        object.__setattr__(self, "args", tuple(self.args))

    def __add__(self, other):
        """Add a cost term."""
        return Expr("add", (self, expression(other)))

    def __radd__(self, other):
        """Support sums of formula terms."""
        return expression(other) + self

    def __sub__(self, other):
        """Subtract a cost term without clipping signed amounts."""
        return self + (-expression(other))

    def __rsub__(self, other):
        """Subtract this expression from another term."""
        return expression(other) - self

    def __neg__(self):
        """Negate a signed cost term."""
        return expression(-1) * self

    def __mul__(self, other):
        """Multiply row-level inputs, preserving their joint identity."""
        return Expr("mul", (self, expression(other)))

    def __rmul__(self, other):
        """Multiply a formula by a constant or another formula."""
        return expression(other) * self

    def to_dict(self):
        """Return a JSON-compatible expression graph."""
        return {
            "op": self.op,
            "args": [
                a.to_dict() if isinstance(a, Expr) else a for a in self.args
            ],
        }

    @classmethod
    def from_dict(cls, value):
        """Read only the supported expression grammar."""
        if not isinstance(value, dict) or set(value) != {"op", "args"}:
            raise ValueError("Invalid expression object")
        op, args = value["op"], value["args"]
        if op == "field" and len(args) == 1:
            return field(args[0])
        if op == "constant" and len(args) == 1:
            return expression(args[0])
        if op in {"add", "mul"} and len(args) == 2:
            return cls(op, tuple(cls.from_dict(a) for a in args))
        raise ValueError(f"Unsupported expression operation: {op}")

    def __str__(self):
        """Render an analyst-readable formula."""
        if self.op in {"field", "constant"}:
            return str(self.args[0])
        symbol = {"add": "+", "mul": "*"}.get(self.op, self.op)
        return f"({self.args[0]} {symbol} {self.args[1]})"


def field(name):
    """Declare a reference to an input named in the model specification."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Formula fields must be Python-style identifiers")
    return Expr("field", (name,))


def expression(value):
    """Coerce a finite numeric constant or an existing expression."""
    if isinstance(value, Expr):
        return value
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise ValueError("Expressions accept only fields and finite constants")
    return Expr("constant", (float(value),))


def monomials(expr):
    """Expand supported arithmetic into coefficients and joint field products.

    Limit expansion to 256 terms; a larger expression needs explicit model
    simplification. No aggregation or statistical independence is inferred.
    """
    if expr.op == "field":
        field(expr.args[0])
        return [(1.0, (expr.args[0],))]
    if expr.op == "constant":
        return [(expression(expr.args[0]).args[0], ())]
    if expr.op not in {"add", "mul"} or len(expr.args) != 2:
        raise ValueError("Only addition, subtraction and multiplication work")
    left, right = (monomials(a) for a in expr.args)
    if expr.op == "add":
        result = left + right
    else:
        result = [(a * b, x + y) for a, x in left for b, y in right]
    if any(not math.isfinite(coefficient) for coefficient, _ in result):
        raise ValueError("Formula arithmetic produced a nonfinite coefficient")
    if len(result) > 256:
        raise ValueError("Formula expands past 256 terms; simplify it")
    return [(c, names) for c, names in result if c != 0]
