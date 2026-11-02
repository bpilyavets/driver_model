"""Data-free mathematical previews of authored workforce expressions.

Notation is presentation configuration, never part of model identity.
This module uses Jupyter's MIME protocol without importing IPython.
"""

import html
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .expressions import Expr, expression
from .graph import Aggregate

_NAMESPACES = ("fields", "factors", "games", "dimensions", "populations")
_QUALIFIERS = dict(zip(_NAMESPACES, ("fld", "fac", "cost", "dim", "pop")))


def _tex(text):
    """Escape plain text for a TeX text or upright-symbol group."""
    escapes = {
        "\\": r"\backslash{}", "{": r"\{", "}": r"\}",
        "_": r"\_", "^": r"\textasciicircum{}", "%": r"\%",
        "&": r"\&", "#": r"\#", "$": r"\$",
        "~": r"\textasciitilde{}", " ": r"\ ",
        "|": r"\vert{}", "<": r"\lt{}", ">": r"\gt{}",
        "\n": " ", "\r": " ",
    }
    return "".join(escapes.get(c, c) for c in str(text))


def _markdown(text):
    """Escape untrusted labels in Markdown, including table delimiters."""
    return "".join(
        f"&#{ord(c)};" if c in r"\`*_{}[]()#$|!"
        else html.escape(c, quote=False)
        for c in str(text)
    ).replace("\n", " ").replace("\r", " ")


@dataclass(frozen=True)
class Notation:
    """Optional plain-text symbol overrides, keyed by declaration ID.

    The five mappings are independent identifier namespaces. Explicit
    symbols must be distinct across namespaces. Mappings are copied;
    reusing a configuration never mutates it or the model. Symbols may
    exceed three characters, but raw TeX and control characters are not
    accepted. Unknown IDs may be included for later authoring steps.
    """

    fields: Mapping = field(default_factory=dict)
    factors: Mapping = field(default_factory=dict)
    games: Mapping = field(default_factory=dict)
    dimensions: Mapping = field(default_factory=dict)
    populations: Mapping = field(default_factory=dict)

    def __post_init__(self):
        """Own and validate overrides, including cross-namespace clashes."""
        owners = {}
        for namespace in _NAMESPACES:
            supplied = getattr(self, namespace)
            if not isinstance(supplied, Mapping):
                raise ValueError(f"notation.{namespace}: expected a mapping")
            values = dict(supplied)
            for name, symbol in values.items():
                path = f"{namespace}.{name}"
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError(f"{path}: expected a declaration ID")
                if (
                    not isinstance(symbol, str) or not symbol.strip()
                    or symbol != symbol.strip() or "\\" in symbol
                    or not symbol.isprintable()
                ):
                    raise ValueError(f"{path}: use a plain-text symbol")
                if symbol in owners:
                    raise ValueError(
                        f"Ambiguous symbol {symbol!r}: "
                        f"{owners[symbol]} and {path}"
                    )
                owners[symbol] = path
            object.__setattr__(self, namespace, MappingProxyType(values))


@dataclass(frozen=True)
class FormulaPreview:
    """Rendered equations, Markdown, and structured symbol metadata.

    ``latex`` contains math without dollar delimiters. ``legend`` is a
    tuple of read-only mappings; unknown metadata is None. ``markdown``
    contains the equation sheet and legend. Notebook display selects the
    Markdown representation; terminals receive readable plain text.
    """

    latex: str
    markdown: str
    legend: tuple
    plain_text: str

    def __str__(self):
        """Return the terminal representation without Markdown tables."""
        return self.plain_text

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Offer notebook Markdown/math with a plain-text fallback."""
        bundle = {
            "text/markdown": self.markdown, "text/plain": self.plain_text,
        }
        return {
            key: value for key, value in bundle.items()
            if (include is None or key in include)
            and (exclude is None or key not in exclude)
        }


def _mnemonic(name):
    """Suggest at most three letters without interpreting business names."""
    words = [x for x in name.split("_") if x]
    text = "".join(x[0] for x in words) if len(words) > 1 else name
    return text[:3].upper() or "X"


def _references(expr, path="expression"):
    """Validate the closed display grammar and collect typed references."""
    if not isinstance(expr, (Expr, Aggregate)):
        raise ValueError(f"{path}: expected Expr or Aggregate")
    arities = {"constant": 1, "add": 2, "mul": 2}
    arities.update(
        {"field": 1} if isinstance(expr, Expr) else {
            "value": 1, "child": 1, "sum": 2, "select": 3,
            "aligned_mean": 2,
        }
    )
    op, args = expr.op, expr.args
    if op not in arities or len(args) != arities[op]:
        raise ValueError(f"{path}: unsupported operation/arity {op!r}")
    if op == "constant":
        expression(args[0])
        return set()
    namespaces = {"field": "fields", "value": "factors", "child": "games"}
    if op in namespaces:
        if not isinstance(args[0], str) or not args[0].isidentifier():
            raise ValueError(f"{path}.{op}: expected a declaration ID")
        return {(namespaces[op], args[0])}
    result = set()
    operands = args if op in {"add", "mul"} else args[:1]
    for index, operand in enumerate(operands):
        if type(operand) is not type(expr):
            raise ValueError(f"{path}.{op}: incompatible expression node")
        result.update(_references(operand, f"{path}.{op}[{index}]"))
    if op == "sum":
        if not isinstance(args[1], (tuple, list)):
            raise ValueError(f"{path}.sum: expected named dimensions")
        result.update(("dimensions", x) for x in args[1])
    if op == "select":
        result.add(("dimensions", args[1]))
    if op == "aligned_mean":
        result.add(("factors", args[1]))
    if any(not isinstance(n, str) or not n.isidentifier() for _, n in result):
        raise ValueError(f"{path}.{op}: invalid reference or dimension")
    return result


def _model_graph(model, *, validate=False):
    """Obtain native draft metadata or a compiled compatibility graph."""
    from .compiler import CompiledModel, compile_model
    from .spec import ModelSpec

    if model is None:
        return None
    if isinstance(model, CompiledModel):
        return model.plan.graph
    if not isinstance(model, ModelSpec):
        raise ValueError("model: expected ModelSpec or CompiledModel")
    if validate or model.graph is None:
        return compile_model(model).plan.graph
    return model.graph


class _Renderer:
    """Resolve one stable symbol vocabulary and render expression trees."""

    def __init__(self, graph, notation, expressions):
        """Gather the complete context before resolving any short symbols."""
        if notation is None:
            notation = Notation()
        if not isinstance(notation, Notation):
            raise ValueError("notation: expected Notation")
        self.graph = graph
        self.declarations = {key: {} for key in _NAMESPACES}
        if graph is not None:
            for key in ("fields", "factors", "populations"):
                self.declarations[key] = {
                    item.name: item for item in getattr(graph, key)
                }
            self.declarations["populations"]["all"] = None
            self.declarations["games"] = {
                item.name: item for item in (
                    *graph.games, graph.organization, graph.local
                )
            }
            dims = {"team", "employment_type", *dict(graph.source.dimensions)}
            for factor in graph.factors:
                dims.update(factor.dimensions + factor.given)
            self.declarations["dimensions"] = dict.fromkeys(dims)
        keys = {
            (namespace, name) for namespace, items in self.declarations.items()
            for name in items
        }
        for expr in expressions:
            keys.update(_references(expr))
        overrides = {
            (namespace, name): symbol for namespace in _NAMESPACES
            for name, symbol in getattr(notation, namespace).items()
        }
        keys.update(overrides)
        self.symbols = self._symbols(keys, overrides)
        self.used = set()
        self.memberships = set()

    @staticmethod
    def _symbols(keys, overrides):
        """Allocate unique explicit or qualified deterministic symbols."""
        counts = Counter(_mnemonic(name) for key, name in keys
                         if (key, name) not in overrides)
        occupied = set(overrides.values())
        symbols = {key: (value, rf"\mathrm{{{_tex(value)}}}")
                   for key, value in overrides.items()}
        for key in sorted(keys - overrides.keys()):
            namespace, name = key
            base = _mnemonic(name)
            qualifier = ""
            if counts[base] > 1 or base in occupied:
                qualifier = _QUALIFIERS[namespace]
            text = base + (f"[{qualifier}]" if qualifier else "")
            serial = 1
            while text in occupied:
                serial += 1
                qualifier = _QUALIFIERS[namespace] + str(serial)
                text = f"{base}[{qualifier}]"
            occupied.add(text)
            tex = rf"\mathrm{{{_tex(base)}}}"
            if qualifier:
                tex += rf"^{{\mathrm{{{qualifier}}}}}"
            symbols[key] = text, tex
        return symbols

    def symbol(self, namespace, name):
        """Record the referenced declaration and return its TeX symbol."""
        if (namespace, name) not in self.symbols:
            raise ValueError(f"{namespace}.{name}: missing display context")
        self.used.add((namespace, name))
        return self.symbols[namespace, name][1]

    def indices(self, dimensions):
        """Use the same dimension symbols in factors and their reductions."""
        return ",".join(self.symbol("dimensions", x) for x in dimensions)

    def factor(self, name):
        """Show conditional structure and actual declared factor axes."""
        symbol = self.symbol("factors", name)
        item = self.declarations["factors"].get(name)
        if item is None:
            return symbol
        if item.kind == "aligned":
            return symbol + "_{i}"
        context = ("team",) if item.scope == "team" else ()
        if item.kind in {"distribution", "population_mix"}:
            args = self.indices(item.dimensions)
            if item.given:
                args += r"\mid " + self.indices(item.given)
            if context:
                args += ";" + self.indices(context)
            return symbol + rf"\!\left({args}\right)"
        axes = context + item.given + item.dimensions
        return symbol + ("_{" + self.indices(axes) + "}" if axes else "")

    def addition(self, expr, expand, stack):
        """Wrap left-associated additions without flattening right groups."""
        terms = []
        while expr.op == "add":
            terms.append(expr.args[1])
            expr = expr.args[0]
        terms.append(expr)
        lines = []
        line = ""
        for index, term in enumerate(reversed(terms)):
            part = self.render(term, expand=expand, stack=stack)
            if term.op == "add":
                part = rf"\left({part}\right)"
            prefix = " + " if index else ""
            if index and part.startswith("-"):
                prefix, part = " - ", part[1:]
            if line and len(line) + len(part) > 160:
                lines.append(line)
                line = "{}" + prefix + part
            else:
                line += prefix + part
        lines.append(line)
        if len(lines) == 1:
            return line
        return (r"\begin{aligned}&" + r" \\ &".join(lines)
                + r"\end{aligned}")

    def render(self, expr, *, expand=False, stack=()):
        """Render one node without algebraic expansion or reordering."""
        op, args = expr.op, expr.args
        if op == "constant":
            number = float(args[0])
            text = str(int(number)) if number.is_integer() else repr(number)
            if "e" in text:
                mantissa, exponent = text.split("e")
                text = rf"{mantissa}\times 10^{{{int(exponent)}}}"
            return text
        if op == "field":
            return self.symbol("fields", args[0])
        if op == "value":
            return self.factor(args[0])
        if op == "child":
            name = args[0]
            symbol = self.symbol("games", name)
            if expand:
                game = self.declarations["games"].get(name)
                if game is None or name in stack:
                    raise ValueError(
                        f"child {name}: missing definition or cyclic expansion"
                    )
                _references(game.formula, f"game {name}")
                inner = self.render(
                    game.formula, expand=True, stack=(*stack, name)
                )
                return rf"\left({inner}\right)"
            if self.graph is not None:
                symbol += "_{" + self.indices(("team",)) + "}"
            return symbol
        if op == "add":
            return self.addition(expr, expand, stack)
        if op == "mul":
            parts = []
            unary = any(x.op == "constant" and x.args[0] == -1 for x in args)
            operands = args if unary else _multiplicands(expr)
            for child in operands:
                part = self.render(child, expand=expand, stack=stack)
                if child.op == "add" or part.startswith("-"):
                    part = rf"\left({part}\right)"
                parts.append(part)
            if args[0].op == "constant" and args[0].args[0] == -1:
                return "-" + parts[1]
            if args[1].op == "constant" and args[1].args[0] == -1:
                return "-" + parts[0]
            return _product(parts)
        inner = self.render(args[0], expand=expand, stack=stack)
        if op == "sum":
            if not args[1]:
                return inner
            axes = self.indices(args[1])
            return rf"\sum_{{{axes}}}\left({inner}\right)"
        if op == "select":
            dim = self.symbol("dimensions", args[1])
            category = rf"\mathrm{{{_tex(args[2])}}}"
            return rf"\left[{inner}\right]_{{{dim}={category}}}"
        if op == "aligned_mean":
            name = args[1]
            membership = self.symbol("factors", name)
            profile = self.declarations["factors"].get(name)
            if profile is None:
                return (rf"\operatorname{{AlignedMean}}_{{{membership}}}"
                        rf"\!\left({inner}\right)")
            if profile.kind != "membership":
                raise ValueError(f"aligned_mean {name}: not a membership")
            self.memberships.add(name)
            context = ("team",) if profile.scope == "team" else ()
            axes = self.indices(context + profile.dimensions)
            index = "i" + (r"\mid " + axes if axes else "")
            weight = rf"w^{{({membership})}}_{{{index}}}"
            conversion = rf"a^{{({membership})}}_{{{index}}}"
            paired = _product([weight, conversion, rf"\left({inner}\right)"])
            return rf"\sum_{{i}}\left({paired}\right)"
        raise ValueError(f"expression: unsupported operation {op!r}")

    def legend(self):
        """Describe all displayed symbols without inventing metadata."""
        entries = []
        for namespace, name in sorted(self.used):
            item = self.declarations[namespace].get(name)
            source = getattr(item, "source", None)
            units = getattr(item, "units", None)
            dimensions = None
            kind = namespace[:-1]
            population = getattr(item, "population", None)
            scope = getattr(item, "scope", None)
            basis = getattr(item, "basis", None)
            if namespace == "factors" and item is not None:
                kind = item.kind
                dimensions = item.given + item.dimensions
                if item.kind in {"aligned", "policy"}:
                    original = self.declarations["fields"].get(item.source)
                    source = getattr(original, "source", None)
                    units = getattr(original, "units", None)
                elif item.kind in {"distribution", "population_mix"}:
                    units = "dimensionless"
                elif item.kind == "payment":
                    units = "currency/exposure"
                    source = str(item.formula)
                elif item.kind == "total":
                    units = "exposure"
                elif item.kind == "membership":
                    conversion = ("1/exposure" if item.basis == "actual"
                                  else "dimensionless")
                    units = f"weights: dimensionless; conversion: {conversion}"
                if item.kind == "aligned":
                    dimensions = ("employee/employment_type",)
                if item.kind not in {"membership", "payment"}:
                    basis = None
            if namespace == "games" and item is not None:
                outer = item in (self.graph.organization, self.graph.local)
                units = "currency" if outer else "currency/exposure"
                scope = ("organization" if item == self.graph.organization
                         else "local team" if outer else "team")
            if namespace == "dimensions" and self.graph is not None:
                source = dict(self.graph.source.dimensions).get(name)
                if name == "team":
                    source = self.graph.source.team
                if name == "employment_type":
                    source = self.graph.source.employment_type
                units = "category"
            entries.append(MappingProxyType({
                "namespace": namespace, "identifier": name, "kind": kind,
                "symbol": self.symbols[namespace, name][0],
                "latex": self.symbols[namespace, name][1],
                "source": source, "population": population,
                "dimensions": dimensions, "units": units,
                "basis": basis, "scope": scope,
            }))
        return tuple(entries)


def _multiplicands(expr):
    """Read a product in authored order, keeping unary negatives grouped."""
    if expr.op != "mul" or any(
        x.op == "constant" and x.args[0] == -1 for x in expr.args
    ):
        return (expr,)
    return tuple(term for child in expr.args for term in _multiplicands(child))


def _product(parts):
    """Wrap long products at operator boundaries without distributing them."""
    lines = []
    line = parts[0]
    for part in parts[1:]:
        if len(line) + len(part) > 240:
            lines.append(line)
            line = r"{}\cdot " + part
        else:
            line += r" \cdot " + part
    lines.append(line)
    if len(lines) == 1:
        return line
    return (r"\begin{aligned}&" + r" \\ &".join(lines)
            + r"\end{aligned}")


def _preview(renderer, equations, notes):
    """Assemble the public data object and notebook/terminal displays."""
    legend = renderer.legend()
    sections = []
    plain = []
    for title, latex, context in equations:
        sections.append(f"### {_markdown(title)}\n\n$$\n{latex}\n$$\n")
        if context:
            sections.append(_markdown(context) + "\n")
        plain.extend((title, latex, context))
    if renderer.memberships:
        notes = list(notes) + [
            "i identifies an aligned employee/employment-type record. "
            "Each w is a normalized sparse membership weight. The matching "
            "a belongs to that same membership object and switches with it."
        ]
        for name in sorted(renderer.memberships):
            basis = renderer.declarations["factors"][name].basis
            conversion = ("inverse donor-row exposure" if basis == "actual"
                          else "1")
            notes.append(f"{name}: {basis} basis; a = {conversion}.")
    sections.extend(_markdown(note) + "\n" for note in notes)
    plain.extend(notes)
    if legend:
        sections.append(
            "### Symbol legend\n\n"
            "Symbol | Kind / identifier | Source | Population / scope | "
            "Dimensions | Units / basis\n"
            "--- | --- | --- | --- | --- | ---"
        )
        for entry in legend:
            values = [
                entry["kind"] + ": " + entry["identifier"], entry["source"],
                " / ".join(str(entry[x] or "unspecified")
                           for x in ("population", "scope")),
                (", ".join(entry["dimensions"]) or "scalar")
                if entry["dimensions"] is not None else None,
                " / ".join(str(entry[x] or "unspecified")
                           for x in ("units", "basis")),
            ]
            sections.append(
                "$" + entry["latex"] + "$ | "
                + " | ".join(_markdown(x if x is not None else "unspecified")
                             for x in values)
            )
            plain.append(f"{entry['symbol']}: {values[0]} ({values[-1]})")
    latex = (r" \\" + "\n").join(item[1] for item in equations)
    if len(equations) > 1:
        latex = r"\begin{gathered}" + latex + r"\end{gathered}"
    return FormulaPreview(latex, "\n".join(sections), legend,
                          "\n".join(x for x in plain if x))


def render_formula(expr, *, model=None, notation=None, expand_children=False):
    """Preview a row or aggregate expression without data or attribution.

    Parameters
    ----------
    expr : Expr or Aggregate
        Authored expression, including an unfinished model's formula.
    model : ModelSpec or CompiledModel, optional
        Metadata for dimensions, bases and child definitions. Native draft
        specifications need not compile; this preview is not validation.
    notation : Notation, optional
        Reusable symbol overrides, independent of model identity.
    expand_children : bool
        Substitute child definitions; requires model context.

    Returns
    -------
    FormulaPreview
        Notebook math, raw LaTeX, and a structured legend.

    Raises
    ------
    ValueError
        Unsupported expression, ambiguous notation or unavailable expansion.
    """
    if expand_children and model is None:
        raise ValueError("Child expansion requires model context")
    graph = _model_graph(model)
    renderer = _Renderer(graph, notation, (expr,))
    latex = renderer.render(expr, expand=expand_children)
    notes = ["Expression preview; this does not validate a model."]
    if graph is None:
        notes.append(
            "Structural preview: dimensions, units and amount bases are "
            "unspecified without model context."
        )
    return _preview(renderer, [("Formula", latex, "")], notes)


def render_model(model, *, notation=None):
    """Render a validated declaration as an equation sheet, without data.

    Parameters
    ----------
    model : ModelSpec or CompiledModel
        V1 or v2 declaration. Specifications pass through compile_model.
    notation : Notation, optional
        Notebook presentation settings; never serialized into the model.

    Returns
    -------
    FormulaPreview
        Row and aggregate equations, game switches, and symbol legend.

    Raises
    ------
    ValueError
        Invalid model declarations or ambiguous display overrides.
    """
    graph = _model_graph(model, validate=True)
    if graph is None:
        raise ValueError("model: expected ModelSpec or CompiledModel")
    games = (graph.organization, graph.local, *graph.games)
    expressions = [row.formula for row in graph.row_costs]
    expressions.extend(game.formula for game in games)
    renderer = _Renderer(graph, notation, expressions)
    equations = []
    for number, row in enumerate(graph.row_costs, 1):
        pop = renderer.symbol("populations", row.population)
        lhs = rf"\operatorname{{RowCost}}^{{({number})}}_{{{pop}}}"
        latex = lhs + " = " + renderer.render(row.formula)
        equations.append((
            f"Row cost {number}: {row.population}", latex,
            "Actual employee/employment-type/month amount; currency. "
            "Overlapping applicable row components add.",
        ))
    for game in games:
        outer = game in (graph.organization, graph.local)
        scope = ("organization" if game == graph.organization else
                 "local team" if outer else "team")
        lhs = renderer.symbol("games", game.name)
        if not outer:
            lhs += "_{" + renderer.indices(("team",)) + "}"
        rhs = renderer.render(game.formula)
        switches = []
        for switch in game.switches:
            target = (f"child {switch.child}" if switch.child is not None
                      else ", ".join(switch.factors) + " (whole objects)")
            label = f" ({switch.label})" if switch.label else ""
            switches.append(f"{switch.name}{label} -> {target}")
        context = f"Scope: {scope}; population: {game.population}. "
        if game == graph.local:
            context += "Team-axis sums use the selected local team only. "
        context += "Switches: " + "; ".join(switches)
        equations.append((f"Cost: {game.name}", lhs + " = " + rhs, context))
    spec = getattr(model, "spec", model)
    notes = [
        "Compilation validates declarations, not business correctness or "
        "data reconciliation. Organizational and local costs are separate "
        "views; do not add their allocations."
    ]
    if spec.graph is None:
        notes.append(
            "V1 compatibility model: aggregate equations and supporting "
            "row components are translated by the existing compiler."
        )
    return _preview(renderer, equations, notes)
