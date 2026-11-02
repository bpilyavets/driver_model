"""Semantic and notebook contracts for data-free formula presentation."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from labor_cost import (
    Aggregate,
    Expr,
    Notation,
    RowCost,
    aligned_mean,
    child,
    compile_model,
    field,
    labor_cost_model,
    render_formula,
    render_model,
    select,
    sum_over,
    value,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.util.spec_from_file_location(
    "presentation_authors", ROOT / "examples" / "author_models.py"
)
AUTHORS = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(AUTHORS)


def symbols(preview):
    """Index legend symbols by their typed declaration identities."""
    return {(x["namespace"], x["identifier"]): x["symbol"]
            for x in preview.legend}


def test_row_precedence_signs_constants_and_groups():
    """Rendering retains grouping and does not expand polynomial terms."""
    a, b, c = field("a"), field("b"), field("c")
    assert render_formula(a * (b - c)).latex == (
        r"\mathrm{A} \cdot \left(\mathrm{B} - \mathrm{C}\right)"
    )
    assert render_formula(a - (b + c)).latex == (
        r"\mathrm{A} - \left(\mathrm{B} + \mathrm{C}\right)"
    )
    assert render_formula(a + (b + c)).latex == (
        r"\mathrm{A} + \left(\mathrm{B} + \mathrm{C}\right)"
    )
    assert render_formula(-a * -2.5).latex == (
        r"\left(-\mathrm{A}\right) \cdot \left(-2.5\right)"
    )
    assert r"10^{-8}" in render_formula(1e-8 * a).latex
    assert render_formula(a * 0).latex.endswith(r"\cdot 0")


def test_symbols_are_unique_deterministic_and_overridable():
    """Colliding initials and namespaces do not hide distinct quantities."""
    first, second = field("salary"), field("sales")
    forward = render_formula(first + second)
    reverse = render_formula(second + first)
    assert symbols(forward) == symbols(reverse)
    assert len(set(symbols(forward).values())) == 2
    assert all("[fld" in x for x in symbols(forward).values())
    supplied = {"salary": "LaborCost"}
    notation = Notation(fields=supplied)
    supplied["salary"] = "changed"
    assert render_formula(first, notation=notation).latex == (
        r"\mathrm{LaborCost}"
    )
    with pytest.raises(TypeError):
        notation.fields["salary"] = "changed"
    for kwargs in (
        {"fields": {"salary": "S", "sales": "S"}},
        {"fields": {"salary": "S"}, "factors": {"salary": "S"}},
    ):
        with pytest.raises(ValueError, match="Ambiguous symbol.*fields"):
            Notation(**kwargs)
    with pytest.raises(ValueError, match="plain-text"):
        Notation(fields={"salary": r"\frac{a}{b}"})


@pytest.mark.parametrize("schema", [1, 2])
def test_model_preview_identity_and_fragment_consistency(schema):
    """Both schemas share symbols without changing declarations or identity."""
    model = labor_cost_model(schema_version=schema)
    before = json.dumps(model.to_dict(), sort_keys=True)
    fingerprint = model.fingerprint
    compiled = compile_model(model)
    preview = render_model(compiled)
    assert preview.markdown == render_model(model).markdown
    formula = compiled.plan.graph.games[0].formula
    fragment = render_formula(formula, model=model)
    whole = symbols(preview)
    assert all(whole[key] == val for key, val in symbols(fragment).items())
    assert len(set(whole.values())) == len(whole)
    assert model.fingerprint == fingerprint
    assert json.dumps(model.to_dict(), sort_keys=True) == before
    assert "V1 compatibility" in preview.markdown if schema == 1 else (
        "V1 compatibility" not in preview.markdown
    )
    assert "selected local team only" in preview.markdown
    assert "Cost: organization" in preview.markdown
    assert "Cost: local" in preview.markdown


@pytest.mark.parametrize("grade_first", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_conditional_axes_paired_products_and_switches(grade_first, grouped):
    """General model notation reflects the actual chosen factorization."""
    model = AUTHORS.hourly_model(grade_first=grade_first, grouped=grouped)
    notation = Notation(
        dimensions={"grade": "g", "region": "r", "team": "t"},
        factors={"rates": "R", "hours": "H", "members": "M"},
    )
    preview = render_model(model, notation=notation)
    cond = (r"\mathrm{r}\mid \mathrm{g}" if grade_first else
            r"\mathrm{g}\mid \mathrm{r}")
    assert cond in preview.latex
    assert r"\sum_{i}" in preview.latex
    assert r"\mathrm{R}_{i} \cdot \mathrm{H}_{i}" in preview.latex
    assert r"w^{(\mathrm{M})}" in preview.latex
    assert r"a^{(\mathrm{M})}" in preview.latex
    assert "inverse donor-row exposure" in preview.markdown
    assert ("earned_pay_inputs" in preview.plain_text) == grouped
    rates = next(x for x in preview.legend if x["identifier"] == "rates")
    assert rates["source"] == "hourly_rate"
    assert rates["units"] == "currency/time"


def test_membership_bases_and_multiple_profiles():
    """Each sparse mean retains its own weights and conversion contract."""
    model = AUTHORS.hourly_model()
    members = next(x for x in model.graph.factors if x.kind == "membership")
    other = replace(members, name="other_members", basis="per_exposure")
    draft = replace(model, graph=replace(
        model.graph, factors=(*model.graph.factors, other)
    ))
    formula = (
        aligned_mean(value("rates") * value("hours"), "members")
        + aligned_mean(value("rates") * value("hours"), "other_members")
    )
    preview = render_formula(formula, model=draft)
    assert preview.latex.count(r"\sum_{i}") == 2
    assert "members: actual basis" in preview.markdown
    assert "other_members: per_exposure basis; a = 1" in preview.plain_text
    assert r"a^{(\mathrm{MEM})}" in preview.latex
    assert r"a^{(\mathrm{OM})}" in preview.latex
    structural = render_formula(formula)
    assert structural.latex.count(r"\operatorname{AlignedMean}") == 2
    assert r"\sum_{i}" not in structural.latex
    assert all(x["units"] is None for x in structural.legend)


def test_selections_dimensions_and_plain_text_escaping():
    """Category labels and symbol overrides cannot become TeX or HTML."""
    expr = select(sum_over(value("mix"), "grade"), "region", "A_$|<x>")
    preview = render_formula(
        expr, notation=Notation(factors={"mix": "M_$|<x>"})
    )
    assert r"\sum_{\mathrm{GRA}}" in preview.latex
    assert r"\mathrm{REG}=\mathrm{A\_\$\vert{}\lt{}x\gt{}}" in preview.latex
    assert "<x>" not in preview.markdown
    assert r"\mathrm{M\_\$\vert{}\lt{}x\gt{}}" in preview.latex


def test_child_expansion_and_incomplete_drafts():
    """Named child costs expand only on request with usable definitions."""
    model = AUTHORS.hourly_model(nested=True)
    formula = model.graph.games[-1].formula
    compact = render_formula(formula, model=model)
    expanded = render_formula(formula, model=model, expand_children=True)
    assert r"\sum_{i}" not in compact.latex
    assert r"\sum_{i}" in expanded.latex
    assert "earnings" in [x["identifier"] for x in compact.legend]
    with pytest.raises(ValueError, match="requires model"):
        render_formula(formula, expand_children=True)
    with pytest.raises(ValueError, match="missing definition"):
        render_formula(child("missing"), model=model, expand_children=True)
    cyclic_game = replace(model.graph.games[0], formula=child("earnings"))
    cyclic = replace(model, graph=replace(
        model.graph, games=(cyclic_game, model.graph.games[1])
    ))
    with pytest.raises(ValueError, match="cyclic"):
        render_formula(child("earnings"), model=cyclic, expand_children=True)
    # Draft previews work even while the overall model fails compilation.
    draft = replace(model, graph=replace(model.graph, row_costs=()))
    assert render_formula(field("rate"), model=draft).legend[0]["source"]
    with pytest.raises(ValueError, match="row_costs"):
        render_model(draft)


def test_overlapping_rows_have_distinct_equations():
    """Rows sharing a population are not collapsed to one misleading LHS."""
    model = AUTHORS.hourly_model()
    rows = (RowCost("all", field("rate") * field("hours")),
            RowCost("all", field("allowance")))
    model = replace(model, graph=replace(model.graph, row_costs=rows))
    preview = render_model(model)
    assert "Row cost 1: all" in preview.markdown
    assert "Row cost 2: all" in preview.markdown
    assert r"\operatorname{RowCost}^{(1)}" in preview.latex
    assert r"\operatorname{RowCost}^{(2)}" in preview.latex


def test_large_formula_wrapping_retains_every_term():
    """Long sums wrap once at this level and keep all distinct inputs."""
    names = [f"payment_{index}" for index in range(25)]
    expr = field(names[0])
    for name in names[1:]:
        expr += field(name)
    preview = render_formula(expr)
    assert preview.latex.count(r"\begin{aligned}") == 1
    assert r"\\ &{} + " in preview.latex
    assert len(preview.legend) == len(names)
    assert all(preview.latex.count(x["latex"]) == 1 for x in preview.legend)
    assert preview.latex.count(" + ") == len(names) - 1


def test_long_products_wrap_without_distributing_sum():
    """Conditional factors cannot push a large nested sum off the page."""
    model = labor_cost_model(schema_version=2)
    game = next(x for x in model.graph.games if x.name == "permanent")
    preview = render_formula(game.formula, model=model)
    assert r"\\ &{}\cdot " in preview.latex
    # The conditional distributions remain outside one paired mean.
    assert preview.latex.count(r"\sum_{i}") == 1
    assert preview.latex.count(r"\sum_{\mathrm{REG},\mathrm{GRA}}") == 1


@pytest.mark.parametrize("expr", [
    Expr("divide", (field("a"), field("b"))),
    Aggregate("power", (value("a"), value("b"))),
    Expr("add", (field("a"),)),
    Expr("constant", (float("inf"),)),
    Aggregate("sum", (value("a"), (None,))),
])
def test_unsupported_nodes_fail_explicitly(expr):
    """Unknown grammar never silently produces plausible mathematics."""
    with pytest.raises(ValueError):
        render_formula(expr)


def test_mime_protocol_filters_and_no_optional_imports():
    """Rich output is available without loading notebook or SHAP modules."""
    preview = render_formula(field("salary"))
    assert preview._repr_mimebundle_(include=["text/plain"]) == {
        "text/plain": str(preview)
    }
    assert set(preview._repr_mimebundle_(exclude=["text/plain"])) == {
        "text/markdown"
    }
    code = """
import sys
from labor_cost import labor_cost_model, render_model
for schema in (1, 2):
    assert render_model(labor_cost_model(schema_version=schema)).latex
assert not {'IPython', 'shap', 'matplotlib', 'jupyterlab'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
