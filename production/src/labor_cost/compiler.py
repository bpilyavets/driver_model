"""Validate and lower declared row formulas to coherent workforce states."""

import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .engine.common import PARENT
from .engine.inputs import adapt_production, finite_numeric
from .engine.kernels import employee_vector, profile_mean
from .engine.registry import (
    DRIVERS,
    PRODUCTION_COLUMNS,
    Driver,
    payment_driver,
)
from .expressions import monomials
from .spec import InputSpec, ModelSpec, WorkforceStructure


@dataclass(frozen=True)
class CompiledModel:
    """Validated declarations and generated engine registrations."""

    spec: ModelSpec
    registry: tuple
    terms: tuple
    plan: object | None = None

    @property
    def fingerprint(self):
        """Return the content identity of the declared model."""
        return self.spec.fingerprint

    def inspect(self):
        """Describe the formula, representations, atomic players and games."""
        if self.spec.graph is not None:
            return {
                "model_id": self.spec.model_id,
                "version": self.spec.version,
                "fingerprint": self.fingerprint,
                **self.plan.inspect(),
            }
        return {
            "model_id": self.spec.model_id,
            "version": self.spec.version,
            "fingerprint": self.fingerprint,
            "formula": str(self.spec.formula),
            "structure": self.spec.to_dict()["structure"],
            "inputs": self.spec.to_dict()["inputs"],
            "players": [
                {
                    "id": d.name,
                    "label": d.label,
                    "parent": d.parent,
                    "representation": d.kind,
                    "cost_dependencies": list(d.dependencies),
                }
                for d in self.registry
            ],
            "missing_economics": self.spec.missing_economics,
            "exposure_column": self.spec.exposure_column,
            "interpretation": "Accounting attribution, not causal effects",
        }

    def row_cost(self, bound):
        """Evaluate applicable actual row payroll using declared arithmetic.

        ``bound`` uses canonical formula names before exposure conversion.
        This is a reconstructed reference, not independent source evidence.
        V1 declarations apply to permanent rows; v2 uses each component's
        population predicate and sums all applicable row components.
        """
        if self.spec.graph is not None:
            from .graph_states import population_masks, row_evaluate

            masks = population_masks(self.plan.graph, bound)
            result = np.zeros(len(bound), dtype=float)
            for component in self.plan.graph.row_costs:
                selected = masks[component.population].to_numpy()
                result[selected] += row_evaluate(
                    component.formula, bound.loc[selected]
                )
            return result
        result = np.zeros(len(bound), dtype=float)
        for coefficient, names in self.terms:
            term = np.full(len(bound), coefficient, dtype=float)
            for name in names:
                term *= bound[name].to_numpy(dtype=float)
            result += term
        if not np.isfinite(result).all():
            raise ValueError("Formula produced nonfinite row payroll")
        return result

    def adapt(self, frame, *, source_cost_column=None, exposure_column=None):
        """Copy and adapt actual rows to per-exposure state inputs.

        Validate declared constraints on applicable population rows. Preserve
        salary-like inputs as employee observations, not normalized amounts.
        Source totals are either supplied independently or reconstructed
        before any exposure conversion. Caller data remain unchanged.
        """
        if self.spec.graph is not None:
            from .contracts import AnalysisRequest
            from .graph_states import adapt_graph

            settings = AnalysisRequest(
                "2000-01",
                "2000-01",
                source_cost_column=source_cost_column,
                exposure_column=exposure_column,
            )
            canonical, _ = adapt_graph(self.plan.graph, frame, settings)
            return canonical
        bound = frame.copy(deep=True)
        reverse = {v: k for k, v in PRODUCTION_COLUMNS.items()}
        if "motivation_type" not in bound:
            raise ValueError("Missing production column: motivation_type")
        structural = set(self.spec.structure.grade_columns)
        if self.spec.structure.region_column is not None:
            structural.add(self.spec.structure.region_column)
        if structural - set(frame):
            raise ValueError("Missing declared composition source columns")
        permanent = bound["motivation_type"].eq("ТК")
        for item in self.spec.inputs:
            if item.source not in frame:
                raise ValueError(f"Missing declared source: {item.source}")
            values = frame[item.source]
            finite_numeric(
                values.loc[permanent],
                item.name,
                lower=0 if item.constraint != "signed" else None,
                strict=item.constraint == "positive",
            )
            bound[reverse.get(item.name, item.name)] = values
        if source_cost_column is None:
            name = "__model_reference__"
            if name in bound:
                raise ValueError(f"Reserved source column: {name}")
            canonical = pd.DataFrame(
                {
                    x.name: frame.loc[permanent, x.source]
                    for x in self.spec.inputs
                }
            )
            bound[name] = np.where(
                bound.motivation_type.eq("ГПД"), bound.gpd_pay, bound.pkc_pay
            )
            bound.loc[permanent, name] = self.row_cost(canonical)
            reference = name
        else:
            reference = source_cost_column
        result = adapt_production(
            bound,
            source_cost_column=reference,
            exposure_column=exposure_column,
            registry=self.registry,
        )
        structure = self.spec.structure
        if structure.region_column is not None:
            result.loc[permanent, "region"] = frame.loc[
                permanent, structure.region_column
            ]
        if structure.grade_columns != ("grade",):
            columns = list(structure.grade_columns)
            values = frame.loc[permanent, columns]
            if values.isna().any().any():
                raise ValueError("Missing applicable composition dimension")
            result.loc[permanent, "grade"] = values.apply(
                lambda row: json.dumps(row.tolist(), ensure_ascii=False),
                axis=1,
            )
        if source_cost_column is None:
            result["reference_basis"] = np.where(
                permanent,
                "reconstructed declared payroll",
                "supplied contract payroll",
            )
        return result


def _validate_structure(structure):
    """Require coherent region-first composition and affine child games."""
    baseline = WorkforceStructure()
    if (
        structure.hierarchy != baseline.hierarchy
        or structure.propagation != "affine_child"
    ):
        raise ValueError(
            "Unsupported hierarchy: use organization/team/contract/permanent "
            "with affine_child propagation; nonlinear parents need an "
            "engine extension"
        )
    declared = [
        (d.driver, d.dimensions, d.given) for d in structure.distributions
    ]
    supported = [
        (d.driver, d.dimensions, d.given) for d in baseline.distributions
    ]
    if declared != supported:
        raise ValueError(
            "Unsupported conditioning: declare whole team and contract "
            "vectors, region within team/permanent, and grade within region"
        )
    if not structure.grade_columns or len(set(structure.grade_columns)) != len(
        structure.grade_columns
    ):
        raise ValueError("Grade composition needs unique source dimensions")
    if len(set(structure.fallback_order)) != len(
        structure.fallback_order
    ) or any(
        x not in baseline.fallback_order for x in structure.fallback_order
    ):
        raise ValueError("Fallbacks must stay within grade/region/team")


def _aligned_cost(terms, inputs):
    """Construct a pure cell contribution preserving all employee pairing."""

    def contribution(state):
        """Evaluate declared joint products before profile aggregation."""
        shape = (
            len(state["_policy"]["regions"]),
            len(state["_policy"]["grades"]),
        )
        result = np.zeros(shape)
        for coefficient, names, regional in terms:
            values = np.full(len(state["_employees"]), coefficient)
            for name in names:
                values *= state[inputs[name].driver_id]
            cells = profile_mean(state, values, base_pay=True)
            if regional:
                cells *= state["_policy"]["regional_coefficients"][:, None]
            result += cells
        return result

    return contribution


def _vector_builder(column):
    """Bind an employee-vector builder without capturing mutable state."""

    def build(context):
        """Build observed values on the declared employee axis."""
        return employee_vector(context, column)

    return build


def _payment_cost(name, coefficient):
    """Bind the linear contribution of one per-exposure payment matrix."""

    def contribution(state):
        """Scale a payment without changing its signed values."""
        return state[name] * coefficient

    return contribution


def compile_model(spec):
    """Compile a declarative labor-cost model into engine registrations.

    Parameters
    ----------
    spec : ModelSpec
        Row formula, input representations and workforce structure.

    Returns
    -------
    CompiledModel
        Serializable declarations plus pure costing/state-builder kernels.

    Raises
    ------
    ValueError
        Units, identifiers, dependencies, aggregation or hierarchy cannot
        support the declared counterfactuals. Products of separately
        averaged inputs are never assumed to preserve row-level products.
    """
    if not isinstance(spec, ModelSpec):
        raise ValueError("Expected ModelSpec")
    if not all(
        isinstance(v, str) and v.strip() for v in (spec.model_id, spec.version)
    ):
        raise ValueError("Model ID and version must be nonempty strings")
    if spec.graph is not None:
        from .graph_compile import compile_graph

        if spec.formula is not None or spec.inputs:
            raise ValueError(
                "Do not mix legacy payroll and v2 graph declarations"
            )
        if (
            spec.structure != WorkforceStructure()
            or spec.missing_economics != "carry_observed"
            or spec.exposure_column is not None
        ):
            raise ValueError(
                "For v2 models declare structure, missing_economics and "
                "source exposure inside ModelGraph; legacy options would "
                "otherwise be ignored"
            )
        return CompiledModel(spec, (), (), compile_graph(spec.graph))
    if not isinstance(spec.structure, WorkforceStructure):
        raise ValueError("Expected a WorkforceStructure declaration")
    if not all(isinstance(x, InputSpec) for x in spec.inputs):
        raise ValueError("Inputs must be InputSpec declarations")
    if not all(isinstance(x.name, str) for x in spec.inputs):
        raise ValueError("Input names must be strings")
    _validate_structure(spec.structure)
    if spec.missing_economics not in {"carry_observed", "require_reference"}:
        raise ValueError("Unknown missing-economics policy")
    inputs = {x.name: x for x in spec.inputs}
    if len(inputs) != len(spec.inputs):
        raise ValueError("Input names must be unique")
    fixed = {d.name for d in DRIVERS if not d.columns}
    reserved = {
        "period",
        "employee_id",
        "team",
        "contract_type",
        "exposure",
        "source_cost",
        "base_pay_factor",
        "contract_reimbursement",
        "gpd_pay",
        "pkc_pay",
        "reference_basis",
        "region",
        "grade",
    }
    driver_ids = []
    for item in spec.inputs:
        if (
            not isinstance(item.name, str)
            or not item.name.isidentifier()
            or item.name in reserved
            or item.name.startswith("_")
            or not isinstance(item.source, str)
            or not item.source
            or not isinstance(item.label, str)
            or not item.label.strip()
        ):
            raise ValueError("Inputs need valid names, sources and labels")
        if item.representation not in {"aligned", "payment", "policy"}:
            raise ValueError("Use aligned, payment or fixed policy inputs")
        if item.units not in {"currency", "dimensionless"}:
            raise ValueError("Declare currency or dimensionless units")
        if item.constraint not in {"signed", "positive", "nonnegative"}:
            raise ValueError("Unknown input constraint")
        if item.representation == "policy":
            if item.name != "regional_coefficient":
                raise ValueError(
                    "Only fixed regional coefficient is supported"
                )
        else:
            if (
                not isinstance(item.driver_id, str)
                or not item.driver_id.isidentifier()
                or item.driver_id in fixed
                or item.driver_id.startswith("_")
            ):
                raise ValueError("Invalid or reserved driver identifier")
            driver_ids.append(item.driver_id)
        if item.representation == "payment" and item.units != "currency":
            raise ValueError("Payment means must have currency units")
    if len({x.source for x in spec.inputs}) != len(spec.inputs):
        raise ValueError(
            "Inputs sharing a source must reuse one field and atomic driver"
        )
    if len(driver_ids) != len(set(driver_ids)):
        raise ValueError("Each input must have one unique atomic driver")
    required = {
        "base_salary": (
            "aligned",
            "currency",
            "positive",
            "underlying_salary_level",
        ),
        "worked_fraction": (
            "aligned",
            "dimensionless",
            "nonnegative",
            "worked_hours",
        ),
        "regional_coefficient": (
            "policy",
            "dimensionless",
            "positive",
            "regional_coefficient",
        ),
    }
    for name, expected in required.items():
        item = inputs.get(name)
        if (
            item is None
            or (
                item.representation,
                item.units,
                item.constraint,
                item.driver_id,
            )
            != expected
        ):
            raise ValueError(f"Preserve the declared {name} input semantics")
    terms = tuple(monomials(spec.formula))
    aligned, payments = [], {}
    for coefficient, names in terms:
        if any(name not in inputs for name in names):
            raise ValueError("Formula contains an unbound input")
        if sum(inputs[n].units == "currency" for n in names) != 1:
            raise ValueError("Every payroll term must have currency units")
        if names.count("regional_coefficient") > 1:
            raise ValueError("Apply the regional coefficient exactly once")
        regional = "regional_coefficient" in names
        factors = tuple(n for n in names if n != "regional_coefficient")
        representations = {inputs[n].representation for n in factors}
        if representations == {"aligned"}:
            aligned.append((coefficient, factors, regional))
        elif representations == {"payment"} and len(factors) == 1:
            name = factors[0]
            old = payments.get(name, (0.0, regional))
            if old[1] != regional:
                raise ValueError("Payment regional eligibility must be fixed")
            payments[name] = (old[0] + coefficient, regional)
        else:
            raise ValueError(
                "Cannot multiply separately aggregated payment means; "
                "declare jointly aligned inputs for row-level products"
            )
    if not aligned:
        raise ValueError("The workforce template requires an aligned pay term")
    registry = [d for d in DRIVERS if not d.columns]
    labels = {d.driver: d.label for d in spec.structure.distributions}
    labels["within_cell_workforce_mix"] = spec.structure.membership_label
    registry = [
        replace(d, label=labels.get(d.name, d.label)) for d in registry
    ]
    dependencies = tuple(
        dict.fromkeys(
            ["within_cell_workforce_mix"]
            + [inputs[n].driver_id for _, names, _ in aligned for n in names]
        )
    )
    original_salary = next(
        d for d in DRIVERS if d.name == "underlying_salary_level"
    )
    baseline = (
        len(aligned) == 1
        and aligned[0][0] == 1
        and sorted(aligned[0][1]) == ["base_salary", "worked_fraction"]
        and aligned[0][2]
    )
    for item in spec.inputs:
        if item.representation == "aligned":
            contribution = None
            regional = False
            if item.name == "base_salary":
                contribution = (
                    original_salary.cell_cost
                    if baseline
                    else _aligned_cost(tuple(aligned), inputs)
                )
                regional = baseline
            registry.append(
                Driver(
                    item.driver_id,
                    item.label,
                    PARENT,
                    _vector_builder(item.name),
                    "employee_" + item.constraint,
                    (item.name,),
                    contribution,
                    regional,
                    dependencies if contribution is not None else (),
                )
            )
        elif item.representation == "payment" and item.name in payments:
            coefficient, regional = payments[item.name]
            driver = payment_driver(
                item.driver_id,
                item.label,
                item.name,
                apply_regional_coefficient=regional,
            )
            registry.append(
                replace(
                    driver,
                    cell_cost=_payment_cost(item.driver_id, coefficient),
                    dependencies=(item.driver_id,),
                )
            )
    from .graph_compile import compile_graph
    from .graph_templates import legacy_graph

    plan = compile_graph(legacy_graph(spec))
    return CompiledModel(spec, tuple(registry), terms, plan)
