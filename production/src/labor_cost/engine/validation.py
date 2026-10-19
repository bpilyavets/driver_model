"""Numeric constraints for workforce state objects."""

from itertools import product

import numpy as np

from .common import CONTRACTS, PROB_ATOL


def validate_value(
    value, kind, policy=None, team_count=None, employee_count=None
):
    """Validate a scalar/vector/matrix using its declared semantic
    kind.
    """
    if kind == "membership_probability":
        shape = (len(policy["regions"]), len(policy["grades"]))
        cells = set(product(range(shape[0]), range(shape[1])))
        if not isinstance(value, dict) or set(value) != cells:
            raise ValueError("Membership profiles do not match cell axes")
        for profile in value.values():
            required = {"indices", "weights"}
            if not required <= set(profile) or not set(profile) <= (
                required | {"base_pay_factors"}
            ):
                raise ValueError(
                    "Membership profile needs indices and weights"
                )
            indices = np.asarray(profile["indices"])
            weights = np.asarray(profile["weights"], dtype=float)
            if (
                indices.ndim != 1
                or len(indices) == 0
                or indices.dtype.kind not in "iu"
                or len(np.unique(indices)) != len(indices)
                or (indices < 0).any()
                or (indices >= employee_count).any()
                or weights.shape != indices.shape
            ):
                raise ValueError("Invalid membership employee indices")
            validate_value(
                weights, "team_probability", team_count=len(indices)
            )
            factors = np.asarray(
                profile.get("base_pay_factors", np.ones(len(indices))),
                dtype=float,
            )
            if (
                factors.shape != indices.shape
                or not np.isfinite(factors).all()
                or (factors <= 0).any()
            ):
                raise ValueError("Invalid membership base_pay_factors")
        return
    values = np.asarray(value, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {kind}")
    shapes = {
        "scalar_signed": (),
        "scalar_nonnegative": (),
        "contract_probability": (len(CONTRACTS),),
        "team_probability": (team_count,),
        "employee_positive": (employee_count,),
        "employee_nonnegative": (employee_count,),
        "employee_signed": (employee_count,),
    }
    if policy is not None:
        shape = (len(policy["regions"]), len(policy["grades"]))
        shapes.update(
            region_probability=(shape[0],),
            grade_probability=shape,
            cell_positive=shape,
            cell_nonnegative=shape,
            cell_signed=shape,
        )
    if kind not in shapes or values.shape != shapes[kind]:
        raise ValueError(f"Wrong {kind} shape: {values.shape}")
    if kind.endswith("probability"):
        normalized = np.allclose(
            values.sum(axis=-1), 1, rtol=0, atol=PROB_ATOL
        )
        if (values < 0).any() or (values > 1).any() or not normalized:
            raise ValueError(f"Invalid probability distribution: {kind}")
    if kind in {
        "scalar_nonnegative",
        "cell_nonnegative",
        "employee_nonnegative",
    }:
        if (values < 0).any():
            raise ValueError(f"Negative {kind}")
    if kind in {"cell_positive", "employee_positive"} and (values <= 0).any():
        raise ValueError("Base salary cells must be positive")
