"""Observed cell aggregates and auditable zero-workload completion."""

import numpy as np


def aggregate_cell(spec, donor, context, cell, donor_dimensions):
    """Construct a mean from one donor pool and retain its evidence.

    Actual quantities are summed before division by exposure. Normalized
    quantities are first multiplied by exposure. Weighted rates retain
    their declared units; exposure applies to the weighting quantity only.
    """
    source = donor[spec.source].to_numpy(dtype=float)
    exposure = donor.exposure.to_numpy(dtype=float)
    total_exposure = float(exposure.sum())
    scale = np.ones(len(donor)) if spec.basis == "actual" else exposure
    method = "observed"
    if tuple(donor_dimensions) != spec.dimensions:
        method = "sparse_fallback"
    fallback_numerator = fallback_denominator = None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if spec.kind == "mean":
            numerator = float(np.dot(scale, source))
            denominator = total_exposure
        else:
            weight = donor[spec.weight].to_numpy(dtype=float)
            if not np.isfinite(weight).all() or (weight < 0).any():
                raise ValueError(f"{context}: invalid weights in {cell}")
            weights = scale * weight
            denominator = float(weights.sum())
            numerator = float(np.dot(weights, source))
        if not np.isfinite([numerator, denominator]).all():
            raise ValueError(f"{context}: nonfinite aggregate in {cell}")
        if denominator == 0:
            if spec.zero_weight == "error":
                raise ValueError(
                    f"{context}: zero total weight in {cell}; "
                    "declare zero_weight='exposure_mean'"
                )
            fallback_numerator = float(np.dot(exposure, source))
            fallback_denominator = total_exposure
            value = fallback_numerator / fallback_denominator
            method = "zero_weight_exposure_mean"
        else:
            value = numerator / denominator
    if not np.isfinite(value):
        raise ValueError(f"{context}: nonfinite aggregate in {cell}")
    record = {
        **context,
        "coordinates": cell,
        "kind": spec.kind,
        "source": spec.source,
        "weight": spec.weight,
        "basis": spec.basis,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "fallback_numerator": fallback_numerator,
        "fallback_denominator": fallback_denominator,
        "donor_rows": len(donor),
        "donor_exposure": total_exposure,
        "donor_dimensions": tuple(donor_dimensions),
        "method": method,
        "source_period": context["period"],
        "source_team": context["team"],
    }
    return value, record
