"""Direct numerical comparison against the unchanged notebook oracle."""

import pandas as pd
import pytest


@pytest.mark.parity
@pytest.mark.parametrize("explainer", ["exact", "permutation"])
def test_notebook_parity(
    model, notebook_model, complete_frame, months, explainer
):
    """Keep both estimators and hierarchical results equal after extraction."""
    options = dict(
        source_cost_column="actual_cost", explainer=explainer, n_permutations=8
    )
    original = notebook_model["analyze_labor_cost"](
        complete_frame, *months, **options
    )
    extracted = model["analyze_labor_cost"](complete_frame, *months, **options)
    for key in (
        "bridge",
        "driver_bridge",
        "hierarchy",
        "team_summary",
        "assumptions",
        "frame",
        "validation",
    ):
        pd.testing.assert_frame_equal(original[key], extracted[key])
    for team in original["local_bridges"]:
        pd.testing.assert_frame_equal(
            original["local_bridges"][team], extracted["local_bridges"][team]
        )
