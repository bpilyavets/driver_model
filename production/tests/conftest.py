"""Read-only PoC fixtures and independent production-engine fixtures."""

import ast
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

POC_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "poc_fixtures", POC_ROOT / "tests" / "conftest.py"
)
poc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(poc)
payroll_row = poc.payroll_row
assert_equal_states = poc.assert_equal_states
complete_frame = poc.complete_frame
months = poc.months
notebook_model = poc.model


@pytest.fixture(scope="session")
def model():
    """Expose the extracted engine to the unchanged analytical regressions."""
    namespace = {}
    for name in (
        "common",
        "validation",
        "kernels",
        "registry",
        "inputs",
        "costing",
        "states",
        "attribution",
        "diagnostics",
        "workflow",
    ):
        module = importlib.import_module("labor_cost.engine." + name)
        namespace.update(vars(module))
    # One existing exposure test also exercises the PoC presentation. Load
    # only that presentation function into this test namespace; production
    # code never imports or executes notebook cells.
    notebook = json.loads(
        (POC_ROOT / "labor_cost_decomposition.ipynb").read_text()
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name == "report":
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]),
                        "poc_presentation",
                        "exec",
                    ),
                    namespace,
                )
    return namespace


@pytest.fixture
def public_result(complete_frame, months):
    """Run the documented model once for result/bundle integration tests."""
    from labor_cost import (
        AnalysisRequest,
        DatasetSnapshot,
        analyze,
        labor_cost_model,
    )

    return analyze(
        labor_cost_model(),
        DatasetSnapshot("payroll", "v1", complete_frame),
        AnalysisRequest(
            *map(str, months),
            source_cost_column="actual_cost",
            explainer="permutation",
            n_permutations=4,
        ),
    )
