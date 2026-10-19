"""Run existing analytical tests against the extracted package unchanged."""

import importlib.util
import sys
from pathlib import Path

POC_TESTS = Path(__file__).resolve().parents[2] / "tests"
for filename in (
    "test_analysis",
    "test_salary_time",
    "test_shap_semantics",
    "test_explainers",
    "test_exposure",
):
    spec = importlib.util.spec_from_file_location(
        "poc_" + filename, POC_TESTS / (filename + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[filename] = module
    spec.loader.exec_module(module)
    for name, value in vars(module).items():
        if name.startswith("test_"):
            globals()[filename + "__" + name] = value
        elif hasattr(value, "_get_wrapped_function"):
            globals()[name] = value

# Bridge-summary semantics are part of the calculation contract. Visual
# waterfall/report tests remain with the unchanged PoC presentation.
spec = importlib.util.spec_from_file_location(
    "poc_driver_reporting", POC_TESTS / "test_driver_reporting.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
team_analysis = module.team_analysis
for name in (
    "test_organizational_summary_not_local_sum",
    "test_cancelling_driver_keeps_signed_team_detail",
    "test_aggregation_uses_registry_without_recomputation",
    "test_reject_local_rows_and_parents",
    "test_reconciliation_detects_misallocated_summary",
):
    globals()["test_driver_reporting__" + name] = getattr(module, name)
