"""Static, installation and thin-notebook workflow checks."""

import ast
import json
import sys
from pathlib import Path

import nbformat
import pycodestyle
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from labor_cost import DatasetSnapshot, labor_cost_model, publish_model

ROOT = Path(__file__).resolve().parents[1]


def test_style_and_docstrings():
    """Require readable package code and docstrings on named definitions."""
    files = [
        p
        for folder in ("src", "tests", "examples")
        for p in (ROOT / folder).rglob("*.py")
    ]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assert ast.get_docstring(node), (path, node.name)
    style = pycodestyle.StyleGuide(quiet=True, max_line_length=79)
    assert style.check_files([str(p) for p in files]).total_errors == 0


@pytest.mark.parametrize("mode", [None, "exact", "permutation"])
def test_thin_notebook_fresh_kernel(
    tmp_path, complete_frame, public_result, mode
):
    """Run unconfigured and configured clients in isolated fresh kernels."""
    notebook_path = ROOT / "examples" / "notebook_client.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
    if mode is not None:
        publish_model(
            labor_cost_model(),
            tmp_path / "model.json",
            validated_result=public_result,
        )
        snapshot = DatasetSnapshot("payroll", "v1", complete_frame)
        complete_frame.to_parquet(tmp_path / "dataset.parquet", index=True)
        config = {
            "models": [
                {
                    "id": "labor-cost",
                    "version": "1.0.0",
                    "path": "model.json",
                    "fingerprint": labor_cost_model().fingerprint,
                }
            ],
            "datasets": [
                {
                    "id": "payroll",
                    "version": "v1",
                    "path": "dataset.parquet",
                    "fingerprint": snapshot.fingerprint,
                }
            ],
        }
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(config))
        notebook.cells[1].source += (
            f"\ncatalog_path = {str(catalog_path)!r}\n"
            "dataset_version = 'v1'\nperiod_a = '2025-03'\n"
            "period_b = '2025-04'\nsource_cost_column = 'actual_cost'\n"
            f"explainer = {mode!r}\nselected_teams = ['Alpha']\n"
        )
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                "assert result.table('validation').passed.all()\n"
                "assert result.select_teams(['Alpha']).run_id "
                "== result.run_id\n"
            )
        )
    else:
        notebook.cells.append(
            nbformat.v4.new_code_cell("assert result is None")
        )
    kernels = tmp_path / "kernels"
    kernel = kernels / "labor-test"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "Labor test",
                "language": "python",
            }
        )
    )
    manager = KernelManager(
        kernel_name="labor-test",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
    )
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=180,
        kernel_name="labor-test",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    assert all(
        output.output_type != "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
