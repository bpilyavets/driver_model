"""Validate the delivered input workflow, docstrings and PEP-8 style."""

import ast
import json

import nbformat
from nbclient import NotebookClient
import pandas as pd
import pycodestyle
import pytest

from conftest import NOTEBOOK, ROOT, payroll_row


def test_notebook_style_and_docstrings(tmp_path):
    """Lint all notebook code and require docstrings on named definitions."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = (
        "\n\n\n".join(
            cell.source.rstrip()
            for cell in notebook.cells
            if cell.cell_type == "code"
        )
        + "\n"
    )
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            assert ast.get_docstring(node), node.name
    path = tmp_path / "notebook_code.py"
    path.write_text(code)
    style = pycodestyle.StyleGuide(quiet=True, max_line_length=79)
    files = [str(path)] + [str(p) for p in (ROOT / "tests").glob("*.py")]
    assert style.check_files(files).total_errors == 0
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert not cell.outputs
    implementation = [
        cell
        for cell in notebook.cells
        if "implementation" in cell.metadata.get("tags", [])
    ]
    assert implementation
    assert "synthetic_production" not in code
    assert "costs_poc.csv" not in code
    assert "PERIOD_A" not in code and "PERIOD_B" not in code


@pytest.mark.kernel
@pytest.mark.parametrize("configured", [False, True])
def test_fresh_kernel_input_workflow(tmp_path, configured):
    """Run fresh kernels with empty and signed production-schema inputs."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    frame = pd.DataFrame(
        [
            payroll_row(
                "2025-03-01",
                "p",
                bonus_kpi_comp=-500,
            ),
            payroll_row(
                "2025-04-01",
                "p",
                bonus_kpi_comp=-300,
            ),
        ]
    )
    if configured:
        encoded = frame.to_json(orient="records")
        for cell in notebook.cells:
            if "user-input" in cell.metadata.get("tags", []):
                cell.source = (
                    "import json\n"
                    f"input_df = pd.DataFrame(json.loads({encoded!r}))\n"
                    'period_a = "2025-03"\n'
                    'period_b = "2025-04"\n'
                    'source_cost_column = "actual_cost"\n'
                    'selected_team = "Alpha"\n'
                    "make_plots = True\n"
                    "show_detailed_diagnostics = True\n"
                    "references = None\n"
                    'missing_economics = "carry_observed"\n'
                )
    final_check = (
        "assert analysis['validation'].passed.all()\n"
        "assert analysis['upper']['base'] == -400\n"
        "assert analysis['upper']['end'] == -200\n"
        "assert analysis['periods'][0] == pd.Timestamp('2025-03-01')\n"
        if configured
        else "assert analysis is None\n"
    )
    notebook.cells.append(nbformat.v4.new_code_cell(final_check))
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    outputs = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    ]
    assert not any(output.output_type == "error" for output in outputs)
    if configured:
        figures = [
            output
            for output in outputs
            if "image/png" in output.get("data", {})
        ]
        assert len(figures) == 2
    else:
        assert any(
            "Set input_df" in output.get("text", "") for output in outputs
        )
    # Executed fixtures remain temporary; do not save real/example payroll
    # into the user-facing notebook.
    nbformat.write(notebook, tmp_path / "executed.ipynb")
