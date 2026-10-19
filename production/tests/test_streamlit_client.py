"""Optional Streamlit client integration against the same library contract."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from labor_cost import (
    DatasetSnapshot,
    labor_cost_model,
    load_run,
    publish_model,
)


@pytest.mark.filterwarnings("ignore:.*missing ScriptRunContext.*")
def test_streamlit_run_select_export(
    tmp_path, complete_frame, public_result, monkeypatch
):
    """Reuse one library run for selection and export without new impacts."""
    testing = pytest.importorskip("streamlit.testing.v1")
    publish_model(
        labor_cost_model(),
        tmp_path / "model.json",
        validated_result=public_result,
    )
    complete_frame.to_parquet(tmp_path / "data.parquet", index=True)
    snapshot = DatasetSnapshot("payroll", "v1", complete_frame)
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
                "path": "data.parquet",
                "fingerprint": snapshot.fingerprint,
            }
        ],
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(config))
    script = Path(__file__).resolve().parents[1] / "examples/streamlit_app.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(catalog)])
    app = testing.AppTest.from_file(script, default_timeout=90).run()
    assert not app.exception
    app.text_input[0].set_value("actual_cost")
    app.selectbox[4].set_value("permutation")
    app.number_input[0].set_value(4)
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["result"]
    pd.testing.assert_frame_equal(
        result.table("bridge"), public_result.table("bridge")
    )
    app.multiselect[0].set_value(["Alpha"]).run()
    assert app.session_state["result"].run_id == result.run_id
    app.text_input[1].set_value(str(tmp_path / "run"))
    app.button[1].click().run()
    assert not app.exception
    assert load_run(tmp_path / "run").run_id == result.run_id
