"""Thin selection and execution UI; calculation belongs to labor_cost."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from labor_cost import AnalysisRequest, Catalog, analyze, export_run

st.title("Labor-cost analysis")
if len(sys.argv) < 2:
    st.info(
        "Supply the published catalog path after -- when starting the app."
    )
    st.stop()

catalog = Catalog(Path(sys.argv[1]))
models, datasets = catalog.list_models(), catalog.list_datasets()
if not models or not datasets:
    st.info("Publish a model and dataset before running an analysis.")
    st.stop()

model_entry = st.selectbox(
    "Model", models, format_func=lambda x: f"{x['id']} @ {x['version']}"
)
data_entry = st.selectbox(
    "Dataset", datasets, format_func=lambda x: f"{x['id']} @ {x['version']}"
)
model = catalog.model(model_entry["id"], model_entry["version"])
snapshot = catalog.dataset(data_entry["id"], data_entry["version"])
periods = sorted(
    pd.to_datetime(snapshot.frame[model.plan.graph.source.period])
    .dt.strftime("%Y-%m")
    .unique()
)
if not periods:
    st.info("The published dataset contains no comparison periods.")
    st.stop()
period_a = st.selectbox("Period A", periods)
period_b = st.selectbox("Period B", periods, index=len(periods) - 1)
with st.expander("Calculation settings"):
    source_column = (
        st.text_input("Independent total column (optional)") or None
    )
    method = st.selectbox("Attribution method", ["exact", "permutation"])
    cycles = st.number_input("Permutation cycles", min_value=1, value=256)
request = AnalysisRequest(
    period_a,
    period_b,
    source_cost_column=source_column,
    explainer=method,
    n_permutations=int(cycles),
)
key = (model.fingerprint, snapshot.fingerprint, request)
if st.session_state.get("request_key") != key:
    st.session_state.pop("result", None)
if st.button("Run analysis"):
    try:
        with st.spinner(
            "Calculating and validating full-organization results"
        ):
            result = analyze(model, snapshot, request)
        st.session_state["result"] = result
        st.session_state["request_key"] = key
    except (ValueError, AssertionError) as exc:
        st.error(str(exc))

result = st.session_state.get("result")
if result is not None:
    st.success(f"Validated run {result.run_id}")
    teams = st.multiselect("Teams to view", result.table("team_summary").team)
    if teams:
        view = result.select_teams(teams)
        st.dataframe(view.team_summary, hide_index=True)
    st.caption("Team selection displays existing full-organization results.")
    destination = st.text_input(
        "Run bundle directory", "runs/" + result.run_id
    )
    if st.button("Export for dashboard"):
        try:
            saved = export_run(result, destination)
            st.success(f"Saved {saved}")
        except (ValueError, OSError) as exc:
            st.error(str(exc))
