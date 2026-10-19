"""Frontend-independent analysis orchestration and standalone costing."""

from .compiler import CompiledModel, compile_model
from .contracts import (
    AnalysisRequest,
    DatasetSnapshot,
)
from .engine.attribution import validate_explainer_options
from .engine.costing import cost as engine_cost
from .engine.registry import DRIVERS
from .engine.workflow import analyze_labor_cost as analyze_labor_cost
from .models import labor_cost_model


def analyze(model, dataset, request):
    """Run one published model over both full-organization periods.

    Parameters
    ----------
    model : ModelSpec or CompiledModel
        The versioned payroll, state and attribution declarations.
    dataset : DatasetSnapshot
        Predefined source rows. The selected-period snapshot is included
        with the result; source data are never changed or filtered by team.
    request : AnalysisRequest
        Comparison periods and explicit calculation settings.

    Returns
    -------
    AnalysisResult
        Signed currency bridges, canonical/source data, validation and
        assumptions. Invalid inputs or counterfactuals raise ValueError;
        reconciliation failures raise AssertionError.
    """
    if not isinstance(dataset, DatasetSnapshot):
        raise ValueError("Expected a versioned DatasetSnapshot")
    if not isinstance(request, AnalysisRequest):
        raise ValueError("Expected AnalysisRequest")
    compiled = (
        model if isinstance(model, CompiledModel) else compile_model(model)
    )
    validate_explainer_options(
        request.explainer, request.n_permutations, request.random_seed
    )
    from .graph_api import analyze_graph

    return analyze_graph(compiled, dataset, request)


def cost(state, model=None):
    """Evaluate signed currency cost independently of history and SHAP.

    Positive-share categories need explicit economics. No history is copied
    here. Supply the compiled model when the state uses custom declarations.
    """
    if state.get("_graph_state"):
        from .graph_cost import graph_cost

        compiled = (
            default_model()
            if model is None
            else (
                model
                if isinstance(model, CompiledModel)
                else compile_model(model)
            )
        )
        if state.get("_model_fingerprint") != compiled.fingerprint:
            raise ValueError(
                "Supply the matching compiled model: fingerprints differ"
            )
        return graph_cost(compiled.plan, state)
    identity = state.get("_model_fingerprint")
    if model is None:
        if identity is not None and identity != labor_cost_model().fingerprint:
            raise ValueError("Supply the compiled model for this custom state")
        registry = DRIVERS
    else:
        compiled = (
            model if isinstance(model, CompiledModel) else compile_model(model)
        )
        if identity is not None and identity != compiled.fingerprint:
            raise ValueError("State and model fingerprints differ")
        registry = compiled.registry
    return engine_cost(state, registry)


def default_model():
    """Compile the documented, observed-salary model."""
    return compile_model(labor_cost_model())


def build_state(
    model, dataset, period, *, source_cost_column=None, exposure_column=None
):
    """Build one observed state for standalone costing without attribution.

    Parameters
    ----------
    model : ModelSpec or CompiledModel
        Declared payroll and state representations.
    dataset : DatasetSnapshot
        Employee/employment-type/month records.
    period : str or Timestamp
        One observed month. No other month supplies economics or inputs.
    source_cost_column, exposure_column : str, optional
        Independent payroll reference and explicit exposure source.

    Returns
    -------
    dict
        Validated observed state. Missing categories remain unavailable;
        positive-share unknown economics fail. No SHAP import is needed.
    """
    import pandas as pd

    from .engine.common import check
    from .engine.inputs import month_start
    from .graph_cost import graph_cost
    from .graph_states import adapt_graph, build_states

    compiled = (
        model if isinstance(model, CompiledModel) else compile_model(model)
    )
    graph = compiled.plan.graph
    month = month_start(period)
    source = dataset.frame
    dates = pd.to_datetime(source[graph.source.period])
    rows = source.loc[dates.dt.to_period("M").eq(month.to_period("M"))]
    if rows.empty:
        raise ValueError("No observed records for the requested month")
    request = AnalysisRequest(
        str(month),
        str(month),
        source_cost_column=source_cost_column,
        exposure_column=exposure_column,
    )
    canonical, _ = adapt_graph(graph, rows, request)
    states, _ = build_states(compiled.plan, canonical, (month, month), request)
    state = states[0]
    state["_graph_state"] = True
    state["_model_fingerprint"] = compiled.fingerprint
    check(
        "Standalone source cost",
        graph_cost(compiled.plan, state),
        canonical.source_cost.sum(),
    )
    return state
