"""Declarative workforce cost modeling and reconciled attribution."""

from .api import analyze, analyze_labor_cost, build_state, cost, default_model
from .bundle import export_run, load_run
from .catalog import Catalog, publish_model
from .compiler import CompiledModel, compile_model
from .contracts import (
    AnalysisRequest,
    AnalysisResult,
    DatasetSnapshot,
    ResultView,
)
from .expressions import Expr, field
from .graph import (
    Aggregate,
    Factor,
    FieldSpec,
    Game,
    ModelGraph,
    Population,
    RowCost,
    SourceSchema,
    Switch,
    aligned_mean,
    child,
    select,
    sum_over,
    value,
)
from .graph_templates import local_game, organization_game, outer_factors
from .models import labor_cost_model
from .presentation import Notation, render_formula, render_model
from .reporting import (
    NotebookReport,
    plot_driver_by_team,
    plot_organization_waterfall,
    plot_team_waterfall,
    report,
)
from .spec import DistributionSpec, InputSpec, ModelSpec, WorkforceStructure

__version__ = "0.2.0"
__all__ = [
    "build_state",
    "Aggregate",
    "Factor",
    "FieldSpec",
    "Game",
    "ModelGraph",
    "Population",
    "RowCost",
    "SourceSchema",
    "Switch",
    "aligned_mean",
    "child",
    "select",
    "sum_over",
    "value",
    "local_game",
    "organization_game",
    "outer_factors",
    "AnalysisRequest",
    "AnalysisResult",
    "Catalog",
    "CompiledModel",
    "DatasetSnapshot",
    "DistributionSpec",
    "Expr",
    "InputSpec",
    "ModelSpec",
    "Notation",
    "NotebookReport",
    "ResultView",
    "WorkforceStructure",
    "analyze",
    "analyze_labor_cost",
    "compile_model",
    "cost",
    "default_model",
    "export_run",
    "field",
    "labor_cost_model",
    "load_run",
    "publish_model",
    "plot_driver_by_team",
    "plot_organization_waterfall",
    "plot_team_waterfall",
    "report",
    "render_formula",
    "render_model",
]
