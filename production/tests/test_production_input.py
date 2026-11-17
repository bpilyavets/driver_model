"""Public preparation limits, audit ownership and production integration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import labor_cost as lca

EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "prod_model_example.py"
)
namespace = {}
exec(
    compile(
        EXAMPLE.read_text().split("compiled = lca.compile_model(model)")[0],
        str(EXAMPLE),
        "exec",
    ),
    namespace,
)
production_model = namespace["model"]
REQUEST = lca.AnalysisRequest(
    "2026-09", "2026-10", explainer="permutation", n_permutations=4
)


def source_rows():
    """Provide 200 source rows, two defective and two without teams."""
    rows = []
    categories = ("ТК", "ТК", "ГПД", "ПКЦ", "ПКЦ") + ("ТК",) * 195
    for i, category in enumerate(categories):
        row = {field.source: 1.0 for field in production_model.graph.fields}
        row.update(
            {
                "period": pd.Timestamp("2026-09-01"),
                "mdm_employee_rk_hash": f"private-person-{i}",
                "motivation_type": category,
                "lvl7_mapped_management_unit_nm": "Operations",
                "grade": "G1",
            }
        )
        rows.append(row)
    rows[1]["base_stavka"] = np.nan
    rows[1]["reg_coef"] = np.nan
    rows[2]["gpd_adjusted_minute_rate"] = 0.0
    rows[2]["all_effect_min_gpd"] = 0.0
    rows[3]["lvl7_mapped_management_unit_nm"] = None
    rows[4]["lvl7_mapped_management_unit_nm"] = "  "
    return pd.DataFrame(rows)


def prepare(raw, model=production_model, request=REQUEST, **kwargs):
    """Call the public API with a versioned, immutable source snapshot."""
    return lca.prepare_input(
        model,
        lca.DatasetSnapshot("audit-test", "1", raw),
        request,
        missing_team="Unassigned",
        **kwargs,
    )


def test_exclusions_are_visible_and_recoverable(capsys):
    """Count rows once, preserve source evidence and retain unknown teams."""
    raw = source_rows()
    original = raw.copy(deep=True)
    prepared = prepare(raw)
    retained, audit = prepared.frame, prepared.audit
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(audit["source_snapshot"], original)
    assert len(retained) == 198
    assert audit["summary"].excluded_rows.sum() == 2
    assert audit["summary"].retained_rows.sum() == 198
    assert audit["excluded_rows"].audit_source_row.tolist() == [1, 2]
    assert "permanent_rate" in audit["excluded_rows"].audit_reasons.iloc[0]
    assert "reg_coefficient" in audit["excluded_rows"].audit_reasons.iloc[0]
    assert len(audit["team_assignments"]) == 2
    assert (
        retained.loc[[3, 4], "lvl7_mapped_management_unit_nm"]
        .eq("Unassigned")
        .all()
    )
    output = capsys.readouterr().out
    assert "Excluded 2 of 200" in output and "gpd_rate" in output
    assert "private-person-" not in output
    assert "after exclusions" in output
    assert prepared.preparation["drop_fraction"] == 0.01
    assert (
        prepared.preparation["raw_fingerprint"]
        == lca.DatasetSnapshot("audit-test", "1", original).fingerprint
    )
    audit["excluded_rows"].iloc[0, 0] = -1
    assert prepared.audit["excluded_rows"].audit_source_row.iloc[0] == 1
    retained.iloc[0, 0] = -1000
    assert prepared.frame.iloc[0, 0] != -1000


@pytest.mark.parametrize(
    "count,passes", [(199, False), (200, True), (201, True)]
)
def test_strict_one_percent_boundary(count, passes):
    """Count rows once and accept exactly one percent without epsilon."""
    raw = source_rows()
    if count == 201:
        extra = raw.iloc[[-1]].copy()
        extra["mdm_employee_rk_hash"] = "extra"
        raw = pd.concat([raw, extra], ignore_index=True)
    raw = raw.iloc[:count]
    if passes:
        assert len(prepare(raw).frame) == count - 2
    else:
        with pytest.raises(lca.InputPreparationError, match="1% limit") as exc:
            prepare(raw)
        assert exc.value.preparation["excluded_rows"] == 2
        assert len(exc.value.audit["excluded_rows"]) == 2


def test_missing_values_are_scoped_and_signed_pay_is_retained():
    """Retain signed corrections and permit absent inapplicable columns."""
    raw = source_rows().iloc[[0, 3]].copy()
    raw.loc[0, "sum_pay_bl_reg_base"] = -25.0
    raw.loc[3, "base_stavka"] = np.nan
    raw.loc[3, "grade"] = None
    raw = raw.drop(columns="gpd_adjusted_minute_rate")
    prepared = prepare(raw)
    assert len(prepared.frame) == 2 and prepared.audit["excluded_rows"].empty
    assert prepared.frame.loc[0, "sum_pay_bl_reg_base"] == -25.0


def test_nullable_inputs_outside_months_and_duplicate_index():
    """Outside months cannot dilute the cutoff for nullable inputs."""
    raw = source_rows().iloc[[0, 3]].copy()
    raw["vyrabotka_percent"] = pd.Series(
        [pd.NA, pd.NA], index=raw.index, dtype="Float64"
    )
    outside = pd.concat([raw.iloc[[0]]] * 200, ignore_index=True)
    outside["period"] = pd.Timestamp("2026-08-01")
    raw = pd.concat([outside, raw])
    raw.index = [0] * len(raw)
    with pytest.raises(lca.InputPreparationError) as exc:
        prepare(raw)
    assert exc.value.preparation["input_rows"] == 2
    assert exc.value.audit["excluded_rows"].audit_source_row.tolist() == [200]


def test_unmapped_categories_and_missing_columns_fail():
    """Schema mistakes require correction, not excluded workforces."""
    raw = source_rows()
    with pytest.raises(ValueError, match="missing source base_stavka"):
        prepare(raw.drop(columns="base_stavka"))
    raw.loc[0, "motivation_type"] = "unmapped"
    with pytest.raises(ValueError, match="unrecognized population"):
        prepare(raw)


def test_no_rows_survive_is_explicit(capsys):
    """Stop with an inspectable audit when all selected rows are defective."""
    with pytest.raises(lca.InputPreparationError, match="analysis stopped"):
        prepare(source_rows().iloc[[1, 2]])
    assert "Excluded 2 of 2" in capsys.readouterr().out


def test_audit_evidence_survives_export(tmp_path):
    """Analysis attaches audit evidence without example-side table mutation."""
    first = source_rows()
    second = first.copy()
    second["period"] = pd.Timestamp("2026-10-01")
    raw = pd.concat([first, second], ignore_index=True)
    prepared = prepare(raw)
    result = lca.analyze(production_model, prepared, REQUEST)
    lca.export_run(result, tmp_path / "run")
    loaded = lca.load_run(tmp_path / "run")
    for name, table in prepared.audit.items():
        pd.testing.assert_frame_equal(
            loaded.table("input_quality/" + name),
            table,
        )
    assert loaded.manifest["input_preparation"] == prepared.preparation
    assert loaded.table("validation").passed.all()
    assert "local_bridges/Unassigned" in loaded.tables
    pd.testing.assert_frame_equal(
        loaded.table("source_snapshot"), prepared.frame
    )
    state = lca.build_state(production_model, prepared, "2026-09")
    assert np.isfinite(lca.cost(state, production_model))


@pytest.mark.parametrize(
    "bad_period", [pd.NaT, "2026-09-15", "2026-09-01 01:00:00"]
)
def test_invalid_period_cannot_disappear_during_selection(bad_period):
    """Periods cannot be silently omitted or normalized by preparation."""
    raw = source_rows()
    raw.loc[0, "period"] = bad_period
    with pytest.raises(ValueError, match="[Pp]eriod"):
        prepare(raw)


def test_context_cannot_change_or_audit_be_lost():
    """Require preparation again when model, periods or validation changes."""
    prepared = prepare(source_rows())
    with pytest.raises(ValueError, match="original snapshot"):
        lca.prepare_input(production_model, prepared, REQUEST)
    for changes in (
        {"period_b": "2026-11"},
        {"exposure_column": "weight"},
        {"source_cost_column": "payroll"},
    ):
        with pytest.raises(ValueError, match="prepare the raw snapshot"):
            lca.analyze(
                production_model, prepared, replace(REQUEST, **changes)
            )
    with pytest.raises(ValueError, match="model_fingerprint"):
        lca.analyze(
            replace(production_model, version="other"), prepared, REQUEST
        )


def independent_model():
    """Declare a simple all-population signed-payment graph."""
    return lca.ModelSpec(
        "independent",
        "1",
        graph=lca.ModelGraph(
            source=lca.SourceSchema(),
            populations=(),
            fields=(lca.FieldSpec("pay", "amount"),),
            row_costs=(lca.RowCost("all", lca.field("pay")),),
            factors=lca.outer_factors()
            + (
                lca.Factor(
                    "pay",
                    "payment",
                    formula=lca.field("pay"),
                ),
            ),
            games=(lca.Game("unit", lca.value("pay"), (lca.Switch("pay"),)),),
            team_game="unit",
            organization=lca.organization_game("unit"),
            local=lca.local_game("unit"),
        ),
    )


def independent_frame():
    """Provide two clean periods with an implicit all population."""
    return pd.DataFrame(
        {
            "period": ["2026-09-01", "2026-10-01"],
            "employee_id": ["a", "a"],
            "employment_type": ["custom", "custom"],
            "team": ["A", "A"],
            "amount": [-10.0, 20.0],
        }
    )


def test_independent_model_and_strict_analyze():
    """Preparation supports implicit all; ordinary analysis never cleans."""
    model = independent_model()
    raw = independent_frame()
    prepared = prepare(raw, model)
    result = lca.analyze(model, prepared, REQUEST)
    assert result.table("bridge").impact.sum() == pytest.approx(30)
    raw.loc[0, "team"] = None
    with pytest.raises(ValueError, match="identifiers"):
        lca.analyze(model, lca.DatasetSnapshot("raw", "1", raw), REQUEST)
    with pytest.raises(lca.InputPreparationError):
        lca.prepare_input(model, lca.DatasetSnapshot("raw", "1", raw), REQUEST)
    assert prepare(raw, model).frame.team.iloc[0] == "Unassigned"


@pytest.mark.parametrize("kind", ["duplicate", "team_conflict"])
def test_structural_conflicts_are_not_auto_repaired(kind):
    """Never choose arbitrarily between duplicate keys or team assignments."""
    raw = independent_frame()
    extra = raw.iloc[[0]].copy()
    if kind == "team_conflict":
        extra["employment_type"] = "second"
        extra["team"] = "B"
    raw = pd.concat([raw, extra], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate|exactly one team"):
        prepare(raw, independent_model())


@pytest.mark.parametrize("setting", ["exposure_column", "source_cost_column"])
def test_requested_numeric_sources_are_validated(setting):
    """Explicit exposure and source payroll cannot bypass input validation."""
    raw = independent_frame()
    raw["reference"] = [np.nan, 1.0]
    with pytest.raises(lca.InputPreparationError) as exc:
        prepare(
            raw,
            independent_model(),
            replace(REQUEST, **{setting: "reference"}),
        )
    assert setting in exc.value.audit["excluded_rows"].audit_reasons.iloc[0]


@pytest.mark.parametrize("version", [1, 2])
def test_builtin_declarations_and_joint_categories(
    complete_frame, months, version
):
    """Both compiled declaration versions support joint grade mappings."""
    model = lca.labor_cost_model(schema_version=version)
    if version == 1:
        model = replace(
            model,
            structure=replace(
                model.structure,
                grade_columns=("grade", "second_grade"),
            ),
        )
    else:
        dimensions = dict(model.graph.source.dimensions)
        dimensions["grade"] = ("grade", "second_grade")
        model = replace(
            model,
            graph=replace(
                model.graph,
                source=replace(
                    model.graph.source,
                    dimensions=tuple(dimensions.items()),
                ),
            ),
        )
    raw = complete_frame.copy()
    raw["second_grade"] = "level"
    request = lca.AnalysisRequest(*map(str, months))
    prepared = prepare(raw, lca.compile_model(model), request)
    assert prepared.audit["excluded_rows"].empty
    pd.testing.assert_frame_equal(prepared.frame, raw)


def test_predicate_categories_are_scoped():
    """A grade predicate for permanent staff cannot reject other categories."""
    model = independent_model()
    model = replace(
        model,
        graph=replace(
            model.graph,
            source=lca.SourceSchema(dimensions={"grade": "grade"}),
            populations=(
                lca.Population(
                    "graded",
                    {
                        "employment_type": "permanent",
                        "grade": "G1",
                    },
                ),
            ),
        ),
    )
    raw = independent_frame()
    raw["grade"] = None
    raw["team"] = raw.team.astype("string")
    prepared = prepare(raw, model)
    pd.testing.assert_frame_equal(prepared.frame, raw)


def test_unknown_type_is_not_hidden_by_another_defect():
    """Unknown workforce mappings still fail when an employee ID is missing."""
    raw = source_rows()
    raw.loc[0, "motivation_type"] = "unmapped"
    raw.loc[0, "mdm_employee_rk_hash"] = None
    with pytest.raises(ValueError, match="unrecognized population"):
        prepare(raw)


@pytest.mark.parametrize("kind", ["duplicate", "team_conflict"])
def test_defective_row_cannot_hide_a_structural_conflict(kind):
    """Removing invalid pay cannot quietly choose among conflicting rows."""
    raw = source_rows().iloc[[0] + list(range(5, 200))].copy()
    extra = raw.iloc[[0]].copy()
    extra["base_stavka"] = np.nan
    if kind == "team_conflict":
        extra["motivation_type"] = "ПКЦ"
        extra["lvl7_mapped_management_unit_nm"] = "Other"
        extra["pkc_pay"] = np.nan
    raw = pd.concat([raw, extra], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate|exactly one team"):
        prepare(raw)
