# Documentation reconciliation — 2026-11-22

Reviewed implementation: repository revision `43ce7b5`, with the
documentation-only corrections recorded here. This audit uses runtime
behavior as the factual baseline for descriptions of what is implemented.
The [methodology](../../docs/labor-cost-decomposition.md) remains
authoritative, as required by [AGENTS.MD](../../AGENTS.MD). Unimplemented
requirements remain requirements; this exercise does not waive them.

## Scope and corrections

The review covered contributor instructions, both READMEs, methodology,
authoring and reporting references, the permanent/temporary walkthrough,
deployment and bundle contracts, notebook instructions, public API
docstrings, and historical verification/accuracy records. Existing source
formulas, declaration identities, tests, user experiments and saved notebook
outputs are preserved. The only Python changes are docstrings.

| Documentation surface | Implementation evidence | Resolution |
| --- | --- | --- |
| Methodology version table and deployment provenance | [pyproject.toml](../pyproject.toml), [contracts.py](../src/labor_cost/contracts.py), [package exports](../src/labor_cost/__init__.py), [bootstrap.py](../runtime/bootstrap.py) | Distribution and engine are 0.3.0; module `__version__` remains 0.2.0. Describe each surface and retain the inconsistency as an implementation issue. |
| AGENTS pipeline and README entry points | [preparation.py](../src/labor_cost/preparation.py), [api.py](../src/labor_cost/api.py) | Add optional audited preparation, the selected-period 1% exclusion limit, explicit missing-team handling and strict ordinary analysis. |
| README reporting overview and methodology API/reporting sections | [reporting.py](../src/labor_cost/reporting.py) | Document default ranked impacts with separate totals, red increases/green decreases, individual-impact tooltips and top-N waterfalls with signed Other. Team arguments select presentation views. |
| AGENTS frontend boundary | [package exports](../src/labor_cost/__init__.py), maintained Jupyter examples | Describe reporting/bundle consumers; an included dashboard is not an implemented interface. |
| Authoring primitive inventory, units and field references | [graph_compile.py](../src/labor_cost/graph_compile.py), [aggregation.py](../src/labor_cost/aggregation.py) | Include mean/weighted_mean, source/weight IDs, actual versus per-exposure payment aggregation, and time/exposure and 1/exposure units. |
| Authoring reconciliation wording | [graph_api.py](../src/labor_cost/graph_api.py), reproduced GAP-001 probe | Qualify endpoint checks: offsetting population errors can pass. Keep independent population reconciliation required. |
| Authoring publication guarantee | [catalog.py](../src/labor_cost/catalog.py) | Publication refuses an existing destination file; it is not a global version-uniqueness registry. |
| Bundle and methodology diagnostic inventories; gap evidence map | [graph_api.py](../src/labor_cost/graph_api.py), [api.py](../src/labor_cost/api.py) | Include aggregate_factors, zero-weight assumptions and optional input_quality tables. Distinguish them from still-missing support/movement summaries. |
| Bundle manifest and replay description | [bundle.py](../src/labor_cost/bundle.py), [preparation.py](../src/labor_cost/preparation.py) | Distinguish the coalition_audit table from manifest metadata. Rebuilding a plain snapshot reproduces retained inputs, not the preparation audit attachment or the full original dataset fingerprint. |
| Public snapshot/result copy docstrings and bundle usage | [contracts.py](../src/labor_cost/contracts.py), direct nested-object probe | Describe pandas deep copies accurately: nested Python objects in object-dtype cells are not recursively copied. Reports separately copy nested values. |
| Walkthrough installation/publication examples | [GUIDE.MD](../examples/GUIDE.MD), isolated execution | Replace workstation-specific absolute paths with repository-relative installation and a caller-relative publication directory. Preserve the 6,575 → 6,982 fixture and +407 change. |
| Notebook-output claims | Actual notebook JSON and [quality tests](../tests/test_quality.py) | Preserve the convention for new notebooks, identify saved executions in model_authoring/reporting_showcase, and retain existing outputs. Add the version caveat to direct-loading notebook guidance only. |
| Historical verification and accuracy claims | Dated [verification records](verification.md), [accuracy study](../../docs/explainer-accuracy.md) | Keep measured counts/timings unchanged; clarify historical status and the PoC permanent-game scope of the accuracy measurements. |

Notation validation, rich-display behavior, named-axis/conditional formulas,
aligned aggregation, grouped switches, explicit references, exact/permutation
settings, bundle schema compatibility and reporting signatures were also
checked against their implementation. No runtime changes were made to make
documentation claims pass.

## Remaining implementation issues

- `labor_cost.__version__` is inconsistent with distribution and engine
  versions. The source loader records that stale module value. Keep commit
  and file hashes as code provenance; correcting the constant is separate
  from this documentation-only change.
- All four [architecture gaps](architecture-gaps.md) remain open:
  independent population aggregate reconciliation; complete support and
  movement diagnostics; independent template M/K/L checks and a complete
  validation ledger; and direct graph coverage of outstanding guarantees.
- The intentionally invalid GAP-001 example still passes all recorded
  validation checks while attributing the wrong population amounts. New
  aggregate/preparation diagnostics and reports do not close this gap.
- Current quality tests conflict with preserved user content: the authoring
  notebook contains saved executions, and an existing experiment has two
  style violations. These failures are reported below, not repaired by
  deleting outputs or editing an unrelated experiment.

## Verification performed for this audit

Commands below ran from the repository root with the existing production
environment:

```bash
production/.venv/bin/python -m pytest -c production/pyproject.toml \
  production/tests/test_production_input.py \
  production/tests/test_aggregates.py production/tests/test_reporting.py -q
production/.venv/bin/python -m pytest -c production/pyproject.toml \
  production/tests/test_quality.py -q
```

- Preparation, aggregate and reporting tests: **114 passed** in 37.46 s.
  These include the fixed exclusion limit, aggregate identities and old
  fingerprints, zero-weight policy, portability, chart semantics and a
  fresh-kernel report check.
- Quality tests: **7 passed, 2 failed** in 52.68 s. The passing cases run
  the production client and comparison notebook in fresh kernels,
  including configured exact/permutation comparisons. Failures are
  `test_style_and_docstrings` (the unchanged
  `examples/poc_extented_model_test.py`: E501 at line 194, W292 at line 418)
  and `test_authoring_notebook_fresh_kernel` (its initial saved-output
  assertion, before execution).
- Separately executed an in-memory copy of the authoring notebook after
  clearing only the copy's outputs: fresh-kernel execution succeeded with
  six rich mathematical outputs. The original file stayed byte-for-byte
  unchanged.
- Executed the walkthrough's first 26 Python blocks in a temporary
  directory, suppressing display only. Verified previews, observed costing,
  exact analysis, reports, publication and export/load. Also ran its
  permutation request: both bridges reconcile to +407. The later fragments
  requiring analyst-supplied production data were syntax-checked, not run
  against a DWH or production catalog.
- Re-executed the GAP-001 probe and confirmed the documented limitation.
  Checked module/engine version values and nested-object copy semantics.
- Parsed 20 first-party documents/notebooks with Mistune: checked 137 local
  link targets/heading anchors and syntax for 60 Python fenced blocks,
  ignoring code when extracting links. Reviewed notebook guidance without
  rewriting code cells or saved outputs.
- Checked the final diff for whitespace, unchanged executable Python AST,
  notebook code/outputs/metadata preservation, unchanged numbered methodology
  headings and unchanged documentation-authority paragraphs. Ruff and
  pycodestyle checks apply to the edited Python file.

This is focused verification of documentation claims, not a new full
production/PoC regression run, deployment certification, visual redesign
review or statistical accuracy study. Earlier full-suite and browser
verification results remain dated evidence in [verification.md](verification.md).
