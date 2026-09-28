"""Reproduce external SHAP accuracy benchmarks; never run in the notebook.

Install requirements-dev.txt, then run this script from the repository root.
Accuracy sweeps use validated component lookup tables. Timings use the actual
notebook engine without those benchmark-only tables or coalition memoization.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from conftest import complete_frame, model, payroll_row  # noqa: E402

MONTHS = (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01"))
BUDGETS = (64, 256, 1024)


def stress_frame():
    """Build external payroll stress data with movement and signed payments."""
    payments = (
        "sum_pay_bl_reg_base",
        "sum_pay_otpusk_reg_base",
        "sum_pay_prazdnik_reg_base",
        "sum_pay_oklad_night_reg_base",
        "bonus_kpi_comp",
        "bonus_overtime_comp",
        "bonus_sharing_comp_reg_base",
        "bonus_quarterly_comp_reg_base",
        "sum_pay_compacted",
    )
    rows = []
    for side, month in enumerate(MONTHS):
        ids = range(8) if side == 0 else (*range(6), *range(8, 12))
        for i in ids:
            coefficient = (
                (1 if i < 6 else 1.4)
                if side == 0
                else (1 if i % 3 == 0 else 1.4)
            )
            grade = (
                ("G2" if i % 4 == 0 else "G1")
                if side == 0
                else ("G2" if i % 2 else "G1")
            )
            base = (40000 + 12000 * i) * (
                1 if side == 0 else 1.10 + 0.08 * (i % 3)
            )
            worked = (
                [0.25, 0.5, 0.75, 1.0] if side == 0 else [1.2, 0.95, 0.4, 0.1]
            )[i % 4]
            amounts = {
                name: (j + 1)
                * 350
                * (1 + i % 3)
                * (1 if side == 0 else 1.6)
                * (-1 if (i + j + side) % 5 == 0 else 1)
                for j, name in enumerate(payments)
            }
            rows.append(
                payroll_row(
                    str(month.date()),
                    f"p{i}",
                    base=base,
                    coefficient=coefficient,
                    worked=worked,
                    grade=grade,
                    **amounts,
                )
            )
        for contract, count in [("ГПД", 2 + side), ("ПКЦ", 1 + side)]:
            rows.extend(
                payroll_row(
                    str(month.date()),
                    f"{contract}{i}",
                    contract=contract,
                    rate=(30000 if contract == "ГПД" else 60000)
                    * (1 + side * 0.2),
                )
                for i in range(count)
            )
        rows.extend(
            payroll_row(
                str(month.date()),
                f"anchor{i}",
                team="Beta",
                contract="ГПД",
                rate=45000 - 5000 * side,
            )
            for i in range(3 if side == 0 else 1)
        )
    return pd.DataFrame(rows)


def add_six_payments(frame, ns):
    """Add six payments through source payroll and registry entries."""
    frame, registry = frame.copy(), ns["DRIVERS"]
    permanent = frame.motivation_type.eq("ТК")
    for i in range(6):
        key = f"extra_payment_{i+1}"
        frame[key] = np.where(
            frame.period.eq("2025-03-01"), -1200 * (i + 1), 1700 * (i + 1)
        )
        frame[key] *= np.where(frame.grade.eq("G2"), 1.8, 1)
        eligible = i % 2 == 0
        frame.loc[permanent, "actual_cost"] += frame.loc[permanent, key] * (
            frame.loc[permanent, "reg_coef"] if eligible else 1
        )
        registry += (
            ns["payment_driver"](
                key, key, key, apply_regional_coefficient=eligible
            ),
        )
    return frame, registry


def component_reference(ns, a, b, registry):
    """Compute exact component Shapley values and a validated lookup engine.

    This factorization is specific to the registered additive payment model.
    It is an external benchmark reference/accelerator, not a production mode.
    Every table entry uses the notebook's shared cell cost engine.
    """
    specs = ns["drivers_at"](ns["PARENT"], registry)
    reference = pd.Series(0.0, index=[d.name for d in specs])
    tables = []
    salary_names = {
        "regional_mix",
        "grade_mix_within_region",
        *ns["SALARY_KEYS"],
    }
    for contribution in (d for d in specs if d.cell_cost is not None):
        names = (
            salary_names
            if contribution.name == "underlying_salary_level"
            else {"regional_mix", "grade_mix_within_region", contribution.name}
        )
        component = tuple(d for d in specs if d.name in names)
        keys = [d.name for d in component]
        table = []
        for bits in product((0, 1), repeat=len(keys)):
            state = dict(a)
            state.update(
                {key: (a, b)[bit][key] for key, bit in zip(keys, bits)}
            )
            table.append(ns["permanent_cell_costs"](state, component).ravel())
        table = np.asarray(table)
        powers = 2 ** np.arange(len(keys) - 1, -1, -1)
        tables.append((keys, powers, table))

        def engine(state):
            """Evaluate one registered contribution through shared costing."""
            return ns["permanent_cost"](state, component)

        game = ns["exact_game"](a, b, component, engine)
        reference.loc[keys] += game["impacts"]

    def lookup(state):
        """Reuse validated component tables on fixed A/B business switches."""
        result = np.zeros_like(tables[0][2][0])
        for keys, powers, table in tables:
            bits = [int(state[key] is b[key]) for key in keys]
            result += table[int(np.dot(bits, powers))]
        return result

    rng = np.random.default_rng(984)
    for bits in [
        np.zeros(len(specs), dtype=int),
        np.ones(len(specs), dtype=int),
        *rng.integers(0, 2, size=(100, len(specs))),
    ]:
        state = dict(a)
        state.update(
            {d.name: (a, b)[bit][d.name] for d, bit in zip(specs, bits)}
        )
        ns["check"](
            "Benchmark tables vs shared engine",
            lookup(state),
            ns["permanent_cell_costs"](state, registry).ravel(),
        )
    ns["check"](
        "Exact component efficiency",
        reference.sum(),
        ns["permanent_cost"](b, registry) - ns["permanent_cost"](a, registry),
    )
    return reference, lookup


def error_metrics(reference, estimate, atol=1e-6):
    """Measure allocation error without dividing by small net cost changes."""
    truth, estimated = np.asarray(reference), np.asarray(estimate)
    error = np.abs(estimated - truth)
    gross = np.abs(truth).sum()
    material = (
        np.abs(truth) >= 0.01 * gross
        if gross > atol
        else np.zeros_like(truth, dtype=bool)
    )
    material &= np.abs(truth) > atol
    count = min(3, len(truth))
    threshold = np.sort(np.abs(truth))[-count]
    mandatory = set(np.flatnonzero(np.abs(truth) > threshold + atol))
    eligible = set(np.flatnonzero(np.abs(truth) >= threshold - atol))
    chosen = set(np.argsort(-np.abs(estimated), kind="stable")[:count])
    has_tie = len(eligible) > count
    return dict(
        normalized_l1=(error.sum() / gross if gross > atol else np.nan),
        maximum_absolute_error=error.max(initial=0),
        maximum_material_relative_error=(
            np.max(error[material] / np.abs(truth[material]))
            if material.any()
            else np.nan
        ),
        material_sign_errors=int(
            np.sum(np.sign(truth[material]) != np.sign(estimated[material]))
        ),
        top_three_match=mandatory <= chosen <= eligible,
        top_three_boundary_tie=has_tie,
    )


def record_run(
    run_rows,
    driver_rows,
    name,
    budget,
    seed,
    reference,
    game,
    factors,
    timings,
):
    """Record unit and propagated errors using existing hierarchy games."""
    for view, factor in factors.items():
        truth, estimate = factor * reference, factor * game["impacts"]
        run_rows.append(
            dict(
                case=name,
                view=view,
                cycles=budget,
                seed=seed,
                **error_metrics(truth, estimate),
                coalitions=game["audit"]["coalitions"],
                model_evaluations=game["audit"]["model_evaluations"],
                accelerated_seconds=game["audit"]["seconds"],
                **timings,
            )
        )
        gross = np.abs(truth).sum()
        for driver in truth.index:
            actual, predicted = truth[driver], estimate[driver]
            material = gross > 1e-6 and abs(actual) >= max(1e-6, 0.01 * gross)
            driver_rows.append(
                dict(
                    case=name,
                    view=view,
                    cycles=budget,
                    seed=seed,
                    driver=driver,
                    exact_impact=actual,
                    absolute_error=abs(predicted - actual),
                    relative_error=(
                        abs(predicted - actual) / abs(actual)
                        if abs(actual) > 1e-6
                        else np.nan
                    ),
                    material=material,
                    sign_error=material
                    and np.sign(actual) != np.sign(predicted),
                )
            )


def run_payroll_case(ns, name, frame, registry, seeds, run_rows, driver_rows):
    """Compare sampled allocations with exact references and real timings."""
    canonical = ns["adapt_production"](
        frame, source_cost_column="actual_cost", registry=registry
    )
    comparison = ns["build_comparison_states"](
        canonical, *MONTHS, registry=registry
    )
    state_a, state_b = comparison["states"]
    a, b = [
        state["teams"]["Alpha"]["permanent"] for state in comparison["states"]
    ]
    specs = ns["drivers_at"](ns["PARENT"], registry)
    started = perf_counter()
    reference, lookup = component_reference(ns, a, b, registry)
    reference_seconds = perf_counter() - started
    exact_seconds = np.nan
    if len(specs) <= 15:
        exact = ns["permanent_games"](state_a, state_b, registry)["Alpha"]
        ns["check"](
            "Component/full exact attribution",
            reference,
            exact["impacts"].reindex(reference.index),
        )
        exact_seconds = exact["audit"]["seconds"]
    upper = ns["upper_games"](state_a, state_b, registry)
    multipliers = upper["teams"]["Alpha"]
    factors = dict(
        permanent_unit=1,
        organization=multipliers["M"] * multipliers["K"],
        local=multipliers["L"] * multipliers["K"],
    )
    cache = ns["prepare_salary_cache"](a, b)

    def direct(state):
        """Use the real, validated notebook evaluator for timing."""
        return ns["permanent_cell_costs"](
            dict(state, _salary_cache=cache), registry
        ).ravel()

    for budget in BUDGETS:
        # Median of three uncached real-path game runs, already JIT-warmed.
        real_runs = [
            ns["permutation_game"](
                a,
                b,
                specs,
                direct,
                n_permutations=budget,
                random_seed=ns["team_random_seed"](i, "Alpha"),
            )
            for i in range(3)
        ]
        timings = dict(
            warm_real_seconds=float(
                np.median([g["audit"]["seconds"] for g in real_runs])
            ),
            exact_seconds=exact_seconds,
            reference_seconds=reference_seconds,
        )
        for seed in range(seeds):
            game = ns["permutation_game"](
                a,
                b,
                specs,
                lookup,
                n_permutations=budget,
                random_seed=ns["team_random_seed"](seed, "Alpha"),
            )
            if seed < 3:
                ns["check"](
                    "Accelerated/real sampled attribution",
                    game["impacts"],
                    real_runs[seed]["impacts"],
                )
            result = ns["assemble_bridges"](
                state_a, state_b, {"Alpha": game}, upper, registry, name
            )
            assert ns["reconcile"](
                canonical, state_a, state_b, result, registry
            ).passed.all()
            record_run(
                run_rows,
                driver_rows,
                name,
                budget,
                seed,
                reference,
                game,
                factors,
                timings,
            )
        print(f"{name}: {budget} cycles completed", flush=True)
    return dict(
        drivers=len(specs),
        permanent_unit_a=ns["permanent_cost"](a, registry),
        permanent_unit_b=ns["permanent_cost"](b, registry),
        organization_a=ns["cost"](state_a, registry),
        organization_b=ns["cost"](state_b, registry),
        factors=factors,
        exact_seconds=None if np.isnan(exact_seconds) else exact_seconds,
        component_reference_seconds=reference_seconds,
    )


def run_control(ns, name, degree, cancellation, seeds, run_rows, driver_rows):
    """Expose interaction error and additivity under net cancellation."""
    keys = [str(i) for i in range(degree)]
    specs = tuple(ns["Driver"](key, key, "controlled") for key in keys)
    a, b = dict.fromkeys(keys, 0), dict.fromkeys(keys, 1)

    def engine(state):
        """Return one controlled interaction, optionally offset to zero net."""
        return 600 * np.prod([state[k] for k in keys]) - (
            600 * state["0"] if cancellation else 0
        )

    exact = ns["exact_game"](a, b, specs, engine)
    for budget in BUDGETS:
        for seed in range(seeds):
            game = ns["permutation_game"](
                a, b, specs, engine, n_permutations=budget, random_seed=seed
            )
            record_run(
                run_rows,
                driver_rows,
                name,
                budget,
                seed,
                exact["impacts"],
                game,
                {"controlled": 1},
                dict(
                    warm_real_seconds=game["audit"]["seconds"],
                    exact_seconds=exact["audit"]["seconds"],
                    reference_seconds=np.nan,
                ),
            )
    return dict(drivers=degree, base=exact["base"], end=exact["end"])


def summarize(run_rows, driver_rows, output):
    """Write compact per-driver and per-scenario summaries across seeds."""
    runs, drivers = pd.DataFrame(run_rows), pd.DataFrame(driver_rows)
    summary = (
        runs.groupby(["case", "view", "cycles"])
        .agg(
            normalized_l1_median=("normalized_l1", "median"),
            normalized_l1_p95=("normalized_l1", lambda x: x.quantile(0.95)),
            max_absolute_error_p95=(
                "maximum_absolute_error",
                lambda x: x.quantile(0.95),
            ),
            max_material_relative_error_p95=(
                "maximum_material_relative_error",
                lambda x: x.quantile(0.95),
            ),
            material_sign_errors=("material_sign_errors", "sum"),
            top_three_match_rate=("top_three_match", "mean"),
            top_three_has_tie=("top_three_boundary_tie", "any"),
            unique_coalitions_median=("coalitions", "median"),
            model_evaluations=("model_evaluations", "first"),
            warm_real_seconds=("warm_real_seconds", "median"),
            exact_seconds=("exact_seconds", "first"),
        )
        .reset_index()
    )
    detail = (
        drivers.groupby(["case", "view", "cycles", "driver"])
        .agg(
            exact_impact=("exact_impact", "first"),
            absolute_error_median=("absolute_error", "median"),
            absolute_error_p95=("absolute_error", lambda x: x.quantile(0.95)),
            absolute_error_max=("absolute_error", "max"),
            relative_error_p95=("relative_error", lambda x: x.quantile(0.95)),
            material=("material", "first"),
            sign_errors=("sign_error", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    detail.to_csv(output / "driver_errors.csv", index=False)
    return summary


def main():
    """Run the external assessment and save reproducible results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmarks" / "results"
    )
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    ns = model.__wrapped__()
    # First use records Numba initialization separately from warm timings.
    initial = perf_counter()
    ns["permutation_game"](
        {"x": 0},
        {"x": 1},
        (ns["Driver"]("x", "x", "warmup"),),
        lambda s: s["x"],
        n_permutations=1,
    )
    first_call = perf_counter() - initial
    rows, details, cases = [], [], {}
    stress = stress_frame()
    extended, extended_registry = add_six_payments(stress, ns)
    for name, frame, registry in [
        ("existing_14", complete_frame.__wrapped__(), ns["DRIVERS"]),
        ("stress_14", stress, ns["DRIVERS"]),
        ("extended_20", extended, extended_registry),
    ]:
        cases[name] = run_payroll_case(
            ns, name, frame, registry, args.seeds, rows, details
        )
    for name, degree, cancellation in [
        ("cubic", 3, False),
        ("quintic", 5, False),
        ("zero_net_cubic", 3, True),
    ]:
        cases[name] = run_control(
            ns, name, degree, cancellation, args.seeds, rows, details
        )
    summary = summarize(rows, details, args.output_dir)
    metadata = dict(
        seeds=args.seeds,
        budgets=BUDGETS,
        cases=cases,
        first_permutation_call_seconds=first_call,
        total_seconds=perf_counter() - started,
        python=platform.python_version(),
        platform=platform.platform(),
        shap=ns["shap"].__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        timing="Median of 3 real payroll runs; lookup tables excluded",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
