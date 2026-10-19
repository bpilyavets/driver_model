"""Attribution for the standalone workforce cost engine."""

from hashlib import sha256
from time import perf_counter

import numpy as np
import pandas as pd

from .common import (
    ATOL,
    MAX_EXACT_DRIVERS,
    PARENT,
    TEAM_PARENT,
    UNIT_KEYS,
    check,
)
from .costing import (
    aligned_comparison,
    cost,
    local_cost_from_units,
    organization_cost_terms,
    permanent_cell_costs,
    permanent_cost,
    team_unit_from_units,
    team_units,
)
from .kernels import (
    ensure_same_policy,
    prepare_salary_cache,
)
from .registry import (
    DRIVERS,
    drivers_at,
)


def active_unit_specs(a, b, registry=DRIVERS):
    """Exclude zero-share-in-both economics players, retaining full
    mix.
    """
    shares = a["contract_type_mix"] + b["contract_type_mix"]
    active = {key for key, share in zip(UNIT_KEYS, shares) if share > 0}
    active.add("contract_type_mix")
    return tuple(
        driver
        for driver in drivers_at(TEAM_PARENT, registry)
        if driver.name in active
    )


def switch_model(a, b, specs, engine, *, vector_output=False):
    """Build a binary A/B switch model over complete driver values.

    Returns the callable, evaluated-mask set and observed cost extrema.
    Every constructed hybrid is validated by the supplied cost engine.
    """
    names = [d.name for d in specs]
    if "_policy" in a:
        ensure_same_policy(a["_policy"], b["_policy"])
        if a["_employees"] != b["_employees"]:
            raise ValueError("A/B permanent employee axes differ")
    seen, extrema = set(), [np.inf, -np.inf]

    def model(switches):
        """Evaluate complete binary counterfactual driver switches."""
        x = np.asarray(switches)
        if (
            x.ndim != 2
            or x.shape[1] != len(names)
            or not np.isin(x, [0, 1]).all()
        ):
            raise ValueError(
                "Expected a binary batch with one feature per complete driver"
            )
        values = []
        for mask in x:
            hybrid = dict(a)
            hybrid.update(
                {
                    name: (b if bit else a)[name]
                    for name, bit in zip(names, mask)
                }
            )
            raw = engine(hybrid)
            value = (
                np.atleast_1d(raw).astype(float)
                if vector_output
                else float(raw)
            )
            if vector_output and value.ndim != 1:
                raise ValueError("Expected a flat vector of cell costs")
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite counterfactual cost")
            total = float(np.sum(value))
            model.evaluations += 1
            seen.add(sum(int(bit) << j for j, bit in enumerate(mask)))
            extrema[0], extrema[1] = (
                min(extrema[0], total),
                max(extrema[1], total),
            )
            values.append(value)
        return np.asarray(values)

    model.evaluations = 0
    return model, seen, extrema


def validate_explainer_options(explainer, n_permutations, random_seed):
    """Validate explicit estimator settings, including inactive options.

    A permutation is one forward/reverse cycle, not a sampled coalition.
    Sampling settings are used only by the permutation estimator.
    """
    if explainer not in ("exact", "permutation"):
        raise ValueError("explainer must be 'exact' or 'permutation'")
    for name, value, minimum in (
        ("n_permutations", n_permutations, 1),
        ("random_seed", random_seed, 0),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")


def team_random_seed(random_seed, team):
    """Derive a stable NumPy seed independently of team processing order."""
    payload = f"{int(random_seed)}:{team}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def permutation_game(
    a, b, specs, engine, *, n_permutations=256, random_seed=0
):
    """Sample Shapley values of scalar or flat vector cost contributions.

    Parameters
    ----------
    a, b : dict
        Aligned endpoints for complete business-driver switches.
    specs : tuple of Driver
        Business players; no exact-enumeration size restriction applies.
    engine : callable
        Validated cost evaluator returning a scalar or a flat cell vector.
    n_permutations : int
        Positive number of antithetic forward/reverse cycles.
    random_seed : int
        Nonnegative reproducibility seed; caller RNG state is preserved.

    Returns
    -------
    dict
        Driver impacts, sampled cell impacts, endpoints and run details.
        Cell impacts and their summed driver impacts use identical samples.

    Notes
    -----
    With the pinned SHAP version, one cycle consumes a budget of 2*d+1.
    Efficiency is numerical reconciliation, not a claim of exact impacts.
    No residual correction or post-hoc attribution rescaling is applied.
    """
    import shap

    validate_explainer_options("permutation", n_permutations, random_seed)
    n = len(specs)
    if n < 1:
        raise ValueError("A game needs at least one driver")
    budget = int(n_permutations) * (2 * n + 1)
    started = perf_counter()
    model, seen, extrema = switch_model(
        a, b, specs, engine, vector_output=True
    )
    background = np.zeros((1, n))
    seed = int(random_seed) % (2**32)
    rng_state = np.random.get_state()
    try:
        explainer = shap.PermutationExplainer(
            model,
            shap.maskers.Independent(background),
            feature_names=[d.name for d in specs],
            seed=seed,
        )
        explanation = explainer(
            np.ones((1, n)),
            max_evals=budget,
            batch_size=2 * n + 1,
            silent=True,
        )
    finally:
        np.random.set_state(rng_state)
    cell_impacts = np.asarray(explanation.values[0]).reshape(n, -1)
    base_cells, end_cells = model(background)[0], model(np.ones((1, n)))[0]
    check("Permutation base", explanation.base_values[0], base_cells)
    check(
        "Permutation cell efficiency",
        cell_impacts.sum(axis=0),
        end_cells - base_cells,
    )
    impacts = pd.Series(
        cell_impacts.sum(axis=1), index=[d.name for d in specs], name="impact"
    )
    base, end = float(base_cells.sum()), float(end_cells.sum())
    check("Permutation efficiency", impacts.sum(), end - base)
    return dict(
        impacts=impacts,
        cell_impacts=cell_impacts,
        base=base,
        end=end,
        audit=dict(
            drivers=n,
            method="permutation",
            exhaustive=False,
            n_permutations=int(n_permutations),
            random_seed=seed,
            evaluation_budget=budget,
            model_evaluations=model.evaluations,
            coalitions=len(seen),
            minimum_cost=extrema[0],
            maximum_cost=extrema[1],
            extrema_scope="evaluated states only",
            seconds=perf_counter() - started,
            passed=True,
        ),
    )


def exact_game(a, b, specs, engine):
    """Enumerate exact Shapley impacts with one zero background.

    Parameters
    ----------
    a, b : dict
        Aligned endpoint values for complete registered drivers.
    specs : tuple of Driver
        Active business players, not columns or individual employees.
    engine : callable
        Validated scalar cost function, accepting signed outputs.

    Returns
    -------
    dict
        Impacts, endpoint costs and actual coalition/runtime audit.

    Raises
    ------
    ValueError
        More than 15 players, invalid states or nonfinite costs.
    AssertionError
        Enumeration coverage or numerical efficiency does not reconcile.
    """
    import shap

    n = len(specs)
    if not 1 <= n <= MAX_EXACT_DRIVERS:
        raise ValueError(
            f"Exact PoC supports 1–{MAX_EXACT_DRIVERS} drivers per game; "
            "select explainer='permutation' for larger permanent games"
        )
    started = perf_counter()
    model, seen, extrema = switch_model(a, b, specs, engine)
    background = np.zeros((1, n))
    explainer = shap.ExactExplainer(
        model,
        shap.maskers.Independent(background),
        feature_names=[d.name for d in specs],
    )
    explanation = explainer(np.ones((1, n)), max_evals=2**n, batch_size=2**n)
    impacts = pd.Series(
        explanation.values[0], index=[d.name for d in specs], name="impact"
    )
    base, end = model(background)[0], model(np.ones((1, n)))[0]
    check("SHAP base", explanation.base_values[0], base)
    check("SHAP efficiency", impacts.sum(), end - base)
    assert len(seen) == 2**n, (
        "Every binary coalition must be evaluated and validated"
    )
    return dict(
        impacts=impacts,
        base=base,
        end=end,
        audit=dict(
            drivers=n,
            method="exact",
            exhaustive=True,
            n_permutations=None,
            random_seed=None,
            evaluation_budget=2**n,
            model_evaluations=model.evaluations,
            extrema_scope="all binary coalitions",
            coalitions=len(seen),
            minimum_cost=extrema[0],
            maximum_cost=extrema[1],
            seconds=perf_counter() - started,
            passed=True,
        ),
    )


def auxiliary_multiplier(a, b, specs, engine, parent):
    """Measure a parent sensitivity using an auxiliary 0-to-1 game."""
    game = exact_game(
        dict(a, **{parent: 0.0}), dict(b, **{parent: 1.0}), specs, engine
    )
    return float(game["impacts"][parent]), game


def team_term_cost(state, team_index):
    """Evaluate one additive team contribution using the shared
    engine.
    """
    units = np.zeros(
        len(state["team_mix"])
    )  # Isolate a cost term, not an imputed workforce state.
    units[team_index] = state[TEAM_PARENT]
    return float(
        organization_cost_terms(
            state["workforce_scale"], state["team_mix"], units
        )[team_index]
    )


def permanent_games(
    a,
    b,
    registry=DRIVERS,
    *,
    explainer="exact",
    n_permutations=256,
    random_seed=0,
):
    """Explain active permanent branches using the selected estimator.

    Small upper games and auxiliary multipliers remain exact. Sampling
    seeds depend on team identity, not loop order. Cached salary scenarios
    use the shared cost helper and live only inside this synchronous call.
    """
    validate_explainer_options(explainer, n_permutations, random_seed)
    aligned_comparison(a, b, registry)
    games = {}
    for team in a["_teams"]:
        left, right = a["teams"][team], b["teams"][team]
        if left["contract_type_mix"][0] + right["contract_type_mix"][0] == 0:
            continue
        pa, pb = left["permanent"], right["permanent"]
        cache = prepare_salary_cache(pa, pb)
        specs = drivers_at(PARENT, registry)
        if explainer == "exact":
            games[team] = exact_game(
                pa,
                pb,
                specs,
                lambda state: permanent_cost(
                    dict(state, _salary_cache=cache), registry
                ),
            )
        else:
            games[team] = permutation_game(
                pa,
                pb,
                specs,
                lambda state: permanent_cell_costs(
                    dict(state, _salary_cache=cache), registry
                ).ravel(),
                n_permutations=n_permutations,
                random_seed=team_random_seed(random_seed, team),
            )
    return games


def upper_games(a, b, registry=DRIVERS):
    """Run team-unit, organizational term and local two-player games.

    M, K and L are auxiliary 0-to-1 multipliers. No propagation divides
    by a parent's change. Inactive permanent branches need no K game.
    """
    aligned_comparison(a, b, registry)
    term_specs = drivers_at("total", registry)
    local_specs = tuple(
        driver for driver in term_specs if driver.name != "team_mix"
    )
    outer = pd.Series({"workforce_scale": 0.0, "team_mix": 0.0}, name="impact")
    teams = {}
    for index, team in enumerate(a["_teams"]):
        units_a, units_b = [
            team_units(s["teams"][team], registry) for s in (a, b)
        ]
        unit_specs = active_unit_specs(units_a, units_b, registry)
        unit = exact_game(
            units_a,
            units_b,
            unit_specs,
            lambda state: team_unit_from_units(state, registry),
        )
        permanent_multiplier, permanent_auxiliary = 0.0, None
        if PARENT in unit["impacts"]:
            permanent_multiplier, permanent_auxiliary = auxiliary_multiplier(
                units_a,
                units_b,
                unit_specs,
                lambda state: team_unit_from_units(state, registry),
                PARENT,
            )
        term_a, term_b = [
            {
                "workforce_scale": state["workforce_scale"],
                "team_mix": state["team_mix"],
                TEAM_PARENT: team_unit_from_units(units, registry),
            }
            for state, units in ((a, units_a), (b, units_b))
        ]

        def term_engine(state):
            """Evaluate this team's additive term in the outer game."""
            return team_term_cost(state, index)

        term = exact_game(term_a, term_b, term_specs, term_engine)
        organization_multiplier, organization_auxiliary = auxiliary_multiplier(
            term_a, term_b, term_specs, term_engine, TEAM_PARENT
        )
        local_a, local_b = [
            {
                "workforce_scale": state["workforce_scale"]
                * state["team_mix"][index],
                TEAM_PARENT: term_state[TEAM_PARENT],
            }
            for state, term_state in ((a, term_a), (b, term_b))
        ]
        local = exact_game(
            local_a, local_b, local_specs, local_cost_from_units
        )
        local_multiplier, local_auxiliary = auxiliary_multiplier(
            local_a, local_b, local_specs, local_cost_from_units, TEAM_PARENT
        )
        common = ["workforce_scale", "team_mix"]
        outer.loc[common] += term["impacts"].loc[common]
        outer.loc["team::" + team] = term["impacts"][TEAM_PARENT]
        teams[team] = {
            "unit": unit,
            "organization_term": term,
            "local": local,
            "M": organization_multiplier,
            "K": permanent_multiplier,
            "L": local_multiplier,
            "M_game": organization_auxiliary,
            "K_game": permanent_auxiliary,
            "L_game": local_auxiliary,
        }
    check(
        "Organizational outer efficiency",
        outer.sum(),
        cost(b, registry) - cost(a, registry),
    )
    return {
        "impacts": outer,
        "base": cost(a, registry),
        "end": cost(b, registry),
        "teams": teams,
    }


def game_audit_table(permanent, upper):
    """Summarize actual enumeration coverage, extrema and runtime."""
    rows = [
        dict(team=team, game="permanent", **game["audit"])
        for team, game in permanent.items()
    ]
    for team, values in upper["teams"].items():
        for name in (
            "unit",
            "organization_term",
            "local",
            "M_game",
            "K_game",
            "L_game",
        ):
            if values[name] is not None:
                rows.append(
                    dict(team=team, game=name, **values[name]["audit"])
                )
    return pd.DataFrame(rows)


def aggregate_driver_bridge(bridge, registry=DRIVERS):
    """Sum organizational leaf impacts by registered business driver.

    Parameters
    ----------
    bridge : pandas.DataFrame
        Organizational leaf bridge for one comparison, in currency.
        Local rows and intermediate hierarchy parents are not accepted.
    registry : tuple of Driver
        Driver identifiers, display labels and output order.

    Returns
    -------
    pandas.DataFrame
        Separate organizational summary with no team assignment. Signed
        impacts sum across teams; endpoints and units are preserved.
        Shares use the organizational net change and are undefined near
        zero. Active zero-impact drivers remain; absent drivers do not
        acquire artificial rows. The input bridge is unchanged.

    Raises
    ------
    ValueError
        Rows are empty, non-organizational, non-leaf, from different
        comparisons, or contain drivers missing from the registry.

    Notes
    -----
    This summarizes the existing hierarchy, including its interactions.
    It performs no costing, pooling, Shapley estimation or rescaling.
    """
    if (
        bridge.empty
        or not (bridge.view.eq("organization") & bridge.leaf.eq(True)).all()
    ):
        raise ValueError("Aggregate only organizational leaf bridge rows")
    context = ["dataset", "period_A_cost", "period_B_cost", "units"]
    if len(bridge[context].drop_duplicates()) != 1:
        raise ValueError("Aggregate one organizational comparison at a time")
    drivers_at("total", registry)
    known = {spec.name for spec in registry}
    if not set(bridge.driver).issubset(known):
        raise ValueError("Organizational bridge contains unregistered drivers")
    rows = []
    for spec in registry:
        contributions = bridge.loc[bridge.driver.eq(spec.name)]
        if contributions.empty:
            continue
        row = contributions.iloc[0].to_dict()
        row.update(
            team=None,
            label=spec.label,
            parent=spec.parent,
            path=("organization", "organization", spec.parent, spec.name),
            impact=float(contributions.impact.sum()),
        )
        rows.append(row)
    summary = pd.DataFrame(rows, columns=bridge.columns)
    delta = summary.period_B_cost - summary.period_A_cost
    summary["share_of_net_change"] = summary.impact / delta.where(
        delta.abs() > ATOL
    )
    return summary


def assemble_bridges(
    a, b, permanent, upper, registry=DRIVERS, dataset="comparison"
):
    """Replace parents by propagated children for both reporting views.

    Organizational children use M and M*K; local children use L and L*K.
    Inactive permanent branches have no child rows. Shares use each
    view's net change and are undefined near zero.
    """
    labels = {d.name: d.label for d in registry}
    rows = []

    def add(view, team, driver, parent, level, impact, leaf=True):
        """Append one labeled hierarchy row with its view and parent
        path.
        """
        label = labels[driver]
        if driver == TEAM_PARENT:
            label = f"{team} economics"
        path = (view, team or "organization", parent, driver)
        rows.append(
            dict(
                dataset=dataset,
                view=view,
                team=team,
                driver=driver,
                label=label,
                path=path,
                parent=parent,
                level=level,
                impact=float(impact),
                leaf=leaf,
                units="currency",
            )
        )

    for driver in ("workforce_scale", "team_mix"):
        add("organization", None, driver, "total", 1, upper["impacts"][driver])
    for team, games in upper["teams"].items():
        add(
            "organization",
            team,
            TEAM_PARENT,
            "total",
            1,
            upper["impacts"]["team::" + team],
            False,
        )
        add(
            "local",
            team,
            "workforce_scale",
            "team_total",
            1,
            games["local"]["impacts"]["workforce_scale"],
        )
        add(
            "local",
            team,
            TEAM_PARENT,
            "team_total",
            1,
            games["local"]["impacts"][TEAM_PARENT],
            False,
        )
        for view, multiplier in (
            ("organization", games["M"]),
            ("local", games["L"]),
        ):
            for driver, impact in games["unit"]["impacts"].items():
                add(
                    view,
                    team,
                    driver,
                    TEAM_PARENT,
                    2,
                    multiplier * impact,
                    driver != PARENT,
                )
            for driver, impact in permanent.get(team, {"impacts": {}})[
                "impacts"
            ].items():
                add(
                    view,
                    team,
                    driver,
                    PARENT,
                    3,
                    multiplier * games["K"] * impact,
                )
    hierarchy = pd.DataFrame(rows)
    hierarchy["period_A_cost"] = [
        (
            upper["base"]
            if row.view == "organization"
            else upper["teams"][row.team]["local"]["base"]
        )
        for row in hierarchy.itertuples()
    ]
    hierarchy["period_B_cost"] = [
        (
            upper["end"]
            if row.view == "organization"
            else upper["teams"][row.team]["local"]["end"]
        )
        for row in hierarchy.itertuples()
    ]
    delta = hierarchy.period_B_cost - hierarchy.period_A_cost
    hierarchy["share_of_net_change"] = hierarchy.impact / delta.where(
        delta.abs() > ATOL
    )
    bridge = hierarchy.loc[
        hierarchy.leaf & hierarchy.view.eq("organization")
    ].copy()
    local_bridges = {
        t: hierarchy.loc[
            hierarchy.leaf & hierarchy.view.eq("local") & hierarchy.team.eq(t)
        ].copy()
        for t in a["_teams"]
    }
    return dict(
        hierarchy=hierarchy,
        bridge=bridge,
        driver_bridge=aggregate_driver_bridge(bridge, registry),
        local_bridges=local_bridges,
        permanent=permanent,
        upper=upper,
        coalition_audit=game_audit_table(permanent, upper),
    )


def decompose(
    a,
    b,
    registry=DRIVERS,
    *,
    dataset="comparison",
    explainer="exact",
    n_permutations=256,
    random_seed=0,
):
    """Explain completed states with exact or sampled permanent attribution.

    Parameters
    ----------
    a, b : dict
        Completed, aligned workforce states in comparison order.
    registry : tuple of Driver
        Registered business drivers and payroll contributions.
    dataset : str
        Reporting label.
    explainer : {'exact', 'permutation'}
        Permanent-game estimator. All upper games remain exact.
    n_permutations : int
        Forward/reverse cycles, used only in permutation mode.
    random_seed : int
        Nonnegative base seed for deterministic per-team sampling.

    Returns
    -------
    dict
        Organizational/local bridges, aggregated driver_bridge, games,
        run details and the requested configuration.
        Numerical reconciliation does not establish sampling accuracy.
    """
    validate_explainer_options(explainer, n_permutations, random_seed)
    result = assemble_bridges(
        a,
        b,
        permanent_games(
            a,
            b,
            registry,
            explainer=explainer,
            n_permutations=n_permutations,
            random_seed=random_seed,
        ),
        upper_games(a, b, registry),
        registry,
        dataset,
    )
    result["explainer_config"] = dict(
        permanent=explainer,
        upper="exact",
        n_permutations=int(n_permutations),
        random_seed=int(random_seed),
    )
    return result
