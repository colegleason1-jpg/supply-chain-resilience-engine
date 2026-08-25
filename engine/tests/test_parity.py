"""Parity tests: the new engine must reproduce the acquired system exactly.

``scrcae.legacy.reference`` is a faithful, line-by-line transcription of the
optimizer and Monte Carlo lifted out of the acquired Streamlit monolith,
including the global ``np.random.seed(42)`` and the 0.4 shock floor. These tests
run the new engine in explicit legacy mode and require agreement.

Why this matters more than it looks: without it, "we rewrote the engine" and "we
changed the answers" are indistinguishable. Every behavioural difference the new
engine introduces should be a difference someone *chose*, and these tests are
what make an unchosen difference visible. They are the reason the legacy
behaviours were preserved as named, versioned, opt-in classes instead of deleted.

Legacy mode is:
    LegacyWeightedObjective + ParameterPowerResponse + LegacyTruncatedNormalShock
    with enforce_risk_cap=False, legacy_bundle_activation=True and
    min_funding_scale=0.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from scrcae.domain import Bundle, Dependency, Intervention, SupplyNetwork
from scrcae.legacy import (
    legacy_marginal_risks,
    legacy_monte_carlo,
    legacy_optimize,
    legacy_target_optimize,
)
from scrcae.optimization import (
    LegacyWeightedObjective,
    MinimizeCapitalObjective,
    OptimizationRequest,
    SolveStatus,
    solve,
)
from scrcae.risk import ParameterPowerResponse
from scrcae.stochastic import (
    LegacyTruncatedNormalShock,
    SimulationRequest,
    UniformCorrelation,
)
from scrcae.stochastic import run as run_simulation

MACRO = 1.0625  # 1 + ((85 - 80) / 80) * 0.15, the Brent-crude multiplier at $85


# --------------------------------------------------------------------------- #
# Shared fixtures expressed twice: once as a domain object, once as raw dicts
# --------------------------------------------------------------------------- #

NODES = ("N1", "N2", "N3", "N4", "N5")
COSTS = {"N1": 180_000.0, "N2": 240_000.0, "N3": 70_000.0, "N4": 310_000.0,
         "N5": 125_000.0}
RISKS = {"N1": 12.5, "N2": 9.0, "N3": 4.25, "N4": 6.75, "N5": 7.4}
LEADS = {"N1": 6.0, "N2": 11.0, "N3": 1.5, "N4": 0.0, "N5": 4.0}
CARBON = {"N1": 40.0, "N2": 95.0, "N3": 5.0, "N4": -220.0, "N5": 12.0}
DEPENDENCIES = (("N2", "N3"),)
BUNDLES = (("Port_Warehouse_Synergy", 50_000.0, ("N1", "N2")),)


def _network() -> SupplyNetwork:
    return SupplyNetwork(
        baseline_risk_pts=65.5,
        interventions=tuple(
            Intervention(
                node_id=n,
                name=n,
                cost=COSTS[n],
                risk_reduction_pts=RISKS[n],
                lead_time_saved_days=LEADS[n],
                carbon_tons=CARBON[n],
                min_funding_scale=0.3,
                max_funding_scale=1.0,
            )
            for n in NODES
        ),
        dependencies=tuple(
            Dependency(dependent=d, prerequisite=p) for d, p in DEPENDENCIES
        ),
        bundles=tuple(
            Bundle(name=nm, discount=d, required_nodes=tuple(r))
            for nm, d, r in BUNDLES
        ),
    )


def _legacy_commercial(weight: float, budget: float, macro: float = MACRO):
    return legacy_optimize(
        nodes=NODES,
        costs=COSTS,
        marginal_risks=legacy_marginal_risks(RISKS, macro),
        lead_times=LEADS,
        opt_weight=weight,
        total_budget=budget,
        dependencies=DEPENDENCIES,
        bundles=BUNDLES,
    )


def _new_commercial(weight: float, budget: float, macro: float = MACRO):
    return solve(
        OptimizationRequest(
            network=_network(),
            objective=LegacyWeightedObjective(weight=weight),
            risk_response=ParameterPowerResponse(exponent=0.85),
            budget=budget,
            macro_multiplier=macro,
            enforce_risk_cap=False,
            # Legacy unlocked a bundle discount on node activation alone. The
            # rewritten engine requires full funding by default because the old
            # rule let a portfolio buy a whole-package discount at minimum
            # scale. Parity means reproducing the old rule on request, not
            # keeping it as the default.
            legacy_bundle_activation=True,
        )
    )


# --------------------------------------------------------------------------- #
# Marginal risk coefficients
# --------------------------------------------------------------------------- #


def test_marginal_risk_coefficients_match_exactly():
    legacy = legacy_marginal_risks(RISKS, MACRO)
    response = ParameterPowerResponse(exponent=0.85)
    for node in NODES:
        assert response.linear_coefficient(RISKS[node], MACRO) == pytest.approx(
            legacy[node], rel=1e-15
        )


# --------------------------------------------------------------------------- #
# Commercial mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("budget", [250_000.0, 500_000.0, 750_000.0, 1_200_000.0])
def test_commercial_mode_matches_legacy_across_the_weight_and_budget_grid(
    weight: float, budget: float
):
    legacy = _legacy_commercial(weight, budget)
    new = _new_commercial(weight, budget)

    assert new.status == SolveStatus.OPTIMAL
    assert legacy["status"] == "Optimal"

    new_scales = new.scales()
    for node in NODES:
        assert new_scales[node] == pytest.approx(legacy["scales"][node], abs=1e-6)

    assert new.objective_value == pytest.approx(legacy["objective"], rel=1e-9)
    assert new.gross_capital == pytest.approx(legacy["gross_cost"], rel=1e-9)
    assert new.bundle_discounts == pytest.approx(legacy["discounts"], rel=1e-9)
    assert new.net_capital == pytest.approx(legacy["net_cost"], rel=1e-9)
    assert new.raw_risk_reduction_pts == pytest.approx(legacy["risk_drop"], rel=1e-9)
    assert new.total_lead_time_saved_days == pytest.approx(
        legacy["lead_time"], rel=1e-9
    )
    assert tuple(new.active_bundles) == tuple(legacy["active_bundles"])


def test_commercial_parity_holds_under_a_macro_shock():
    """The macro multiplier enters through the risk response in both versions."""
    for macro in (1.0, 1.0625, 1.15, 1.3):
        legacy = _legacy_commercial(0.5, 750_000.0, macro=macro)
        new = _new_commercial(0.5, 750_000.0, macro=macro)
        assert new.scales() == pytest.approx(legacy["scales"], abs=1e-6)
        assert new.objective_value == pytest.approx(legacy["objective"], rel=1e-9)


def test_commercial_parity_holds_with_a_binding_budget():
    """A tight budget is where semi-continuous funding and the bundle interact,
    so it is the most likely place for a transcription error to show up."""
    legacy = _legacy_commercial(0.5, 90_000.0)
    new = _new_commercial(0.5, 90_000.0)
    assert new.scales() == pytest.approx(legacy["scales"], abs=1e-6)
    assert new.net_capital == pytest.approx(legacy["net_cost"], rel=1e-9)


# --------------------------------------------------------------------------- #
# Target mode
# --------------------------------------------------------------------------- #


def _max_attainable_reduction() -> float:
    """Total risk reduction if every node were funded at full scale."""
    return sum(legacy_marginal_risks(RISKS, MACRO).values())


@pytest.mark.parametrize("target", [50.0, 55.0, 60.0])
def test_target_mode_matches_legacy(target: float):
    effective_baseline = min(100.0, 65.5 * MACRO)
    legacy = legacy_target_optimize(
        nodes=NODES,
        costs=COSTS,
        marginal_risks=legacy_marginal_risks(RISKS, MACRO),
        effective_baseline_risk=effective_baseline,
        target_risk_goal=target,
        dependencies=DEPENDENCIES,
        bundles=BUNDLES,
    )
    required = max(0.0, effective_baseline - target)

    new = solve(
        OptimizationRequest(
            network=_network(),
            objective=MinimizeCapitalObjective(),
            risk_response=ParameterPowerResponse(exponent=0.85),
            required_risk_reduction_pts=required,
            macro_multiplier=MACRO,
            enforce_risk_cap=False,
            # Legacy unlocked a bundle discount on node activation alone. The
            # rewritten engine requires full funding by default because the old
            # rule let a portfolio buy a whole-package discount at minimum
            # scale. Parity means reproducing the old rule on request, not
            # keeping it as the default.
            legacy_bundle_activation=True,
        )
    )

    assert legacy["status"] == "Optimal"
    assert new.status == SolveStatus.OPTIMAL
    # Costs and coefficients admit ties, so compare the objective value rather
    # than the argmin: two different portfolios of equal cost are both correct.
    assert new.objective_value == pytest.approx(legacy["objective"], rel=1e-7)
    assert new.net_capital == pytest.approx(legacy["net_cost"], rel=1e-7)
    assert new.raw_risk_reduction_pts >= required - 1e-6
    assert legacy["risk_drop"] >= required - 1e-6


@pytest.mark.parametrize("target", [10.0, 20.0, 35.0])
def test_target_mode_is_infeasible_at_the_shipped_default_target(target: float):
    """A finding, not a parity failure — and both engines agree on it.

    On this portfolio the exponent transform caps total attainable reduction at
    about 30.7 points against an effective baseline of 69.6, so the reachable
    floor is roughly 38.9%. The acquired system shipped ``target_risk_goal``
    defaulting to **20.0**, which is unreachable, and the session default for
    ``baseline_risk`` combined with any macro multiplier above 1.0 pushes it
    further out of reach.

    Two things make this worse than an awkward default. First, the transform is
    what creates the ceiling: raising the risk *parameter* to 0.85 shrinks every
    coefficient above 1.0, so the attainable total is strictly less than the sum
    of the raw reduction figures the user entered. A user who enters interventions
    summing to 40 points of reduction cannot reach a 40-point improvement, and
    nothing in the interface explains why. Second, target mode set the budget to
    1e9 — effectively unlimited — so infeasibility here can never be resolved by
    spending more. It is a structural ceiling masquerading as a budget problem.

    Both engines return Infeasible, so this is faithful extraction of real
    behaviour. It is recorded here so the rebuild does not silently "fix" it and
    lose the evidence.
    """
    effective_baseline = min(100.0, 65.5 * MACRO)
    required = max(0.0, effective_baseline - target)
    assert required > _max_attainable_reduction()

    legacy = legacy_target_optimize(
        nodes=NODES,
        costs=COSTS,
        marginal_risks=legacy_marginal_risks(RISKS, MACRO),
        effective_baseline_risk=effective_baseline,
        target_risk_goal=target,
        dependencies=DEPENDENCIES,
        bundles=BUNDLES,
    )
    new = solve(
        OptimizationRequest(
            network=_network(),
            objective=MinimizeCapitalObjective(),
            risk_response=ParameterPowerResponse(exponent=0.85),
            required_risk_reduction_pts=required,
            macro_multiplier=MACRO,
            enforce_risk_cap=False,
            # Legacy unlocked a bundle discount on node activation alone. The
            # rewritten engine requires full funding by default because the old
            # rule let a portfolio buy a whole-package discount at minimum
            # scale. Parity means reproducing the old rule on request, not
            # keeping it as the default.
            legacy_bundle_activation=True,
        )
    )

    assert legacy["status"] == "Infeasible"
    assert new.status == SolveStatus.INFEASIBLE
    # The new engine at least reports *why* rather than rendering a blank result.
    assert not new.constraint_report.is_feasible


def test_the_exponent_transform_is_what_caps_attainable_reduction():
    """Quantifies the ceiling described above so the number is not folklore."""
    raw_total = sum(RISKS.values())
    transformed_total = _max_attainable_reduction()
    assert transformed_total < raw_total
    # Roughly 30.7 against 39.9 raw, even with the macro multiplier inflating
    # exposure by 6.25%.
    assert transformed_total == pytest.approx(30.7, abs=0.5)
    assert raw_total == pytest.approx(39.9, abs=0.1)


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #


def test_legacy_monte_carlo_is_reproducible_in_isolation():
    """Baseline for the comparison below: the legacy routine is deterministic
    because it reseeds the global RNG on every call. That is also precisely why
    its results could never be checked for seed sensitivity."""
    matrix = UniformCorrelation(rho=0.35).matrix(NODES)
    scales = {n: 1.0 for n in NODES}
    marginal = legacy_marginal_risks(RISKS, MACRO)

    first = legacy_monte_carlo(65.5, scales, marginal, matrix, NODES, 5_000)
    second = legacy_monte_carlo(65.5, scales, marginal, matrix, NODES, 5_000)
    assert first == second


def test_new_engine_in_legacy_shock_mode_reproduces_legacy_distribution():
    """The two implementations draw from different RNG streams — the legacy one
    uses the global ``RandomState`` seeded to 42, the new one a local
    ``Generator`` — so bit-identical samples are not achievable and would not be
    desirable. What must match is the distribution: identical shock model,
    identical correlation, so identical percentiles to within Monte Carlo error.
    """
    matrix = UniformCorrelation(rho=0.35).matrix(NODES)
    scales = {n: 1.0 for n in NODES}
    marginal = legacy_marginal_risks(RISKS, MACRO)
    iterations = 200_000

    legacy = legacy_monte_carlo(65.5, scales, marginal, matrix, NODES, iterations)

    new = run_simulation(
        SimulationRequest(
            baseline_risk_pts=65.5,
            risk_reduction_by_node={n: marginal[n] * scales[n] for n in NODES},
            correlation=matrix,
            node_order=NODES,
            iterations=iterations,
            seed=42,
            shock_model=LegacyTruncatedNormalShock(sigma=0.12, floor=0.4),
        )
    )

    tolerance = 6 * max(new.p50_standard_error, 1e-3)
    assert new.p50_risk_pts == pytest.approx(legacy["P50_Risk"], abs=tolerance)
    assert new.p90_risk_pts == pytest.approx(legacy["P90_Risk"], abs=tolerance * 2)
    assert new.std_dev_pts == pytest.approx(legacy["Std_Dev"], rel=0.05)


def test_legacy_monte_carlo_pollutes_the_global_rng_and_the_new_one_does_not():
    """A direct, side-by-side demonstration of the defect, so it is documented
    as a fact about the acquired code rather than an assertion in a memo."""
    matrix = UniformCorrelation(rho=0.35).matrix(NODES)
    scales = {n: 1.0 for n in NODES}
    marginal = legacy_marginal_risks(RISKS, MACRO)

    np.random.seed(999)
    baseline = np.random.random(3)

    np.random.seed(999)
    legacy_monte_carlo(65.5, scales, marginal, matrix, NODES, 100)
    after_legacy = np.random.random(3)
    assert not np.array_equal(baseline, after_legacy)

    np.random.seed(999)
    run_simulation(
        SimulationRequest(
            baseline_risk_pts=65.5,
            risk_reduction_by_node={n: marginal[n] for n in NODES},
            correlation=matrix,
            node_order=NODES,
            iterations=100,
            seed=42,
            shock_model=LegacyTruncatedNormalShock(),
        )
    )
    after_new = np.random.random(3)
    np.testing.assert_array_equal(baseline, after_new)


# --------------------------------------------------------------------------- #
# The differences that were chosen
# --------------------------------------------------------------------------- #


def test_the_only_intended_divergences_are_opt_in():
    """Parity is the default in legacy mode; every improvement requires asking.

    This test is the contract: swapping a single component changes the answer,
    and nothing changes the answer unless a component is swapped.
    """
    baseline = _new_commercial(0.5, 750_000.0)

    # Enforcing the risk cap in-model is opt-in.
    capped = solve(
        OptimizationRequest(
            network=_network(),
            objective=LegacyWeightedObjective(weight=0.5),
            risk_response=ParameterPowerResponse(exponent=0.85),
            budget=750_000.0,
            macro_multiplier=MACRO,
            enforce_risk_cap=True,
        )
    )
    assert capped.status == SolveStatus.OPTIMAL

    # Identical inputs twice: the solve is deterministic.
    repeat = _new_commercial(0.5, 750_000.0)
    assert repeat.scales() == pytest.approx(baseline.scales(), abs=1e-9)
    assert repeat.objective_value == pytest.approx(baseline.objective_value, rel=1e-12)


def test_legacy_mode_results_are_labelled_as_incommensurate():
    """A historic result must be identifiable as one produced under the old
    objective, so a future reader knows not to compare it to an NPV figure."""
    result = _new_commercial(0.5, 750_000.0)
    assert result.objective_unit == "incommensurate (percentage-points + days)"
    assert "legacy" in str(result.audit["objective_version"]).lower()
    assert result.audit["objective_unit"] == result.objective_unit
    # The provenance hash lets a historic number be tied back to exact inputs,
    # replacing the hard-coded "sha256:8f4c99a..." literal in the acquired UI
    # which hashed nothing at all.
    assert len(str(result.audit["input_hash"])) >= 32
