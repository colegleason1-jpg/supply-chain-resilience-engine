"""Known-answer optimization tests.

Every case here has an optimum that can be derived by hand in a few lines. That
is the point: if the solver, the constraint construction, or the objective
assembly regresses, these fail with an answer a human can adjudicate. The
acquired system had no such anchor, so there was no way to tell a modelling
change from a bug.
"""

from __future__ import annotations

import math

import pytest

from scrcae.domain import Bundle, Dependency, Intervention, SupplyNetwork
from scrcae.optimization import (
    MinimizeCapitalObjective,
    MonetaryNPVObjective,
    OptimizationRequest,
    PriceBook,
    SolveStatus,
    solve,
)
from scrcae.risk import AllocationConcaveResponse, LinearResponse

TOL = 1e-6


def _prices(**kwargs: float) -> PriceBook:
    base = {
        "value_per_risk_point": 100_000.0,
        "value_per_lead_time_day": 0.0,
        "price_per_carbon_ton": 0.0,
        "benefit_horizon_years": 5,
        "discount_rate": 0.10,
    }
    base.update(kwargs)
    return PriceBook(**base)  # type: ignore[arg-type]


def _request(network: SupplyNetwork, **kwargs: object) -> OptimizationRequest:
    defaults: dict[str, object] = {
        "objective": MonetaryNPVObjective(prices=_prices()),
        "risk_response": LinearResponse(),
        "enforce_risk_cap": False,
    }
    defaults.update(kwargs)
    return OptimizationRequest(network=network, **defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# KA-1  Binary knapsack: budget admits exactly one of two candidates
# --------------------------------------------------------------------------- #


def test_ka1_binary_knapsack_selects_higher_risk_reducer():
    """Both cost 100k, budget admits one. A removes 10 points, B removes 5.

    Hand solution: fund A, leave B. Objective = AF * 100k * 10 - 100k.
    """
    network = SupplyNetwork(
        baseline_risk_pts=50.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0,
                         min_funding_scale=1.0),
            Intervention("B", cost=100_000.0, risk_reduction_pts=5.0,
                         min_funding_scale=1.0),
        ),
    )
    result = solve(_request(network, budget=100_000.0))

    assert result.status == SolveStatus.OPTIMAL
    assert result.constraint_report.is_feasible, result.constraint_report.summary()
    assert result.scales() == pytest.approx({"A": 1.0, "B": 0.0}, abs=TOL)

    annuity = _prices().annuity_factor
    expected = annuity * 100_000.0 * 10.0 - 100_000.0
    assert result.objective_value == pytest.approx(expected, rel=1e-9)
    assert result.risk_reduction_pts == pytest.approx(10.0, abs=TOL)
    assert result.optimized_risk_pts == pytest.approx(40.0, abs=TOL)


# --------------------------------------------------------------------------- #
# KA-2  Semi-continuous funding and the minimum economic scale
# --------------------------------------------------------------------------- #


def test_ka2_partial_funding_is_budget_limited():
    """One node costing 100k, budget 50k, MES 0.3. Optimum is exactly x = 0.5."""
    network = SupplyNetwork(
        baseline_risk_pts=50.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0,
                         min_funding_scale=0.3),
        ),
    )
    result = solve(_request(network, budget=50_000.0))

    assert result.status == SolveStatus.OPTIMAL
    assert result.scales()["A"] == pytest.approx(0.5, abs=1e-6)
    assert result.net_capital == pytest.approx(50_000.0, abs=1e-3)


def test_ka2b_budget_below_minimum_economic_scale_funds_nothing():
    """Budget 20k cannot reach the 30k minimum, so the node stays unfunded.

    This is the behaviour that distinguishes a semi-continuous variable from a
    plain continuous one, and it is the reason the model is mixed-integer at all.
    """
    network = SupplyNetwork(
        baseline_risk_pts=50.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0,
                         min_funding_scale=0.3),
        ),
    )
    result = solve(_request(network, budget=20_000.0))

    assert result.status == SolveStatus.OPTIMAL
    assert result.scales()["A"] == pytest.approx(0.0, abs=TOL)
    assert result.net_capital == pytest.approx(0.0, abs=TOL)
    assert result.constraint_report.is_feasible


# --------------------------------------------------------------------------- #
# KA-3  Prerequisite cascade capping
# --------------------------------------------------------------------------- #


def test_ka3_dependency_cascade_forces_equal_funding():
    """B depends on A, so x_B <= x_A.

    Costs are 100k each, budget 60k, MES 0.3, risk A = 1 point, B = 20 points.
    Feasible set: x_A + x_B <= 0.6, x_B <= x_A, each 0 or >= 0.3.
    Maximising 1*x_A + 20*x_B gives x_A = x_B = 0.3, value 6.3, versus 0.6 for
    spending everything on A alone. The cascade cap is what makes the valuable
    node unable to run ahead of its prerequisite.
    """
    network = SupplyNetwork(
        baseline_risk_pts=50.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=1.0),
            Intervention("B", cost=100_000.0, risk_reduction_pts=20.0),
        ),
        dependencies=(Dependency(dependent="B", prerequisite="A"),),
    )
    result = solve(_request(network, budget=60_000.0))

    scales = result.scales()
    assert result.status == SolveStatus.OPTIMAL
    assert scales["A"] == pytest.approx(0.3, abs=1e-6)
    assert scales["B"] == pytest.approx(0.3, abs=1e-6)
    assert result.risk_reduction_pts == pytest.approx(6.3, abs=1e-6)
    assert result.constraint_report.is_feasible


# --------------------------------------------------------------------------- #
# KA-4  Bundle synergy relaxes the budget
# --------------------------------------------------------------------------- #


def test_ka4_bundle_discount_unlocks_full_funding():
    """Two nodes at 100k, budget 150k, bundle pays 50k when both are active.

    Without the bundle the affordable total scale is 1.5. With it, net spend is
    100k(x_A + x_B) - 50k <= 150k, so total scale 2.0 becomes affordable and both
    nodes reach full funding.
    """
    network = SupplyNetwork(
        baseline_risk_pts=80.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0),
            Intervention("B", cost=100_000.0, risk_reduction_pts=9.0),
        ),
        bundles=(Bundle("Synergy", discount=50_000.0, required_nodes=("A", "B")),),
    )
    result = solve(_request(network, budget=150_000.0))

    scales = result.scales()
    assert result.status == SolveStatus.OPTIMAL
    assert scales["A"] == pytest.approx(1.0, abs=1e-6)
    assert scales["B"] == pytest.approx(1.0, abs=1e-6)
    assert result.active_bundles == ("Synergy",)
    assert result.gross_capital == pytest.approx(200_000.0, abs=1e-3)
    assert result.bundle_discounts == pytest.approx(50_000.0, abs=TOL)
    assert result.net_capital == pytest.approx(150_000.0, abs=1e-3)
    assert result.constraint_report.is_feasible


def test_ka4b_bundle_cannot_claim_discount_without_its_nodes():
    """A bundle requiring an unaffordable node must not pay out.

    Verified independently of the solver by the constraint report, which checks
    bundle activation against the returned allocation vector.
    """
    network = SupplyNetwork(
        baseline_risk_pts=80.0,
        interventions=(
            Intervention("A", cost=50_000.0, risk_reduction_pts=10.0),
            Intervention("B", cost=5_000_000.0, risk_reduction_pts=0.1),
        ),
        bundles=(Bundle("Synergy", discount=40_000.0, required_nodes=("A", "B")),),
    )
    result = solve(_request(network, budget=60_000.0))

    assert result.status == SolveStatus.OPTIMAL
    assert result.active_bundles == ()
    assert result.bundle_discounts == pytest.approx(0.0, abs=TOL)
    assert result.constraint_report.is_feasible


# --------------------------------------------------------------------------- #
# KA-5  Target-risk mode
# --------------------------------------------------------------------------- #


def test_ka5_target_mode_buys_the_cheapest_route_to_the_floor():
    """Required reduction 10 points. A costs 100k for 10 points, B costs 50k for
    the same 10. Minimum-capital solution is B alone at full scale."""
    network = SupplyNetwork(
        baseline_risk_pts=50.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0),
            Intervention("B", cost=50_000.0, risk_reduction_pts=10.0),
        ),
    )
    result = solve(
        _request(
            network,
            objective=MinimizeCapitalObjective(),
            required_risk_reduction_pts=10.0,
        )
    )

    scales = result.scales()
    assert result.status == SolveStatus.OPTIMAL
    assert scales["A"] == pytest.approx(0.0, abs=1e-6)
    assert scales["B"] == pytest.approx(1.0, abs=1e-6)
    assert result.net_capital == pytest.approx(50_000.0, abs=1e-3)
    assert result.objective_unit == "currency (net capital outlay)"
    assert result.constraint_report.is_feasible


def test_ka5b_target_mode_requires_a_risk_floor():
    """Minimising capital with no risk floor is degenerate, so it is rejected at
    construction rather than silently returning an empty portfolio."""
    network = SupplyNetwork(
        baseline_risk_pts=50.0,
        interventions=(Intervention("A", cost=1.0, risk_reduction_pts=1.0),),
    )
    with pytest.raises(ValueError, match="required_risk_reduction_pts"):
        OptimizationRequest(network=network, objective=MinimizeCapitalObjective())


def test_ka5c_unreachable_target_is_reported_infeasible():
    """Asking for more reduction than the whole portfolio can deliver must fail
    loudly. The acquired system would have returned a non-optimal solve and
    rendered the resulting numbers anyway."""
    network = SupplyNetwork(
        baseline_risk_pts=90.0,
        interventions=(Intervention("A", cost=10_000.0, risk_reduction_pts=5.0),),
    )
    result = solve(
        _request(
            network,
            objective=MinimizeCapitalObjective(),
            required_risk_reduction_pts=50.0,
        )
    )
    assert result.status == SolveStatus.INFEASIBLE
    assert not result.constraint_report.is_feasible


# --------------------------------------------------------------------------- #
# KA-6  The baseline risk cap
# --------------------------------------------------------------------------- #


def test_ka6_risk_cap_prevents_buying_unusable_reduction():
    """Baseline risk is 10 points; two nodes each claim 20 points of reduction.

    With the cap enforced in-model, total reduction stops at the baseline. With
    it disabled — the acquired system's behaviour — the optimizer funds reduction
    it cannot use and the result flags that its capital efficiency is overstated.
    """
    network = SupplyNetwork(
        baseline_risk_pts=10.0,
        interventions=(
            Intervention("A", cost=10_000.0, risk_reduction_pts=20.0),
            Intervention("B", cost=10_000.0, risk_reduction_pts=20.0),
        ),
    )

    capped = solve(_request(network, budget=1_000_000.0, enforce_risk_cap=True))
    uncapped = solve(_request(network, budget=1_000_000.0, enforce_risk_cap=False))

    assert capped.raw_risk_reduction_pts <= 10.0 + 1e-6
    assert not capped.risk_cap_was_binding
    assert capped.constraint_report.is_feasible

    assert uncapped.raw_risk_reduction_pts > 10.0
    assert uncapped.risk_cap_was_binding
    assert uncapped.risk_reduction_pts == pytest.approx(10.0, abs=TOL)
    assert uncapped.optimized_risk_pts == pytest.approx(0.0, abs=TOL)
    # The uncapped run spends strictly more capital for identical delivered risk.
    assert uncapped.net_capital > capped.net_capital


# --------------------------------------------------------------------------- #
# KA-7  Concave-in-funding response changes the decision
# --------------------------------------------------------------------------- #


def test_ka7_concave_response_spreads_funding_linear_does_not():
    """Two identical nodes, budget covering one node's full cost.

    Under a linear response the objective is indifferent between concentrating
    and splitting: 1.0 + 0.0 and 0.5 + 0.5 both deliver 10 points.

    Under a genuinely concave response with alpha = 0.85, splitting delivers
    2 * 10 * 0.5**0.85 = 11.09 points against 10.0 for concentrating, so the
    optimum splits. This is what diminishing returns to funding actually implies,
    and the acquired system could not express it: its exponent was applied to the
    risk parameter, leaving the model linear in funding.
    """
    network = SupplyNetwork(
        baseline_risk_pts=80.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0),
            Intervention("B", cost=100_000.0, risk_reduction_pts=10.0),
        ),
    )

    linear = solve(
        _request(network, budget=100_000.0, risk_response=LinearResponse())
    )
    concave = solve(
        _request(
            network,
            budget=100_000.0,
            risk_response=AllocationConcaveResponse(exponent=0.85),
        )
    )

    linear_scales = linear.scales()
    concave_scales = concave.scales()

    # Linear: total scale is 1.0 however it is distributed.
    assert sum(linear_scales.values()) == pytest.approx(1.0, abs=1e-6)
    assert linear.risk_reduction_pts == pytest.approx(10.0, abs=1e-6)

    # Concave: both nodes funded, close to an even split.
    # The split lands within one facet width of the linearised response. With 13
    # breakpoints over [0.3, 1.0] a facet spans about 0.058, so 0.06 is the bound
    # the approximation can actually honour. (An earlier revision asserted half a
    # facet. That was right for the tangent envelope, whose facets meet away from
    # the breakpoints, but the objective now uses a chord interpolant whose kinks
    # sit *on* the breakpoints, so the indifference region is a full segment. The
    # looser bound is not a regression: it is the honest bound for a conservative
    # approximation that no longer over-states what each node delivers.)
    # the linearisation can actually promise. Reported risk reduction is computed
    # from the exact concave response, not the envelope.
    assert concave_scales["A"] > 0.3
    assert concave_scales["B"] > 0.3
    assert abs(concave_scales["A"] - concave_scales["B"]) < 0.06
    assert sum(concave_scales.values()) == pytest.approx(1.0, abs=1e-5)

    expected = 2 * 10.0 * math.pow(0.5, 0.85)
    assert concave.risk_reduction_pts == pytest.approx(expected, rel=2e-3)
    assert concave.risk_reduction_pts > linear.risk_reduction_pts
    assert concave.constraint_report.is_feasible


def test_ka7b_tangent_envelope_never_understates_ground_truth():
    """The linearisation is an outer approximation, so the linear programme's
    view of risk reduction must never be *below* the true concave value at the
    solved point. Reported figures come from ground truth regardless, and the
    residual gap is bounded."""
    response = AllocationConcaveResponse(exponent=0.85)
    coefficient = 10.0
    tangents = response.tangents(coefficient, breakpoints=(0.3, 0.45, 0.6, 0.75, 1.0))
    for x in (0.3, 0.37, 0.5, 0.62, 0.8, 1.0):
        truth = response.evaluate(coefficient, x)
        envelope = min(t.at(x) for t in tangents)
        assert envelope >= truth - 1e-9
        assert envelope - truth < 0.02 * coefficient
