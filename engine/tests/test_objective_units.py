"""Tests for the objective reformulation.

The headline test is :func:`test_legacy_objective_decision_flips_on_a_pure_unit_change`.
It demonstrates the incoherence concretely: relabelling lead-time savings from
days to years — which changes nothing about the business — reverses the portfolio
the legacy objective selects, while the monetary objective is invariant once the
price is expressed in the matching unit.

That is the argument for doing this before calibration. Any parameter tuned
against the legacy objective was tuned against the accident of the input data's
units.
"""

from __future__ import annotations

import pytest

from scrcae.domain import Intervention, SupplyNetwork
from scrcae.optimization import (
    LegacyWeightedObjective,
    MonetaryNPVObjective,
    OptimizationRequest,
    PriceBook,
    solve,
)
from scrcae.risk import LinearResponse, ParameterPowerResponse

DAYS_PER_YEAR = 365.0


def _network(lead_time_unit_divisor: float) -> SupplyNetwork:
    """Two candidates, one risk-led and one lead-time-led.

    ``lead_time_unit_divisor`` rescales the lead-time column only. Dividing by
    365 relabels the same savings from days into years. No business fact changes.
    """
    return SupplyNetwork(
        baseline_risk_pts=60.0,
        interventions=(
            Intervention(
                "RISK_LED",
                cost=100_000.0,
                risk_reduction_pts=20.0,
                lead_time_saved_days=1.0 / lead_time_unit_divisor,
                min_funding_scale=1.0,
            ),
            Intervention(
                "LEAD_LED",
                cost=100_000.0,
                risk_reduction_pts=5.0,
                lead_time_saved_days=30.0 / lead_time_unit_divisor,
                min_funding_scale=1.0,
            ),
        ),
    )


def _chosen(result) -> str:
    active = [a.node_id for a in result.active_allocations]
    assert len(active) == 1, f"expected exactly one funded node, got {active}"
    return active[0]


def test_legacy_objective_decision_flips_on_a_pure_unit_change():
    """Same preferences, same data, different unit label, opposite decision."""
    objective = LegacyWeightedObjective(weight=0.5)
    response = ParameterPowerResponse(exponent=0.85)

    in_days = solve(
        OptimizationRequest(
            network=_network(1.0),
            objective=objective,
            risk_response=response,
            budget=100_000.0,
            enforce_risk_cap=False,
        )
    )
    in_years = solve(
        OptimizationRequest(
            network=_network(DAYS_PER_YEAR),
            objective=objective,
            risk_response=response,
            budget=100_000.0,
            enforce_risk_cap=False,
        )
    )

    assert _chosen(in_days) == "LEAD_LED"
    assert _chosen(in_years) == "RISK_LED"

    # The objective is self-identifying as incommensurate in the audit ledger, so
    # any historic result carrying it can be found and reviewed.
    assert "incommensurate" in in_days.objective_unit


def test_monetary_objective_is_invariant_under_the_same_unit_change():
    """Convert the price alongside the data and the decision does not move."""
    per_day = PriceBook(
        value_per_risk_point=100_000.0,
        value_per_lead_time_day=500.0,
        benefit_horizon_years=5,
        discount_rate=0.10,
        source="test",
    )
    per_year = PriceBook(
        value_per_risk_point=100_000.0,
        value_per_lead_time_day=500.0 * DAYS_PER_YEAR,
        benefit_horizon_years=5,
        discount_rate=0.10,
        source="test",
    )

    in_days = solve(
        OptimizationRequest(
            network=_network(1.0),
            objective=MonetaryNPVObjective(prices=per_day),
            risk_response=LinearResponse(),
            budget=100_000.0,
            enforce_risk_cap=False,
        )
    )
    in_years = solve(
        OptimizationRequest(
            network=_network(DAYS_PER_YEAR),
            objective=MonetaryNPVObjective(prices=per_year),
            risk_response=LinearResponse(),
            budget=100_000.0,
            enforce_risk_cap=False,
        )
    )

    assert _chosen(in_days) == _chosen(in_years)
    assert in_days.objective_value == pytest.approx(in_years.objective_value, rel=1e-9)
    assert in_days.objective_unit == "currency (net present value)"


def test_monetary_objective_declines_value_destroying_portfolios():
    """When benefits are worth less than the capital they consume, fund nothing.

    The legacy objective could never do this. It never priced capital, so with any
    budget at all it always spent the budget — the optimum was "spend it" by
    construction. Pricing capital means the engine can now say no, which is the
    single most valuable thing a capital allocation model can say.
    """
    network = SupplyNetwork(
        baseline_risk_pts=60.0,
        interventions=(
            Intervention("A", cost=1_000_000.0, risk_reduction_pts=1.0),
            Intervention("B", cost=2_000_000.0, risk_reduction_pts=2.0),
        ),
    )
    prices = PriceBook(
        value_per_risk_point=100.0,  # a point of risk is worth almost nothing here
        benefit_horizon_years=5,
        discount_rate=0.10,
        source="test",
    )
    result = solve(
        OptimizationRequest(
            network=network,
            objective=MonetaryNPVObjective(prices=prices),
            risk_response=LinearResponse(),
            budget=5_000_000.0,
        )
    )

    assert result.active_allocations == ()
    assert result.net_capital == pytest.approx(0.0, abs=1e-9)
    assert result.objective_value == pytest.approx(0.0, abs=1e-6)


def test_legacy_objective_always_spends_the_budget():
    """The companion to the test above: the legacy objective funds the same
    value-destroying portfolio, because nothing in it opposes spending."""
    network = SupplyNetwork(
        baseline_risk_pts=60.0,
        interventions=(
            Intervention("A", cost=1_000_000.0, risk_reduction_pts=1.0),
            Intervention("B", cost=2_000_000.0, risk_reduction_pts=2.0),
        ),
    )
    result = solve(
        OptimizationRequest(
            network=network,
            objective=LegacyWeightedObjective(weight=1.0),
            risk_response=ParameterPowerResponse(),
            budget=5_000_000.0,
            enforce_risk_cap=False,
        )
    )
    assert len(result.active_allocations) == 2
    assert result.net_capital == pytest.approx(3_000_000.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Price book arithmetic
# --------------------------------------------------------------------------- #


def test_annuity_factor_matches_closed_form():
    assert PriceBook(1.0, discount_rate=0.0, benefit_horizon_years=5).annuity_factor == (
        pytest.approx(5.0)
    )
    # sum of 1.1^-t for t = 1..5
    assert PriceBook(
        1.0, discount_rate=0.10, benefit_horizon_years=5
    ).annuity_factor == pytest.approx(3.790786769, rel=1e-9)
    assert PriceBook(
        1.0, discount_rate=0.10, benefit_horizon_years=1
    ).annuity_factor == pytest.approx(1.0 / 1.1, rel=1e-12)


def test_annuity_factor_is_increasing_in_horizon_and_decreasing_in_rate():
    short = PriceBook(1.0, discount_rate=0.10, benefit_horizon_years=3).annuity_factor
    long = PriceBook(1.0, discount_rate=0.10, benefit_horizon_years=10).annuity_factor
    cheap = PriceBook(1.0, discount_rate=0.05, benefit_horizon_years=10).annuity_factor
    assert long > short
    assert cheap > long


def test_price_book_from_annual_exposure_divides_across_the_risk_scale():
    prices = PriceBook.from_annual_exposure(
        10_000_000.0, benefit_horizon_years=5, discount_rate=0.10
    )
    assert prices.value_per_risk_point == pytest.approx(100_000.0)


def test_strategic_multipliers_scale_the_monetary_base():
    prices = PriceBook(
        value_per_risk_point=1_000.0,
        value_per_lead_time_day=10.0,
        benefit_horizon_years=1,
        discount_rate=0.0,
    )
    plain = MonetaryNPVObjective(prices=prices)
    weighted = MonetaryNPVObjective(prices=prices, strategic_risk_multiplier=1.5)

    assert plain.risk_point_npv == pytest.approx(1_000.0)
    assert weighted.risk_point_npv == pytest.approx(1_500.0)
    # Lead time is untouched, so the multiplier expresses a preference about risk
    # rather than a rescaling of the whole objective.
    assert weighted.lead_time_day_npv == pytest.approx(plain.lead_time_day_npv)


def test_negative_prices_are_rejected():
    with pytest.raises(ValueError):
        PriceBook(value_per_risk_point=-1.0)
    with pytest.raises(ValueError):
        PriceBook(value_per_risk_point=1.0, benefit_horizon_years=0)
    with pytest.raises(ValueError):
        MonetaryNPVObjective(
            prices=PriceBook(1.0), strategic_risk_multiplier=-0.5
        )


def test_legacy_weight_bounds_are_enforced():
    with pytest.raises(ValueError):
        LegacyWeightedObjective(weight=1.5)


def test_carbon_price_penalises_emitting_interventions():
    """A high internal carbon price should reject an otherwise attractive node."""
    network = SupplyNetwork(
        baseline_risk_pts=60.0,
        interventions=(
            Intervention(
                "DIRTY",
                cost=100_000.0,
                risk_reduction_pts=10.0,
                carbon_tons=5_000.0,
                min_funding_scale=1.0,
            ),
        ),
    )

    def run(carbon_price: float):
        return solve(
            OptimizationRequest(
                network=network,
                objective=MonetaryNPVObjective(
                    prices=PriceBook(
                        value_per_risk_point=50_000.0,
                        price_per_carbon_ton=carbon_price,
                        benefit_horizon_years=5,
                        discount_rate=0.10,
                    )
                ),
                risk_response=LinearResponse(),
                budget=100_000.0,
            )
        )

    assert len(run(0.0).active_allocations) == 1
    assert run(500.0).active_allocations == ()
