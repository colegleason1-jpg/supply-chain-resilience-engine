"""The price of risk: derivation, provenance, and refusal to launder a guess."""

from __future__ import annotations

import math

import pytest

from app.services.pricing import (
    DAYS_PER_YEAR,
    RISK_SCALE_MAX,
    DisruptionExposure,
    PricingError,
    derive_prices,
    unsourced_price_book,
)


def _exposure(**overrides) -> DisruptionExposure:
    kwargs = dict(
        annual_revenue=400_000_000.0,
        revenue_at_risk_share=0.35,
        gross_margin=0.28,
        disruption_days=21.0,
        fixed_cost_per_event=500_000.0,
        basis="FY25 statutory accounts; duration from 2023 port closure review",
    )
    kwargs.update(overrides)
    return DisruptionExposure(**kwargs)


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #


def test_cost_of_one_event_follows_the_stated_formula():
    exposure = _exposure()
    at_risk = 400_000_000.0 * 0.35
    daily = at_risk * 0.28 / DAYS_PER_YEAR
    assert exposure.at_risk_revenue == pytest.approx(at_risk)
    assert exposure.daily_gross_profit_at_risk == pytest.approx(daily)
    assert exposure.cost_of_one_event == pytest.approx(daily * 21.0 + 500_000.0)


def test_one_risk_point_is_one_hundredth_of_an_event():
    """The load-bearing assumption: at 100 points a disruption is certain."""
    derivation = derive_prices(_exposure(), annual_disruption_probability=0.655)
    assert derivation.value_per_risk_point == pytest.approx(
        derivation.cost_of_one_event / RISK_SCALE_MAX
    )


def test_the_derivation_agrees_with_the_engines_own_reading_of_the_scale():
    """pricing.py and PriceBook.from_annual_exposure must not drift apart."""
    from scrcae import PriceBook

    exposure = _exposure()
    derived = derive_prices(exposure, annual_disruption_probability=0.5)
    engine_side = PriceBook.from_annual_exposure(
        exposure.cost_of_one_event, risk_scale_max=RISK_SCALE_MAX
    )
    assert derived.value_per_risk_point == pytest.approx(
        engine_side.value_per_risk_point
    )


def test_lead_time_is_priced_at_the_event_probability():
    exposure = _exposure()
    derivation = derive_prices(exposure, annual_disruption_probability=0.4)
    assert derivation.value_per_lead_time_day == pytest.approx(
        exposure.daily_gross_profit_at_risk * 0.4
    )


def test_a_day_of_lead_time_is_worth_more_when_disruption_is_likelier():
    low = derive_prices(_exposure(), annual_disruption_probability=0.1)
    high = derive_prices(_exposure(), annual_disruption_probability=0.8)
    assert high.value_per_lead_time_day > low.value_per_lead_time_day


def test_the_risk_point_price_does_not_use_the_probability():
    """The risk scale already expresses probability; using it again prices it twice."""
    low = derive_prices(_exposure(), annual_disruption_probability=0.05)
    high = derive_prices(_exposure(), annual_disruption_probability=0.95)
    assert low.value_per_risk_point == pytest.approx(high.value_per_risk_point)


def test_certain_disruption_prices_lead_time_at_a_full_day_of_contribution():
    exposure = _exposure()
    derivation = derive_prices(exposure, annual_disruption_probability=1.0)
    assert derivation.value_per_lead_time_day == pytest.approx(
        exposure.daily_gross_profit_at_risk
    )


def test_a_longer_disruption_raises_the_price_of_risk_proportionally():
    short = derive_prices(
        _exposure(disruption_days=10.0, fixed_cost_per_event=0.0),
        annual_disruption_probability=0.5,
    )
    long = derive_prices(
        _exposure(disruption_days=20.0, fixed_cost_per_event=0.0),
        annual_disruption_probability=0.5,
    )
    assert long.value_per_risk_point == pytest.approx(2 * short.value_per_risk_point)


def test_fixed_cost_lands_in_the_price_without_being_scaled_by_duration():
    without = _exposure(fixed_cost_per_event=0.0)
    with_fixed = _exposure(fixed_cost_per_event=1_000_000.0)
    assert with_fixed.cost_of_one_event - without.cost_of_one_event == pytest.approx(
        1_000_000.0
    )


def test_a_business_with_no_at_risk_revenue_prices_only_its_fixed_costs():
    exposure = _exposure(revenue_at_risk_share=0.0, fixed_cost_per_event=250_000.0)
    assert exposure.cost_of_one_event == pytest.approx(250_000.0)


# --------------------------------------------------------------------------- #
# Refusing inputs that cannot mean anything
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides",
    [
        {"gross_margin": 1.4},          # 140% entered as a fraction
        {"gross_margin": -0.1},
        {"revenue_at_risk_share": 1.2},
        {"revenue_at_risk_share": -0.5},
        {"disruption_days": 400.0},     # longer than the year it is amortised over
        {"disruption_days": -3.0},
        {"annual_revenue": -1.0},
        {"fixed_cost_per_event": -1.0},
        {"gross_margin": math.nan},
        {"annual_revenue": math.inf},
    ],
)
def test_out_of_range_inputs_are_refused_rather_than_clamped(overrides):
    """A margin of 1.4 is a units mistake, not a slightly optimistic margin."""
    with pytest.raises(PricingError):
        _exposure(**overrides)


def test_the_refusal_names_the_field():
    with pytest.raises(PricingError, match="gross_margin"):
        _exposure(gross_margin=2.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5, math.nan])
def test_an_impossible_probability_is_refused(bad):
    with pytest.raises(PricingError):
        derive_prices(_exposure(), annual_disruption_probability=bad)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_a_derived_price_book_is_sourced():
    derivation = derive_prices(_exposure(), annual_disruption_probability=0.5)
    assert derivation.prices.is_sourced


def test_the_source_string_carries_the_arithmetic_not_just_a_label():
    """It flows into the audit ledger, so it has to be checkable there."""
    derivation = derive_prices(_exposure(), annual_disruption_probability=0.5)
    source = derivation.prices.source
    assert "cost of one event" in source
    assert "margin" in source
    assert "days" in source
    assert "FY25 statutory accounts" in source


def test_unattributed_figures_are_called_out_in_the_assumptions():
    derivation = derive_prices(
        _exposure(basis=""), annual_disruption_probability=0.5
    )
    assert any("unattributed" in a for a in derivation.assumptions)


def test_attributed_figures_do_not_carry_that_warning():
    derivation = derive_prices(_exposure(), annual_disruption_probability=0.5)
    assert not any("unattributed" in a for a in derivation.assumptions)


def test_the_linearity_assumption_is_stated_rather_than_buried():
    derivation = derive_prices(_exposure(), annual_disruption_probability=0.5)
    assert any("linear in expected loss" in a for a in derivation.assumptions)


def test_the_workings_show_every_step_of_the_derivation():
    derivation = derive_prices(_exposure(), annual_disruption_probability=0.5)
    workings = derivation.workings()
    assert len(workings) == 6
    assert any("At-risk revenue" in w for w in workings)
    assert any("Cost of one event" in w for w in workings)
    assert any("annuity factor" in w for w in workings)


def test_a_typed_price_with_no_source_is_not_sourced():
    """Typing a price in is legitimate. Looking derived when it is not, is not."""
    book = unsourced_price_book(value_per_risk_point=50_000.0)
    assert not book.is_sourced
    assert book.source == "unstated"


def test_a_typed_price_with_a_stated_source_records_both_facts():
    book = unsourced_price_book(
        value_per_risk_point=50_000.0, stated_source="Group risk model v3"
    )
    assert book.is_sourced
    assert "Entered directly" in book.source
    assert "Group risk model v3" in book.source


def test_whitespace_is_not_a_source():
    assert not unsourced_price_book(
        value_per_risk_point=1.0, stated_source="   "
    ).is_sourced


def test_horizon_and_discount_rate_reach_the_price_book():
    derivation = derive_prices(
        _exposure(),
        annual_disruption_probability=0.5,
        benefit_horizon_years=7,
        discount_rate=0.06,
    )
    assert derivation.prices.benefit_horizon_years == 7
    assert derivation.prices.discount_rate == pytest.approx(0.06)


def test_the_derivation_is_immutable_and_reproducible():
    first = derive_prices(_exposure(), annual_disruption_probability=0.5)
    second = derive_prices(_exposure(), annual_disruption_probability=0.5)
    assert first.prices == second.prices
    with pytest.raises((AttributeError, TypeError)):
        first.prices.value_per_risk_point = 1.0  # type: ignore[misc]
