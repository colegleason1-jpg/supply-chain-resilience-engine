"""Inputs. Collects values and returns them; decides nothing.

The acquired system built `st.sidebar.multiselect` widgets *inside* the optimizer
loop, with different widget keys per mode (`bundle_req_target_{idx}` vs
`bundle_req_{idx}`). Toggling the mode switch therefore silently reset bundle
membership, because the new keys had no stored state and fell back to `nodes[:2]`
(F5). Every widget in this app is created exactly once, in this module, before any
solving happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from scrcae import PriceBook

from app import adapters
from app.services.macro import MacroReading, fetch_macro_reading
from app.services.pricing import (
    DisruptionExposure,
    PriceDerivation,
    PricingError,
    derive_prices,
    unsourced_price_book,
)


@dataclass(frozen=True, slots=True)
class Inputs:
    """Everything the user chose, in one object."""

    mode: str
    baseline_risk_pts: float
    budget: float | None
    target_risk_pts: float | None
    macro: MacroReading
    apply_macro: bool
    exponent: float
    prices: PriceBook
    iterations: int
    seed: int
    run_simulation: bool
    show_sweep: bool
    apply_market_exposure: bool = False
    price_derivation: PriceDerivation | None = None

    @property
    def is_target_mode(self) -> bool:
        return self.target_risk_pts is not None

    # The three price fields these replace are kept as read-only views, so callers
    # that only need a number are unaffected by where the number came from.
    @property
    def value_per_risk_point(self) -> float:
        return self.prices.value_per_risk_point

    @property
    def value_per_lead_time_day(self) -> float:
        return self.prices.value_per_lead_time_day

    @property
    def price_source(self) -> str:
        return self.prices.source

    @property
    def macro_multiplier(self) -> float:
        return self.macro.multiplier if self.apply_macro else 1.0


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_macro() -> MacroReading:
    """Fetched server-side and cached, so a rerun is not a new HTTP request."""
    return fetch_macro_reading()


def render() -> Inputs:
    st.sidebar.title("Assumptions")

    mode_label = st.sidebar.radio(
        "Model",
        ["Monetary NPV (recommended)", "Legacy parity"],
        key="mode",
        help=(
            "Monetary NPV values risk reduction and lead time in currency, so the "
            "objective is a single coherent quantity. Legacy parity reproduces the "
            "previous system's arithmetic exactly, including its known defects, and "
            "exists so results can be reconciled against it — not for new decisions."
        ),
    )
    mode = adapters.MODE_LEGACY if mode_label == "Legacy parity" else adapters.MODE_MONETARY
    if mode == adapters.MODE_LEGACY:
        st.sidebar.warning(
            "Legacy parity mixes percentage points with days in one objective and "
            "disables the risk cap. Figures are reproducible, not defensible."
        )

    st.sidebar.subheader("Risk")
    baseline = st.sidebar.number_input(
        "Baseline risk (%)",
        min_value=0.0,
        max_value=100.0,
        value=65.5,
        step=0.5,
        key="baseline_risk",
    )

    macro = _cached_macro()
    apply_macro = st.sidebar.checkbox(
        "Apply macro adjustment",
        value=macro.is_live,
        disabled=not macro.is_live,
        key="apply_macro",
    )
    if macro.is_live:
        st.sidebar.caption(f"Live: {macro.multiplier:g}x — {macro.provenance}")
    else:
        # The acquired system showed a green feed indicator next to a hardcoded 1.0.
        st.sidebar.caption(f"{macro.status}. {macro.provenance}")

    st.sidebar.subheader("Objective")
    target_mode = st.sidebar.checkbox(
        "Hit a risk target for least capital",
        value=False,
        key="target_mode",
        help=(
            "Off: spend the budget to buy the most value. On: name a risk target and "
            "find the cheapest portfolio that reaches it."
        ),
    )

    budget: float | None = None
    target: float | None = None
    if target_mode:
        target = st.sidebar.number_input(
            "Target risk (%)",
            min_value=0.0,
            max_value=100.0,
            value=45.0,
            step=0.5,
            key="target_risk",
        )
        cap = st.sidebar.checkbox("Cap the spend", value=False, key="cap_spend")
        if cap:
            budget = st.sidebar.number_input(
                "Maximum capital",
                min_value=0.0,
                value=750_000.0,
                step=25_000.0,
                key="max_capital",
            )
    else:
        budget = st.sidebar.number_input(
            "Capital budget",
            min_value=0.0,
            value=750_000.0,
            step=25_000.0,
            key="budget",
        )

    prices = unsourced_price_book(value_per_risk_point=50_000.0)
    derivation: PriceDerivation | None = None
    if mode == adapters.MODE_MONETARY:
        prices, derivation = _render_prices(baseline)

    apply_market_exposure = st.sidebar.checkbox(
        "Apply market exposure",
        value=False,
        key="apply_market_exposure",
        help=(
            "Adjusts each intervention's effectiveness using the market exposures you "
            "set in the Market exposure table, priced off a live feed. Applied at "
            "solve time only — your intervention figures are never modified."
        ),
    )

    with st.sidebar.expander("Advanced"):
        exponent = st.number_input(
            "Diminishing-returns exponent",
            min_value=0.1,
            max_value=1.0,
            value=0.85,
            step=0.05,
            key="exponent",
            help=(
                "1.0 means a linear response to funding. Below 1.0 means partial "
                "funding delivers proportionally more than full funding. Inherited "
                "from the previous system and not empirically calibrated."
            ),
        )
        show_sweep = st.checkbox(
            "Compute budget sensitivity", value=True, key="show_sweep"
        )
        run_sim = st.checkbox("Run Monte Carlo", value=True, key="run_simulation")
        iterations = st.number_input(
            "Iterations",
            min_value=1_000,
            max_value=200_000,
            value=10_000,
            step=1_000,
            key="iterations",
        )
        seed = st.number_input("Seed", min_value=0, value=42, step=1, key="seed")

    return Inputs(
        mode=mode,
        baseline_risk_pts=float(baseline),
        budget=float(budget) if budget is not None else None,
        target_risk_pts=float(target) if target is not None else None,
        macro=macro,
        apply_macro=bool(apply_macro),
        exponent=float(exponent),
        prices=prices,
        price_derivation=derivation,
        apply_market_exposure=bool(apply_market_exposure),
        iterations=int(iterations),
        seed=int(seed),
        run_simulation=bool(run_sim),
        show_sweep=bool(show_sweep),
    )


def _render_prices(baseline_risk_pts: float) -> tuple[PriceBook, PriceDerivation | None]:
    """The price of risk: derived from the business's own numbers, or entered.

    Both paths are always rendered so that every widget exists on every run — the
    direct-entry fields are disabled rather than removed when derivation is on.
    That keeps the copilot's proposals (which write to these widget keys) coherent
    regardless of which path is selected, and avoids the class of bug that made
    bundle membership reset on a mode switch (F5).
    """
    st.sidebar.subheader("Prices")
    st.sidebar.caption(
        "These convert risk and time into money. Every currency figure in the app "
        "is only as defensible as they are."
    )
    derive = st.sidebar.checkbox(
        "Derive from disruption exposure",
        value=False,
        key="derive_prices",
        help=(
            "Builds the price of risk from revenue, margin and expected disruption "
            "duration — figures a finance function already reports. Preferred over "
            "an industry benchmark, which describes a population of companies rather "
            "than this one."
        ),
    )

    with st.sidebar.expander("Disruption exposure", expanded=derive):
        annual_revenue = st.number_input(
            "Annual revenue",
            min_value=0.0,
            value=400_000_000.0,
            step=1_000_000.0,
            key="annual_revenue",
        )
        share_pct = st.number_input(
            "Share of revenue through this network (%)",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            step=5.0,
            key="revenue_at_risk_share_pct",
        )
        margin_pct = st.number_input(
            "Gross margin on that revenue (%)",
            min_value=0.0,
            max_value=100.0,
            value=28.0,
            step=1.0,
            key="gross_margin_pct",
            help="Margin, not revenue: a disruption defers contribution, and the "
            "cost of goods not sold is largely not incurred either.",
        )
        disruption_days = st.number_input(
            "Expected days of interrupted supply per event",
            min_value=0.0,
            max_value=365.0,
            value=21.0,
            step=1.0,
            key="disruption_days",
        )
        fixed_cost = st.number_input(
            "Fixed cost per event",
            min_value=0.0,
            value=0.0,
            step=100_000.0,
            key="fixed_cost_per_event",
            help="Penalties, expedited freight, remediation, customer credits — "
            "costs incurred regardless of how long the disruption lasts.",
        )
        probability_pct = st.number_input(
            "Annual probability of a disruption (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(min(100.0, max(0.0, baseline_risk_pts))),
            step=1.0,
            key="event_probability_pct",
            help=(
                "Prices lead time only. Defaults to your baseline risk, which reads "
                "the risk scale as an annual probability — change it if you hold a "
                "separately estimated event frequency."
            ),
        )
        basis = st.text_input(
            "Where these figures came from",
            value="",
            placeholder="e.g. FY26 statutory accounts; duration from 2023 review",
            key="exposure_basis",
        )

    entered_point = st.sidebar.number_input(
        "Value per risk point (annual)",
        min_value=0.0,
        value=50_000.0,
        step=5_000.0,
        key="value_per_risk_point",
        disabled=derive,
    )
    entered_day = st.sidebar.number_input(
        "Value per lead-time day (annual)",
        min_value=0.0,
        value=0.0,
        step=500.0,
        key="value_per_day",
        disabled=derive,
    )
    entered_source = st.sidebar.text_input(
        "Where these came from",
        value="",
        placeholder="e.g. FY26 cost-of-disruption study",
        help="Left blank, the app will flag every currency figure as unsourced.",
        key="price_source",
        disabled=derive,
    )

    if not derive:
        return (
            unsourced_price_book(
                value_per_risk_point=float(entered_point),
                value_per_lead_time_day=float(entered_day),
                stated_source=entered_source,
            ),
            None,
        )

    try:
        derivation = derive_prices(
            DisruptionExposure(
                annual_revenue=float(annual_revenue),
                revenue_at_risk_share=float(share_pct) / 100.0,
                gross_margin=float(margin_pct) / 100.0,
                disruption_days=float(disruption_days),
                fixed_cost_per_event=float(fixed_cost),
                basis=basis,
            ),
            annual_disruption_probability=float(probability_pct) / 100.0,
        )
    except PricingError as exc:
        # Falls back to the entered figures rather than to a guess, and says which
        # numbers are actually in force. Substituting a plausible default here would
        # leave the user reading currency figures they believe they derived.
        st.sidebar.error(
            f"The exposure figures cannot be priced: {exc}. Using the entered "
            f"value per risk point instead."
        )
        return (
            unsourced_price_book(
                value_per_risk_point=float(entered_point),
                value_per_lead_time_day=float(entered_day),
                stated_source=entered_source,
            ),
            None,
        )

    st.sidebar.metric(
        "Value per risk point (annual)", f"{derivation.value_per_risk_point:,.0f}"
    )
    st.sidebar.metric(
        "Value per lead-time day (annual)",
        f"{derivation.value_per_lead_time_day:,.0f}",
    )
    with st.sidebar.expander("How this was derived"):
        for line in derivation.workings():
            st.caption(line)
        st.caption("**Assumptions**")
        for assumption in derivation.assumptions:
            st.caption(f"- {assumption}")
    return derivation.prices, derivation
