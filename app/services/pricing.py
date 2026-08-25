"""Deriving the price of risk from figures a finance function already owns.

Why this module exists
---------------------
Reformulating the objective in money (ADR-001) moved the problem rather than
solving it. The acquired system's incoherent ``w = 0.65`` was replaced by
``value_per_risk_point``, which is coherent, auditable and — as shipped — still a
number somebody typed into a box. Every currency figure the engine reports is
directly proportional to it. If it is invented, so is the headline, and the fact
that it is now expressed in dollars makes it *more* persuasive rather than more
defensible. That is a worse failure mode than an obviously meaningless weight.

What this module deliberately does not do
----------------------------------------
It does not ship an industry benchmark as a default. Published figures for the
average cost of a disruption are population statistics: a median across firms of
different size, margin, sector and network depth. Substituting one for this
company's price of risk would produce a number that *looks* sourced — it would
have a citation next to it — while being no more applicable to the business than
the 0.85 exponent was. Borrowed authority is the specific defect this rebuild
exists to remove, and a footnote is not a calibration.

What it does instead
--------------------
It derives the price from quantities a CFO can already produce, defend to an
auditor, and disagree with line by line:

* annual revenue, and the share of it that flows through this network
* gross margin on that revenue
* how long a disruption is expected to interrupt supply
* any fixed per-event cost (penalties, expedited freight, remediation)

From those::

    cost of one event = revenue x share x (days / 365) x margin + fixed cost
    value per risk point (per year) = cost of one event / 100

The second line is the load-bearing assumption and it is worth stating plainly.
The risk scale runs 0-100, and one point is read as one percentage point of
annual probability that a disruption occurs. At 100 points a disruption is
certain, so the expected annual loss equals the cost of one event; the scale is
therefore linear in expected loss. That is an assumption about the scale, not a
finding about the world, and it is the single place to change if a calibration
exercise ever establishes a non-linear loss curve. The engine's
``PriceBook.from_annual_exposure`` encodes the same reading, so this module and
the engine cannot drift apart on it.

Note what the derivation does *not* depend on: the network's current baseline
risk. One point is one point regardless of how many the business currently
carries. A price of risk that moved with the baseline would make the objective
non-stationary in the risk being optimised, and the resulting numbers would not
be comparable between two runs of the same portfolio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scrcae import PriceBook

__all__ = [
    "DAYS_PER_YEAR",
    "RISK_SCALE_MAX",
    "DisruptionExposure",
    "PriceDerivation",
    "PricingError",
    "derive_prices",
    "unsourced_price_book",
]

#: Denominator for converting an annual figure into a daily one. 365 rather than a
#: working-year figure because a disruption does not observe weekends.
DAYS_PER_YEAR = 365.0

#: The risk scale's upper bound. One point is 1/100th of certainty.
RISK_SCALE_MAX = 100.0


class PricingError(ValueError):
    """An input to the derivation was outside the range it can mean anything in."""


def _require(name: str, value: float, *, low: float, high: float) -> float:
    """Range-check an input, naming it.

    Deliberately raises rather than clamping. A gross margin of 1.4 is not a
    slightly optimistic margin to be quietly rounded to 1.0 — it is a units
    mistake (140% entered as a fraction), and silently accepting it would put a
    figure inflated by 40% into every currency number the app reports.
    """
    number = float(value)
    if not math.isfinite(number):
        raise PricingError(f"{name} must be a finite number, got {value!r}")
    if not low <= number <= high:
        raise PricingError(f"{name} must lie in [{low:g}, {high:g}], got {number:g}")
    return number


@dataclass(frozen=True, slots=True)
class DisruptionExposure:
    """What one disruption costs this business, in its own numbers.

    Every field is a quantity a finance function already reports or can derive
    from something it reports. That is the whole selection criterion: an input
    nobody can source is an input that will be guessed, and a guess laundered
    through a derivation is harder to spot than a guess typed into a box.
    """

    #: Total annual revenue, in the same currency as intervention costs.
    annual_revenue: float
    #: Share of that revenue dependent on this supply network, in [0, 1].
    revenue_at_risk_share: float
    #: Gross margin on the at-risk revenue, in [0, 1]. Margin rather than revenue
    #: because a disruption defers contribution, not turnover: the cost of goods
    #: not sold is largely not incurred either.
    gross_margin: float
    #: Expected days of interrupted supply in one disruption event.
    disruption_days: float
    #: Costs incurred per event regardless of duration — contractual penalties,
    #: expedited freight, remediation, customer credits.
    fixed_cost_per_event: float = 0.0
    #: Free-text note on where these figures came from, e.g. "FY25 statutory
    #: accounts; duration from 2023 port closure post-mortem".
    basis: str = ""

    def __post_init__(self) -> None:
        _require("annual_revenue", self.annual_revenue, low=0.0, high=math.inf)
        _require("revenue_at_risk_share", self.revenue_at_risk_share, low=0.0, high=1.0)
        _require("gross_margin", self.gross_margin, low=0.0, high=1.0)
        # An upper bound of one year: beyond that the annualised framing stops
        # holding, because a disruption lasting longer than the period it is
        # amortised over is a going-concern question rather than a resilience
        # investment question.
        _require("disruption_days", self.disruption_days, low=0.0, high=DAYS_PER_YEAR)
        _require("fixed_cost_per_event", self.fixed_cost_per_event, low=0.0, high=math.inf)

    @property
    def at_risk_revenue(self) -> float:
        return self.annual_revenue * self.revenue_at_risk_share

    @property
    def daily_gross_profit_at_risk(self) -> float:
        """Contribution earned per day on the revenue this network carries."""
        return self.at_risk_revenue * self.gross_margin / DAYS_PER_YEAR

    @property
    def cost_of_one_event(self) -> float:
        """Total cost of a single disruption: lost contribution plus fixed costs."""
        return (
            self.daily_gross_profit_at_risk * self.disruption_days
            + self.fixed_cost_per_event
        )


@dataclass(frozen=True, slots=True)
class PriceDerivation:
    """A price book plus the arithmetic and assumptions that produced it."""

    prices: PriceBook
    exposure: DisruptionExposure
    annual_disruption_probability: float
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cost_of_one_event(self) -> float:
        return self.exposure.cost_of_one_event

    @property
    def value_per_risk_point(self) -> float:
        return self.prices.value_per_risk_point

    @property
    def value_per_lead_time_day(self) -> float:
        return self.prices.value_per_lead_time_day

    def workings(self) -> tuple[str, ...]:
        """The derivation, line by line, for display next to the result.

        Shown rather than summarised because the purpose is to be argued with. A
        CFO who disagrees with the duration should be able to see exactly which
        line to push on.
        """
        e = self.exposure
        return (
            f"At-risk revenue: {e.annual_revenue:,.0f} x "
            f"{e.revenue_at_risk_share:.0%} = {e.at_risk_revenue:,.0f}",
            f"Daily contribution at risk: {e.at_risk_revenue:,.0f} x "
            f"{e.gross_margin:.0%} / {DAYS_PER_YEAR:.0f} = "
            f"{e.daily_gross_profit_at_risk:,.0f}",
            f"Cost of one event: {e.daily_gross_profit_at_risk:,.0f} x "
            f"{e.disruption_days:,.0f} days + {e.fixed_cost_per_event:,.0f} fixed = "
            f"{e.cost_of_one_event:,.0f}",
            f"Value of one risk point per year: {e.cost_of_one_event:,.0f} / "
            f"{RISK_SCALE_MAX:.0f} = {self.value_per_risk_point:,.0f}",
            f"Value of one lead-time day per year: "
            f"{e.daily_gross_profit_at_risk:,.0f} x "
            f"{self.annual_disruption_probability:.1%} probability = "
            f"{self.value_per_lead_time_day:,.0f}",
            f"Present value over {self.prices.benefit_horizon_years} years at "
            f"{self.prices.discount_rate:.1%}: annuity factor "
            f"{self.prices.annuity_factor:.3f}",
        )


def derive_prices(
    exposure: DisruptionExposure,
    *,
    annual_disruption_probability: float,
    benefit_horizon_years: int = 5,
    discount_rate: float = 0.10,
    price_per_carbon_ton: float = 0.0,
) -> PriceDerivation:
    """Turn a disruption exposure into a sourced price book.

    ``annual_disruption_probability`` prices lead time only. A day of lead time
    removed is worth a day of contribution *in the scenario where supply is
    actually interrupted*, so its value scales with how likely that scenario is.
    The risk-point price deliberately does not use it: the risk scale already
    expresses probability, and multiplying by it again would price the same
    likelihood twice.

    The caller supplies the probability rather than the network's baseline risk
    figure so that the two are not silently assumed to be the same quantity. They
    usually are — the baseline is a probability in points — but a business with a
    separately estimated event frequency should be able to use it, and a caller
    passing the baseline is then making that identification explicitly.
    """
    probability = _require(
        "annual_disruption_probability", annual_disruption_probability, low=0.0, high=1.0
    )

    cost_of_event = exposure.cost_of_one_event
    value_per_day = exposure.daily_gross_profit_at_risk * probability

    assumptions = [
        "The 0-100 risk scale is linear in expected loss, and one point is one "
        "percentage point of annual probability that a disruption occurs.",
        "A disruption defers gross contribution rather than revenue, so cost of "
        "goods not sold is treated as not incurred.",
        "Benefits recur annually and are annuitised over the stated horizon; "
        "capital is spent once.",
        "A day of lead time removed is worth a day of contribution only in the "
        "disruption scenario, so it is priced at the stated event probability.",
    ]
    if not exposure.basis.strip():
        assumptions.append(
            "The exposure figures themselves are unattributed: no source was "
            "recorded for revenue, margin or duration."
        )

    source = (
        f"Derived from disruption exposure: cost of one event "
        f"{cost_of_event:,.0f} = {exposure.at_risk_revenue:,.0f} at-risk revenue "
        f"x {exposure.gross_margin:.0%} margin x {exposure.disruption_days:,.0f} "
        f"days / {DAYS_PER_YEAR:.0f} + {exposure.fixed_cost_per_event:,.0f} fixed; "
        f"one risk point = event cost / {RISK_SCALE_MAX:.0f}; lead-time day priced "
        f"at {probability:.1%} annual event probability"
    )
    if exposure.basis.strip():
        source = f"{source}. Figures: {exposure.basis.strip()}"

    prices = PriceBook.from_annual_exposure(
        cost_of_event,
        risk_scale_max=RISK_SCALE_MAX,
        value_per_lead_time_day=value_per_day,
        price_per_carbon_ton=price_per_carbon_ton,
        benefit_horizon_years=benefit_horizon_years,
        discount_rate=discount_rate,
        source=source,
    )
    return PriceDerivation(
        prices=prices,
        exposure=exposure,
        annual_disruption_probability=probability,
        assumptions=tuple(assumptions),
    )


def unsourced_price_book(
    *,
    value_per_risk_point: float,
    value_per_lead_time_day: float = 0.0,
    price_per_carbon_ton: float = 0.0,
    benefit_horizon_years: int = 5,
    discount_rate: float = 0.10,
    stated_source: str = "",
) -> PriceBook:
    """A price book from directly entered figures, honestly labelled.

    Typing the price in is a legitimate thing to want: a business may have its own
    model, or be exploring. What is not legitimate is the result looking identical
    to a derived one. When no source is stated the book's ``source`` stays at
    ``"unstated"``, which sets ``PriceBook.is_sourced`` to False and carries that
    fact into the audit ledger and every export.
    """
    stated = stated_source.strip()
    return PriceBook(
        value_per_risk_point=float(value_per_risk_point),
        value_per_lead_time_day=float(value_per_lead_time_day),
        price_per_carbon_ton=float(price_per_carbon_ton),
        benefit_horizon_years=int(benefit_horizon_years),
        discount_rate=float(discount_rate),
        source=(
            f"Entered directly. {stated}"
            if stated
            else "unstated"
        ),
    )
