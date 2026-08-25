"""Objective functions for the capital allocation problem.

The unit-coherence problem
--------------------------
The acquired system maximised::

    sum_n [ w * R*_n + (1 - w) * L_n ] * x_n

where ``R*_n`` is a transformed risk reduction in percentage points and ``L_n``
is lead-time saving in days. That expression adds percentage points to days.

The consequences are concrete, not stylistic:

1. **``w`` has no interpretable value.** It simultaneously encodes the decision
   maker's preference *and* the conversion factor between two incommensurate
   units. There is no value of ``w`` that means "weight risk and lead time
   equally", because equality is undefined without a common unit.
2. **``w`` is not portable.** A ``w`` tuned on a dataset whose lead-time savings
   are measured in tens of days behaves completely differently on one measured
   in hundreds, even when the decision maker's preferences are identical. The
   parameter silently absorbs the scale of the input data.
3. **It poisons everything calibrated beneath it.** Calibrating the risk
   response exponent, or the stochastic shock parameters, against an objective
   whose scale is arbitrary calibrates them against nothing. This is why the
   reformulation has to land before the calibration work, not after.
4. **The optimum is an artefact.** Because the two coefficient families have
   different scales, the argmax is driven by whichever family happens to carry
   larger raw numbers, and moving ``w`` across most of [0, 1] frequently does
   not change the selected portfolio at all — then flips it abruptly.

The reformulation
-----------------
Every term is converted into currency before it is summed. Preferences are
elicited as *prices*, which a finance function can source, defend and audit:

* ``value_per_risk_point`` — currency value of removing one percentage point of
  system risk per year (expected avoided disruption loss).
* ``value_per_lead_time_day`` — currency value of removing one day of lead time
  per year (working-capital release plus expediting avoided).
* ``price_per_carbon_ton`` — internal carbon price.

Benefits recur; capital is spent once. So benefits are annuitised over a stated
horizon at a stated discount rate, and the objective becomes a net present
value::

    maximise  sum_n [ AF * ( lambda_risk  * value_per_risk_point  * dR_n(x_n)
                           + lambda_lead  * value_per_lead_time_day * L_n * x_n
                           - price_per_carbon_ton * K_n * x_n ) ]
              - sum_n C_n x_n
              + sum_b D_b * b_v

    where AF = sum_{t=1..T} (1 + d)^-t

``lambda_*`` are optional strategic multipliers defaulting to 1.0. They let a
decision maker deliberately over-weight resilience beyond its measured monetary
value — a legitimate stance — but they now sit *on top of* a coherent monetary
base instead of substituting for one. A lambda of 1.5 means "we value this at
150% of its computed financial value", which is a statement someone can agree
or disagree with. ``w = 0.65`` was not.

Objective value is now denominated in currency, which also means it is
comparable across scenarios, auditable against a business case, and directly
interpretable: a negative optimum means the portfolio destroys value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import pulp as pl

__all__ = [
    "ObjectiveModel",
    "NodeTerms",
    "ProblemTerms",
    "PriceBook",
    "LegacyWeightedObjective",
    "MonetaryNPVObjective",
    "MinimizeCapitalObjective",
]


@dataclass(frozen=True)
class NodeTerms:
    """Solver expressions describing one node's contribution."""

    node_id: str
    #: Risk reduction in percentage points (linear expression or bounded var).
    risk_points: pl.LpAffineExpression
    #: Lead-time saving in days.
    lead_time_days: pl.LpAffineExpression
    #: Carbon impact in tonnes (positive = emitted).
    carbon_tons: pl.LpAffineExpression
    #: Capital deployed in currency units.
    capital: pl.LpAffineExpression


@dataclass(frozen=True)
class ProblemTerms:
    """Everything an objective may reference."""

    nodes: Mapping[str, NodeTerms]
    total_capital: pl.LpAffineExpression
    total_discounts: pl.LpAffineExpression

    @property
    def net_capital(self) -> pl.LpAffineExpression:
        return self.total_capital - self.total_discounts


@runtime_checkable
class ObjectiveModel(Protocol):
    """A scalar objective over :class:`ProblemTerms`."""

    version: str

    @property
    def sense(self) -> int:
        """``pulp.LpMaximize`` or ``pulp.LpMinimize``."""
        ...

    @property
    def unit(self) -> str:
        """Human-readable unit of the objective value, for the audit ledger."""
        ...

    def build(self, terms: ProblemTerms) -> pl.LpAffineExpression:
        ...


# --------------------------------------------------------------------------- #
# Legacy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LegacyWeightedObjective:
    """Bit-faithful reproduction of the acquired weighted-sum objective.

    Retained only to prove the extraction is faithful and to reproduce historic
    results. ``unit`` is deliberately reported as ``"incommensurate"`` so that
    any result carrying this objective is self-identifying in the audit ledger.
    """

    weight: float = 0.5
    version: str = "objective/legacy-weighted@1.0.0"

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must lie in [0, 1], got {self.weight}")

    @property
    def sense(self) -> int:
        return pl.LpMaximize

    @property
    def unit(self) -> str:
        return "incommensurate (percentage-points + days)"

    def build(self, terms: ProblemTerms) -> pl.LpAffineExpression:
        w = self.weight
        return pl.lpSum(
            w * node.risk_points + (1.0 - w) * node.lead_time_days
            for node in terms.nodes.values()
        )


# --------------------------------------------------------------------------- #
# Monetary reformulation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PriceBook:
    """Monetary prices that make the objective's terms commensurable.

    Every field is an elicitable business quantity with a defensible source,
    which is the whole point of the reformulation.
    """

    #: Currency per percentage point of system risk removed, per year.
    value_per_risk_point: float
    #: Currency per day of lead time removed, per year.
    value_per_lead_time_day: float = 0.0
    #: Internal carbon price, currency per tonne, per year.
    price_per_carbon_ton: float = 0.0
    #: Years over which benefits accrue.
    benefit_horizon_years: int = 5
    #: Annual discount rate, e.g. 0.10 for 10%.
    discount_rate: float = 0.10
    #: Provenance for the audit ledger.
    source: str = "unstated"

    def __post_init__(self) -> None:
        if self.value_per_risk_point < 0:
            raise ValueError("value_per_risk_point must be non-negative")
        if self.value_per_lead_time_day < 0:
            raise ValueError("value_per_lead_time_day must be non-negative")
        if self.price_per_carbon_ton < 0:
            raise ValueError("price_per_carbon_ton must be non-negative")
        if self.benefit_horizon_years < 1:
            raise ValueError("benefit_horizon_years must be at least 1")
        if self.discount_rate <= -1.0:
            raise ValueError("discount_rate must exceed -1")

    @classmethod
    def from_annual_exposure(
        cls,
        annual_disruption_exposure: float,
        *,
        risk_scale_max: float = 100.0,
        **kwargs: object,
    ) -> PriceBook:
        """Derive ``value_per_risk_point`` from total annual exposure.

        ``annual_disruption_exposure`` is the expected annual disruption loss
        the business would carry at the top of the risk scale. One percentage
        point of the scale is therefore worth ``exposure / risk_scale_max``.

        This is a linear reading of the risk scale. It is an assumption, it is
        stated here rather than buried, and it is the single place to change if
        a future calibration establishes a non-linear loss curve.
        """
        if risk_scale_max <= 0:
            raise ValueError("risk_scale_max must be positive")
        return cls(
            value_per_risk_point=annual_disruption_exposure / risk_scale_max,
            **kwargs,  # type: ignore[arg-type]
        )

    @property
    def annuity_factor(self) -> float:
        """Present value of one currency unit received annually for T years."""
        d = self.discount_rate
        t = self.benefit_horizon_years
        if d == 0.0:
            return float(t)
        return float((1.0 - (1.0 + d) ** -t) / d)

    @property
    def is_sourced(self) -> bool:
        """Whether anyone has said where these prices came from.

        Every currency figure the engine reports is proportional to
        ``value_per_risk_point``. If that number is unsourced then so is the
        headline, and a reader is entitled to know that before quoting it. This is
        deliberately a property of the price book rather than a UI concern: the
        fact travels with the result into the audit ledger and any export.
        """
        return self.source.strip().lower() not in ("", "unstated")


@dataclass(frozen=True, slots=True)
class MonetaryNPVObjective:
    """Maximise portfolio net present value, denominated in currency."""

    prices: PriceBook
    strategic_risk_multiplier: float = 1.0
    strategic_lead_time_multiplier: float = 1.0
    version: str = "objective/monetary-npv@1.0.0"

    def __post_init__(self) -> None:
        if self.strategic_risk_multiplier < 0:
            raise ValueError("strategic_risk_multiplier must be non-negative")
        if self.strategic_lead_time_multiplier < 0:
            raise ValueError("strategic_lead_time_multiplier must be non-negative")

    @property
    def sense(self) -> int:
        return pl.LpMaximize

    @property
    def unit(self) -> str:
        return "currency (net present value)"

    # -- coefficient helpers, also used for reporting ----------------------- #

    @property
    def risk_point_npv(self) -> float:
        """Present value of removing one percentage point of risk."""
        return (
            self.prices.annuity_factor
            * self.strategic_risk_multiplier
            * self.prices.value_per_risk_point
        )

    @property
    def lead_time_day_npv(self) -> float:
        """Present value of removing one day of lead time."""
        return (
            self.prices.annuity_factor
            * self.strategic_lead_time_multiplier
            * self.prices.value_per_lead_time_day
        )

    @property
    def carbon_ton_npv(self) -> float:
        """Present value cost of one tonne of carbon emitted."""
        return self.prices.annuity_factor * self.prices.price_per_carbon_ton

    @property
    def provenance(self) -> dict[str, object]:
        """Readable pricing provenance for the audit ledger.

        The price book already enters ``objective_parameters_hash``, so a changed
        price is already detectable. This makes it *legible*: a hash proves two
        results used the same price of risk, and proves nothing about whether that
        price was ever defensible. The acquired system's failure was not that its
        numbers were unhashed, it was that nobody could see where they came from.
        """
        return {
            "value_per_risk_point_per_year": self.prices.value_per_risk_point,
            "value_per_lead_time_day_per_year": self.prices.value_per_lead_time_day,
            "price_per_carbon_ton_per_year": self.prices.price_per_carbon_ton,
            "benefit_horizon_years": self.prices.benefit_horizon_years,
            "discount_rate": self.prices.discount_rate,
            "annuity_factor": self.prices.annuity_factor,
            "risk_point_npv": self.risk_point_npv,
            "strategic_risk_multiplier": self.strategic_risk_multiplier,
            "strategic_lead_time_multiplier": self.strategic_lead_time_multiplier,
            "price_source": self.prices.source,
            "prices_are_sourced": self.prices.is_sourced,
        }

    def build(self, terms: ProblemTerms) -> pl.LpAffineExpression:
        benefit = pl.lpSum(
            self.risk_point_npv * node.risk_points
            + self.lead_time_day_npv * node.lead_time_days
            - self.carbon_ton_npv * node.carbon_tons
            for node in terms.nodes.values()
        )
        return benefit - terms.total_capital + terms.total_discounts


# --------------------------------------------------------------------------- #
# Target-risk mode
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MinimizeCapitalObjective:
    """Minimise net capital outlay. Pairs with a required-risk-reduction floor.

    This reproduces the acquired system's target-risk mode, which was already
    unit-coherent: it minimises currency subject to a risk constraint, so it
    never had to trade points against days.
    """

    version: str = "objective/minimize-capital@1.0.0"

    @property
    def sense(self) -> int:
        return pl.LpMinimize

    @property
    def unit(self) -> str:
        return "currency (net capital outlay)"

    def build(self, terms: ProblemTerms) -> pl.LpAffineExpression:
        return terms.net_capital


@dataclass(frozen=True, slots=True)
class MaximizeRiskReductionObjective:
    """Maximise total in-model risk reduction, ignoring cost entirely.

    This is not a decision objective and must never be offered to a user as one:
    it will always spend every available currency unit, because capital does not
    appear in it. It exists to answer a different and narrower question — "what is
    the most this portfolio could ever deliver?" — which is what turns a bare
    ``infeasible`` into an attainable-frontier statement a CFO can act on.

    Because the engine linearises a concave response with an inner (chord)
    approximation, the ceiling this reports is very slightly conservative: it may
    understate the true maximum by up to one facet width, never overstate it. That
    is the correct direction of error for a number used to tell someone what they
    cannot reach.
    """

    version: str = "objective/maximize-risk-reduction@1.0.0"

    @property
    def sense(self) -> int:
        return pl.LpMaximize

    @property
    def unit(self) -> str:
        return "risk percentage points (diagnostic ceiling, not a decision)"

    def build(self, terms: ProblemTerms) -> pl.LpAffineExpression:
        return pl.lpSum(node.risk_points for node in terms.nodes.values())
