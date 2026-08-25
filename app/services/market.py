"""Per-market quotes, and turning them into per-node risk multipliers (F11).

What the acquired system did
---------------------------
A `MarketFeedbackBridge` walked an eleven-row market table and, for each row,
built a mask over the node table with ``nodes['Node Name'] == sector``. Three
things followed from that line:

* Sector names and user-typed node names essentially never match, so the mapping
  silently did nothing in the normal case.
* When the ``Node Name`` column was absent the mask fell back to
  ``[True] * len(nodes)`` — one market's volatility applied to the entire network.
* The multiplier was ``1 + volatility * 0.15 - min(daily_change, 0) * 0.5``, which
  *increases* an intervention's risk reduction whether the market rises or falls,
  because the second term subtracts a negative. Risk reduction could only ratchet
  upwards.

And the result was written straight back into the user's own input table
(``st.session_state.nodes_df = bridge.apply_market_feedback(...)``), so each
30-minute sync compounded on the last one and the originally entered figures were
unrecoverable.

The market table itself initialised every row to ``Live Price ($) 100.00`` with a
``0.00%`` change, and a failed fetch set the status to "Holding Last Known" — so a
feed that had never once succeeded displayed eleven plausible prices and a status
implying they were recent.

What this module does
--------------------
Quotes are fetched per symbol and each carries its own status. There is no
initial price, so an unavailable market has no number rather than a placeholder
one. Exposure comes from an explicit table (``adapters.build_node_exposures``),
and the multipliers this module computes are handed to the engine as
``OptimizationRequest.node_macro_multipliers`` — a solve-time overlay. The user's
inputs are never touched, so nothing compounds and the original figures always
remain the source of record.

Direction
--------
Deviations are read one-sided by default: a market above its anchor raises the
exposed node's multiplier, one below it does nothing. This matches the asymmetry
already documented in ``services.macro`` and is retained for the same reason —
that cheap inputs make a supply chain *safer* is not a claim anybody established.
It is a stated assumption with a switch, not a fact.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from app.adapters import NodeExposure
from app.services.feeds import PriceHistory, QuoteProvider, resolve_provider

__all__ = [
    "MIN_MULTIPLIER",
    "MarketQuote",
    "QuoteSet",
    "ExposureEffect",
    "ExposureOverlay",
    "fetch_history",
    "fetch_quotes",
    "quotes_from_prices",
    "build_overlay",
]

#: Floor on a node multiplier. The engine rejects non-positive multipliers, and a
#: multiplier at or below zero would silently delete an intervention's entire
#: benefit while leaving its full cost in the portfolio.
MIN_MULTIPLIER = 0.01

#: How long a quote stays usable before it is refetched.
CACHE_TTL_SECONDS = 1800.0


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """One market's price, or an explicit statement that we do not have one.

    ``price`` is ``None`` when the quote is unavailable. There is deliberately no
    default price: the acquired system's 100.00 placeholder was indistinguishable
    on screen from a real reading, and the whole failure mode of that feed was
    that it looked like it was working.
    """

    symbol: str
    price: float | None
    status: str
    is_live: bool
    fetched_at: float | None = None

    @property
    def is_usable(self) -> bool:
        return self.is_live and self.price is not None and self.price > 0.0


@dataclass(frozen=True, slots=True)
class QuoteSet:
    """Quotes keyed by symbol, with the failures kept rather than dropped."""

    quotes: Mapping[str, MarketQuote] = field(default_factory=dict)

    def get(self, symbol: str) -> MarketQuote | None:
        return self.quotes.get(symbol.upper())

    @property
    def usable(self) -> tuple[str, ...]:
        return tuple(sorted(s for s, q in self.quotes.items() if q.is_usable))

    @property
    def unavailable(self) -> tuple[str, ...]:
        return tuple(sorted(s for s, q in self.quotes.items() if not q.is_usable))

    @property
    def any_live(self) -> bool:
        return bool(self.usable)


@dataclass(frozen=True, slots=True)
class ExposureEffect:
    """What one exposure row did, or why it did nothing."""

    node_id: str
    symbol: str
    anchor: float
    elasticity: float
    price: float | None
    deviation: float | None
    contribution: float
    applied: bool
    note: str

    @property
    def multiplier_effect(self) -> float:
        return 1.0 + self.contribution


@dataclass(frozen=True, slots=True)
class ExposureOverlay:
    """Per-node multipliers plus a full account of how each was reached.

    ``multipliers`` contains only nodes whose multiplier differs from 1.0, so an
    overlay with nothing to say hands the engine an empty mapping and is provably
    a no-op rather than a mapping of ones that merely behaves like one.
    """

    multipliers: Mapping[str, float] = field(default_factory=dict)
    effects: tuple[ExposureEffect, ...] = field(default_factory=tuple)
    clamped: tuple[str, ...] = field(default_factory=tuple)
    one_sided: bool = True

    def __bool__(self) -> bool:
        return bool(self.multipliers)

    @property
    def applied_effects(self) -> tuple[ExposureEffect, ...]:
        return tuple(e for e in self.effects if e.applied)

    @property
    def skipped_effects(self) -> tuple[ExposureEffect, ...]:
        """Exposures the user asserted that could not be applied.

        Surfaced as prominently as the applied ones. A mapping the user configured
        and believes is active, silently inert because a feed is down, is the exact
        condition the acquired system shipped in permanently.
        """
        return tuple(e for e in self.effects if not e.applied)

    def summary(self) -> str:
        if not self.effects:
            return "No market exposure configured."
        applied = len(self.applied_effects)
        skipped = len(self.skipped_effects)
        if not applied:
            return (
                f"No market adjustment applied. {skipped} configured "
                f"exposure(s) had no usable quote."
            )
        nodes = len(self.multipliers)
        text = (
            f"{applied} exposure(s) applied across {nodes} node(s); "
            f"risk parameters adjusted at solve time."
        )
        if skipped:
            text += f" {skipped} exposure(s) had no usable quote and did nothing."
        if self.clamped:
            text += f" Floored at {MIN_MULTIPLIER:g}: {', '.join(self.clamped)}."
        return text


def quotes_from_prices(prices: Mapping[str, float | None]) -> QuoteSet:
    """Build a quote set from prices already in hand.

    The seam that makes this module testable without a network, and the reason the
    fetching and the arithmetic are separate functions.
    """
    quotes: dict[str, MarketQuote] = {}
    for raw_symbol, price in prices.items():
        symbol = str(raw_symbol).upper()
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            quotes[symbol] = MarketQuote(
                symbol=symbol,
                price=None,
                status="No usable price",
                is_live=False,
            )
        else:
            quotes[symbol] = MarketQuote(
                symbol=symbol,
                price=float(price),
                status="Live",
                is_live=True,
                fetched_at=time.time(),
            )
    return QuoteSet(quotes=quotes)


def fetch_quotes(
    symbols: Iterable[str],
    *,
    provider: QuoteProvider | None = None,
    timeout: float = 4.0,
) -> QuoteSet:
    """Fetch one quote per symbol, reporting each failure individually.

    Never raises and never invents a price. A per-symbol status matters because
    partial availability is the normal case: with one global status, three working
    markets and one broken one either read as "live" — hiding the gap — or as
    "offline", discarding three usable readings.

    The provider is injectable and defaults to the configured one. This function's
    remaining job is translation: a provider says what it saw, and this decides
    whether that counts as a usable quote. Keeping the judgement here means a new
    provider cannot accidentally widen what the app treats as a live price.
    """
    wanted = tuple(dict.fromkeys(str(s).upper() for s in symbols if str(s).strip()))
    if not wanted:
        return QuoteSet()

    source = provider or resolve_provider()
    readings = source.latest(wanted, timeout=timeout)
    stamped = time.time()

    quotes: dict[str, MarketQuote] = {}
    for reading in readings:
        symbol = str(reading.symbol).upper()
        price = reading.price
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            quotes[symbol] = MarketQuote(
                symbol=symbol,
                price=None,
                status=f"{reading.status} [{source.name}]",
                is_live=False,
            )
            continue
        quotes[symbol] = MarketQuote(
            symbol=symbol,
            price=float(price),
            status=f"{reading.status} [{source.name}]",
            is_live=True,
            fetched_at=stamped,
        )

    # A provider that returns fewer readings than it was asked for must not leave
    # symbols simply absent: a missing key reads downstream as "not configured"
    # rather than "asked for and not answered".
    for symbol in wanted:
        if symbol not in quotes:
            quotes[symbol] = MarketQuote(
                symbol=symbol,
                price=None,
                status=f"Feed returned no reading for this symbol [{source.name}]",
                is_live=False,
            )
    return QuoteSet(quotes=quotes)


def fetch_history(
    symbol: str,
    *,
    provider: QuoteProvider | None = None,
    months: int = 60,
    timeout: float = 8.0,
) -> PriceHistory:
    """Monthly closes for one symbol, for calibrating that market's elasticity.

    Separate from ``fetch_quotes`` because the two have different failure meanings: a
    missing quote makes today's overlay inert, while a missing history makes the
    elasticity uncalibratable. Collapsing them would let a provider with spot prices
    and no history look like a total feed failure.
    """
    source = provider or resolve_provider()
    if not str(symbol).strip():
        return PriceHistory(symbol=str(symbol), status="No symbol given")
    return source.history(str(symbol).upper(), months=months, timeout=timeout)


def build_overlay(
    exposures: Sequence[NodeExposure],
    quotes: QuoteSet,
    *,
    one_sided: bool = True,
) -> ExposureOverlay:
    """Combine asserted exposures with available quotes into per-node multipliers.

    Contributions for a node with several exposures are summed. Summation is the
    assumption that exposures are independent and additive at the margin; it is
    stated here because the alternative (compounding them multiplicatively) gives
    materially different answers once a node carries three or four exposures, and
    neither is calibrated.
    """
    contributions: dict[str, float] = {}
    effects: list[ExposureEffect] = []

    for exposure in exposures:
        quote = quotes.get(exposure.symbol)
        if quote is None:
            effects.append(
                _inert(exposure, "no quote was requested for this market")
            )
            continue
        if not quote.is_usable:
            effects.append(_inert(exposure, quote.status))
            continue

        price = float(quote.price)  # is_usable guarantees a positive float
        deviation = (price - exposure.anchor) / exposure.anchor
        effective = max(deviation, 0.0) if one_sided else deviation
        contribution = exposure.elasticity * effective

        if contribution == 0.0:
            note = (
                f"{exposure.symbol} at {price:,.2f} is at or below its "
                f"{exposure.anchor:,.2f} anchor"
                if one_sided and deviation <= 0.0
                else f"{exposure.symbol} is at its anchor"
            )
            effects.append(
                ExposureEffect(
                    node_id=exposure.node_id,
                    symbol=exposure.symbol,
                    anchor=exposure.anchor,
                    elasticity=exposure.elasticity,
                    price=price,
                    deviation=deviation,
                    contribution=0.0,
                    applied=True,
                    note=note,
                )
            )
            contributions.setdefault(exposure.node_id, 0.0)
            continue

        contributions[exposure.node_id] = (
            contributions.get(exposure.node_id, 0.0) + contribution
        )
        effects.append(
            ExposureEffect(
                node_id=exposure.node_id,
                symbol=exposure.symbol,
                anchor=exposure.anchor,
                elasticity=exposure.elasticity,
                price=price,
                deviation=deviation,
                contribution=contribution,
                applied=True,
                note=(
                    f"{exposure.symbol} at {price:,.2f} is {deviation:+.1%} vs "
                    f"anchor {exposure.anchor:,.2f}; elasticity "
                    f"{exposure.elasticity:g} gives {contribution:+.4f}"
                ),
            )
        )

    multipliers: dict[str, float] = {}
    clamped: list[str] = []
    for node_id, total in contributions.items():
        multiplier = 1.0 + total
        if multiplier < MIN_MULTIPLIER:
            multiplier = MIN_MULTIPLIER
            clamped.append(node_id)
        # Exact ones are dropped rather than passed through, so an overlay that
        # changes nothing is an empty mapping the engine can be proven to ignore.
        if multiplier != 1.0:
            multipliers[node_id] = multiplier

    return ExposureOverlay(
        multipliers=multipliers,
        effects=tuple(effects),
        clamped=tuple(sorted(clamped)),
        one_sided=one_sided,
    )


def _inert(exposure: NodeExposure, why: str) -> ExposureEffect:
    return ExposureEffect(
        node_id=exposure.node_id,
        symbol=exposure.symbol,
        anchor=exposure.anchor,
        elasticity=exposure.elasticity,
        price=None,
        deviation=None,
        contribution=0.0,
        applied=False,
        note=why,
    )
