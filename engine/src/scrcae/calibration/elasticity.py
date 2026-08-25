"""Estimating a node's risk elasticity from its own delivery history.

Why this exists
--------------
``NodeExposure.elasticity`` says how much more a node's intervention is worth when
its market runs above anchor. Shipped as a number a user types, it is the same
class of object as the 0.85 response exponent: an assumption wearing a decimal
point. This module replaces it with an estimate from the company's own delivery
records, and — just as importantly — refuses to produce one when the records
cannot support it.

The model
--------
For each period *t* with market price :math:`p_t` and realised disruption rate
:math:`y_t`, define the one-sided deviation from anchor

.. math:: x_t = \\max\\left(\\frac{p_t - a}{a},\\ 0\\right)

and fit

.. math:: y_t = f_0\\,(1 + e\\,x_t) = f_0 + (f_0 e)\\,x_t.

That is ordinary least squares of ``y`` on ``x`` with an intercept: the intercept
estimates the baseline disruption rate :math:`f_0` when the market sits at or below
anchor, the slope estimates :math:`f_0 e`, and the elasticity is their ratio. Two
consequences are worth stating because they are checkable by hand: the fit is
linear despite the parameter of interest being a ratio, and the elasticity is
scale-free in ``y`` — measuring disruption in percent or in fraction gives the same
answer.

The one-sided deviation matches the runtime policy in ``app/services/market.py``.
It is not a statistical choice. It is the same asymmetry stated there: nobody has
established that cheap inputs make a supply chain safer, so periods below anchor
inform :math:`f_0` and are not allowed to pull the slope.

Three things this estimate is not
--------------------------------
**It is not causal.** Delivery failures and commodity prices share causes. A
hurricane raises crude and closes a port in the same week, and this fit will
attribute the closure to the price. The estimate is an association between market
stress and realised disruption, which is genuinely what the multiplier is used for
— but it must not be described as the effect of price on reliability.

**It is not the elasticity of intervention effectiveness.** What delivery records
observe is the risk a node *carries*, not how much of that risk a given
intervention *removes*. The engine's multiplier scales the latter. Treating this
estimate as the latter assumes an intervention averts a constant fraction of node
risk rather than a fixed number of points. That assumption is recorded in every
fit's ``notes`` so it travels into the UI and the audit ledger rather than living
in a docstring nobody reads.

**It is not precise.** Delivery periods are serially correlated: a bad quarter
follows a bad quarter. Textbook OLS standard errors assume independent draws and
are therefore too narrow here, sometimes by a lot. The interval reported is a
moving-block bootstrap, which resamples contiguous blocks of periods and so keeps
short-run dependence intact. It is still an approximation, and with the sample
sizes a single company can supply it will usually be wide. A wide interval is the
finding, not a defect in the reporting of it.

How good the interval actually is
---------------------------------
Measured rather than asserted, over four hundred simulated histories with a known
elasticity, counting how often the reported interval contains it. At sixty monthly
periods the nominal 95% interval covers the truth about 91% of the time when prices
are drawn independently, and about 88% of the time when prices follow an AR(1) process
with persistence 0.85 — which is what a commodity actually does. Raising the bootstrap
draw count fourfold moves those figures by under a point, so the gap is the resampling
scheme rather than Monte Carlo error in it. The estimator itself is essentially
unbiased in both settings; it is the interval that is a little optimistic, and it gets
more optimistic as the market becomes more persistent.

Read the label accordingly: what is printed as a 95% interval behaves like a 90%
interval, give or take. That is close enough to be useful and not close enough to
round off in conversation, which is why it is written down here and pinned by a test
instead of living in somebody's memory.

Longer resampling blocks were tried as a fix and made coverage slightly worse, so the
:math:`n^{1/3}` rule stands. The shortfall is a property of estimating a ratio from
sixty correlated observations, not a bug with a patch. Two consequences are therefore
baked into the design rather than left to the reader: the interval is reported
everywhere the elasticity is, and a fit whose interval merely clears zero is treated
as the weakest kind of usable rather than as an established fact.

One more thing the simulations show, and it matters more than the coverage gap: with
realistically persistent prices the spread of the point estimate itself widens by about
threefold against independent prices at the same sample size. Sixty months of history
is not a lot of information about a market that moves slowly. Two teams calibrating the
same node on overlapping decades can legitimately land some way apart, and neither is
wrong.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

__all__ = [
    "CALIBRATION_VERSION",
    "CONFIDENCE_LEVEL",
    "MIN_OBSERVATIONS",
    "MIN_PERIODS_ABOVE_ANCHOR",
    "MIN_DISTINCT_DEVIATIONS",
    "DeliveryObservation",
    "ElasticityFit",
    "FitQuality",
    "calibrate_elasticity",
]

#: Bumped whenever the estimator changes, so a stored elasticity can be traced to the
#: method that produced it. Recorded in every fit and carried into the audit ledger.
CALIBRATION_VERSION = "elasticity-ols-blockbootstrap-1"

#: Two-sided coverage of the reported interval.
#:
#: 95% rather than 90% because of what a false positive costs here. The interval is
#: not decoration: excluding zero is what promotes an elasticity from "asserted" to
#: "calibrated", after which it multiplies a risk reduction that is priced in dollars
#: and put in front of a CFO. Measured against pure noise, this estimator flags about
#: 8.5% of histories as usable at 90% and about 5.5% at 95% — well calibrated, but one
#: in twelve is too often to launder a coincidence into a money figure. The remaining
#: one in eighteen is why a calibrated elasticity is still labelled with its interval,
#: its R² and its period count everywhere it is shown.
CONFIDENCE_LEVEL = 0.95

#: Below this, an interval is theatre. Eight is not a statistical threshold — no
#: threshold would be — but it is the point below which a block bootstrap has fewer
#: distinct resamples than draws, and the interval starts describing the resampling
#: rather than the data.
MIN_OBSERVATIONS = 8

#: The slope is identified only by periods above anchor. With none, ``x`` is all
#: zeros and the elasticity is not estimable at any sample size; with one or two, it
#: is a line through the leverage of a single episode.
MIN_PERIODS_ABOVE_ANCHOR = 3

#: Distinct values of ``x``, so the slope is not carried by repeated readings of one
#: price level.
MIN_DISTINCT_DEVIATIONS = 3

#: Bootstrap resamples. Fixed rather than configurable at the call site's whim: the
#: number is part of the method, and varying it silently varies the interval.
BOOTSTRAP_DRAWS = 2_000


class FitQuality(Enum):
    """Whether the fit may be used, and if not, precisely what is wrong.

    A single boolean would collapse "you have four data points" and "your data say
    the effect is indistinguishable from nothing" into the same message, and those
    call for opposite actions: collect more, versus stop assuming an effect.
    """

    USABLE = "usable"
    TOO_FEW_OBSERVATIONS = "too-few-observations"
    TOO_FEW_PERIODS_ABOVE_ANCHOR = "too-few-periods-above-anchor"
    NO_DEVIATION_VARIATION = "no-deviation-variation"
    NON_POSITIVE_BASELINE = "non-positive-baseline"
    WRONG_SIGN = "wrong-sign"
    NOT_DISTINGUISHABLE_FROM_ZERO = "not-distinguishable-from-zero"


@dataclass(frozen=True, slots=True)
class DeliveryObservation:
    """One period of one node's realised delivery performance, and the market then.

    ``disruption_rate`` is whatever share-of-shipments-gone-wrong measure the
    business already keeps — late, short, failed, diverted. The estimator is
    scale-free in it, so percent and fraction give the same elasticity; what matters
    is that the same definition is used in every period.
    """

    period: str
    market_price: float
    disruption_rate: float

    def __post_init__(self) -> None:
        for name, value in (
            ("market_price", self.market_price),
            ("disruption_rate", self.disruption_rate),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.market_price <= 0.0:
            raise ValueError(
                f"market_price must be positive, got {self.market_price!r}"
            )
        if self.disruption_rate < 0.0:
            raise ValueError(
                f"disruption_rate cannot be negative, got {self.disruption_rate!r}"
            )


@dataclass(frozen=True, slots=True)
class ElasticityFit:
    """An elasticity estimate, its interval, and the reason it may not be trusted."""

    quality: FitQuality
    anchor: float
    observations: int
    periods_above_anchor: int
    elasticity: float | None = None
    baseline_disruption_rate: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    r_squared: float | None = None
    method: str = CALIBRATION_VERSION
    confidence_level: float = CONFIDENCE_LEVEL
    one_sided: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_usable(self) -> bool:
        return self.quality is FitQuality.USABLE

    @property
    def interval_spans_zero(self) -> bool:
        if self.interval_low is None or self.interval_high is None:
            return True
        return self.interval_low <= 0.0 <= self.interval_high

    def summary(self) -> str:
        """One line a non-statistician can act on."""
        if self.elasticity is None:
            return _REFUSALS[self.quality].format(
                n=self.observations,
                above=self.periods_above_anchor,
                min_n=MIN_OBSERVATIONS,
                min_above=MIN_PERIODS_ABOVE_ANCHOR,
            )
        interval = ""
        if self.interval_low is not None and self.interval_high is not None:
            interval = (
                f" ({self.confidence_level:.0%} interval "
                f"{self.interval_low:.2f} to {self.interval_high:.2f})"
            )
        headline = f"Elasticity {self.elasticity:.2f}{interval}"
        if self.r_squared is not None:
            headline += f", R\u00b2 {self.r_squared:.2f}"
        headline += (
            f", from {self.observations} periods of which "
            f"{self.periods_above_anchor} above anchor."
        )
        if self.quality is FitQuality.NOT_DISTINGUISHABLE_FROM_ZERO:
            headline += (
                " The interval includes zero, so these records do not establish that "
                "this market moves this node's risk at all."
            )
        elif self.quality is FitQuality.WRONG_SIGN:
            headline += (
                " The estimate is negative: in these records, a market above anchor "
                "went with fewer disruptions. That is a hedge claim and needs a "
                "mechanism before it is used."
            )
        return headline

    def provenance(self) -> dict[str, object]:
        """Flat, JSON-safe, and hashable by ``audit.canonical_json``."""
        return {
            "method": self.method,
            "quality": self.quality.value,
            "elasticity": self.elasticity,
            "baseline_disruption_rate": self.baseline_disruption_rate,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "confidence_level": self.confidence_level,
            "r_squared": self.r_squared,
            "observations": self.observations,
            "periods_above_anchor": self.periods_above_anchor,
            "anchor": self.anchor,
            "one_sided": self.one_sided,
        }


_REFUSALS: dict[FitQuality, str] = {
    FitQuality.TOO_FEW_OBSERVATIONS: (
        "Not enough history to estimate an elasticity: {n} periods, and at least "
        "{min_n} are needed before an interval means anything."
    ),
    FitQuality.TOO_FEW_PERIODS_ABOVE_ANCHOR: (
        "Only {above} of {n} periods had the market above its anchor. The elasticity "
        "is identified entirely by those periods, and at least {min_above} are needed."
    ),
    FitQuality.NO_DEVIATION_VARIATION: (
        "The market barely moved above its anchor across these periods, so the slope "
        "would rest on a single price level rather than on a relationship."
    ),
    FitQuality.NON_POSITIVE_BASELINE: (
        "The estimated baseline disruption rate is zero or negative, so the "
        "elasticity — a ratio to that baseline — is not defined. This usually means "
        "the disruption column is empty or constant."
    ),
}


def _deviation(price: float, anchor: float, *, one_sided: bool) -> float:
    raw = (price - anchor) / anchor
    return max(raw, 0.0) if one_sided else raw


def _ols(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float] | None:
    """Intercept and slope, or None when x has no variance.

    Written out rather than delegated so the arithmetic is visible: this is the one
    place a wrong sign convention would silently invert every elasticity in the app.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(ys) / n
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def _r_squared(xs: Sequence[float], ys: Sequence[float], intercept: float, slope: float) -> float | None:
    mean_y = math.fsum(ys) / len(ys)
    sst = math.fsum((y - mean_y) ** 2 for y in ys)
    if sst <= 0.0:
        return None
    sse = math.fsum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sse / sst


def _block_length(n: int) -> int:
    """Standard :math:`n^{1/3}` rule, floored at 1 and capped so blocks are resampled.

    The point of blocks is to carry serial correlation into the resample. A block as
    long as the sample would resample the sample.
    """
    return max(1, min(n // 2, round(n ** (1.0 / 3.0))))


def _bootstrap_interval(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float | None, float | None]:
    """Moving-block bootstrap percentile interval for the elasticity.

    Uses its own ``random.Random``. The engine never touches the global RNG — an
    earlier version of this system seeded ``random`` and ``np.random`` globally, so
    running a simulation changed the results of anything else that drew a number
    (F10).
    """
    n = len(xs)
    block = _block_length(n)
    rng = random.Random(seed)
    estimates: list[float] = []
    starts = max(1, n - block + 1)

    for _ in range(draws):
        idx: list[int] = []
        while len(idx) < n:
            start = rng.randrange(starts)
            idx.extend(range(start, min(start + block, n)))
        idx = idx[:n]
        rx = [xs[i] for i in idx]
        ry = [ys[i] for i in idx]
        fit = _ols(rx, ry)
        if fit is None:
            continue
        intercept, slope = fit
        if intercept <= 0.0:
            # A resample implying a non-positive baseline cannot produce a ratio.
            # Dropping it rather than substituting a value keeps the interval an
            # interval over estimable resamples, and the count is reported below.
            continue
        estimates.append(slope / intercept)

    if len(estimates) < draws // 4:
        # Too many resamples were inestimable for a percentile to describe anything.
        return None, None

    estimates.sort()
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    high = estimates[min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))]
    return low, high


_STANDING_NOTES = (
    "This is an association between market level and realised disruption, not the "
    "causal effect of price on reliability: a shock that moves the market and breaks "
    "the network in the same period is attributed here to the market.",
    "Delivery records measure the risk this node carries, not the share of it a given "
    "intervention removes. Using this figure as an effectiveness multiplier assumes an "
    "intervention averts a constant fraction of node risk rather than a fixed number "
    "of risk points.",
    "The interval is a moving-block bootstrap, which allows for one bad period "
    "following another. It is wider than a textbook standard error and that is the "
    "point.",
)


def calibrate_elasticity(
    observations: Sequence[DeliveryObservation],
    *,
    anchor: float,
    one_sided: bool = True,
    seed: int = 20260825,
    draws: int = BOOTSTRAP_DRAWS,
) -> ElasticityFit:
    """Fit one node's risk elasticity to one market, or explain why it cannot be fit.

    Always returns a fit. It never raises on thin or awkward data, because the
    caller's job is to show the user what is wrong with their records, and an
    exception would push that decision into a ``try`` block in the view layer.

    ``seed`` makes the interval reproducible: the same records give the same
    interval, which is a precondition for the audit ledger's claim that a run can be
    reproduced from its inputs.
    """
    if anchor <= 0.0 or not math.isfinite(anchor):
        raise ValueError(f"anchor must be a positive, finite price, got {anchor!r}")

    rows = tuple(observations)
    n = len(rows)
    xs = [_deviation(o.market_price, anchor, one_sided=one_sided) for o in rows]
    ys = [float(o.disruption_rate) for o in rows]
    above = sum(1 for x in xs if x > 0.0)

    def refuse(quality: FitQuality) -> ElasticityFit:
        return ElasticityFit(
            quality=quality,
            anchor=float(anchor),
            observations=n,
            periods_above_anchor=above,
            one_sided=one_sided,
            notes=_STANDING_NOTES,
        )

    if n < MIN_OBSERVATIONS:
        return refuse(FitQuality.TOO_FEW_OBSERVATIONS)
    if above < MIN_PERIODS_ABOVE_ANCHOR:
        return refuse(FitQuality.TOO_FEW_PERIODS_ABOVE_ANCHOR)
    # Rounded before counting, so readings that differ in the twelfth decimal are not
    # mistaken for genuine variation in the market.
    if len({round(x, 9) for x in xs}) < MIN_DISTINCT_DEVIATIONS:
        return refuse(FitQuality.NO_DEVIATION_VARIATION)

    fit = _ols(xs, ys)
    if fit is None:
        return refuse(FitQuality.NO_DEVIATION_VARIATION)
    intercept, slope = fit
    if intercept <= 0.0:
        return refuse(FitQuality.NON_POSITIVE_BASELINE)

    elasticity = slope / intercept
    low, high = _bootstrap_interval(xs, ys, seed=seed, draws=draws)
    r2 = _r_squared(xs, ys, intercept, slope)

    if elasticity < 0.0:
        quality = FitQuality.WRONG_SIGN
    elif low is None or high is None or low <= 0.0 <= high:
        quality = FitQuality.NOT_DISTINGUISHABLE_FROM_ZERO
    else:
        quality = FitQuality.USABLE

    notes = list(_STANDING_NOTES)
    if r2 is not None and r2 < 0.2:
        notes.append(
            f"The market explains only {r2:.0%} of the variation in disruption across "
            f"these periods. Most of what happens to this node is not this market."
        )
    if above < n // 4:
        notes.append(
            f"Only {above} of {n} periods sat above anchor, so the estimate leans on a "
            f"small part of the record."
        )

    return ElasticityFit(
        quality=quality,
        anchor=float(anchor),
        observations=n,
        periods_above_anchor=above,
        elasticity=elasticity,
        baseline_disruption_rate=intercept,
        interval_low=low,
        interval_high=high,
        r_squared=r2,
        one_sided=one_sided,
        notes=tuple(notes),
    )
