"""Risk response models — how funding scale converts into risk reduction.

Forensic background
-------------------
The acquired system computed, once per node, before optimization::

    marginal_risks[n] = (live_macro_multiplier * risks[n]) ** 0.85

and then used ``marginal_risks[n] * x[n]`` linearly inside the optimizer.

That has two separable problems, and the audit conflated them:

1. **The 0.85 exponent is uncalibrated.** It is a model assumption with no
   stated empirical basis.
2. **The exponent is applied to the wrong argument.** Applying it to the risk
   *parameter* and then scaling linearly by ``x`` produces a model that is
   perfectly *linear* in funding. Funding a node at 50% yields exactly half the
   risk reduction of funding it at 100%. There are no diminishing returns to
   funding at all — only a constant rescaling of each node's coefficient.

Problem 2 is the more serious finding, because the exponent's stated purpose in
the acquired system's own equation ledger was to create diminishing returns.
It does not. It creates a *relative reweighting between nodes* (compressing
large risk reducers relative to small ones) which is a different economic claim
entirely, and one nobody has articulated or defended.

This module separates the two ideas so each can be chosen and tested
independently:

``LinearResponse``
    ``f(x) = beta * R * x``. No transform. The honest baseline.

``ParameterPowerResponse``
    ``f(x) = (beta * R) ** alpha * x``. Bit-exact reproduction of the acquired
    behaviour, retained solely so the extraction can be proven faithful and so
    legacy results remain reproducible. Not recommended for new work.

``AllocationConcaveResponse``
    ``f(x) = (beta * R) * x ** alpha``. Genuinely concave in funding, which is
    what "diminishing utility" was supposed to mean. Non-linear, so it is
    exposed to the mixed-integer optimizer as a set of tangent constraints
    forming an outer concave envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

__all__ = [
    "RiskResponseModel",
    "LinearResponse",
    "ParameterPowerResponse",
    "AllocationConcaveResponse",
    "Tangent",
    "check_scale",
]


def check_scale(scale: float) -> float:
    """Reject funding scales outside the domain the response models define.

    A negative scale is not a small numerical nuisance here: with a linear
    response it silently returns *negative risk reduction*, i.e. an intervention
    that increases the risk it was funded to reduce. The acquired system had no
    such guard anywhere. Failing loudly is the point.
    """
    if scale < 0.0:
        raise ValueError(f"funding scale must be non-negative, got {scale}")
    if not math.isfinite(scale):
        raise ValueError(f"funding scale must be finite, got {scale}")
    return scale


@dataclass(frozen=True, slots=True)
class Tangent:
    """A linear upper bound ``value <= slope * x + intercept``."""

    slope: float
    intercept: float

    def at(self, x: float) -> float:
        return self.slope * x + self.intercept


@runtime_checkable
class RiskResponseModel(Protocol):
    """Maps a funding scale in [0, 1] to risk reduction in percentage points."""

    #: Recorded in the audit ledger so a result can be reproduced.
    version: str

    @property
    def is_linear(self) -> bool:
        """True when ``evaluate`` is exactly ``linear_coefficient * scale``."""
        ...

    def evaluate(self, risk_reduction_pts: float, scale: float, macro: float = 1.0) -> float:
        """Ground-truth risk reduction. Used by simulation and verification."""
        ...

    def linear_coefficient(self, risk_reduction_pts: float, macro: float = 1.0) -> float:
        """Coefficient for the linear form. Only meaningful when ``is_linear``."""
        ...

    def tangents(
        self,
        risk_reduction_pts: float,
        macro: float = 1.0,
        *,
        breakpoints: Sequence[float] = (),
    ) -> tuple[Tangent, ...]:
        """Linear outer approximation of ``evaluate`` over scale in [0, 1]."""
        ...


DEFAULT_BREAKPOINTS: tuple[float, ...] = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)


@dataclass(frozen=True, slots=True)
class LinearResponse:
    """``f(x) = macro * R * x``."""

    calibration_source: str = "not-applicable-no-free-parameters"
    version: str = "risk-response/linear@1.0.0"

    @property
    def is_linear(self) -> bool:
        return True

    def evaluate(self, risk_reduction_pts: float, scale: float, macro: float = 1.0) -> float:
        return macro * risk_reduction_pts * check_scale(scale)

    def linear_coefficient(self, risk_reduction_pts: float, macro: float = 1.0) -> float:
        return macro * risk_reduction_pts

    def tangents(
        self,
        risk_reduction_pts: float,
        macro: float = 1.0,
        *,
        breakpoints: Sequence[float] = (),
    ) -> tuple[Tangent, ...]:
        return (Tangent(self.linear_coefficient(risk_reduction_pts, macro), 0.0),)


@dataclass(frozen=True, slots=True)
class ParameterPowerResponse:
    """``f(x) = (macro * R) ** alpha * x`` — the acquired system's behaviour.

    Retained for parity and legacy reproducibility only. Note that this is
    linear in ``x``: it produces no diminishing returns to funding.
    """

    exponent: float = 0.85
    calibration_source: str = "uncalibrated-assumption"
    version: str = "risk-response/parameter-power@1.0.0"

    def __post_init__(self) -> None:
        if self.exponent <= 0:
            raise ValueError(f"exponent must be positive, got {self.exponent}")

    @property
    def is_linear(self) -> bool:
        return True

    def evaluate(self, risk_reduction_pts: float, scale: float, macro: float = 1.0) -> float:
        return self.linear_coefficient(risk_reduction_pts, macro) * check_scale(scale)

    def linear_coefficient(self, risk_reduction_pts: float, macro: float = 1.0) -> float:
        base = macro * risk_reduction_pts
        if base <= 0.0:
            return 0.0
        return float(base**self.exponent)

    def tangents(
        self,
        risk_reduction_pts: float,
        macro: float = 1.0,
        *,
        breakpoints: Sequence[float] = (),
    ) -> tuple[Tangent, ...]:
        return (Tangent(self.linear_coefficient(risk_reduction_pts, macro), 0.0),)


@dataclass(frozen=True, slots=True)
class AllocationConcaveResponse:
    """``f(x) = (macro * R) * x ** alpha`` with ``0 < alpha <= 1``.

    Concave in funding scale, so the first dollar into a node buys more risk
    reduction than the last. This is what the acquired system's ledger claimed
    the 0.85 exponent was doing.

    ``alpha`` remains an uncalibrated assumption. It is now a named, versioned
    parameter with a calibration slot rather than a literal buried in an
    expression, so a future calibration exercise can replace it without
    touching the optimizer.
    """

    exponent: float = 0.85
    calibration_source: str = "uncalibrated-assumption"
    version: str = "risk-response/allocation-concave@1.0.0"

    def __post_init__(self) -> None:
        if not 0.0 < self.exponent <= 1.0:
            raise ValueError(f"exponent must lie in (0, 1], got {self.exponent}")

    @property
    def is_linear(self) -> bool:
        return self.exponent == 1.0

    def evaluate(self, risk_reduction_pts: float, scale: float, macro: float = 1.0) -> float:
        if check_scale(scale) == 0.0:
            return 0.0
        return macro * risk_reduction_pts * float(scale**self.exponent)

    def linear_coefficient(self, risk_reduction_pts: float, macro: float = 1.0) -> float:
        if not self.is_linear:
            raise TypeError(
                "AllocationConcaveResponse with exponent < 1 is non-linear; "
                "use tangents() instead of linear_coefficient()"
            )
        return macro * risk_reduction_pts

    def tangents(
        self,
        risk_reduction_pts: float,
        macro: float = 1.0,
        *,
        breakpoints: Sequence[float] = (),
    ) -> tuple[Tangent, ...]:
        coefficient = macro * risk_reduction_pts
        if coefficient <= 0.0:
            return (Tangent(0.0, 0.0),)
        if self.is_linear:
            return (Tangent(coefficient, 0.0),)

        points = tuple(breakpoints) or DEFAULT_BREAKPOINTS
        alpha = self.exponent
        out: list[Tangent] = []
        for p in points:
            if not 0.0 < p <= 1.0:
                raise ValueError(f"breakpoints must lie in (0, 1], got {p}")
            # f(p) + f'(p)(x - p) with f(x) = c * x**alpha
            value = coefficient * math.pow(p, alpha)
            slope = coefficient * alpha * math.pow(p, alpha - 1.0)
            out.append(Tangent(slope=slope, intercept=value - slope * p))
        return tuple(out)
