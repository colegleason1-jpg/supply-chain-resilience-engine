"""Shock models — how a correlated normal draw becomes a delivery multiplier.

Forensic background
-------------------
The acquired system applied::

    shocks = 1.0 + 0.12 * correlated_z
    delivered = marginal_risk * scale * max(0.4, shocks[i])

Two defects, one of which materially biases published tail-risk numbers.

**1. The floor biases the mean upward.** ``1 + 0.12 Z`` has mean 1.0, so a naive
reading says the simulation is centred on the deterministic answer. But
``max(0.4, .)`` truncates only the left tail. Every draw below 0.4 is lifted to
0.4, and nothing is ever pushed down to compensate, so the realised multiplier
has mean strictly greater than 1. Simulated risk *reduction* is therefore
systematically larger than the deterministic reduction, and simulated *remaining
risk* is systematically lower. The engine's headline P50 risk is optimistic
relative to its own deterministic answer, by construction rather than by
modelling choice.

With sigma = 0.12 the floor sits at 5 sigma, so the bias is tiny and nobody
would notice. That is exactly what makes it dangerous: the moment sigma is
raised during calibration — and calibrating sigma is on the roadmap — the bias
grows fast and silently. At sigma = 0.30 the floor is 2 sigma out and the
distortion becomes material.

**2. A normal multiplier can go negative.** ``1 + sigma Z`` is negative for
``Z < -1/sigma``. A negative multiplier means an intervention *increases* the
risk it was funded to reduce. The floor at 0.4 conceals this rather than
addressing it, which is why the floor exists at all.

The fix
-------
:class:`LognormalShock` uses ``exp(sigma * Z - sigma**2 / 2)``, which is:

* strictly positive for every draw, so no floor is needed
* exactly mean-one for any sigma, so the simulation is centred on the
  deterministic result by construction and the two are legitimately comparable
* right-skewed, which better matches the empirical behaviour of delivery
  shortfalls than a symmetric normal does

:class:`LegacyTruncatedNormalShock` reproduces the original exactly so the two
can be compared numerically and the bias demonstrated in a test rather than
argued about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ShockModel",
    "LognormalShock",
    "LegacyTruncatedNormalShock",
    "DeterministicShock",
]


@runtime_checkable
class ShockModel(Protocol):
    """Transforms correlated standard normals into non-negative multipliers."""

    version: str

    @property
    def is_mean_preserving(self) -> bool:
        """True when the multiplier's expectation is exactly 1."""
        ...

    def apply(self, correlated_z: np.ndarray) -> np.ndarray:
        """Map an array of correlated standard normals to multipliers."""
        ...


@dataclass(frozen=True, slots=True)
class LognormalShock:
    """``m = exp(sigma * Z - sigma^2 / 2)``. Positive and exactly mean-one."""

    sigma: float = 0.12
    calibration_source: str = "uncalibrated-assumption"
    version: str = "shock/lognormal@1.0.0"

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    @property
    def is_mean_preserving(self) -> bool:
        return True

    def apply(self, correlated_z: np.ndarray) -> np.ndarray:
        if self.sigma == 0.0:
            return np.ones_like(correlated_z)
        return np.exp(self.sigma * correlated_z - 0.5 * self.sigma**2)


@dataclass(frozen=True, slots=True)
class LegacyTruncatedNormalShock:
    """``m = max(floor, 1 + sigma * Z)`` — the acquired system's shock.

    Retained for parity and to quantify the truncation bias. Not recommended.
    """

    sigma: float = 0.12
    floor: float = 0.4
    version: str = "shock/legacy-truncated-normal@1.0.0"

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    @property
    def is_mean_preserving(self) -> bool:
        return False

    def apply(self, correlated_z: np.ndarray) -> np.ndarray:
        return np.maximum(self.floor, 1.0 + self.sigma * correlated_z)

    @property
    def analytic_mean(self) -> float:
        """Closed-form expectation of the truncated multiplier.

        For ``M = max(f, 1 + sigma Z)`` with ``Z`` standard normal and
        ``a = (f - 1) / sigma``::

            E[M] = f * Phi(a) + (1 - Phi(a)) + sigma * phi(a)

        The final term is the upward bias introduced by truncation. It is
        strictly positive for every finite ``a``, which proves the bias exists
        analytically rather than only in simulation.
        """
        if self.sigma == 0.0:
            return max(self.floor, 1.0)
        a = (self.floor - 1.0) / self.sigma
        phi = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))
        return self.floor * Phi + (1.0 - Phi) + self.sigma * phi

    @property
    def analytic_bias(self) -> float:
        """``E[M] - 1``: how much the floor inflates delivered risk reduction."""
        return self.analytic_mean - 1.0


@dataclass(frozen=True, slots=True)
class DeterministicShock:
    """Always 1.0. Used to prove that simulation collapses onto the
    deterministic result when uncertainty is switched off."""

    version: str = "shock/deterministic@1.0.0"

    @property
    def is_mean_preserving(self) -> bool:
        return True

    def apply(self, correlated_z: np.ndarray) -> np.ndarray:
        return np.ones_like(correlated_z)
