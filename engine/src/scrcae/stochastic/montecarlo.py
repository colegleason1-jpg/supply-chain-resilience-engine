"""Correlated Monte Carlo runner and tail-risk analysis.

Changes from the acquired implementation, beyond leaving Streamlit:

* **Reproducibility without global state.** The original called
  ``np.random.seed(42)`` — a hard-coded seed mutating the process-wide generator,
  so any other component drawing randomness afterwards was silently affected,
  and the seed could not be varied to test seed sensitivity. This uses an
  explicit ``numpy.random.Generator``.
* **Vectorised.** The original looped in Python, drawing ``k`` normals per
  iteration, for up to 50,000 iterations. This draws one
  ``(iterations, k)`` block and uses matrix products.
* **Convergence is reported, not assumed.** The original exposed an iteration
  slider from 1,000 to 50,000 with no indication of whether the chosen count was
  sufficient. Percentile standard errors are now returned alongside the
  estimates.
* **Consistent baseline clipping.** The deterministic path clipped total risk
  reduction at the baseline while the simulation path clipped each realisation
  separately, so the two were not strictly comparable. Both now clip identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .correlation import RepairReport, cholesky_factor, repair_to_correlation
from .shock import LognormalShock, ShockModel

__all__ = ["SimulationRequest", "SimulationResult", "run"]


@dataclass(frozen=True)
class SimulationRequest:
    """Inputs for one Monte Carlo run.

    Parameters
    ----------
    baseline_risk_pts:
        Pre-intervention system risk.
    risk_reduction_by_node:
        Deterministic risk reduction delivered per node, in percentage points,
        already evaluated through the risk response model at the chosen funding
        scale. Passing ground-truth values keeps the simulation independent of
        any linearisation used inside the optimizer.
    correlation:
        Square correlation matrix, ordered to match ``node_order``.
    node_order:
        Node identifiers fixing the correlation matrix's row and column order.
    iterations:
        Number of realisations.
    seed:
        Explicit seed. Same seed plus same inputs gives bit-identical output.
    shock_model:
        How correlated normals become delivery multipliers.
    """

    baseline_risk_pts: float
    risk_reduction_by_node: Mapping[str, float]
    correlation: np.ndarray
    node_order: Sequence[str]
    iterations: int = 10_000
    seed: int = 42
    shock_model: ShockModel = field(default_factory=LognormalShock)

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.baseline_risk_pts < 0:
            raise ValueError("baseline_risk_pts must be non-negative")
        missing = [n for n in self.node_order if n not in self.risk_reduction_by_node]
        if missing:
            raise ValueError(f"risk_reduction_by_node is missing nodes: {missing}")
        k = len(self.node_order)
        if np.asarray(self.correlation).shape != (k, k):
            raise ValueError(
                f"correlation shape {np.asarray(self.correlation).shape} does not match "
                f"{k} nodes"
            )


@dataclass(frozen=True)
class SimulationResult:
    """Distribution of post-intervention system risk."""

    p50_risk_pts: float
    p90_risk_pts: float
    mean_risk_pts: float
    std_dev_pts: float
    p10_risk_pts: float
    expected_shortfall_90_pts: float
    deterministic_risk_pts: float
    mean_delivered_reduction_pts: float
    deterministic_reduction_pts: float
    iterations: int
    seed: int
    shock_version: str
    correlation_repair: RepairReport
    p50_standard_error: float
    p90_standard_error: float
    samples: np.ndarray = field(repr=False, compare=False)

    @property
    def reduction_bias_pts(self) -> float:
        """Mean simulated reduction minus deterministic reduction.

        A mean-preserving shock model should hold this near zero to within Monte
        Carlo error. A materially non-zero value means the shock model is
        shifting the central estimate, not merely adding dispersion around it.
        """
        return self.mean_delivered_reduction_pts - self.deterministic_reduction_pts

    @property
    def tail_spread_pts(self) -> float:
        """P90 minus P50: how much worse the tail is than the central case."""
        return self.p90_risk_pts - self.p50_risk_pts

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.samples, q))

    def summary(self) -> str:
        return (
            f"P50 {self.p50_risk_pts:.2f}% (SE {self.p50_standard_error:.3f}), "
            f"P90 {self.p90_risk_pts:.2f}% (SE {self.p90_standard_error:.3f}), "
            f"deterministic {self.deterministic_risk_pts:.2f}%, "
            f"bias {self.reduction_bias_pts:+.3f} pts, "
            f"N={self.iterations:,}, seed={self.seed}"
        )


def _percentile_standard_error(samples: np.ndarray, q: float) -> float:
    """Asymptotic standard error of a sample quantile via a density estimate.

    ``SE = sqrt(p (1 - p) / n) / f(x_p)``, with ``f`` estimated from the spacing
    of order statistics around the quantile. Returned so a user can tell whether
    an iteration count actually supports the precision being reported.
    """
    n = samples.size
    if n < 100:
        return float("nan")
    p = q / 100.0
    ordered = np.sort(samples)
    index = int(np.clip(round(p * (n - 1)), 0, n - 1))
    window = max(1, int(0.02 * n))
    low = int(np.clip(index - window, 0, n - 1))
    high = int(np.clip(index + window, 0, n - 1))
    spread = ordered[high] - ordered[low]
    if spread <= 0:
        return 0.0
    density = (high - low) / (n * spread)
    if density <= 0:
        return float("nan")
    return float(np.sqrt(p * (1.0 - p) / n) / density)


def run(request: SimulationRequest) -> SimulationResult:
    """Execute a correlated Monte Carlo simulation of post-intervention risk."""
    node_order = list(request.node_order)
    reductions = np.array(
        [float(request.risk_reduction_by_node[n]) for n in node_order], dtype=float
    )
    baseline = float(request.baseline_risk_pts)
    deterministic_reduction = float(min(baseline, reductions.sum()))
    deterministic_risk = max(0.0, baseline - deterministic_reduction)

    k = len(node_order)
    if k == 0 or reductions.sum() == 0.0:
        samples = np.full(request.iterations, baseline, dtype=float)
        empty_repair = RepairReport(False, 1.0, 1.0, 0.0, 0.0)
        return SimulationResult(
            p50_risk_pts=baseline,
            p90_risk_pts=baseline,
            mean_risk_pts=baseline,
            std_dev_pts=0.0,
            p10_risk_pts=baseline,
            expected_shortfall_90_pts=baseline,
            deterministic_risk_pts=baseline,
            mean_delivered_reduction_pts=0.0,
            deterministic_reduction_pts=0.0,
            iterations=request.iterations,
            seed=request.seed,
            shock_version=getattr(request.shock_model, "version", "unknown"),
            correlation_repair=empty_repair,
            p50_standard_error=0.0,
            p90_standard_error=0.0,
            samples=samples,
        )

    repaired, repair_report = repair_to_correlation(np.asarray(request.correlation))
    chol = cholesky_factor(repaired)

    rng = np.random.default_rng(request.seed)
    z = rng.standard_normal(size=(request.iterations, k))
    correlated = z @ chol.T
    multipliers = request.shock_model.apply(correlated)

    delivered = multipliers @ reductions
    delivered = np.clip(delivered, 0.0, baseline)
    samples = baseline - delivered

    p90 = float(np.percentile(samples, 90))
    worst = samples[samples >= p90]

    return SimulationResult(
        p50_risk_pts=float(np.percentile(samples, 50)),
        p90_risk_pts=p90,
        mean_risk_pts=float(samples.mean()),
        std_dev_pts=float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        p10_risk_pts=float(np.percentile(samples, 10)),
        expected_shortfall_90_pts=float(worst.mean()) if worst.size else p90,
        deterministic_risk_pts=deterministic_risk,
        mean_delivered_reduction_pts=float(delivered.mean()),
        deterministic_reduction_pts=deterministic_reduction,
        iterations=request.iterations,
        seed=request.seed,
        shock_version=getattr(request.shock_model, "version", "unknown"),
        correlation_repair=repair_report,
        p50_standard_error=_percentile_standard_error(samples, 50.0),
        p90_standard_error=_percentile_standard_error(samples, 90.0),
        samples=samples,
    )
