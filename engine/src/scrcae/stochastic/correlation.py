"""Correlation models and positive-definiteness repair.

Forensic background
-------------------
The acquired system built a correlation matrix from a user-edited grid, silently
substituting 0.35 for any cell it failed to parse, and repaired
non-positive-definiteness with::

    min_eig = np.min(np.real(np.linalg.eigvals(corr_matrix)))
    corr_matrix += np.eye(k) * (-min_eig + 1e-5)

That ridge does restore positive definiteness, but it adds a constant to every
diagonal entry, so the diagonal is no longer 1. The matrix stops being a
correlation matrix and becomes a covariance matrix with inflated, unequal
variances. Downstream, the Cholesky factor then scales every node's shock by
``sqrt(1 + ridge)``, silently widening the simulated distribution — by 6.9% per
node for a ridge of 0.15, for example. Tail risk gets worse purely because the
input matrix was awkward, which is not a property of the business.

:func:`repair_to_correlation` clips the spectrum and then renormalises the
diagonal back to unity, which keeps the object a correlation matrix and leaves
per-node marginal variances untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "CorrelationError",
    "RepairReport",
    "repair_to_correlation",
    "naive_ridge_repair",
    "CorrelationModel",
    "UniformCorrelation",
    "MatrixCorrelation",
]


class CorrelationError(ValueError):
    """Raised when a supplied correlation matrix is structurally invalid."""


@dataclass(frozen=True, slots=True)
class RepairReport:
    """What had to be done to make a matrix usable."""

    repair_applied: bool
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    frobenius_shift: float
    max_diagonal_deviation: float

    def summary(self) -> str:
        if not self.repair_applied:
            return (
                "matrix was already positive definite "
                f"(min eigenvalue {self.min_eigenvalue_before:.6g})"
            )
        return (
            f"repaired: min eigenvalue {self.min_eigenvalue_before:.6g} -> "
            f"{self.min_eigenvalue_after:.6g}, Frobenius shift "
            f"{self.frobenius_shift:.6g}, max diagonal deviation "
            f"{self.max_diagonal_deviation:.3g}"
        )


def _validate(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise CorrelationError(f"correlation matrix must be square, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise CorrelationError("correlation matrix contains non-finite values")
    if not np.allclose(m, m.T, atol=1e-8):
        raise CorrelationError("correlation matrix must be symmetric")
    if np.any(np.abs(m) > 1.0 + 1e-9):
        raise CorrelationError("correlation entries must lie within [-1, 1]")
    if not np.allclose(np.diag(m), 1.0, atol=1e-8):
        raise CorrelationError("correlation matrix must have a unit diagonal")
    return m


def repair_to_correlation(
    matrix: np.ndarray, *, min_eigenvalue: float = 1e-8
) -> tuple[np.ndarray, RepairReport]:
    """Return the nearest usable correlation matrix and a repair report.

    Eigenvalues below ``min_eigenvalue`` are clipped, the matrix is
    reconstructed, symmetrised, and then rescaled so the diagonal returns to
    exactly 1. Marginal variances are therefore preserved, which the ridge
    approach does not do.
    """
    m = _validate(matrix)
    eigenvalues = np.linalg.eigvalsh(m)
    min_before = float(eigenvalues.min())

    if min_before >= min_eigenvalue:
        return m, RepairReport(
            repair_applied=False,
            min_eigenvalue_before=min_before,
            min_eigenvalue_after=min_before,
            frobenius_shift=0.0,
            max_diagonal_deviation=float(np.max(np.abs(np.diag(m) - 1.0))),
        )

    values, vectors = np.linalg.eigh(m)
    clipped = np.clip(values, min_eigenvalue, None)
    rebuilt = vectors @ np.diag(clipped) @ vectors.T
    rebuilt = (rebuilt + rebuilt.T) / 2.0

    # Renormalise to a unit diagonal, which is what keeps this a correlation
    # matrix rather than a variance-inflated covariance matrix.
    scale = np.sqrt(np.clip(np.diag(rebuilt), 1e-12, None))
    rebuilt = rebuilt / np.outer(scale, scale)
    np.fill_diagonal(rebuilt, 1.0)
    rebuilt = np.clip(rebuilt, -1.0, 1.0)
    np.fill_diagonal(rebuilt, 1.0)

    return rebuilt, RepairReport(
        repair_applied=True,
        min_eigenvalue_before=min_before,
        min_eigenvalue_after=float(np.linalg.eigvalsh(rebuilt).min()),
        frobenius_shift=float(np.linalg.norm(rebuilt - m, ord="fro")),
        max_diagonal_deviation=float(np.max(np.abs(np.diag(rebuilt) - 1.0))),
    )


def naive_ridge_repair(matrix: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """The acquired system's repair, reproduced for comparison in tests only.

    Kept so the variance-inflation defect can be demonstrated numerically
    rather than merely asserted in prose.
    """
    m = np.asarray(matrix, dtype=float).copy()
    min_eig = float(np.min(np.real(np.linalg.eigvals(m))))
    if min_eig < 0:
        m = m + np.eye(m.shape[0]) * (-min_eig + epsilon)
    return m


def cholesky_factor(matrix: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor. Assumes an already-repaired matrix."""
    return np.linalg.cholesky(matrix)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class UniformCorrelation:
    """Every off-diagonal pair shares one coefficient.

    The acquired system initialised its grid to 0.35 off-diagonal with no stated
    basis. That default is reproduced here but named, versioned, and marked as
    an assumption so it is visible in the audit ledger instead of appearing to
    be data.
    """

    rho: float = 0.35
    source: str = "uncalibrated-default"
    version: str = "correlation/uniform@1.0.0"

    def __post_init__(self) -> None:
        if not -1.0 <= self.rho <= 1.0:
            raise CorrelationError(
                f"rho must lie in [-1, 1], got {self.rho}"
            )

    def matrix(self, node_ids: Sequence[str]) -> np.ndarray:
        """Build the equicorrelation matrix for ``node_ids``.

        A uniform ``rho`` across ``k`` nodes is only positive semi-definite for
        ``rho >= -1/(k-1)``. Below that bound the matrix the caller is asking for
        does not describe any possible joint distribution, so it is rejected
        rather than silently handed to the repair step — repairing it would
        return a *different* correlation structure than the one requested, which
        is exactly the class of silent substitution this engine is meant to stop.
        """
        k = len(node_ids)
        if k > 1:
            floor = -1.0 / (k - 1)
            if self.rho < floor:
                raise CorrelationError(
                    f"rho={self.rho} is not attainable for {k} nodes: a uniform "
                    f"correlation must be at least {floor:.4f} to be positive "
                    f"semi-definite"
                )
        m = np.full((k, k), float(self.rho))
        np.fill_diagonal(m, 1.0)
        return m


@dataclass(frozen=True, slots=True)
class MatrixCorrelation:
    """An explicitly supplied matrix, ordered to match a node sequence."""

    values: tuple[tuple[float, ...], ...]
    node_ids: tuple[str, ...]
    source: str = "user-supplied"
    version: str = "correlation/matrix@1.0.0"

    def matrix(self, node_ids: Sequence[str]) -> np.ndarray:
        index = {n: i for i, n in enumerate(self.node_ids)}
        missing = [n for n in node_ids if n not in index]
        if missing:
            raise CorrelationError(f"correlation matrix is missing nodes: {missing}")
        raw = np.asarray(self.values, dtype=float)
        order = [index[n] for n in node_ids]
        return raw[np.ix_(order, order)]


CorrelationModel = UniformCorrelation | MatrixCorrelation
