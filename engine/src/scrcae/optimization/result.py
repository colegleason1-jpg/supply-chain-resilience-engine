"""Optimization results and independent constraint verification.

The acquired system displayed strings such as "Verification: zero fractional
violations detected" in its equation ledger. That text was a hard-coded literal
in the UI. Nothing checked it.

:class:`ConstraintReport` performs the check for real, against the solver's
returned allocation vector, using tolerances that are stated rather than
implied. A verification claim is only worth as much as the code that can fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "SolveStatus",
    "Violation",
    "ConstraintReport",
    "NodeAllocation",
    "OptimizationResult",
]


class SolveStatus:
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    NOT_SOLVED = "not_solved"
    UNDEFINED = "undefined"


@dataclass(frozen=True, slots=True)
class Violation:
    """A single failed constraint check."""

    kind: str
    detail: str
    magnitude: float

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail} (by {self.magnitude:.6g})"


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """Outcome of independently re-checking every structural constraint."""

    violations: tuple[Violation, ...] = ()
    tolerance: float = 1e-6
    checks_performed: tuple[str, ...] = ()
    #: Largest observed constraint overshoot that fell within tolerance, keyed by
    #: check name. Recorded rather than discarded: a solution can be feasible to
    #: within solver tolerance while still overspending a budget by a measurable
    #: amount, and whoever signs off on a capital plan is entitled to see that
    #: number instead of being told only that the check "passed".
    residual_slack: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_feasible(self) -> bool:
        return not self.violations

    @property
    def max_residual_slack(self) -> float:
        return max(self.residual_slack.values(), default=0.0)

    def summary(self) -> str:
        if self.is_feasible:
            return (
                f"{len(self.checks_performed)} constraint families verified, "
                f"0 violations (tolerance {self.tolerance:g})"
            )
        return (
            f"{len(self.violations)} violation(s) across "
            f"{len(self.checks_performed)} constraint families: "
            + "; ".join(str(v) for v in self.violations[:5])
        )


@dataclass(frozen=True, slots=True)
class NodeAllocation:
    """Per-node outcome, with every quantity recomputed from ground truth."""

    node_id: str
    name: str
    action: str
    funding_scale: float
    capital: float
    risk_reduction_pts: float
    lead_time_saved_days: float
    carbon_tons: float

    @property
    def is_active(self) -> bool:
        return self.funding_scale > 0.0


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """The full, self-describing outcome of one optimization run."""

    status: str
    allocations: tuple[NodeAllocation, ...]
    baseline_risk_pts: float
    gross_capital: float
    bundle_discounts: float
    active_bundles: tuple[str, ...]
    total_lead_time_saved_days: float
    total_carbon_tons: float
    objective_value: float | None
    objective_unit: str
    constraint_report: ConstraintReport
    audit: Mapping[str, object] = field(default_factory=dict)
    #: Populated only when ``status == SolveStatus.INFEASIBLE``. Explains which
    #: constraint bit and, critically, whether more capital would help. Typed as
    #: ``object`` to avoid a circular import; it is an
    #: ``optimization.diagnostics.InfeasibilityDiagnosis``.
    diagnosis: object | None = None
    #: The baseline after the macro multiplier, i.e. what the acquired system
    #: displayed and what target-mode goals are set against. Defaults to the raw
    #: baseline when no multiplier is in play. Kept separate from
    #: ``baseline_risk_pts`` rather than replacing it, because the raw figure is
    #: what the in-model risk cap constrains -- the two are genuinely different
    #: numbers and merging them would hide that. See ADR-001 F18.
    effective_baseline_risk_pts: float | None = None

    #: Risk reduction before the baseline cap is applied.
    raw_risk_reduction_pts: float = 0.0

    #: Units of each declared resource the allocation draws, recomputed from the
    #: funding scales. Empty for a network without resources.
    resource_use: Mapping[str, float] = field(default_factory=dict)

    def resource_headroom(self, name: str, capacity: float) -> float:
        """Capacity left on ``name`` after this allocation; negative only if a violation was reported."""
        return float(capacity) - float(self.resource_use.get(name, 0.0))

    @property
    def net_capital(self) -> float:
        return self.gross_capital - self.bundle_discounts

    @property
    def risk_reduction_pts(self) -> float:
        """Risk reduction, capped at the baseline. Cannot exceed 100% removal."""
        return min(self.baseline_risk_pts, self.raw_risk_reduction_pts)

    @property
    def optimized_risk_pts(self) -> float:
        return max(0.0, self.baseline_risk_pts - self.risk_reduction_pts)

    @property
    def risk_cap_was_binding(self) -> bool:
        """True when the optimizer bought risk reduction it could not use.

        The acquired system capped total risk reduction *after* solving, with
        ``min(baseline_risk, total_risk_drop)``. The optimizer itself never saw
        the cap, so it could spend capital on reduction beyond the baseline and
        still report that spend as productive. When this flag is True the run's
        capital efficiency is overstated unless the cap was enforced in-model.
        """
        return self.raw_risk_reduction_pts > self.baseline_risk_pts + 1e-9

    @property
    def is_solved(self) -> bool:
        return self.status == SolveStatus.OPTIMAL

    @property
    def active_allocations(self) -> tuple[NodeAllocation, ...]:
        """Funded nodes -- but only when the solve actually succeeded.

        When the solver reports infeasible, CBC still leaves values in the decision
        variables. Those values violate the constraints and mean nothing, yet they
        read as a complete, fully funded portfolio. Surfacing them would be an
        invitation to act on a plan the engine just said is impossible, so a
        non-optimal result presents no allocations at all. The raw values remain on
        :attr:`allocations` for debugging.
        """
        if not self.is_solved:
            return ()
        return tuple(a for a in self.allocations if a.is_active)

    @property
    def reporting_baseline_risk_pts(self) -> float:
        """The baseline to quote to a reader: macro-inflated where applicable."""
        if self.effective_baseline_risk_pts is None:
            return self.baseline_risk_pts
        return self.effective_baseline_risk_pts

    @property
    def effective_optimized_risk_pts(self) -> float:
        """Post-intervention risk against the macro-inflated baseline.

        This is the number the acquired system displayed. It differs from
        :attr:`optimized_risk_pts`, which subtracts from the raw baseline, whenever a
        macro multiplier above 1.0 is active. Quoting the raw version to a user
        understates their residual risk, which is the wrong direction to be wrong in.
        """
        return max(0.0, self.reporting_baseline_risk_pts - self.risk_reduction_pts)

    @property
    def explanation(self) -> str:
        """A sentence fit to show a reader, whatever the status.

        Deliberately never returns the bare word "infeasible": that is a solver
        term, and on its own it invites the reader to reach for more budget even
        when no budget would help.
        """
        if self.is_solved:
            return (
                f"Risk {self.reporting_baseline_risk_pts:.1f} -> "
                f"{self.effective_optimized_risk_pts:.1f} pts for "
                f"{self.net_capital:,.0f} net capital."
            )
        if self.diagnosis is not None:
            return self.diagnosis.summary()  # type: ignore[attr-defined]
        return (
            f"No allocation was produced (solver status: {self.status}). No "
            "diagnosis is available, so none is being inferred."
        )

    def scales(self) -> dict[str, float]:
        return {a.node_id: a.funding_scale for a in self.allocations}

    def to_records(self) -> list[dict[str, object]]:
        """Flat rows suitable for a DataFrame, CSV export or API response."""
        return [
            {
                "node_id": a.node_id,
                "name": a.name,
                "action": a.action,
                "funding_scale": a.funding_scale,
                "capital": a.capital,
                "risk_reduction_pts": a.risk_reduction_pts,
                "lead_time_saved_days": a.lead_time_saved_days,
                "carbon_tons": a.carbon_tons,
            }
            for a in self.allocations
        ]
