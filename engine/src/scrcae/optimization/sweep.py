"""Budget sensitivity, computed by calling the one optimizer repeatedly.

The acquired system drew a budget-sensitivity curve from a *second* copy of the
optimizer, transcribed by hand into the chart-rendering block (F6). The two copies
had already drifted: the headline figure clamped risk reduction at the baseline
while the sweep did not, so the curve and the number above it were computed by
different models. Nobody had changed the model twice on purpose; one copy had simply
been updated and the other missed.

The fix is not to keep the copies in sync. It is to have one optimizer and call it
in a loop, which is what this module does. A sweep is orchestration, not modelling,
so it belongs next to the solver rather than in a view.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .result import SolveStatus


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One budget level and what the optimizer did with it."""

    budget: float
    status: str
    net_capital: float
    risk_reduction_pts: float
    optimized_risk_pts: float
    lead_time_saved_days: float
    active_node_ids: tuple[str, ...]

    @property
    def is_solved(self) -> bool:
        return self.status == SolveStatus.OPTIMAL

    @property
    def capital_unspent(self) -> float:
        """Authorised but unused capital.

        Persistently large values across the top of a sweep mean the portfolio has
        run out of things worth buying, not that the budget is right.
        """
        return max(0.0, self.budget - self.net_capital)


@dataclass(frozen=True, slots=True)
class BudgetSweep:
    """A sweep result, with the saturation point called out.

    ``saturation_budget`` is the lowest budget that already achieves the best
    reduction any budget in the sweep achieved. Above it, authorising more capital
    buys nothing — the single most useful thing a budget curve can tell a reader,
    and the thing a bare line chart makes them squint to find.
    """

    points: tuple[SweepPoint, ...]

    @property
    def solved_points(self) -> tuple[SweepPoint, ...]:
        return tuple(p for p in self.points if p.is_solved)

    @property
    def best_reduction_pts(self) -> float:
        solved = self.solved_points
        return max((p.risk_reduction_pts for p in solved), default=0.0)

    @property
    def saturation_budget(self) -> float | None:
        """Lowest budget reaching the sweep's best reduction, if any solved."""
        best = self.best_reduction_pts
        candidates = [
            p.budget
            for p in self.solved_points
            if p.risk_reduction_pts >= best - 1e-6
        ]
        return min(candidates) if candidates else None

    @property
    def is_saturated(self) -> bool:
        """True when the top of the sweep buys nothing over the saturation point."""
        sat = self.saturation_budget
        if sat is None or not self.points:
            return False
        return sat < max(p.budget for p in self.points) - 1e-9

    def to_records(self) -> list[dict[str, object]]:
        """Rows for tabular display. Formatting stays with the caller."""
        return [
            {
                "budget": p.budget,
                "status": p.status,
                "net_capital": p.net_capital,
                "risk_reduction_pts": p.risk_reduction_pts,
                "optimized_risk_pts": p.optimized_risk_pts,
                "lead_time_saved_days": p.lead_time_saved_days,
                "active_nodes": len(p.active_node_ids),
            }
            for p in self.points
        ]


def budget_levels(
    reference: float, *, steps: int = 9, span: float = 0.8
) -> tuple[float, ...]:
    """Budget levels bracketing ``reference``, evenly spaced and strictly positive.

    Kept separate from :func:`sweep_budget` so a caller can show the levels it is
    about to solve, and so the choice of levels is testable without a solver.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0.0 < span <= 2.0:
        raise ValueError("span must be in (0, 2]")
    reference = max(float(reference), 1.0)
    if steps == 1:
        return (reference,)

    low = max(1.0, reference * (1.0 - span / 2.0))
    high = reference * (1.0 + span / 2.0)
    step = (high - low) / (steps - 1)
    return tuple(low + step * i for i in range(steps))


def sweep_budget(request, levels: Sequence[float] | None = None, **kwargs) -> BudgetSweep:
    """Re-solve ``request`` at each budget in ``levels``.

    Every point comes from the same :func:`solve` the headline figure comes from, so
    the curve and the number above it cannot disagree. Infeasible levels are kept
    rather than dropped: a curve that silently omits the budgets where nothing works
    reads as though everything works.

    Diagnosis is switched off for the sweep. It costs extra solves per point and the
    sweep already answers the question diagnosis would ask.
    """
    from .optimizer import solve

    if levels is None:
        reference = request.budget if request.budget else 1.0
        levels = budget_levels(reference, **kwargs)
    elif kwargs:
        raise TypeError("pass either explicit levels or level-construction kwargs")

    points: list[SweepPoint] = []
    for level in levels:
        result = solve(replace(request, budget=float(level), diagnose=False))
        points.append(
            SweepPoint(
                budget=float(level),
                status=result.status,
                net_capital=result.net_capital if result.is_solved else 0.0,
                risk_reduction_pts=result.risk_reduction_pts if result.is_solved else 0.0,
                optimized_risk_pts=(
                    result.effective_optimized_risk_pts
                    if result.is_solved
                    else result.reporting_baseline_risk_pts
                ),
                lead_time_saved_days=(
                    result.total_lead_time_saved_days if result.is_solved else 0.0
                ),
                active_node_ids=tuple(
                    a.node_id for a in result.active_allocations
                ),
            )
        )
    return BudgetSweep(points=tuple(points))
