"""The app's single point of contact with the engine.

Every call into `scrcae` goes through here. The point is not indirection for its own
sake: it is that the acquired system called PuLP from inside a Streamlit render
function in three separate places, which is how it ended up with two optimizers that
had drifted apart (F6) and a Monte Carlo that seeded the global RNG on every rerun
(F10). Keeping the boundary in one file means "does the UI do any modelling?" is a
question you can answer by reading one module.

Nothing here decides *what* to model. It takes an already-validated request and
returns an already-computed answer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from scrcae.optimization import (
    BudgetSweep,
    OptimizationRequest,
    attainable_frontier,
    budget_levels,
    solve,
    sweep_budget,
)
from scrcae.stochastic import montecarlo


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Everything one screen needs, computed once.

    Streamlit reruns the whole script on every widget interaction. Computing these
    together and passing the bundle down means a rerun cannot leave the headline
    figure and the sweep chart describing different solves.
    """

    result: object
    simulation: object | None = None
    frontier: object | None = None
    sweep: BudgetSweep | None = None


def run_optimization(request: OptimizationRequest):
    """Solve. Diagnosis comes back attached when the request is infeasible."""
    return solve(request)


def run_simulation(simulation_request):
    return montecarlo.run(simulation_request)


def run_frontier(request: OptimizationRequest, *, budget: float | None = None):
    """The portfolio's ceiling, independent of (or subject to) a budget."""
    return attainable_frontier(
        request.network,
        request.risk_response,
        macro_multiplier=request.macro_multiplier,
        budget=budget,
        enforce_risk_cap=request.enforce_risk_cap,
        tangent_points=request.tangent_points,
        legacy_bundle_activation=request.legacy_bundle_activation,
    )


def run_sweep(
    request: OptimizationRequest, *, reference: float | None = None, steps: int = 9
) -> BudgetSweep:
    """Budget sensitivity via repeated calls to the one optimizer (F6).

    The reference point is the current budget where there is one, and the capital the
    solution actually used in target mode — sweeping around a 1e9 placeholder budget
    would produce a chart of nine identical rows.
    """
    if reference is None:
        reference = request.budget or 1.0
    return sweep_budget(request, levels=budget_levels(reference, steps=steps))


def analyse(
    request: OptimizationRequest,
    *,
    simulation_factory: Callable[[object], object] | None = None,
    include_sweep: bool = True,
    sweep_steps: int = 9,
    sweep_reference: float | None = None,
) -> AnalysisBundle:
    """Run the full analysis for one screen.

    The frontier is computed whenever the solve failed, because that is precisely when
    the user needs to know what *is* reachable. It is also computed on success in
    target mode, where "you met your target, and here is the best that was available"
    is a materially different conversation from "you met your target".
    """
    result = run_optimization(request)

    simulation = None
    if simulation_factory is not None and result.is_solved:
        # The simulation describes the portfolio the optimizer chose, so it can only be
        # built after the solve. It is skipped entirely on failure: putting a
        # confidence interval around a plan that does not exist would give the
        # non-existent plan the strongest visual claim to being real.
        simulation = run_simulation(simulation_factory(result))

    frontier = None
    if not result.is_solved:
        frontier = result.diagnosis.attainable if result.diagnosis else run_frontier(request)
    elif request.required_risk_reduction_pts:
        frontier = run_frontier(request)

    sweep = None
    if include_sweep:
        reference = sweep_reference
        if reference is None:
            reference = request.budget if request.budget else None
        if reference is None or reference > 1e8:
            # Target mode pins the budget to something enormous or leaves it unset.
            # Sweep around what the plan actually costs instead.
            reference = result.net_capital if result.is_solved else None
        if reference is None and frontier is not None:
            reference = frontier.capital_at_ceiling
        if reference and reference > 0:
            sweep = run_sweep(request, reference=reference, steps=sweep_steps)

    return AnalysisBundle(
        result=result, simulation=simulation, frontier=frontier, sweep=sweep
    )
