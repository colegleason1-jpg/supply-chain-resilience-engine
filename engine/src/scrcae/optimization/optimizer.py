"""The mixed-integer capital allocation optimizer, headless.

This is the acquired system's highest-value component, lifted out of Streamlit.
In the original it existed twice — once for the live model and once, subtly
divergent, inside the budget sensitivity sweep — with bundle requirements read
from widget state mid-solve. Here it is one function with explicit inputs.

Structural semantics preserved from the acquired system:

* binary activation ``y_n`` and continuous funding scale ``x_n``
* minimum economic scale ``min_n * y_n <= x_n <= max_n * y_n``
* prerequisite cascade capping ``x_dependent <= x_prerequisite``
* bundle activation ``b_v <= y_n`` for every required node
* net budget constraint ``sum(C_n x_n) - sum(D_b b_v) <= B``
* target-risk mode: minimise net capital subject to a risk-reduction floor

Corrected here:

* the baseline risk cap is enforceable *in-model* rather than applied after the
  fact, so the optimizer can no longer buy unusable risk reduction
* risk response is a pluggable model, including genuinely concave-in-funding
  forms handled by tangent-based outer approximation
* every reported quantity is recomputed from the response model's ground-truth
  ``evaluate``, never read back from the linearised objective
* constraints are independently re-verified after the solve
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pulp as pl

from ..audit import build_audit_record
from ..risk.response import LinearResponse, RiskResponseModel, Tangent
from ..domain.network import SupplyNetwork
from .objectives import (
    MinimizeCapitalObjective,
    NodeTerms,
    ObjectiveModel,
    ProblemTerms,
)
from .result import (
    ConstraintReport,
    NodeAllocation,
    OptimizationResult,
    SolveStatus,
    Violation,
)

__all__ = ["OptimizationRequest", "solve", "DEFAULT_TOLERANCE"]

DEFAULT_TOLERANCE = 1e-6
_TANGENT_POINTS = 7

_STATUS_MAP = {
    pl.LpStatusOptimal: SolveStatus.OPTIMAL,
    pl.LpStatusInfeasible: SolveStatus.INFEASIBLE,
    pl.LpStatusUnbounded: SolveStatus.UNBOUNDED,
    pl.LpStatusNotSolved: SolveStatus.NOT_SOLVED,
    pl.LpStatusUndefined: SolveStatus.UNDEFINED,
}


@dataclass(frozen=True)
class OptimizationRequest:
    """A fully specified allocation problem.

    Parameters
    ----------
    network:
        Structural inputs.
    objective:
        Any :class:`~scrcae.optimization.objectives.ObjectiveModel`.
    risk_response:
        How funding scale converts into risk reduction.
    budget:
        Net capital cap. ``None`` means unconstrained, which is only sensible
        when the objective itself prices capital (as the monetary NPV objective
        does) or in target-risk mode.
    required_risk_reduction_pts:
        Risk-reduction floor. Required by target-risk mode.
    macro_multiplier:
        Network-level macro stress applied to every node's risk-reduction
        parameter, and the only multiplier that inflates the reported baseline.
        Preserved from the acquired system's Brent-crude derived scalar.
    node_macro_multipliers:
        Per-node macro stress, multiplied on top of ``macro_multiplier`` for that
        node's risk-reduction parameter only. This is what lets a market feed say
        "the oil-exposed node got riskier and the domestic assembler did not"
        instead of scaling the whole network by one commodity's move.

        Two deliberate constraints:

        *Unknown node ids are rejected, not ignored.* An exposure entry naming a
        node that does not exist is a mapping error, and the acquired system's
        habit of silently matching nothing — or, when a column was missing,
        silently matching *everything* — is precisely how a single market's
        volatility came to be applied to an entire network unnoticed (F11).

        *It does not touch the baseline.* ``effective_baseline_risk_pts`` uses
        the scalar alone. Aggregating per-node multipliers into a baseline
        adjustment would require weights the network does not carry, and the
        obvious candidate — each node's share of total risk reduction — makes the
        baseline move when an intervention is merely *written down*, which is the
        same class of artefact as F17. Per-node exposure changes which
        interventions pay; it does not restate the risk being measured.
    enforce_risk_cap:
        When True, constrains total in-model risk reduction to the baseline. The
        acquired system did not do this and clipped afterwards instead.
    tolerance:
        Absolute tolerance for post-solve constraint verification.
    relative_tolerance:
        Scale-relative slack allowed when verifying constraints, on top of the
        absolute ``tolerance``. Monetary constraints are checked against
        ``max(tolerance, |bound| * relative_tolerance)`` because an absolute
        tolerance of 1e-6 is meaningless against a nine-figure budget: it is far
        below the solver's own floating-point feasibility slack, so it would
        report violations that are numerical noise rather than modelling errors.
        Dimensionless constraints (funding scales, which live in [0, 1]) use the
        absolute tolerance alone.
    diagnose:
        When True (the default) and the solver reports the problem infeasible, the
        engine computes an attainable-frontier diagnosis and attaches it to the
        result. This costs additional solves, so it is disabled on the internal
        solves that diagnosis itself performs. It never changes the reported status:
        an infeasible request stays infeasible.
    legacy_bundle_activation:
        When True, a bundle discount is unlocked by mere node *activation*, as in
        the acquired system, rather than by full funding. This permits a portfolio
        to earn a discount priced against the whole package while paying only each
        node's minimum economic scale, so it is off by default and exists only so
        legacy parity remains demonstrable.
    tangent_points:
        Number of breakpoints per node used to linearise a non-linear risk
        response. Both linearisations are piecewise linear, so along any one
        facet the optimizer is indifferent between nearby funding splits; the
        resulting indeterminacy in a symmetric problem is bounded by roughly one
        facet width, or ``(max_scale - min_scale) / (tangent_points - 1)``. More
        breakpoints narrow that band at the cost of extra constraints and binary
        big-M rows. Reported quantities are always recomputed from the exact
        response, so this affects the chosen allocation's precision, never the
        accuracy of the numbers attributed to it.
    """

    network: SupplyNetwork
    objective: ObjectiveModel
    risk_response: RiskResponseModel = field(default_factory=LinearResponse)
    budget: float | None = None
    required_risk_reduction_pts: float | None = None
    macro_multiplier: float = 1.0
    node_macro_multipliers: Mapping[str, float] = field(default_factory=dict)
    enforce_risk_cap: bool = True
    tolerance: float = DEFAULT_TOLERANCE
    tangent_points: int = 13
    relative_tolerance: float = 1e-6
    legacy_bundle_activation: bool = False
    diagnose: bool = True
    solver_msg: bool = False
    problem_name: str = "capital_allocation"

    @property
    def effective_baseline_risk_pts(self) -> float:
        """Baseline risk after the macro multiplier, as the acquired system defined it.

        The acquired app computed ``min(100, baseline_risk * live_macro_multiplier)``
        and derived target mode's required reduction from *that*, while its in-model
        constraints used the raw baseline. The two therefore disagree whenever the
        macro multiplier is above 1.0, which is exactly when a macro feed is doing
        anything.

        This property exists so the disagreement is named rather than latent. It is
        used for *reporting* risk percentages, so that a target the user typed as
        20% is described back to them as 20%. It is deliberately **not** used for the
        ``enforce_risk_cap`` constraint: changing that would alter results and is a
        modelling decision, not a presentation one. See ADR-001 F18.
        """
        return min(100.0, self.network.baseline_risk_pts * self.macro_multiplier)

    def macro_for(self, node_id: str) -> float:
        """Effective macro multiplier for one node's risk-reduction parameter.

        With no per-node exposure this returns ``macro_multiplier`` unchanged and
        bit-exactly: multiplication by 1.0 is exact in IEEE-754, so introducing
        this indirection cannot move a single historic result.
        """
        return self.macro_multiplier * self.node_macro_multipliers.get(node_id, 1.0)

    def __post_init__(self) -> None:
        if self.macro_multiplier <= 0:
            raise ValueError("macro_multiplier must be positive")
        unknown = sorted(set(self.node_macro_multipliers) - set(self.network.node_ids))
        if unknown:
            raise ValueError(
                "node_macro_multipliers names nodes that are not in the network: "
                f"{', '.join(unknown)}"
            )
        for node_id, value in self.node_macro_multipliers.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"node macro multiplier for {node_id} must be a positive "
                    f"finite number, got {value}"
                )
        if self.tangent_points < 2:
            raise ValueError("tangent_points must be at least 2")
        if self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative")
        if self.budget is not None and self.budget < 0:
            raise ValueError("budget must be non-negative")
        if (
            self.required_risk_reduction_pts is not None
            and self.required_risk_reduction_pts < 0
        ):
            raise ValueError("required_risk_reduction_pts must be non-negative")
        if (
            isinstance(self.objective, MinimizeCapitalObjective)
            and self.required_risk_reduction_pts is None
        ):
            raise ValueError(
                "MinimizeCapitalObjective requires required_risk_reduction_pts; "
                "minimising capital with no risk floor trivially allocates nothing"
            )


def _breakpoints(low: float, high: float, count: int = _TANGENT_POINTS) -> tuple[float, ...]:
    """Tangent points across the node's feasible active funding interval."""
    low = max(low, 1e-3)
    if high <= low:
        return (high,)
    # Interpolate rather than accumulate `low + step * i`. Repeated addition of a
    # step lets floating-point error drift past the interval: for one generated
    # case it produced a final breakpoint of 1.0000000000000002, which the
    # response model correctly rejected as outside its (0, 1] domain. Anchoring
    # the endpoints makes the first and last breakpoints exact by construction.
    last = count - 1
    points = [low + (high - low) * (i / last) for i in range(count)]
    points[0] = low
    points[last] = high
    return tuple(points)


def _chords(
    response: RiskResponseModel,
    risk_reduction_pts: float,
    macro: float,
    breakpoints: Sequence[float],
) -> tuple[Tangent, ...]:
    """Secant lines through consecutive breakpoints of a concave response.

    For a concave function the piecewise-linear interpolant through a set of
    breakpoints lies *below* the function, and being concave it equals the
    minimum of its own segment lines. So ``u <= chord_k(x)`` for every segment
    yields ``u <= f(x)``: an inner (conservative) approximation.

    This is the mirror image of :func:`RiskResponseModel.tangents`, which gives
    an outer approximation. The distinction is not cosmetic — see the comments
    at the linearisation site for why the engine needs both.
    """
    if len(breakpoints) < 2:
        p = breakpoints[0]
        value = response.evaluate(risk_reduction_pts, p, macro)
        slope = value / p if p else 0.0
        return (Tangent(slope=slope, intercept=value - slope * p),)

    out: list[Tangent] = []
    for left, right in zip(breakpoints, breakpoints[1:]):
        f_left = response.evaluate(risk_reduction_pts, left, macro)
        f_right = response.evaluate(risk_reduction_pts, right, macro)
        span = right - left
        if span <= 0:
            continue
        slope = (f_right - f_left) / span
        out.append(Tangent(slope=slope, intercept=f_left - slope * left))
    return tuple(out)


def solve(request: OptimizationRequest) -> OptimizationResult:
    """Solve one allocation problem and return a verified, audited result."""
    network = request.network
    response = request.risk_response
    macro = request.macro_multiplier
    # Per-node effective multiplier. Read once here so every downstream use --
    # objective coefficients, the risk cap's outer approximation, and the
    # recomputation of reported quantities -- is guaranteed to use the same
    # number for a given node. The acquired system's equivalent was applied in
    # one place and forgotten in the sweep, which is how the two optimizers
    # diverged (F6).
    macro_of = {n: request.macro_for(n) for n in network.node_ids}
    node_ids = network.node_ids

    problem = pl.LpProblem(request.problem_name, request.objective.sense)

    y = {n: pl.LpVariable(f"y_{n}", cat="Binary") for n in node_ids}
    x = {
        n: pl.LpVariable(f"x_{n}", lowBound=0.0, upBound=1.0, cat="Continuous")
        for n in node_ids
    }

    # -- minimum economic scale ------------------------------------------- #
    for intervention in network:
        n = intervention.node_id
        problem += x[n] >= intervention.min_funding_scale * y[n], f"mes_low_{n}"
        problem += x[n] <= intervention.max_funding_scale * y[n], f"mes_high_{n}"

    # -- risk reduction expressions --------------------------------------- #
    # A non-linear response cannot be represented exactly in an MILP, so it is
    # approximated. The *direction* of the approximation error decides whether a
    # constraint is sound, and the two scalar risk constraints pull in opposite
    # directions:
    #
    #   * The objective and the required-reduction *floor* need an approximation
    #     that never over-promises. Claiming 40 points and delivering 38 is the
    #     failure mode that matters, so these use an inner (chord) approximation
    #     satisfying ``u <= f(x)``.
    #   * The risk *cap* needs an approximation that never under-reports, because
    #     a cap applied to an under-report is a cap on nothing. It uses an outer
    #     (tangent) approximation satisfying ``w >= f(x)``.
    #
    # The earlier design used one tangent variable for all three roles. That was
    # unsound in a way property testing exposed: the tangent variable is bounded
    # only from *above*, so it rises to its bound solely because the objective
    # pushes it there. Whenever risk carried little or no objective weight -- a
    # lead-time-focused mandate, for instance -- nothing pushed it, so the solver
    # was free to report a low risk figure while funding a portfolio whose true
    # reduction exceeded the baseline. The cap stopped binding exactly when the
    # objective stopped caring about risk, which is precisely when it was needed.
    risk_expr: dict[str, pl.LpAffineExpression] = {}
    cap_expr: dict[str, pl.LpAffineExpression] = {}
    linearisation_used = False
    for intervention in network:
        n = intervention.node_id
        if response.is_linear:
            coefficient = response.linear_coefficient(
                intervention.risk_reduction_pts, macro_of[n]
            )
            risk_expr[n] = coefficient * x[n]
            cap_expr[n] = coefficient * x[n]
            continue

        linearisation_used = True
        points = _breakpoints(
            intervention.min_funding_scale,
            intervention.max_funding_scale,
            request.tangent_points,
        )
        ceiling = response.evaluate(
            intervention.risk_reduction_pts, intervention.max_funding_scale, macro_of[n]
        )

        # -- inner approximation: objective and required-reduction floor ----- #
        u = pl.LpVariable(f"riskpts_{n}", lowBound=0.0, cat="Continuous")
        for k, chord in enumerate(
            _chords(response, intervention.risk_reduction_pts, macro_of[n], points)
        ):
            problem += (
                u <= chord.slope * x[n] + chord.intercept,
                f"risk_chord_{n}_{k}",
            )
        # An inactive node contributes nothing, and no active node may exceed its
        # own ceiling. Without these, the chords' positive intercepts -- produced
        # by extrapolating below the first breakpoint -- would let an unfunded
        # node claim value. This constraint is load-bearing, not redundant.
        problem += u <= ceiling * y[n], f"risk_ceiling_{n}"
        risk_expr[n] = u

        # -- outer approximation: risk cap only ------------------------------ #
        tangents = response.tangents(
            intervention.risk_reduction_pts, macro_of[n], breakpoints=points
        )
        w = pl.LpVariable(f"riskcap_{n}", lowBound=0.0, cat="Continuous")
        # Every tangent to a strictly concave function has a positive intercept,
        # so at x = 0 the tangents would force w above zero and an unfunded node
        # would consume cap headroom it cannot possibly use. The big-M term
        # releases each tangent when the node is switched off; M is the largest
        # intercept, which is exactly the slack required and no more.
        big_m = max(max((t.intercept for t in tangents), default=0.0), 0.0)
        for k, tangent in enumerate(tangents):
            problem += (
                w >= tangent.slope * x[n] + tangent.intercept - big_m * (1 - y[n]),
                f"risk_tangent_{n}_{k}",
            )
        cap_expr[n] = w

    total_risk_expr = pl.lpSum(risk_expr.values())
    total_cap_expr = pl.lpSum(cap_expr.values())

    # -- bundles ----------------------------------------------------------- #
    # A bundle discount is subtracted from the capital requirement, so if it can
    # be earned for less than it is worth the budget constraint stops meaning
    # anything. The acquired system tied the discount to node *activation*
    # (``b <= y``), which is the flaw: a node funded at its 30% minimum still
    # counted as activated, so three nodes bought at minimum scale could claim a
    # discount priced against the full package. Property testing found a network
    # that funded itself at a budget of zero.
    #
    # By default the discount is therefore contingent on the required nodes being
    # funded in full (``b <= x``, and x <= 1). That is also what a volume discount
    # actually is: the vendor rebates you for buying the whole package, not for
    # touching it. Combined with the domain rule that a discount must be strictly
    # less than its bundle's gross cost at full funding, realised spend on a
    # bundle always strictly exceeds the discount it earns.
    bundle_vars: dict[str, pl.LpVariable] = {}
    for index, bundle in enumerate(network.bundles):
        b_var = pl.LpVariable(f"bundle_{index}", cat="Binary")
        bundle_vars[bundle.name] = b_var
        for required in bundle.required_nodes:
            gate = y[required] if request.legacy_bundle_activation else x[required]
            problem += b_var <= gate, f"bundle_{index}_requires_{required}"

    total_discounts = (
        pl.lpSum(
            bundle_vars[bundle.name] * bundle.discount for bundle in network.bundles
        )
        if bundle_vars
        else pl.LpAffineExpression()
    )

    # -- dependencies ------------------------------------------------------ #
    for index, dependency in enumerate(network.dependencies):
        problem += (
            x[dependency.dependent] <= x[dependency.prerequisite],
            f"dependency_{index}_{dependency.dependent}_needs_{dependency.prerequisite}",
        )

    # -- objective --------------------------------------------------------- #
    total_capital = pl.lpSum(
        network[n].cost * x[n] for n in node_ids
    )
    terms = ProblemTerms(
        nodes={
            n: NodeTerms(
                node_id=n,
                risk_points=risk_expr[n],
                lead_time_days=network[n].lead_time_saved_days * x[n],
                carbon_tons=network[n].carbon_tons * x[n],
                capital=network[n].cost * x[n],
            )
            for n in node_ids
        },
        total_capital=total_capital,
        total_discounts=total_discounts,
    )
    problem += request.objective.build(terms), "objective"

    # -- scalar constraints ------------------------------------------------ #
    if request.budget is not None:
        problem += total_capital - total_discounts <= request.budget, "budget"
    if request.required_risk_reduction_pts is not None:
        problem += (
            total_risk_expr >= request.required_risk_reduction_pts,
            "required_risk_reduction",
        )
    if request.enforce_risk_cap:
        problem += total_cap_expr <= network.baseline_risk_pts, "risk_cap"

    # CBC's default primal feasibility tolerance is absolute, so on currency
    # coefficients of order 1e5-1e9 it can return a solution that overspends the
    # budget by a fraction of a cent and consider it feasible. That is correct
    # behaviour for a floating-point solver, but it means "the solver said
    # optimal" and "the constraints hold" are different claims. Tightening the
    # tolerances narrows the gap; the verifier below closes it by measuring the
    # residual slack explicitly rather than assuming it is zero.
    problem.solve(
        pl.PULP_CBC_CMD(
            msg=request.solver_msg,
            options=[
                "primalTolerance", "1e-9",
                "integerTolerance", "1e-9",
                "ratioGap", "0",
            ],
        )
    )
    status = _STATUS_MAP.get(problem.status, SolveStatus.UNDEFINED)

    # -- extract, recomputing every quantity from ground truth ------------- #
    scales: dict[str, float] = {}
    for n in node_ids:
        raw = pl.value(x[n])
        scale = 0.0 if raw is None else float(raw)
        if scale < request.tolerance:
            scale = 0.0
        scales[n] = scale

    allocations: list[NodeAllocation] = []
    raw_risk_reduction = 0.0
    lead_time_total = 0.0
    carbon_total = 0.0
    gross_capital = 0.0
    for intervention in network:
        n = intervention.node_id
        scale = scales[n]
        reduction = response.evaluate(intervention.risk_reduction_pts, scale, macro_of[n])
        capital = intervention.cost * scale
        lead_time = intervention.lead_time_saved_days * scale
        carbon = intervention.carbon_tons * scale

        raw_risk_reduction += reduction
        lead_time_total += lead_time
        carbon_total += carbon
        gross_capital += capital

        allocations.append(
            NodeAllocation(
                node_id=n,
                name=intervention.display_name,
                action=intervention.action,
                funding_scale=scale,
                capital=capital,
                risk_reduction_pts=reduction,
                lead_time_saved_days=lead_time,
                carbon_tons=carbon,
            )
        )

    active_bundles: list[str] = []
    discounts_applied = 0.0
    for bundle in network.bundles:
        value = pl.value(bundle_vars[bundle.name])
        if value is not None and value > 0.5:
            active_bundles.append(bundle.name)
            discounts_applied += bundle.discount

    objective_value = pl.value(problem.objective)

    # An infeasible status is the truthful solver answer and is reported unchanged.
    # What gets added is an account of *why*, because "infeasible" alone points a
    # reader toward more budget even when no budget would help. See diagnostics.py.
    diagnosis = None
    if status == SolveStatus.INFEASIBLE and request.diagnose:
        # Local import: diagnostics re-enters solve(), so importing it at module
        # scope would be circular.
        from .diagnostics import diagnose as _diagnose

        diagnosis = _diagnose(request)

    report = _verify(
        request=request,
        scales=scales,
        gross_capital=gross_capital,
        discounts=discounts_applied,
        active_bundles=tuple(active_bundles),
        raw_risk_reduction=raw_risk_reduction,
        solved=status == SolveStatus.OPTIMAL,
    )

    audit = build_audit_record(
        network=network,
        objective=request.objective,
        risk_response=response,
        parameters={
            "budget": request.budget,
            "required_risk_reduction_pts": request.required_risk_reduction_pts,
            "macro_multiplier": macro,
            "node_macro_multipliers": dict(request.node_macro_multipliers),
            "enforce_risk_cap": request.enforce_risk_cap,
            "tolerance": request.tolerance,
            "tangent_points": request.tangent_points,
            "relative_tolerance": request.relative_tolerance,
            "legacy_bundle_activation": request.legacy_bundle_activation,
        },
        solver="PULP_CBC_CMD",
        extra={
            "solver_status": status,
            "linearisation_used": linearisation_used,
            "constraint_check": report.summary(),
            "constraint_violations": [str(v) for v in report.violations],
        },
    )
    audit["output_hash"] = None  # filled below once the result exists

    result = OptimizationResult(
        status=status,
        allocations=tuple(allocations),
        baseline_risk_pts=network.baseline_risk_pts,
        gross_capital=gross_capital,
        bundle_discounts=discounts_applied,
        active_bundles=tuple(active_bundles),
        total_lead_time_saved_days=lead_time_total,
        total_carbon_tons=carbon_total,
        objective_value=None if objective_value is None else float(objective_value),
        objective_unit=request.objective.unit,
        constraint_report=report,
        diagnosis=diagnosis,
        effective_baseline_risk_pts=request.effective_baseline_risk_pts,
        audit=audit,
        raw_risk_reduction_pts=raw_risk_reduction,
    )

    from ..audit import content_hash

    audit["output_hash"] = content_hash(
        {
            "scales": scales,
            "gross_capital": gross_capital,
            "discounts": discounts_applied,
            "raw_risk_reduction_pts": raw_risk_reduction,
        }
    )
    return result


def _verify(
    *,
    request: OptimizationRequest,
    scales: dict[str, float],
    gross_capital: float,
    discounts: float,
    active_bundles: tuple[str, ...],
    raw_risk_reduction: float,
    solved: bool,
) -> ConstraintReport:
    """Independently re-check the solver's answer against the stated model."""
    network = request.network
    tol = request.tolerance
    violations: list[Violation] = []
    checks = [
        "minimum_economic_scale",
        "funding_scale_bounds",
        "dependency_cascade",
        "bundle_activation",
    ]

    if not solved:
        return ConstraintReport(
            violations=(
                Violation(
                    kind="solver_status",
                    detail=f"solver returned status {request.problem_name}: not optimal",
                    magnitude=0.0,
                ),
            ),
            tolerance=tol,
            checks_performed=tuple(checks),
        )

    for intervention in network:
        n = intervention.node_id
        scale = scales[n]
        if scale < -tol:
            violations.append(
                Violation("funding_scale_bounds", f"{n} funding scale is negative", -scale)
            )
        if scale > intervention.max_funding_scale + tol:
            violations.append(
                Violation(
                    "funding_scale_bounds",
                    f"{n} funding scale {scale:.6g} exceeds max "
                    f"{intervention.max_funding_scale:.6g}",
                    scale - intervention.max_funding_scale,
                )
            )
        if tol < scale < intervention.min_funding_scale - tol:
            violations.append(
                Violation(
                    "minimum_economic_scale",
                    f"{n} funded at {scale:.6g}, below minimum economic scale "
                    f"{intervention.min_funding_scale:.6g}",
                    intervention.min_funding_scale - scale,
                )
            )

    for dependency in network.dependencies:
        dep, pre = scales[dependency.dependent], scales[dependency.prerequisite]
        if dep > pre + tol:
            violations.append(
                Violation(
                    "dependency_cascade",
                    f"{dependency.dependent} funded at {dep:.6g} exceeds prerequisite "
                    f"{dependency.prerequisite} at {pre:.6g}",
                    dep - pre,
                )
            )

    for bundle in network.bundles:
        if bundle.name not in active_bundles:
            continue
        for required in bundle.required_nodes:
            if scales[required] <= tol:
                violations.append(
                    Violation(
                        "bundle_activation",
                        f"bundle {bundle.name!r} claimed its discount while required "
                        f"node {required} is unfunded",
                        bundle.discount,
                    )
                )

    slack: dict[str, float] = {}

    def scaled_tol(bound: float) -> float:
        """Slack allowed on a constraint whose bound has magnitude ``bound``."""
        return max(tol, abs(bound) * request.relative_tolerance)

    if request.budget is not None:
        checks.append("budget")
        net = gross_capital - discounts
        overshoot = net - request.budget
        if overshoot > 0:
            slack["budget"] = overshoot
        if net > request.budget + scaled_tol(request.budget):
            violations.append(
                Violation(
                    "budget",
                    f"net capital {net:.6g} exceeds budget {request.budget:.6g}",
                    net - request.budget,
                )
            )

    if request.required_risk_reduction_pts is not None:
        checks.append("required_risk_reduction")
        floor = request.required_risk_reduction_pts
        shortfall = floor - raw_risk_reduction
        if shortfall > 0:
            slack["required_risk_reduction"] = shortfall
        if raw_risk_reduction < floor - scaled_tol(floor):
            violations.append(
                Violation(
                    "required_risk_reduction",
                    f"risk reduction {raw_risk_reduction:.6g} falls short of required "
                    f"{floor:.6g}",
                    floor - raw_risk_reduction,
                )
            )

    if request.enforce_risk_cap:
        checks.append("risk_cap")
        cap = network.baseline_risk_pts
        cap_overshoot = raw_risk_reduction - cap
        if cap_overshoot > 0:
            slack["risk_cap"] = cap_overshoot
        if raw_risk_reduction > cap + scaled_tol(cap):
            violations.append(
                Violation(
                    "risk_cap",
                    f"risk reduction {raw_risk_reduction:.6g} exceeds baseline {cap:.6g}",
                    raw_risk_reduction - cap,
                )
            )

    return ConstraintReport(
        violations=tuple(violations),
        tolerance=tol,
        checks_performed=tuple(checks),
        residual_slack=slack,
    )
