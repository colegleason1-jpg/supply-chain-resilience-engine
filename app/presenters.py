"""Turning engine results into things a person can read.

Everything here is formatting. No arithmetic on model quantities happens in this
module: if a number needs computing it is computed in the engine and read from the
result, because the acquired system's headline figures and its sweep chart were
computed by two different copies of the same arithmetic and had already drifted apart
(F6).

The one job this module takes seriously beyond formatting is *not lying*. The
acquired system's audit tab printed "Zero fractional violations detected" and
"Verification Hash: sha256:8f4c99a..." as string literals — nothing computed either
(F4). The equivalents here read from `result.constraint_report` and `result.audit`,
so if the engine has not verified something, the UI cannot claim it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scrcae.optimization import BudgetSweep, SolveStatus


def money(value: float | None, symbol: str = "$") -> str:
    if value is None:
        return "—"
    return f"{symbol}{value:,.0f}"


def points(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f} pts"


def percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}%"


def scale(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.0f}%"


# --------------------------------------------------------------------------- #
# Headline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Metric:
    label: str
    value: str
    help: str = ""
    delta: str | None = None


def headline_metrics(result, macro_multiplier: float = 1.0) -> list[Metric]:
    """The four numbers at the top of the screen.

    Risk figures use the *reporting* baseline, i.e. after the macro multiplier, which
    is the baseline the user's target was set against. Quoting the raw baseline here
    understates residual risk (ADR-001 F18).
    """
    baseline = result.reporting_baseline_risk_pts
    macro_note = (
        f"{macro_multiplier:g}x macro applied" if macro_multiplier != 1.0 else "no macro adjustment"
    )

    if not result.is_solved:
        # No plan exists, so no plan is displayed. Showing 0 capital and the baseline
        # risk would read as a legitimate "do nothing" recommendation.
        return [
            Metric("Baseline risk", percent(baseline), macro_note),
            Metric("Optimised risk", "—", "No allocation was produced"),
            Metric("Capital deployed", "—", "No allocation was produced"),
            Metric("Lead time saved", "—", "No allocation was produced"),
        ]

    return [
        Metric("Baseline risk", percent(baseline), macro_note),
        Metric(
            "Optimised risk",
            percent(result.effective_optimized_risk_pts),
            "Deterministic outcome, recomputed from the risk response",
            delta=f"-{result.risk_reduction_pts:.2f} pts",
        ),
        Metric(
            "Capital deployed",
            money(result.net_capital),
            f"Gross {money(result.gross_capital)} less "
            f"{money(result.bundle_discounts)} of bundle discounts",
        ),
        Metric(
            "Lead time saved",
            f"{result.total_lead_time_saved_days:.1f} days",
            "Sum over funded interventions at their chosen scale",
        ),
    ]


def simulation_metrics(simulation) -> list[Metric]:
    if simulation is None:
        return []
    return [
        Metric(
            f"P50 risk ({simulation.iterations:,} runs)",
            percent(simulation.p50_risk_pts),
            f"Standard error {simulation.p50_standard_error:.3f} pts",
        ),
        Metric(
            "P90 tail risk",
            percent(simulation.p90_risk_pts),
            f"Standard error {simulation.p90_standard_error:.3f} pts",
        ),
        Metric(
            "Shock bias vs deterministic",
            f"{simulation.reduction_bias_pts:+.3f} pts",
            "Mean delivered reduction less the deterministic figure. Should be near "
            "zero for an unbiased shock model; positive means the shock model is "
            "flattering the plan.",
        ),
    ]


def portfolio_rows(result) -> list[dict[str, Any]]:
    """One row per funded intervention.

    Empty when the solve failed, because `active_allocations` is empty then — the
    solver leaves constraint-violating values in its variables after reporting
    infeasible, and they read as a fully funded portfolio.
    """
    rows = []
    for rank, alloc in enumerate(
        sorted(result.active_allocations, key=lambda a: -a.capital), start=1
    ):
        rows.append(
            {
                "Rank": rank,
                "Node": alloc.name or alloc.node_id,
                "Node ID": alloc.node_id,
                "Action": alloc.action,
                "Funding": scale(alloc.funding_scale),
                "Capital": money(alloc.capital),
                "Risk reduction": points(alloc.risk_reduction_pts),
                "Lead time saved": f"{alloc.lead_time_saved_days:.1f} d",
                "Carbon": f"{alloc.carbon_tons:.1f} t",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Status and diagnosis
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StatusBanner:
    """What to tell the user about the solve, and how loudly.

    ``level`` is one of ``"success"``, ``"warning"``, ``"error"``, ``"info"``.
    """

    level: str
    headline: str
    detail: str = ""
    remedy: str = ""


def status_banner(result) -> StatusBanner:
    """The single most important string in the app.

    Target mode in the acquired system ran with the budget pinned to 1e9 and then
    printed the solver's status. An unreachable target therefore surfaced as
    ``Infeasible`` with an effectively unlimited budget on screen, and every reader
    drew the same wrong conclusion: ask for more money. On the shipped default no
    amount of money would have helped.

    So this never shows the solver's word. It shows the engine's diagnosis, which
    distinguishes "no budget achieves this" from "this budget does not achieve this",
    and it says which one it is explicitly.
    """
    if result.is_solved:
        return StatusBanner(
            level="success",
            headline="Target met." if result.constraint_report.is_feasible else "Solved.",
            detail=result.explanation,
        )

    diagnosis = result.diagnosis
    if diagnosis is None:
        return StatusBanner(
            level="error",
            headline="No allocation could be produced.",
            detail=result.explanation,
            remedy="This is unexpected. Please report it rather than working around it.",
        )

    # A budget shortfall is actionable and ordinary; a structural ceiling is a
    # different conversation and is styled to stop the reader rather than nudge them.
    level = "warning" if diagnosis.more_capital_would_help else "error"
    headline = (
        "Target not reachable within this budget."
        if diagnosis.more_capital_would_help
        else "Target not reachable with this intervention portfolio."
    )
    return StatusBanner(
        level=level,
        headline=headline,
        detail=diagnosis.explanation,
        remedy=diagnosis.remedy,
    )


def frontier_rows(report) -> list[dict[str, Any]]:
    """The attainable frontier as a small table."""
    if report is None:
        return []
    limited_by = {
        "structure": "The portfolio itself — every intervention is at its cap",
        "budget": "The budget",
        "risk_cap": "The risk cap — reduction is clipped at the baseline",
    }.get(report.limited_by, report.limited_by)
    return [
        {"Quantity": "Lowest attainable risk", "Value": percent(report.attainable_risk_pts)},
        {"Quantity": "Maximum reduction", "Value": points(report.max_reduction_pts)},
        {"Quantity": "Capital at that ceiling", "Value": money(report.capital_at_ceiling)},
        {"Quantity": "Limited by", "Value": limited_by},
    ]


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


def sweep_rows(sweep: BudgetSweep) -> list[dict[str, Any]]:
    rows = []
    for point in sweep.points:
        rows.append(
            {
                "Budget": money(point.budget),
                "Spent": money(point.net_capital) if point.is_solved else "—",
                "Unspent": money(point.capital_unspent) if point.is_solved else "—",
                "Risk": percent(point.optimized_risk_pts) if point.is_solved else "—",
                "Reduction": points(point.risk_reduction_pts) if point.is_solved else "—",
                "Funded": len(point.active_node_ids) if point.is_solved else 0,
                "Status": "solved" if point.is_solved else point.status,
            }
        )
    return rows


def sweep_note(sweep: BudgetSweep) -> str:
    """Say what the curve means, because a flat line is easy to misread.

    A budget curve that has gone flat looks like a well-tuned budget. It usually means
    the opposite: the portfolio has run out of things worth buying, and the extra
    authorisation is unspendable.
    """
    if not sweep.solved_points:
        return "No budget level in this range produced a feasible allocation."
    saturation = sweep.saturation_budget
    if sweep.is_saturated and saturation is not None:
        return (
            f"Reduction stops improving above {money(saturation)}. Authorising more "
            f"than that buys nothing: the constraint is the portfolio, not the budget."
        )
    return (
        "Reduction is still improving at the top of this range, so the budget is "
        "the binding constraint here — a larger authorisation would still buy risk."
    )


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def verification_rows(result, request=None) -> list[dict[str, Any]]:
    """The real constraint check, not a sentence that says one happened.

    ``checks_performed`` and ``violations`` come from the engine's own post-solve
    verification pass. If it did not run, this table is empty and the UI says so
    rather than printing reassurance.
    """
    report = result.constraint_report
    families = tuple(report.checks_performed)
    rows = [
        {"Check": "Constraint families verified", "Result": str(len(families))},
        {
            "Check": "Families",
            "Result": ", ".join(name.replace("_", " ") for name in families) or "none",
        },
        {"Check": "Violations found", "Result": str(len(report.violations))},
        {"Check": "Absolute tolerance", "Result": f"{report.tolerance:g}"},
        {
            "Check": "Largest residual slack",
            "Result": f"{report.max_residual_slack:.3g}",
        },
    ]
    # Money constraints are checked against a scale-relative tolerance, because an
    # absolute 1e-6 sits below CBC's own feasibility slack on a six-figure budget and
    # would report numerical noise as a modelling error (F15). Showing only the
    # absolute figure would misrepresent what was actually verified.
    if request is not None and request.budget:
        allowed = max(report.tolerance, abs(request.budget) * request.relative_tolerance)
        rows.append(
            {
                "Check": "Slack allowed on the budget",
                "Result": f"{allowed:.3g} (relative tolerance {request.relative_tolerance:g})",
            }
        )
    for violation in report.violations:
        rows.append({"Check": "VIOLATION", "Result": str(violation)})
    return rows


def _describe_repair(repair) -> str:
    """Describe what had to be done to make the correlation matrix usable.

    Worth surfacing rather than hiding: the acquired system's ridge repair added a
    constant to the diagonal and never renormalised, so a "repaired" matrix no longer
    had unit variances and every simulated node was quietly rescaled (F9).
    """
    if repair is None:
        return "none required"
    summary = getattr(repair, "summary", None)
    return summary() if callable(summary) else str(repair)


def _elasticity_provenance(exposures) -> str:
    """How many elasticities were fitted and how many are still typed in.

    Reads as "2 calibrated, 1 asserted", counted from the exposures themselves.

    Counted rather than claimed. ``NodeExposure.elasticity_source`` is derived by
    matching the entered elasticity and anchor against a stored fit, so an exposure
    can only be counted as calibrated while the number in the table is still the
    number that was fitted. Overtype it and it moves back into the asserted count on
    the next run, which is the honest direction for an edit to push a label.
    """
    exposures = tuple(exposures)
    if not exposures:
        return "none configured"
    calibrated = sum(1 for exposure in exposures if exposure.is_calibrated)
    asserted = len(exposures) - calibrated
    parts = []
    if calibrated:
        parts.append(f"{calibrated} calibrated")
    if asserted:
        parts.append(f"{asserted} asserted")
    return ", ".join(parts)


def provenance_rows(
    result, *, request=None, simulation=None, exposures=None
) -> list[dict[str, Any]]:
    """Model versions, hashes and calibration provenance, read from the audit record.

    Every uncalibrated constant in the engine carries a ``calibration_source``, so an
    assumption shows up here as an assumption instead of passing for data.

    ``exposures`` are the market exposures as resolved against the stored fits. Left
    out, no elasticity row is written at all rather than a reassuring one: a caller
    that did not supply the exposures has said nothing about their provenance, and
    inventing "asserted" or "calibrated" on its behalf would be a claim about
    evidence this function has not seen.
    """
    audit = result.audit or {}
    rows: list[dict[str, Any]] = [
        {"Item": "Engine version", "Value": str(audit.get("engine_version", "—"))},
        {"Item": "Objective", "Value": str(audit.get("objective_version", "—"))},
        {"Item": "Objective unit", "Value": result.objective_unit},
        {"Item": "Risk response", "Value": str(audit.get("risk_response_version", "—"))},
        {"Item": "Solver", "Value": str(audit.get("solver", "—"))},
        {"Item": "Solver status", "Value": result.status},
        {"Item": "Linearisation used", "Value": str(audit.get("linearisation_used", "—"))},
        {"Item": "Run at (UTC)", "Value": str(audit.get("generated_at", "—"))},
        {"Item": "Network hash", "Value": str(audit.get("network_hash", "—"))},
        {"Item": "Input hash", "Value": str(audit.get("input_hash", "—"))},
        {"Item": "Output hash", "Value": str(audit.get("output_hash", "—"))},
    ]
    if request is not None:
        response = request.risk_response
        source = getattr(response, "calibration_source", None)
        if source:
            rows.append({"Item": "Risk response calibration", "Value": source})
        prices = getattr(request.objective, "prices", None)
        if prices is not None:
            # The full derivation string, not a yes/no. An auditor's question is not
            # "is there a source" but "what was the arithmetic", and for a derived
            # price book the source line contains it.
            rows.append({"Item": "Price of risk", "Value": str(prices.source)})
        exposure = dict(getattr(request, "node_macro_multipliers", {}) or {})
        rows.append(
            {
                "Item": "Market exposure applied",
                "Value": (
                    ", ".join(
                        f"{node} x{value:.4f}"
                        for node, value in sorted(exposure.items())
                    )
                    if exposure
                    else "none"
                ),
            }
        )
    if exposures is not None:
        rows.append(
            {"Item": "Risk elasticities", "Value": _elasticity_provenance(exposures)}
        )
    if simulation is not None:
        rows.extend(
            [
                {"Item": "Shock model", "Value": simulation.shock_version},
                {"Item": "Simulation seed", "Value": str(simulation.seed)},
                {
                    # `correlation_repair` is a RepairReport, not a string. Arrow cannot
                    # serialise the object and silently degrades the whole column, so
                    # every cell in these tables is stringified at the boundary.
                    "Item": "Correlation repair",
                    "Value": _describe_repair(simulation.correlation_repair),
                },
            ]
        )
    return rows


def uncalibrated_warnings(request, simulation=None, *, exposures=None) -> list[str]:
    """Name every assumption in play, so none of them passes for a measurement.

    ``exposures`` narrows the market-exposure warning to the elasticities that are
    still asserted, and names them. Warning about a fully calibrated set would be the
    mirror image of the defect this whole path exists to prevent: a disclosure that
    fires whatever the evidence says is one a reader learns to skip, and then the
    genuine assertion in the twentieth row goes unread with it.

    Omitted, the warning keeps its unconditional form. A caller that supplies no
    exposures has supplied no evidence of calibration either, and asserted is the
    safe direction to fail.
    """
    warnings: list[str] = []
    response = request.risk_response
    if getattr(response, "calibration_source", "") == "uncalibrated-assumption":
        exponent = getattr(response, "exponent", None)
        warnings.append(
            f"The risk response exponent ({exponent}) is inherited from the acquired "
            "system and has no stated empirical basis. It shapes how funding converts "
            "to risk reduction and is not measured."
        )
    objective = request.objective
    prices = getattr(objective, "prices", None)
    if prices is not None and not getattr(prices, "is_sourced", False):
        warnings.append(
            "The price book has no stated source. Every currency figure below is only "
            "as defensible as the value per risk point that produced it."
        )
    if dict(getattr(request, "node_macro_multipliers", {}) or {}):
        asserted = [e for e in (exposures or ()) if not e.is_calibrated]
        if exposures is None:
            warnings.append(
                "Market exposure is adjusting intervention effectiveness. The "
                "elasticities that scale those adjustments are stated assumptions, "
                "not measured sensitivities, and the direction is one-sided by "
                "choice: prices above anchor raise risk, prices below it are not "
                "credited as reducing it."
            )
        elif asserted:
            named = ", ".join(f"{e.node_id}/{e.symbol}" for e in asserted)
            warnings.append(
                "Market exposure is adjusting intervention effectiveness on "
                f"elasticities that are still asserted rather than fitted: {named}. "
                "These are stated assumptions, not measured sensitivities, and the "
                "direction is one-sided by choice: prices above anchor raise risk, "
                "prices below it are not credited as reducing it."
            )
    if simulation is not None and "legacy" in (simulation.shock_version or ""):
        warnings.append(
            "The legacy shock model truncates multipliers at 0.4, which biases the "
            "mean outcome upward. Reported P50 and P90 figures are optimistic by "
            "construction in this mode."
        )
    if request.macro_multiplier != 1.0:
        warnings.append(
            f"A macro multiplier of {request.macro_multiplier:g} is inflating every "
            "node's risk reduction. This figure comes from a commodity price feed and "
            "is a heuristic, not a calibrated relationship."
        )
    return warnings
