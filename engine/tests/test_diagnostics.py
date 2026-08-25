"""Tests for the attainable frontier and infeasibility diagnosis.

The behaviour under test is a product decision as much as an engineering one: a
reader who asks for an unreachable target must be told the truth *and* pointed at
the right remedy. The acquired system did the first and got the second backwards,
because target mode set the budget to 1e9 and then surfaced a bare ``Infeasible``,
which reads as "ask for more money" when no amount of money would help.

So these tests assert two separable things:

1. The engine still reports infeasible. Nothing here softens the machine answer.
2. It also says whether more capital would help — and says *no* when the ceiling
   is structural.
"""

from __future__ import annotations

import pytest

from scrcae.domain import Dependency, Intervention, SupplyNetwork
from scrcae.optimization import (
    InfeasibilityKind,
    MinimizeCapitalObjective,
    OptimizationRequest,
    SolveStatus,
    attainable_frontier,
    diagnose,
    solve,
)
from scrcae.risk import LinearResponse, ParameterPowerResponse


def _network(baseline: float = 60.0) -> SupplyNetwork:
    """Three nodes, 10 points each, 100k each. Ceiling is 30 points for 300k."""
    return SupplyNetwork(
        baseline_risk_pts=baseline,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0),
            Intervention("B", cost=100_000.0, risk_reduction_pts=10.0),
            Intervention("C", cost=100_000.0, risk_reduction_pts=10.0),
        ),
    )


def _target_request(required: float, budget: float | None, **kw) -> OptimizationRequest:
    return OptimizationRequest(
        network=kw.pop("network", _network()),
        objective=MinimizeCapitalObjective(),
        risk_response=kw.pop("risk_response", LinearResponse()),
        required_risk_reduction_pts=required,
        budget=budget,
        **kw,
    )


# --------------------------------------------------------------------------- #
# The frontier itself
# --------------------------------------------------------------------------- #


def test_frontier_finds_the_structural_ceiling():
    report = attainable_frontier(_network(), LinearResponse())

    assert report.max_reduction_pts == pytest.approx(30.0, abs=1e-6)
    assert report.attainable_risk_pts == pytest.approx(30.0, abs=1e-6)
    assert report.capital_at_ceiling == pytest.approx(300_000.0, rel=1e-9)
    assert report.limited_by == "structure"
    assert not report.is_budget_limited


def test_frontier_reports_budget_as_the_binding_constraint_when_it_is():
    report = attainable_frontier(_network(), LinearResponse(), budget=150_000.0)

    assert report.max_reduction_pts == pytest.approx(15.0, abs=1e-6)
    assert report.capital_at_ceiling <= 150_000.0 + 1e-6
    assert report.limited_by == "budget"
    assert report.is_budget_limited


def test_frontier_respects_the_risk_cap():
    """A network that could over-deliver is clipped, and says so."""
    report = attainable_frontier(_network(baseline=20.0), LinearResponse())

    assert report.max_reduction_pts == pytest.approx(20.0, abs=1e-6)
    assert report.attainable_risk_pts == pytest.approx(0.0, abs=1e-6)
    assert report.limited_by == "risk_cap"


def test_frontier_respects_dependencies():
    """A prerequisite held to a low ceiling drags its dependents down with it.

    This is why the frontier has to be solved rather than computed as a sum over
    nodes at full scale. The parity suite's original helper did exactly that sum,
    and it would report 30 points here instead of the true 12.
    """
    network = SupplyNetwork(
        baseline_risk_pts=60.0,
        interventions=(
            Intervention("PRE", cost=100_000.0, risk_reduction_pts=10.0,
                         min_funding_scale=0.0, max_funding_scale=0.2),
            Intervention("DEP1", cost=100_000.0, risk_reduction_pts=10.0,
                         min_funding_scale=0.0),
            Intervention("DEP2", cost=100_000.0, risk_reduction_pts=10.0,
                         min_funding_scale=0.0),
        ),
        dependencies=(
            Dependency(dependent="DEP1", prerequisite="PRE"),
            Dependency(dependent="DEP2", prerequisite="PRE"),
        ),
    )
    report = attainable_frontier(network, LinearResponse())

    # 0.2 scale on all three, 10 pts each => 6.0, not 30.0.
    assert report.max_reduction_pts == pytest.approx(6.0, abs=1e-6)
    assert report.limited_by == "structure"


def test_frontier_never_overstates_under_a_concave_response():
    """Conservative by construction: the inner approximation may understate only.

    A ceiling that overstates what is reachable would let the engine tell someone a
    target is achievable when it is not, which is the one error this feature exists
    to prevent.
    """
    response = ParameterPowerResponse(exponent=0.85)
    network = _network()
    report = attainable_frontier(network, response)

    true_ceiling = sum(
        response.evaluate(node.risk_reduction_pts, node.max_funding_scale, 1.0)
        for node in network
    )
    assert report.max_reduction_pts <= true_ceiling + 1e-9


# --------------------------------------------------------------------------- #
# Structural vs budget: the distinction that matters
# --------------------------------------------------------------------------- #


def test_unreachable_target_is_diagnosed_as_structural_not_budgetary():
    """The acquired system's central confusion, now impossible to reproduce."""
    # 45 points wanted, 30 available, and a budget so large it cannot be the issue.
    result = solve(_target_request(required=45.0, budget=1e9))

    assert result.status == SolveStatus.INFEASIBLE  # the machine answer is unchanged
    assert result.diagnosis is not None
    assert result.diagnosis.kind == InfeasibilityKind.EXCEEDS_STRUCTURAL_CEILING
    assert result.diagnosis.more_capital_would_help is False
    assert result.diagnosis.shortfall_pts == pytest.approx(15.0, abs=1e-6)

    # The remedy must actively steer away from the budget inference.
    assert "not a budget constraint" in result.diagnosis.remedy
    # And it must quote the attainable floor, which is the actionable number.
    assert "30.0" in result.diagnosis.explanation


def test_affordable_target_beyond_budget_is_diagnosed_as_budgetary():
    result = solve(_target_request(required=20.0, budget=50_000.0))

    assert result.status == SolveStatus.INFEASIBLE
    assert result.diagnosis.kind == InfeasibilityKind.EXCEEDS_BUDGET
    assert result.diagnosis.more_capital_would_help is True
    # Structurally reachable, so no points are missing -- only money.
    assert result.diagnosis.shortfall_pts == pytest.approx(0.0, abs=1e-6)
    assert result.diagnosis.capital_required == pytest.approx(200_000.0, rel=1e-6)


def test_the_two_kinds_are_not_confusable():
    """Same engine, same target, opposite diagnoses depending on what actually bit."""
    structural = solve(_target_request(required=45.0, budget=1e9)).diagnosis
    budgetary = solve(_target_request(required=20.0, budget=50_000.0)).diagnosis

    assert structural.kind != budgetary.kind
    assert structural.more_capital_would_help != budgetary.more_capital_would_help
    assert structural.capital_required is None
    assert budgetary.capital_required is not None


def test_requesting_more_reduction_than_baseline_is_its_own_diagnosis():
    result = solve(
        _target_request(required=80.0, budget=1e9, enforce_risk_cap=True)
    )
    assert result.status == SolveStatus.INFEASIBLE
    assert result.diagnosis.kind == InfeasibilityKind.EXCEEDS_RISK_CAP
    assert "cannot fall below zero" in result.diagnosis.explanation


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #


def test_the_target_is_never_silently_relaxed():
    """The worst outcome is a reader who thinks the target was met. Not possible."""
    result = solve(_target_request(required=45.0, budget=1e9))

    assert result.status == SolveStatus.INFEASIBLE
    assert result.active_allocations == ()
    assert result.diagnosis.requested_reduction_pts == pytest.approx(45.0)
    # The attainable figure is reported *alongside* the request, never in place of it.
    assert result.diagnosis.attainable.max_reduction_pts < 45.0
    assert "short by" in result.diagnosis.explanation


def test_the_word_infeasible_is_never_shown_to_a_reader():
    """`status` keeps the solver term; `explanation` must not leak it."""
    result = solve(_target_request(required=45.0, budget=1e9))

    assert result.status == SolveStatus.INFEASIBLE
    assert "infeasible" not in result.explanation.lower()


def test_a_feasible_request_carries_no_diagnosis_and_explains_itself():
    result = solve(_target_request(required=20.0, budget=1e9))

    assert result.status == SolveStatus.OPTIMAL
    assert result.diagnosis is None
    assert "Risk 60.0 ->" in result.explanation


def test_diagnose_is_safe_to_call_on_a_satisfiable_request():
    report = diagnose(_target_request(required=20.0, budget=1e9))
    assert report.kind == InfeasibilityKind.NONE


def test_diagnosis_can_be_switched_off():
    """It costs extra solves, so callers in a hot loop can decline it."""
    result = solve(_target_request(required=45.0, budget=1e9, diagnose=False))
    assert result.status == SolveStatus.INFEASIBLE
    assert result.diagnosis is None


def test_diagnosis_does_not_change_the_reported_status_or_allocations():
    with_diag = solve(_target_request(required=45.0, budget=1e9, diagnose=True))
    without = solve(_target_request(required=45.0, budget=1e9, diagnose=False))

    assert with_diag.status == without.status
    assert with_diag.scales() == without.scales()
    assert with_diag.net_capital == pytest.approx(without.net_capital)


# --------------------------------------------------------------------------- #
# F18: the effective baseline
# --------------------------------------------------------------------------- #


def test_percentages_are_quoted_against_the_baseline_the_target_was_set_from():
    """A target typed as 40% must be described back as 40%, not 35.9%.

    The acquired system inflated the baseline by the macro multiplier and derived
    target mode from the inflated figure, while its in-model constraints used the raw
    one. Reporting against the raw baseline understates residual risk, which is the
    wrong direction to be wrong in when someone is deciding whether to act.
    """
    macro = 1.0625  # Brent at $85 in the acquired system's formula
    network = _network(baseline=60.0)
    effective = 60.0 * macro  # 63.75
    required = effective - 40.0

    result = solve(
        _target_request(
            required=required, budget=1e9, network=network, macro_multiplier=macro
        )
    )
    assert result.status == SolveStatus.OPTIMAL
    assert result.reporting_baseline_risk_pts == pytest.approx(effective, abs=1e-9)
    assert result.effective_optimized_risk_pts == pytest.approx(40.0, abs=1e-6)
    assert "40.0 pts" in result.explanation

    # The raw figures are preserved rather than overwritten, because the in-model
    # risk cap genuinely constrains the raw baseline.
    assert result.baseline_risk_pts == pytest.approx(60.0)
    assert result.optimized_risk_pts < result.effective_optimized_risk_pts


def test_effective_baseline_defaults_to_the_raw_baseline():
    result = solve(_target_request(required=20.0, budget=1e9))
    assert result.reporting_baseline_risk_pts == pytest.approx(60.0)
    assert result.effective_optimized_risk_pts == pytest.approx(
        result.optimized_risk_pts
    )


def test_effective_baseline_is_clamped_at_100():
    request = _target_request(
        required=1.0, budget=1e9, network=_network(baseline=95.0), macro_multiplier=2.0
    )
    assert request.effective_baseline_risk_pts == pytest.approx(100.0)
