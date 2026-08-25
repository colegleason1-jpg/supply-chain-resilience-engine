"""Tests for what the app shows, with emphasis on what it must never show.

Two acquired-system habits are pinned here as prohibitions:

* The audit tab asserted verification that nothing had performed (F4). The
  presenters may only report what the engine actually computed.
* Target mode surfaced the solver's `Infeasible` next to a 1e9 budget, so an
  unreachable target read as a funding shortfall. The banner must never show that
  word, and must say explicitly whether capital is the constraint.
"""

from __future__ import annotations

import pytest

from app.adapters import (
    MODE_LEGACY,
    MODE_MONETARY,
    build_network,
    build_optimization_request,
    default_nodes_frame,
)
from app.engine_client import analyse, run_sweep
from app.presenters import (
    frontier_rows,
    headline_metrics,
    portfolio_rows,
    provenance_rows,
    status_banner,
    sweep_note,
    sweep_rows,
    uncalibrated_warnings,
    verification_rows,
)
from scrcae import PriceBook

PRICES = PriceBook(value_per_risk_point=50_000.0, source="test fixture")


def _request(**kw):
    network, rejections = build_network(default_nodes_frame(), kw.pop("baseline", 65.5))
    assert not rejections
    kw.setdefault("mode", MODE_MONETARY)
    kw.setdefault("prices", PRICES)
    kw.setdefault("budget", 750_000.0)
    return build_optimization_request(network, **kw)


# --------------------------------------------------------------------------- #
# The banner
# --------------------------------------------------------------------------- #


def test_solved_run_reports_success_with_real_numbers():
    bundle = analyse(_request(), include_sweep=False)
    banner = status_banner(bundle.result)

    assert banner.level == "success"
    assert "Risk" in banner.detail


def test_unreachable_target_never_shows_the_word_infeasible():
    """The single most consequential string in the app."""
    bundle = analyse(_request(budget=None, target_risk_pts=1.0), include_sweep=False)
    banner = status_banner(bundle.result)

    assert not bundle.result.is_solved
    assert "infeasible" not in banner.headline.lower()
    assert "infeasible" not in banner.detail.lower()
    assert "infeasible" not in banner.remedy.lower()


def test_structural_ceiling_says_capital_will_not_help():
    """The acquired system pointed the reader at the wrong remedy here."""
    bundle = analyse(_request(budget=None, target_risk_pts=1.0), include_sweep=False)
    banner = status_banner(bundle.result)

    assert banner.level == "error"
    assert "intervention portfolio" in banner.headline
    assert "not a budget constraint" in banner.remedy


def test_budget_shortfall_says_capital_will_help_and_names_the_figure():
    bundle = analyse(
        _request(budget=50_000.0, target_risk_pts=45.0), include_sweep=False
    )
    banner = status_banner(bundle.result)

    assert bundle.result.diagnosis.more_capital_would_help
    assert banner.level == "warning"
    assert "within this budget" in banner.headline
    assert bundle.result.diagnosis.capital_required is not None


def test_the_two_failure_modes_are_visibly_different():
    structural = status_banner(
        analyse(_request(budget=None, target_risk_pts=1.0), include_sweep=False).result
    )
    budgetary = status_banner(
        analyse(_request(budget=50_000.0, target_risk_pts=45.0), include_sweep=False).result
    )

    assert structural.level != budgetary.level
    assert structural.headline != budgetary.headline


# --------------------------------------------------------------------------- #
# Headline metrics
# --------------------------------------------------------------------------- #


def test_failed_solve_shows_dashes_not_zeros():
    """Zero capital and baseline risk would read as a considered 'do nothing'."""
    bundle = analyse(_request(budget=None, target_risk_pts=1.0), include_sweep=False)
    metrics = {m.label: m.value for m in headline_metrics(bundle.result)}

    assert metrics["Capital deployed"] == "—"
    assert metrics["Optimised risk"] == "—"


def test_risk_is_quoted_against_the_macro_inflated_baseline():
    """ADR-001 F18: the raw baseline understates residual risk."""
    macro = 1.0625
    bundle = analyse(_request(macro_multiplier=macro), include_sweep=False)
    metrics = {m.label: m.value for m in headline_metrics(bundle.result, macro)}

    assert metrics["Baseline risk"] == f"{65.5 * macro:.2f}%"
    assert "macro applied" in next(
        m.help for m in headline_metrics(bundle.result, macro) if m.label == "Baseline risk"
    )


def test_portfolio_rows_are_empty_for_a_failed_solve():
    """CBC leaves constraint-violating values in its variables after infeasibility."""
    bundle = analyse(_request(budget=None, target_risk_pts=1.0), include_sweep=False)
    assert portfolio_rows(bundle.result) == []


def test_portfolio_rows_are_ordered_by_capital():
    bundle = analyse(_request(), include_sweep=False)
    rows = portfolio_rows(bundle.result)

    assert rows
    assert [r["Rank"] for r in rows] == list(range(1, len(rows) + 1))


# --------------------------------------------------------------------------- #
# Verification: F4
# --------------------------------------------------------------------------- #


def test_verification_reports_the_engines_own_check_not_a_slogan():
    """F4. 'Zero fractional violations detected' used to be a string literal."""
    bundle = analyse(_request(), include_sweep=False)
    rows = {r["Check"]: r["Result"] for r in verification_rows(bundle.result)}

    assert int(rows["Constraint families verified"]) > 0
    assert rows["Violations found"] == "0"
    assert int(rows["Constraint families verified"]) == len(
        bundle.result.constraint_report.checks_performed
    )
    # The families are named, so the claim is checkable rather than atmospheric.
    assert "budget" in rows["Families"]
    assert "risk cap" in rows["Families"]


def test_hashes_come_from_the_audit_record():
    """F4. 'sha256:8f4c99a...' used to be a literal, truncated to look plausible."""
    bundle = analyse(_request(), include_sweep=False)
    rows = {r["Item"]: r["Value"] for r in provenance_rows(bundle.result)}

    assert rows["Input hash"].startswith("sha256:")
    assert rows["Output hash"].startswith("sha256:")
    assert rows["Input hash"] != rows["Output hash"]
    assert len(rows["Input hash"]) > 20  # not a decorative ellipsis


def test_provenance_names_the_objective_and_its_unit():
    bundle = analyse(_request(), include_sweep=False)
    rows = {r["Item"]: r["Value"] for r in provenance_rows(bundle.result)}

    assert "monetary-npv" in rows["Objective"]
    assert "currency" in rows["Objective unit"]


def test_legacy_mode_is_self_identifying_in_the_audit():
    """A legacy run must be impossible to mistake for a defensible one."""
    bundle = analyse(_request(mode=MODE_LEGACY), include_sweep=False)
    rows = {r["Item"]: r["Value"] for r in provenance_rows(bundle.result)}

    assert "legacy" in rows["Objective"]
    assert rows["Objective unit"].startswith("incommensurate")


# --------------------------------------------------------------------------- #
# Assumptions are named
# --------------------------------------------------------------------------- #


def test_uncalibrated_exponent_is_disclosed():
    request = _request()
    warnings = uncalibrated_warnings(request)
    assert any("no stated empirical basis" in w for w in warnings)


def test_unstated_price_source_is_disclosed():
    request = _request(prices=PriceBook(value_per_risk_point=50_000.0))
    assert any("no stated source" in w for w in uncalibrated_warnings(request))


def test_stated_price_source_is_not_flagged():
    request = _request(prices=PRICES)
    assert not any("no stated source" in w for w in uncalibrated_warnings(request))


def test_macro_multiplier_is_disclosed_as_a_heuristic():
    request = _request(macro_multiplier=1.0625)
    assert any("heuristic" in w for w in uncalibrated_warnings(request))


# --------------------------------------------------------------------------- #
# Sweep: F6
# --------------------------------------------------------------------------- #


def test_sweep_and_headline_agree_because_they_share_one_optimizer():
    """F6. The acquired system's sweep was a second, drifted copy of the optimizer.

    Its headline clamped risk reduction at the baseline and its sweep did not, so the
    chart and the number above it came from different models.
    """
    request = _request(budget=400_000.0)
    bundle = analyse(request, include_sweep=False)
    sweep = run_sweep(request, reference=400_000.0, steps=5)

    at_same_budget = [p for p in sweep.points if abs(p.budget - 400_000.0) < 1.0]
    assert at_same_budget, "the sweep should include its own reference budget"
    point = at_same_budget[0]

    assert point.net_capital == pytest.approx(bundle.result.net_capital, rel=1e-6)
    assert point.risk_reduction_pts == pytest.approx(
        bundle.result.risk_reduction_pts, rel=1e-6
    )
    assert point.optimized_risk_pts == pytest.approx(
        bundle.result.effective_optimized_risk_pts, rel=1e-6
    )


def test_sweep_keeps_infeasible_levels_rather_than_hiding_them():
    """A curve that omits the budgets where nothing works reads as though all work."""
    request = _request(budget=None, target_risk_pts=45.0)
    sweep = run_sweep(request, reference=200_000.0, steps=7)

    assert any(not p.is_solved for p in sweep.points)
    assert len(sweep_rows(sweep)) == len(sweep.points)
    assert any(r["Status"] != "solved" for r in sweep_rows(sweep))


def test_saturation_is_identified_and_explained():
    """A flat curve means the portfolio ran out of things to buy, not a tuned budget."""
    request = _request(budget=5_000_000.0)
    sweep = run_sweep(request, reference=5_000_000.0, steps=7)

    assert sweep.is_saturated
    note = sweep_note(sweep)
    assert "buys nothing" in note
    assert "the portfolio, not the budget" in note


def test_unsaturated_sweep_says_the_budget_is_binding():
    request = _request(budget=200_000.0)
    sweep = run_sweep(request, reference=200_000.0, steps=5)

    if not sweep.is_saturated:
        assert "budget is" in sweep_note(sweep)


def test_sweep_reports_unspent_capital():
    request = _request(budget=5_000_000.0)
    sweep = run_sweep(request, reference=5_000_000.0, steps=5)
    top = sweep.points[-1]
    assert top.capital_unspent > 0


def test_analyse_sweeps_around_actual_spend_in_target_mode():
    """Sweeping around a 1e9 placeholder budget would give nine identical rows."""
    bundle = analyse(
        _request(budget=1e9, target_risk_pts=45.0), include_sweep=True, sweep_steps=5
    )
    assert bundle.sweep is not None
    budgets = [p.budget for p in bundle.sweep.points]
    assert max(budgets) < 1e8


# --------------------------------------------------------------------------- #
# Frontier presentation
# --------------------------------------------------------------------------- #


def test_frontier_rows_explain_what_the_limit_is():
    bundle = analyse(_request(budget=None, target_risk_pts=1.0), include_sweep=False)
    rows = {r["Quantity"]: r["Value"] for r in frontier_rows(bundle.frontier)}

    assert "Lowest attainable risk" in rows
    assert "every intervention is at its cap" in rows["Limited by"]


def test_frontier_is_computed_on_success_in_target_mode():
    """'You met your target' and 'here is what was available' are different facts."""
    bundle = analyse(_request(budget=None, target_risk_pts=50.0), include_sweep=False)
    assert bundle.result.is_solved
    assert bundle.frontier is not None
