"""Per-node macro exposure (F11).

The acquired system had a `MarketFeedbackBridge` that matched a market's sector
name against a node's *name* by exact string equality, and — when the column it
matched on was absent — fell back to a mask of `[True] * len(nodes)`, applying one
market's volatility to the entire network. It then wrote the adjusted values back
into the user's own input table, so every 30-minute sync compounded on the last.

The replacement is an explicit, validated mapping from node id to multiplier.
These tests pin the three properties that make it safe to put a live feed behind:
it cannot silently apply to nodes nobody named, it cannot move the baseline, and
it reaches every solve the engine performs rather than only the headline one.
"""

from __future__ import annotations

import math

import pytest

from scrcae.domain import SupplyNetwork
from scrcae.optimization import (
    MaximizeRiskReductionObjective,
    MonetaryNPVObjective,
    OptimizationRequest,
    PriceBook,
    SolveStatus,
    attainable_frontier,
    diagnose,
    solve,
    sweep_budget,
)
from scrcae.risk.response import AllocationConcaveResponse, LinearResponse

BIG_BUDGET = 10_000_000.0


def _fund_everything(network: SupplyNetwork, **kwargs) -> OptimizationRequest:
    """A request that funds every node at its maximum scale.

    Uses the diagnostic ceiling objective deliberately: it ignores cost, so the
    allocation is fixed by structure alone and a change in reported reduction can
    only have come from the exposure under test rather than from the optimizer
    re-choosing a portfolio.
    """
    return OptimizationRequest(
        network=network,
        objective=MaximizeRiskReductionObjective(),
        risk_response=LinearResponse(),
        budget=BIG_BUDGET,
        enforce_risk_cap=False,
        diagnose=False,
        **kwargs,
    )


def _reduction_by_node(result) -> dict[str, float]:
    return {a.node_id: a.risk_reduction_pts for a in result.allocations}


# --------------------------------------------------------------------------- #
# Composition with the scalar
# --------------------------------------------------------------------------- #


def test_unmapped_node_gets_the_scalar_exactly(small_network):
    """No exposure entry must mean no change whatsoever, not merely no big change."""
    request = _fund_everything(small_network, macro_multiplier=1.0625)
    assert request.macro_for("N1") == 1.0625


def test_exposure_multiplies_on_top_of_the_scalar(small_network):
    request = _fund_everything(
        small_network, macro_multiplier=1.0625, node_macro_multipliers={"N1": 1.2}
    )
    assert request.macro_for("N1") == pytest.approx(1.0625 * 1.2)
    assert request.macro_for("N2") == 1.0625


def test_explicit_ones_are_indistinguishable_from_omission(small_network):
    """The mapping is a no-op at 1.0.

    This is the property that lets the app hand the engine a full mapping for
    every node without having to decide which entries are 'really' active, and it
    is what makes the feature safe to leave switched on.
    """
    without = solve(_fund_everything(small_network, macro_multiplier=1.0625))
    with_ones = solve(
        _fund_everything(
            small_network,
            macro_multiplier=1.0625,
            node_macro_multipliers={n: 1.0 for n in small_network.node_ids},
        )
    )
    assert with_ones.objective_value == without.objective_value
    assert _reduction_by_node(with_ones) == _reduction_by_node(without)


# --------------------------------------------------------------------------- #
# Locality — the actual F11 fix
# --------------------------------------------------------------------------- #


def test_exposure_changes_only_the_node_it_names(small_network):
    """The bug this replaces applied one market's move to every node."""
    base = solve(_fund_everything(small_network))
    exposed = solve(_fund_everything(small_network, node_macro_multipliers={"N1": 1.2}))

    before, after = _reduction_by_node(base), _reduction_by_node(exposed)
    assert after["N1"] == pytest.approx(before["N1"] * 1.2)
    for node_id in ("N2", "N3", "N4", "N5"):
        assert after[node_id] == before[node_id]


def test_reported_reduction_is_recomputed_with_the_node_multiplier(small_network):
    """Reported quantities must come from the response model, not the linearisation."""
    exposure = {"N1": 1.4, "N3": 0.8}
    request = _fund_everything(small_network, node_macro_multipliers=exposure)
    result = solve(request)

    response = LinearResponse()
    by_node = {a.node_id: a for a in result.allocations}
    for intervention in small_network:
        allocation = by_node[intervention.node_id]
        assert allocation.risk_reduction_pts == pytest.approx(
            response.evaluate(
                intervention.risk_reduction_pts,
                allocation.funding_scale,
                request.macro_for(intervention.node_id),
            )
        )


def test_exposure_below_one_reduces_only_that_node(small_network):
    """Exposure is not assumed to be one-sided at this layer.

    Whether a *feed* should ever lower a node's multiplier is a modelling choice
    that belongs to whatever computes the mapping. The engine takes any positive
    number, so that choice stays visible where it is made instead of being
    silently enforced here.
    """
    base = _reduction_by_node(solve(_fund_everything(small_network)))
    lowered = _reduction_by_node(
        solve(_fund_everything(small_network, node_macro_multipliers={"N5": 0.5}))
    )
    assert lowered["N5"] == pytest.approx(base["N5"] * 0.5)
    assert lowered["N1"] == base["N1"]


def test_exposure_can_change_which_portfolio_is_chosen(small_network, prices):
    """The point of per-node exposure is that it changes decisions.

    A test that only checked arithmetic would pass on a mapping the optimizer
    never consulted.
    """
    tight = OptimizationRequest(
        network=small_network,
        objective=MonetaryNPVObjective(prices=prices),
        risk_response=LinearResponse(),
        budget=200_000.0,
        diagnose=False,
    )
    base = solve(tight)
    swung = solve(
        OptimizationRequest(
            network=small_network,
            objective=MonetaryNPVObjective(prices=prices),
            risk_response=LinearResponse(),
            budget=200_000.0,
            node_macro_multipliers={"N4": 8.0},
            diagnose=False,
        )
    )
    assert base.status is SolveStatus.OPTIMAL and swung.status is SolveStatus.OPTIMAL
    funded_before = {a.node_id for a in base.active_allocations}
    funded_after = {a.node_id for a in swung.active_allocations}
    assert "N4" not in funded_before
    assert "N4" in funded_after


# --------------------------------------------------------------------------- #
# Validation — silence was the defect
# --------------------------------------------------------------------------- #


def test_unknown_node_id_is_rejected_and_named(small_network):
    """A mapping entry that matches nothing is an error, not a no-op.

    The acquired version's equivalent silently matched nothing, which is why a
    misspelled sector name looked exactly like a working feed.
    """
    with pytest.raises(ValueError, match="Nsuch"):
        _fund_everything(small_network, node_macro_multipliers={"Nsuch": 1.2})


def test_unknown_ids_are_all_reported_not_just_the_first(small_network):
    with pytest.raises(ValueError) as excinfo:
        _fund_everything(small_network, node_macro_multipliers={"ZZ": 1.1, "YY": 1.1})
    message = str(excinfo.value)
    assert "YY" in message and "ZZ" in message


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.5, math.inf, math.nan])
def test_non_positive_or_non_finite_multipliers_are_rejected(small_network, bad):
    """Zero is rejected too.

    A zero multiplier silently deletes an intervention's entire benefit while
    leaving it in the portfolio at full cost, which reads on screen as an
    intervention that does nothing rather than as a bad input.
    """
    with pytest.raises(ValueError, match="N1"):
        _fund_everything(small_network, node_macro_multipliers={"N1": bad})


# --------------------------------------------------------------------------- #
# The baseline stays put
# --------------------------------------------------------------------------- #


def test_per_node_exposure_does_not_move_the_reported_baseline(small_network):
    """Exposure changes which interventions pay. It does not restate the risk.

    Aggregating per-node multipliers into a baseline adjustment would need weights
    the network does not carry; the obvious candidate — share of total risk
    reduction — would make the baseline move when an intervention is merely
    written down. See F17 for the same artefact in bundle discounts.
    """
    plain = _fund_everything(small_network)
    exposed = _fund_everything(
        small_network, node_macro_multipliers={n: 3.0 for n in small_network.node_ids}
    )
    assert exposed.effective_baseline_risk_pts == plain.effective_baseline_risk_pts


def test_the_scalar_still_moves_the_baseline(small_network):
    """The distinction above is a real one, so the other half must be shown too."""
    scaled = _fund_everything(small_network, macro_multiplier=1.0625)
    assert scaled.effective_baseline_risk_pts == pytest.approx(65.5 * 1.0625)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


def test_audit_records_the_exposure(small_network):
    result = solve(_fund_everything(small_network, node_macro_multipliers={"N1": 1.25}))
    assert result.audit["parameters"]["node_macro_multipliers"] == {"N1": 1.25}


def test_exposure_changes_the_input_hash(small_network):
    """Two results computed under different market conditions must not collide.

    Saved portfolios are re-solved and compared by hash, so an exposure that did
    not enter the hash would let a stale result pass verification.
    """
    plain = solve(_fund_everything(small_network))
    exposed = solve(_fund_everything(small_network, node_macro_multipliers={"N1": 1.25}))
    assert plain.audit["input_hash"] != exposed.audit["input_hash"]


def test_equal_exposures_hash_equal_regardless_of_insertion_order(small_network):
    first = solve(
        _fund_everything(small_network, node_macro_multipliers={"N1": 1.2, "N2": 1.1})
    )
    second = solve(
        _fund_everything(small_network, node_macro_multipliers={"N2": 1.1, "N1": 1.2})
    )
    assert first.audit["input_hash"] == second.audit["input_hash"]


# --------------------------------------------------------------------------- #
# Reach — every solve, not just the headline
# --------------------------------------------------------------------------- #


def test_attainable_frontier_reflects_the_exposure(small_network):
    plain = attainable_frontier(small_network, LinearResponse(), enforce_risk_cap=False)
    exposed = attainable_frontier(
        small_network,
        LinearResponse(),
        node_macro_multipliers={"N1": 2.0},
        enforce_risk_cap=False,
    )
    assert exposed.max_reduction_pts > plain.max_reduction_pts


def test_diagnosis_ceiling_is_computed_under_the_same_exposure(small_network, prices):
    """The frontier is what a CFO sees instead of the word `infeasible`.

    If diagnosis ignored the exposure it would quote a ceiling belonging to a
    different portfolio than the one on screen — a wrong number in the one place
    the product exists to be trusted.
    """
    exposure = {"N1": 2.0, "N2": 1.5}
    unreachable = OptimizationRequest(
        network=small_network,
        objective=MonetaryNPVObjective(prices=prices),
        risk_response=LinearResponse(),
        budget=BIG_BUDGET,
        required_risk_reduction_pts=60.0,
        node_macro_multipliers=exposure,
        enforce_risk_cap=False,
        diagnose=False,
    )
    report = diagnose(unreachable)
    expected = attainable_frontier(
        small_network,
        LinearResponse(),
        node_macro_multipliers=exposure,
        budget=None,
        enforce_risk_cap=False,
    )
    assert report.attainable.max_reduction_pts == pytest.approx(
        expected.max_reduction_pts
    )


def test_the_budget_sweep_carries_the_exposure(small_network, prices):
    """F6 was two optimizers disagreeing. The sweep must not reintroduce it."""
    request = OptimizationRequest(
        network=small_network,
        objective=MonetaryNPVObjective(prices=prices),
        risk_response=LinearResponse(),
        budget=300_000.0,
        node_macro_multipliers={"N1": 1.3},
        diagnose=False,
    )
    sweep = sweep_budget(request, levels=(150_000.0, 300_000.0))
    assert len(sweep.points) == 2
    solved = [p for p in sweep.points if p.status == SolveStatus.OPTIMAL]
    assert solved, "expected at least one solvable level"
    # Same exposure, same budget, same answer as the headline solve.
    headline = solve(request)
    at_headline = [p for p in sweep.points if p.budget == 300_000.0][0]
    assert at_headline.risk_reduction_pts == pytest.approx(headline.risk_reduction_pts)


def test_exposure_works_with_a_linearised_concave_response(small_network):
    """Exposure enters the chords and tangents, not only the linear coefficient.

    The non-linear path builds three separate objects from the multiplier — the
    ceiling, the inner chords and the outer tangents — and a multiplier applied to
    only some of them would produce an unsound risk cap rather than a wrong
    number, which is far harder to notice.
    """
    concave = AllocationConcaveResponse(exponent=0.85)
    plain = solve(
        OptimizationRequest(
            network=small_network,
            objective=MaximizeRiskReductionObjective(),
            risk_response=concave,
            budget=BIG_BUDGET,
            enforce_risk_cap=False,
            diagnose=False,
        )
    )
    exposed = solve(
        OptimizationRequest(
            network=small_network,
            objective=MaximizeRiskReductionObjective(),
            risk_response=concave,
            budget=BIG_BUDGET,
            node_macro_multipliers={"N1": 1.5},
            enforce_risk_cap=False,
            diagnose=False,
        )
    )
    before, after = _reduction_by_node(plain), _reduction_by_node(exposed)
    assert after["N1"] == pytest.approx(before["N1"] * 1.5)
    assert after["N2"] == pytest.approx(before["N2"])
    assert exposed.audit["constraint_check"] == plain.audit["constraint_check"]
