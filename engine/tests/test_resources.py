"""Resource capacities: capped supplies beside capital.

The budget is one scalar. A real portfolio also runs against per-supplier limits (a
vendor's daily quota, a port's slots, a team's hours). Each is a ``Resource`` with a
capacity, each intervention states its usage at full scale, the optimizer adds one row
per resource and the verifier re-checks it. Every test here has a hand-derivable answer,
and the first one pins the property that matters most: a network without resources
produces exactly what it produced before the family existed, hashes included.
"""

from __future__ import annotations

import pytest

from scrcae import (
    Intervention,
    MinimizeCapitalObjective,
    OptimizationRequest,
    Resource,
    SupplyNetwork,
    solve,
)
from scrcae.domain import DomainError
from scrcae.optimization import SolveStatus, attainable_frontier
from scrcae.optimization.diagnostics import InfeasibilityKind
from scrcae.optimization.objectives import MaximizeRiskReductionObjective
from scrcae.risk import LinearResponse

TOL = 1e-6


def _two_nodes(usage_a=None, usage_b=None, resources=()):
    return SupplyNetwork(
        baseline_risk_pts=80.0,
        interventions=(
            Intervention("A", cost=100_000.0, risk_reduction_pts=10.0, min_funding_scale=0.0, usage=usage_a or ()),
            Intervention("B", cost=100_000.0, risk_reduction_pts=6.0, min_funding_scale=0.0, usage=usage_b or ()),
        ),
        resources=tuple(resources),
    )


def _max(network, **kwargs):
    return solve(OptimizationRequest(network=network, objective=MaximizeRiskReductionObjective(), risk_response=LinearResponse(), enforce_risk_cap=False, diagnose=False, **kwargs))


def test_a_network_without_resources_is_unchanged_hashes_included(small_network, npv_objective):
    """The default is empty, so the parity case adds no rows, no checks, and no hash input."""
    request = OptimizationRequest(network=small_network, objective=npv_objective, risk_response=LinearResponse(), budget=400_000.0)
    before = solve(request)
    assert before.resource_use == {} and "resource_capacity" not in before.constraint_report.checks_performed
    assert small_network.resources == () and small_network.resource_names == ()
    # The same problem stated with the field present but empty solves identically.
    again = solve(OptimizationRequest(network=SupplyNetwork(interventions=small_network.interventions, baseline_risk_pts=small_network.baseline_risk_pts, dependencies=small_network.dependencies, bundles=small_network.bundles, resources=()), objective=npv_objective, risk_response=LinearResponse(), budget=400_000.0))
    assert again.audit["input_hash"] == before.audit["input_hash"] and again.audit["output_hash"] == before.audit["output_hash"]
    assert again.scales() == before.scales()


def test_a_shared_capacity_binds_where_the_budget_would_not():
    """A and B both draw 100 units of the vendor at full scale; the vendor allows 150.

    Budget admits both in full (200k). The optimizer must fill A (10 points per 100 units)
    and give B the remaining 50 units: x_A = 1, x_B = 0.5, reduction 13.
    """
    network = _two_nodes({"vendor": 100.0}, {"vendor": 100.0}, [Resource("vendor", 150.0)])
    result = _max(network, budget=200_000.0)
    assert result.status == SolveStatus.OPTIMAL
    assert result.scales()["A"] == pytest.approx(1.0, abs=TOL) and result.scales()["B"] == pytest.approx(0.5, abs=TOL)
    assert result.raw_risk_reduction_pts == pytest.approx(13.0, abs=1e-6)
    assert result.resource_use["vendor"] == pytest.approx(150.0, abs=1e-6) and result.resource_headroom("vendor", 150.0) == pytest.approx(0.0, abs=1e-6)
    assert "resource_capacity" in result.constraint_report.checks_performed and result.constraint_report.is_feasible
    assert result.constraint_report.residual_slack.get("resource:vendor", 0.0) <= 1e-6


def test_a_zero_capacity_switches_its_users_off_and_leaves_the_rest_alone():
    network = _two_nodes({"vendor": 100.0}, None, [Resource("vendor", 0.0)])
    result = _max(network, budget=200_000.0)
    assert result.scales()["A"] == 0.0 and result.scales()["B"] == pytest.approx(1.0, abs=TOL)
    assert result.resource_use["vendor"] == 0.0


def test_capacities_change_the_hashes_and_are_verified():
    loose = _max(_two_nodes({"vendor": 100.0}, {"vendor": 100.0}, [Resource("vendor", 300.0)]), budget=200_000.0)
    tight = _max(_two_nodes({"vendor": 100.0}, {"vendor": 100.0}, [Resource("vendor", 150.0)]), budget=200_000.0)
    assert loose.audit["input_hash"] != tight.audit["input_hash"] and loose.audit["output_hash"] != tight.audit["output_hash"]
    assert loose.scales() == {"A": 1.0, "B": 1.0} and loose.resource_use["vendor"] == pytest.approx(200.0)


def test_target_mode_reports_a_resource_bound_ceiling_not_a_budget_shortfall():
    """Asking for 15 points when the vendor's 150 units cap delivery at 13 is a supply problem.

    The diagnosis must not tell the reader that more capital helps, and it names the resource.
    """
    network = _two_nodes({"vendor": 100.0}, {"vendor": 100.0}, [Resource("vendor", 150.0)])
    result = solve(OptimizationRequest(network=network, objective=MinimizeCapitalObjective(), risk_response=LinearResponse(), enforce_risk_cap=False, budget=1_000_000.0, required_risk_reduction_pts=15.0))
    assert result.status == SolveStatus.INFEASIBLE and result.diagnosis is not None
    assert result.diagnosis.kind == InfeasibilityKind.EXCEEDS_STRUCTURAL_CEILING and not result.diagnosis.more_capital_would_help
    assert "vendor" in result.diagnosis.explanation and "capacity of vendor" in result.diagnosis.remedy
    assert result.diagnosis.attainable.limited_by == "resource:vendor" and result.diagnosis.attainable.max_reduction_pts == pytest.approx(13.0, abs=1e-6)
    frontier = attainable_frontier(network, LinearResponse(), enforce_risk_cap=False)
    assert frontier.limited_by == "resource:vendor" and "resource:vendor" in frontier.summary()
    # With room to spare the frontier is structural again.
    roomy = attainable_frontier(_two_nodes({"vendor": 100.0}, {"vendor": 100.0}, [Resource("vendor", 500.0)]), LinearResponse(), enforce_risk_cap=False)
    assert roomy.limited_by == "structure"


def test_usage_accepts_a_mapping_or_pairs_and_is_stored_sorted():
    a = Intervention("A", usage={"z": 1.0, "m": 2.0})
    b = Intervention("A", usage=(("m", 2), ("z", 1)))
    assert a.usage == (("m", 2.0), ("z", 1.0)) == b.usage and a == b and hash(a) == hash(b)
    assert a.usage_of("m") == 2.0 and a.usage_of("missing") == 0.0


@pytest.mark.parametrize(
    "build",
    [
        lambda: Intervention("A", usage={"v": -1.0}),
        lambda: Intervention("A", usage={"v": float("inf")}),
        lambda: Intervention("A", usage={"": 1.0}),
        lambda: Intervention("A", usage=(("v", 1.0), ("v", 2.0))),
        lambda: Intervention("A", usage=("v",)),
        lambda: Resource("", 1.0),
        lambda: Resource("v", -1.0),
        lambda: SupplyNetwork(interventions=(Intervention("A", usage={"v": 1.0}),), baseline_risk_pts=50.0),
        lambda: SupplyNetwork(interventions=(Intervention("A"),), baseline_risk_pts=50.0, resources=(Resource("v", 1.0), Resource("v", 2.0))),
    ],
)
def test_invalid_resource_inputs_are_rejected_loudly(build):
    with pytest.raises(DomainError):
        build()


def test_network_helpers_report_demand_and_lookup():
    network = _two_nodes({"vendor": 100.0}, {"vendor": 40.0}, [Resource("vendor", 150.0)])
    assert network.resource("vendor").capacity == 150.0 and network.resource_demand_at_full_scale("vendor") == 140.0
    with pytest.raises(KeyError):
        network.resource("port")
