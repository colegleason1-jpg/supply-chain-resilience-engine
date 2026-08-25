"""Tests for the spreadsheet-to-domain boundary.

This is where the acquired system's data bugs lived, so it gets the most tests. The
recurring theme is that bad input must be *reported*, not absorbed: a sheet with one
unusable row should not quietly produce a plan built from the other rows, because the
user has no way to tell that happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import adapters
from app.adapters import (
    COL_BUNDLE_DISCOUNT,
    COL_BUNDLE_NAME,
    COL_BUNDLE_NODES,
    COL_COST,
    COL_DEP_DEPENDENT,
    COL_DEP_PREREQUISITE,
    COL_NODE_ID,
    COL_RISK,
    MODE_LEGACY,
    MODE_MONETARY,
    build_bundles,
    build_correlation_matrix,
    build_dependencies,
    build_interventions,
    build_network,
    build_optimization_request,
    build_risk_response,
    default_bundles_frame,
    default_correlation_frame,
    default_nodes_frame,
    parse_node_list,
)
from scrcae import PriceBook


# --------------------------------------------------------------------------- #
# Node parsing
# --------------------------------------------------------------------------- #


def test_default_portfolio_is_usable():
    interventions, rejections = build_interventions(default_nodes_frame())
    assert len(interventions) == 5
    assert not rejections


def test_missing_cost_is_rejected_not_defaulted_to_zero():
    """A blank cost must never become a free intervention.

    Zero-cost interventions are irresistible to the optimizer: it funds them to the
    maximum regardless of budget. The acquired system's editor passed blanks straight
    through as zeros.
    """
    frame = default_nodes_frame()
    frame.loc[0, COL_COST] = None

    interventions, rejections = build_interventions(frame)

    assert len(interventions) == 4
    assert len(rejections) == 1
    assert "Cost is missing" in rejections.items[0]
    assert all(i.cost > 0 for i in interventions)


@pytest.mark.parametrize("bad", ["", "   ", "not a number", float("nan"), None])
def test_unparseable_risk_is_rejected(bad):
    frame = default_nodes_frame()
    frame[COL_RISK] = frame[COL_RISK].astype(object)
    frame.loc[1, COL_RISK] = bad

    interventions, rejections = build_interventions(frame)

    assert len(interventions) == 4
    assert len(rejections) == 1


def test_currency_and_percent_formatting_is_accepted():
    """Users paste from spreadsheets; '$180,000' is a number to them."""
    frame = default_nodes_frame()
    frame[COL_COST] = frame[COL_COST].astype(object)
    frame.loc[0, COL_COST] = "$180,000"
    frame[COL_RISK] = frame[COL_RISK].astype(object)
    frame.loc[0, COL_RISK] = "12.5%"

    interventions, rejections = build_interventions(frame)

    assert not rejections
    assert interventions[0].cost == pytest.approx(180_000.0)
    assert interventions[0].risk_reduction_pts == pytest.approx(12.5)


def test_duplicate_node_ids_are_rejected():
    frame = default_nodes_frame()
    frame.loc[1, COL_NODE_ID] = "N1"

    interventions, rejections = build_interventions(frame)

    assert [i.node_id for i in interventions] == ["N1", "N3", "N4", "N5"]
    assert any("duplicate" in r for r in rejections)


def test_trailing_blank_rows_are_ignored_silently():
    """An empty row is how editors look, not an error worth shouting about."""
    frame = default_nodes_frame()
    blank = pd.DataFrame([{c: None for c in frame.columns}])
    frame = pd.concat([frame, blank], ignore_index=True)

    interventions, rejections = build_interventions(frame)

    assert len(interventions) == 5
    assert not rejections


def test_a_row_with_values_but_no_id_is_reported():
    frame = default_nodes_frame()
    frame = pd.concat(
        [frame, pd.DataFrame([{COL_COST: 1000.0, COL_RISK: 1.0}])], ignore_index=True
    )

    _, rejections = build_interventions(frame)

    assert any("no Node ID" in r for r in rejections)


def test_inverted_funding_scales_are_rejected():
    frame = default_nodes_frame()
    frame.loc[0, adapters.COL_MIN_SCALE] = 0.9
    frame.loc[0, adapters.COL_MAX_SCALE] = 0.4

    interventions, rejections = build_interventions(frame)

    assert len(interventions) == 4
    assert any("exceeds maximum" in r for r in rejections)


# --------------------------------------------------------------------------- #
# Bundles: F5
# --------------------------------------------------------------------------- #


def test_bundle_membership_comes_from_data_not_row_order():
    """F5. The acquired system used `nodes[:2]` -- whatever the editor showed first.

    Membership determines which discount the model may claim, so deriving it from
    display order meant reordering the table changed the answer.
    """
    frame = pd.DataFrame(
        [{COL_BUNDLE_NAME: "B", COL_BUNDLE_DISCOUNT: 1000.0, COL_BUNDLE_NODES: "N3, N5"}]
    )
    bundles, rejections = build_bundles(frame, ["N1", "N2", "N3", "N4", "N5"])

    assert not rejections
    assert bundles[0].required_nodes == ("N3", "N5")


def test_bundle_naming_an_unknown_node_is_rejected_not_shrunk():
    """Shrinking silently changes what the discount is contingent on."""
    frame = pd.DataFrame(
        [{COL_BUNDLE_NAME: "B", COL_BUNDLE_DISCOUNT: 1000.0, COL_BUNDLE_NODES: "N1, NOPE"}]
    )
    bundles, rejections = build_bundles(frame, ["N1", "N2"])

    assert bundles == ()
    assert any("unknown node" in r for r in rejections)


def test_bundle_without_members_is_rejected_with_an_actionable_message():
    frame = pd.DataFrame(
        [{COL_BUNDLE_NAME: "B", COL_BUNDLE_DISCOUNT: 1000.0, COL_BUNDLE_NODES: ""}]
    )
    bundles, rejections = build_bundles(frame, ["N1"])

    assert bundles == ()
    assert any("Required Nodes" in r and "N1, N2" in r for r in rejections)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("N1, N2", ("N1", "N2")),
        ("N1;N2", ("N1", "N2")),
        (" N1 , , N2 ", ("N1", "N2")),
        ("N1", ("N1",)),
        ("", ()),
        (None, ()),
    ],
)
def test_node_list_parsing(raw, expected):
    assert parse_node_list(raw) == expected


def test_default_bundle_frame_is_valid_against_the_default_portfolio():
    interventions, _ = build_interventions(default_nodes_frame())
    bundles, rejections = build_bundles(
        default_bundles_frame(), [i.node_id for i in interventions]
    )
    assert not rejections
    assert bundles[0].required_nodes == ("N1", "N2")


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def test_self_dependency_is_rejected():
    frame = pd.DataFrame([{COL_DEP_DEPENDENT: "N1", COL_DEP_PREREQUISITE: "N1"}])
    deps, rejections = build_dependencies(frame, ["N1", "N2"])
    assert deps == ()
    assert any("cannot depend on itself" in r for r in rejections)


def test_dependency_on_unknown_node_is_rejected():
    frame = pd.DataFrame([{COL_DEP_DEPENDENT: "N1", COL_DEP_PREREQUISITE: "GHOST"}])
    deps, rejections = build_dependencies(frame, ["N1", "N2"])
    assert deps == ()
    assert any("unknown node" in r for r in rejections)


def test_dependency_cycle_is_rejected_by_the_domain_model():
    """The adapter defers to the domain model rather than re-implementing the check."""
    deps = pd.DataFrame(
        [
            {COL_DEP_DEPENDENT: "N1", COL_DEP_PREREQUISITE: "N2"},
            {COL_DEP_DEPENDENT: "N2", COL_DEP_PREREQUISITE: "N1"},
        ]
    )
    network, rejections = build_network(
        default_nodes_frame(), 60.0, dependencies_frame=deps
    )
    assert network is None
    assert any("cycle" in r.lower() for r in rejections)


# --------------------------------------------------------------------------- #
# Network assembly
# --------------------------------------------------------------------------- #


def test_build_network_returns_none_rather_than_an_empty_network():
    """An empty network solves fine and reports zero risk for zero cost.

    That is indistinguishable on screen from a considered 'do nothing' recommendation,
    so it must not be reachable from broken input.
    """
    network, rejections = build_network(pd.DataFrame(), 60.0)
    assert network is None
    assert any("no usable interventions" in r for r in rejections)


def test_build_network_carries_bundles_and_dependencies_through():
    deps = pd.DataFrame([{COL_DEP_DEPENDENT: "N2", COL_DEP_PREREQUISITE: "N1"}])
    network, rejections = build_network(
        default_nodes_frame(),
        65.5,
        bundles_frame=default_bundles_frame(),
        dependencies_frame=deps,
    )
    assert not rejections
    assert network.baseline_risk_pts == pytest.approx(65.5)
    assert len(network.bundles) == 1
    assert len(network.dependencies) == 1


def test_negative_baseline_is_rejected():
    network, rejections = build_network(default_nodes_frame(), -5.0)
    assert network is None
    assert any("Baseline risk" in r for r in rejections)


# --------------------------------------------------------------------------- #
# Correlation: F11
# --------------------------------------------------------------------------- #


def test_correlation_is_read_by_node_id_in_the_requested_order():
    ids = ["N1", "N2", "N3"]
    frame = default_correlation_frame(ids, rho=0.4)
    frame.loc[frame[COL_NODE_ID] == "N1", "N2"] = 0.9
    frame.loc[frame[COL_NODE_ID] == "N2", "N1"] = 0.9

    matrix, rejections = build_correlation_matrix(frame, ids)

    assert not rejections
    assert matrix[0, 1] == pytest.approx(0.9)
    assert np.allclose(np.diag(matrix), 1.0)


def test_correlation_reordering_follows_node_order_not_sheet_order():
    """Reading positionally would silently transpose the structure."""
    frame = default_correlation_frame(["N1", "N2"], rho=0.0)
    frame.loc[frame[COL_NODE_ID] == "N1", "N2"] = 0.8
    frame.loc[frame[COL_NODE_ID] == "N2", "N1"] = 0.8

    forward, _ = build_correlation_matrix(frame, ["N1", "N2"])
    reversed_, _ = build_correlation_matrix(frame, ["N2", "N1"])

    assert forward[0, 1] == pytest.approx(0.8)
    assert reversed_[0, 1] == pytest.approx(0.8)


def test_missing_correlation_row_is_reported_not_defaulted_to_035():
    """F11. The acquired system substituted 0.35 inside a bare `except`."""
    frame = default_correlation_frame(["N1", "N2"])
    matrix, rejections = build_correlation_matrix(frame, ["N1", "N2", "N3"])

    assert any("no row for node 'N3'" in r for r in rejections)
    # Unknown pairs stay independent rather than acquiring an invented correlation.
    assert matrix[2, 0] == pytest.approx(0.0)


def test_asymmetric_input_is_symmetrised():
    frame = default_correlation_frame(["N1", "N2"], rho=0.0)
    frame.loc[frame[COL_NODE_ID] == "N1", "N2"] = 0.6  # other triangle left at 0

    matrix, _ = build_correlation_matrix(frame, ["N1", "N2"])

    assert matrix[0, 1] == pytest.approx(matrix[1, 0])
    assert matrix[0, 1] == pytest.approx(0.3)


def test_correlations_are_clamped_into_range():
    frame = default_correlation_frame(["N1", "N2"], rho=5.0)
    matrix, _ = build_correlation_matrix(frame, ["N1", "N2"])
    assert abs(matrix[0, 1]) <= 0.99


def test_empty_portfolio_yields_a_zero_by_zero_matrix():
    """The engine validates the shape, so an empty portfolio needs (0, 0)."""
    matrix, _ = build_correlation_matrix(default_correlation_frame([]), [])
    assert matrix.shape == (0, 0)


# --------------------------------------------------------------------------- #
# Mode selection
# --------------------------------------------------------------------------- #


def test_legacy_mode_transforms_the_parameter_and_monetary_the_allocation():
    """F2, in one assertion.

    The acquired system's 0.85 exponent applied to the risk *parameter*, which leaves
    the model perfectly linear in funding -- there were no diminishing returns at all,
    only a cross-sectional reweighting.
    """
    legacy = build_risk_response(MODE_LEGACY, 0.85)
    monetary = build_risk_response(MODE_MONETARY, 0.85)

    assert legacy.is_linear          # linear in funding, despite the exponent
    assert not monetary.is_linear    # genuinely concave in funding


def test_exponent_of_one_gives_a_linear_response_in_monetary_mode():
    assert build_risk_response(MODE_MONETARY, 1.0).is_linear


def test_mode_selects_a_coherent_bundle_of_choices():
    """Modes are not a menu of independent switches; half of each is not defensible."""
    network, _ = build_network(default_nodes_frame(), 65.5)

    legacy = build_optimization_request(
        network, mode=MODE_LEGACY, budget=750_000.0, macro_multiplier=1.0625
    )
    monetary = build_optimization_request(
        network,
        mode=MODE_MONETARY,
        budget=750_000.0,
        prices=PriceBook(value_per_risk_point=50_000.0),
    )

    assert legacy.enforce_risk_cap is False
    assert legacy.legacy_bundle_activation is True
    assert monetary.enforce_risk_cap is True
    assert monetary.legacy_bundle_activation is False


def test_target_mode_minimises_capital_in_either_mode():
    network, _ = build_network(default_nodes_frame(), 65.5)
    for mode in (MODE_LEGACY, MODE_MONETARY):
        request = build_optimization_request(
            network,
            mode=mode,
            budget=None,
            target_risk_pts=40.0,
            prices=PriceBook(value_per_risk_point=50_000.0),
        )
        assert request.objective.unit == "currency (net capital outlay)"
        assert request.required_risk_reduction_pts == pytest.approx(25.5)


def test_target_is_derived_from_the_macro_inflated_baseline():
    """ADR-001 F18. The user sets the target against the baseline they can see."""
    network, _ = build_network(default_nodes_frame(), 65.5)
    macro = 1.0625

    request = build_optimization_request(
        network,
        mode=MODE_MONETARY,
        budget=None,
        macro_multiplier=macro,
        target_risk_pts=40.0,
        prices=PriceBook(value_per_risk_point=50_000.0),
    )

    effective = 65.5 * macro
    assert request.effective_baseline_risk_pts == pytest.approx(effective)
    assert request.required_risk_reduction_pts == pytest.approx(effective - 40.0)


def test_target_at_or_above_baseline_requires_no_reduction():
    network, _ = build_network(default_nodes_frame(), 65.5)
    request = build_optimization_request(
        network,
        mode=MODE_MONETARY,
        budget=None,
        target_risk_pts=90.0,
        prices=PriceBook(value_per_risk_point=50_000.0),
    )
    assert request.required_risk_reduction_pts == pytest.approx(0.0)


def test_monetary_mode_without_prices_is_a_programming_error():
    network, _ = build_network(default_nodes_frame(), 65.5)
    with pytest.raises(ValueError, match="PriceBook"):
        build_optimization_request(network, mode=MODE_MONETARY, budget=1000.0)
