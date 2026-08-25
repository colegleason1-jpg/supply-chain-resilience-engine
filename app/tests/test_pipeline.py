"""End-to-end: spreadsheet in, rendered rows out, with no Streamlit involved.

These exercise the exact sequence `main.py` performs. If they pass, the only thing
left that can break is layout — which is the point of moving every decision out of the
view layer.
"""

from __future__ import annotations

import pytest

from app import adapters
from app.engine_client import analyse
from app.presenters import (
    headline_metrics,
    portfolio_rows,
    simulation_metrics,
    status_banner,
    verification_rows,
)
from app.services.macro import ANCHOR_BRENT_USD, NO_FEED, multiplier_from_brent
from scrcae import PriceBook


def _pipeline(**overrides):
    """Everything main.py does, in order."""
    frames = {
        "nodes": adapters.default_nodes_frame(),
        "bundles": adapters.default_bundles_frame(),
        "dependencies": adapters.default_dependencies_frame(),
    }
    network, rejections = adapters.build_network(
        frames["nodes"],
        overrides.pop("baseline", 65.5),
        bundles_frame=frames["bundles"],
        dependencies_frame=frames["dependencies"],
    )
    assert network is not None and not rejections

    node_ids = [i.node_id for i in network.interventions]
    correlation, corr_rejections = adapters.build_correlation_matrix(
        adapters.default_correlation_frame(node_ids), node_ids
    )
    assert not corr_rejections

    mode = overrides.pop("mode", adapters.MODE_MONETARY)
    simulate = overrides.pop("simulate", True)
    request = adapters.build_optimization_request(
        network,
        mode=mode,
        budget=overrides.pop("budget", 750_000.0),
        prices=PriceBook(value_per_risk_point=50_000.0, source="fixture"),
        **overrides,
    )

    factory = None
    if simulate:
        def factory(result):
            return adapters.build_simulation_request(
                result, network, correlation, mode=mode, iterations=2_000
            )

    return analyse(request, simulation_factory=factory, sweep_steps=5), request


def test_default_configuration_produces_a_funded_plan():
    bundle, _ = _pipeline()

    assert bundle.result.is_solved
    assert portfolio_rows(bundle.result)
    # Compared against the engine's own scale-relative tolerance, not an absolute
    # epsilon: CBC's feasibility slack on a six-figure budget is larger than 1e-6,
    # so an absolute comparison would flag numerical noise as an overspend (F15).
    assert bundle.result.net_capital <= 750_000.0 * (1 + 1e-6) + 1e-6
    assert bundle.sweep is not None
    assert bundle.simulation is not None


def test_every_displayed_number_is_present_and_formatted():
    bundle, request = _pipeline()

    for metric in headline_metrics(bundle.result, request.macro_multiplier):
        assert metric.value and metric.value != "—"
    for row in verification_rows(bundle.result, request):
        assert row["Result"]


def test_simulation_is_skipped_when_no_plan_exists():
    """A confidence interval around a non-existent plan is the worst kind of number."""
    bundle, _ = _pipeline(budget=None, target_risk_pts=1.0)

    assert not bundle.result.is_solved
    assert bundle.simulation is None
    assert simulation_metrics(bundle.simulation) == []


def test_simulation_is_centred_near_the_deterministic_result():
    """The new shock model should not systematically flatter or punish the plan."""
    bundle, _ = _pipeline()
    simulation = bundle.simulation

    assert abs(simulation.reduction_bias_pts) < 0.5
    assert simulation.p90_risk_pts >= simulation.p50_risk_pts


def test_legacy_shocks_are_biased_and_the_app_says_so():
    """F7. The 0.4 floor truncates the downside, so the mean drifts upward."""
    bundle, request = _pipeline(mode=adapters.MODE_LEGACY)

    assert "legacy" in bundle.simulation.shock_version
    from app.presenters import uncalibrated_warnings

    assert any("optimistic by construction" in w for w in uncalibrated_warnings(request, bundle.simulation))


def test_budget_is_never_exceeded_across_the_whole_sweep():
    bundle, _ = _pipeline(budget=300_000.0)
    for point in bundle.sweep.solved_points:
        assert point.net_capital <= point.budget * (1 + 1e-6) + 1e-6


def test_tighter_budget_never_buys_more_risk_reduction():
    """Monotonicity across the curve. The drifted duplicate could violate this."""
    bundle, _ = _pipeline(budget=400_000.0)
    solved = bundle.sweep.solved_points
    reductions = [p.risk_reduction_pts for p in solved]
    assert reductions == sorted(reductions), "reduction should be non-decreasing in budget"


#: Every module that must remain computable without a web framework. `views/` is
#: excluded by design — rendering is the one job that is allowed to need Streamlit.
HEADLESS_MODULES = (
    "adapters.py",
    "presenters.py",
    "engine_client.py",
    "copilot.py",
    "services/macro.py",
    "services/portfolios.py",
    "services/copilot_state.py",
    "storage/models.py",
    "storage/serialization.py",
    "storage/sqlite_store.py",
    "auth/models.py",
    "auth/providers.py",
)


def test_no_streamlit_import_is_required_to_compute_anything():
    """The load-bearing structural claim of this rebuild.

    Checked by reading the source rather than by inspecting `sys.modules`. The earlier
    version of this test asserted `"streamlit" not in sys.modules`, which passed or
    failed according to whether some other test file had imported Streamlit first —
    it was measuring the test runner's import order, not this codebase. `test_render.py`
    legitimately imports Streamlit and broke it.

    An `ast` walk states the actual property: none of these modules names Streamlit at
    any import site, top-level or nested.
    """
    import ast
    from pathlib import Path

    app_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []

    for relative in HEADLESS_MODULES:
        path = app_root / relative
        assert path.exists(), f"{relative} is listed as headless but does not exist"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "streamlit" for name in names):
                offenders.append(f"{relative}:{node.lineno}")

    assert not offenders, f"streamlit imported in compute modules: {offenders}"


def test_the_headless_modules_are_importable_without_streamlit_state():
    """Importing them must not require a running Streamlit script context."""
    import importlib

    for relative in HEADLESS_MODULES:
        module = "app." + relative.removesuffix(".py").replace("/", ".")
        assert importlib.import_module(module) is not None


# --------------------------------------------------------------------------- #
# Macro service: F13
# --------------------------------------------------------------------------- #


def test_unconfigured_feed_is_reported_as_unconfigured():
    """F13. The placeholder key failed silently behind a green status indicator."""
    assert NO_FEED.is_live is False
    assert NO_FEED.multiplier == 1.0
    assert "No macro feed configured" in NO_FEED.status
    assert "No live reading" in NO_FEED.provenance


def test_missing_credential_returns_the_unconfigured_reading(monkeypatch):
    monkeypatch.delenv("SCRCAE_MACRO_API_KEY", raising=False)
    from app.services.macro import fetch_macro_reading

    reading = fetch_macro_reading()
    assert not reading.is_live
    assert reading.multiplier == 1.0


def test_feed_failure_never_raises_into_the_ui(monkeypatch):
    monkeypatch.setenv("SCRCAE_MACRO_API_KEY", "definitely-not-valid")
    from app.services import macro as macro_module

    def explode(*args, **kwargs):
        raise ConnectionError("no network in tests")

    import sys
    import types

    fake = types.ModuleType("requests")
    fake.get = explode
    monkeypatch.setitem(sys.modules, "requests", fake)

    reading = macro_module.fetch_macro_reading()
    assert not reading.is_live
    assert reading.multiplier == 1.0
    assert "ConnectionError" in reading.status


@pytest.mark.parametrize(
    "brent,expected",
    [(ANCHOR_BRENT_USD, 1.0), (40.0, 1.0), (ANCHOR_BRENT_USD * 2, 1.15)],
)
def test_multiplier_is_one_sided(brent, expected):
    """Cheap oil does not make a supply chain safer; nobody established that."""
    assert multiplier_from_brent(brent) == pytest.approx(expected)


def test_multiplier_is_monotone_above_the_anchor():
    prices = [80.0, 90.0, 100.0, 120.0]
    values = [multiplier_from_brent(p) for p in prices]
    assert values == sorted(values)


# --------------------------------------------------------------------------- #
# Market exposure, end to end (F11)
# --------------------------------------------------------------------------- #


def _exposure_pipeline(prices: dict[str, float | None], *, elasticity: float = 1.0):
    """Spreadsheet through feed through solver, exactly as main.py sequences it."""
    from app.services import market as market_service

    nodes = adapters.default_nodes_frame()
    network, rejections = adapters.build_network(nodes, 65.5)
    assert network is not None and not rejections
    node_ids = [i.node_id for i in network.interventions]

    frame = adapters.align_exposure_frame(None, node_ids)
    frame.loc[0, adapters.COL_EXP_SYMBOL] = "OIL"
    frame.loc[0, adapters.COL_EXP_ANCHOR] = 80.0
    frame.loc[0, adapters.COL_EXP_ELASTICITY] = elasticity

    exposures, exposure_rejections = adapters.build_node_exposures(frame, node_ids)
    assert not exposure_rejections

    quotes = market_service.quotes_from_prices(prices)
    overlay = market_service.build_overlay(exposures, quotes)

    request = adapters.build_optimization_request(
        network,
        mode=adapters.MODE_MONETARY,
        budget=750_000.0,
        prices=PriceBook(value_per_risk_point=50_000.0, source="fixture"),
        node_macro_multipliers=overlay.multipliers,
    )
    return analyse(request, include_sweep=False), overlay, node_ids[0], request


def test_a_live_market_above_anchor_reaches_the_solver():
    """The path the acquired system's string comparison never completed."""
    bundle, overlay, first_node, request = _exposure_pipeline({"OIL": 120.0})
    assert overlay.multipliers[first_node] == pytest.approx(1.5)
    assert request.node_macro_multipliers == overlay.multipliers
    assert request.macro_for(first_node) == pytest.approx(1.5)
    assert bundle.result.is_solved


def test_exposure_changes_the_reported_answer():
    with_feed, _, _, _ = _exposure_pipeline({"OIL": 120.0})
    without, _, _, _ = _exposure_pipeline({"OIL": None})
    assert with_feed.result.risk_reduction_pts != pytest.approx(
        without.result.risk_reduction_pts
    )


def test_a_dead_feed_leaves_the_answer_identical_to_no_exposure():
    """Not 'approximately identical'. An unavailable market must be a strict no-op."""
    dead, overlay, _, request = _exposure_pipeline({"OIL": None})
    baseline_request = adapters.build_optimization_request(
        adapters.build_network(adapters.default_nodes_frame(), 65.5)[0],
        mode=adapters.MODE_MONETARY,
        budget=750_000.0,
        prices=PriceBook(value_per_risk_point=50_000.0, source="fixture"),
    )
    plain = analyse(baseline_request, include_sweep=False)
    assert overlay.multipliers == {}
    assert request.node_macro_multipliers == {}
    assert dead.result.audit["input_hash"] == plain.result.audit["input_hash"]
    assert dead.result.risk_reduction_pts == plain.result.risk_reduction_pts


def test_the_exposure_is_recorded_in_the_audit_ledger():
    bundle, _, first_node, _ = _exposure_pipeline({"OIL": 120.0})
    recorded = bundle.result.audit["parameters"]["node_macro_multipliers"]
    assert recorded[first_node] == pytest.approx(1.5)


def test_the_verification_tab_names_the_exposure_and_the_price_source():
    from app.presenters import provenance_rows, uncalibrated_warnings

    bundle, _, first_node, request = _exposure_pipeline({"OIL": 120.0})
    rows = {r["Item"]: r["Value"] for r in provenance_rows(bundle.result, request=request)}
    assert first_node in rows["Market exposure applied"]
    assert "fixture" in rows["Price of risk"]
    assert any("Market exposure is adjusting" in w for w in uncalibrated_warnings(request))


def test_no_exposure_is_reported_as_none_rather_than_omitted():
    """An absent row reads as an oversight; 'none' is an answer."""
    from app.presenters import provenance_rows

    bundle, _, _, request = _exposure_pipeline({"OIL": None})
    rows = {r["Item"]: r["Value"] for r in provenance_rows(bundle.result, request=request)}
    assert rows["Market exposure applied"] == "none"


def test_the_derived_price_book_flows_through_the_whole_pipeline():
    from app.presenters import uncalibrated_warnings
    from app.services.pricing import DisruptionExposure, derive_prices

    derivation = derive_prices(
        DisruptionExposure(
            annual_revenue=400_000_000.0,
            revenue_at_risk_share=0.35,
            gross_margin=0.28,
            disruption_days=21.0,
            basis="FY26 accounts",
        ),
        annual_disruption_probability=0.655,
    )
    network, _ = adapters.build_network(adapters.default_nodes_frame(), 65.5)
    request = adapters.build_optimization_request(
        network,
        mode=adapters.MODE_MONETARY,
        budget=750_000.0,
        prices=derivation.prices,
    )
    bundle = analyse(request, include_sweep=False)
    assert bundle.result.is_solved
    # A derived price book is sourced, so the unsourced warning must be absent.
    assert not any("no stated source" in w for w in uncalibrated_warnings(request))
    provenance = bundle.result.audit["objective_provenance"]
    assert provenance["prices_are_sourced"] is True
    assert "FY26 accounts" in provenance["price_source"]


def test_an_unsourced_price_book_is_flagged_all_the_way_into_the_ledger():
    from app.services.pricing import unsourced_price_book

    network, _ = adapters.build_network(adapters.default_nodes_frame(), 65.5)
    request = adapters.build_optimization_request(
        network,
        mode=adapters.MODE_MONETARY,
        budget=750_000.0,
        prices=unsourced_price_book(value_per_risk_point=50_000.0),
    )
    bundle = analyse(request, include_sweep=False)
    assert bundle.result.audit["objective_provenance"]["prices_are_sourced"] is False
