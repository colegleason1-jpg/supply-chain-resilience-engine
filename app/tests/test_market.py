"""Market exposure: the explicit mapping that replaces F11's string equality."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app import adapters
from app.adapters import (
    COL_EXP_ANCHOR,
    COL_EXP_ELASTICITY,
    COL_EXP_NODE_ID,
    COL_EXP_SYMBOL,
    NodeExposure,
    align_exposure_frame,
    build_node_exposures,
    default_exposure_frame,
)
from app.services.feeds import ApiNinjasProvider, StaticProvider
from app.services.market import (
    MIN_MULTIPLIER,
    MarketQuote,
    QuoteSet,
    build_overlay,
    fetch_history,
    fetch_quotes,
    quotes_from_prices,
)

NODES = ("N1", "N2", "N3")


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(adapters.EXPOSURE_COLUMNS))


def _row(node="N1", symbol="BRENT_CRUDE_OIL", anchor=80.0, elasticity=0.2) -> dict:
    return {
        COL_EXP_NODE_ID: node,
        COL_EXP_SYMBOL: symbol,
        COL_EXP_ANCHOR: anchor,
        COL_EXP_ELASTICITY: elasticity,
    }


# --------------------------------------------------------------------------- #
# Parsing the table
# --------------------------------------------------------------------------- #


def test_a_blank_sheet_yields_no_exposure_and_no_complaints():
    """Most nodes have no market exposure. That is not an error condition."""
    exposures, rejections = build_node_exposures(default_exposure_frame(NODES), NODES)
    assert exposures == ()
    assert not rejections


def test_a_complete_row_parses():
    exposures, rejections = build_node_exposures(_frame([_row()]), NODES)
    assert not rejections
    assert exposures == (
        NodeExposure(
            node_id="N1", symbol="BRENT_CRUDE_OIL", anchor=80.0, elasticity=0.2
        ),
    )


def test_symbols_are_normalised_so_case_cannot_break_the_join():
    """The defect being replaced was a string comparison. Casing must not matter."""
    exposures, _ = build_node_exposures(_frame([_row(symbol="brent_crude_oil")]), NODES)
    assert exposures[0].symbol == "BRENT_CRUDE_OIL"


def test_unknown_node_is_rejected_rather_than_ignored():
    exposures, rejections = build_node_exposures(_frame([_row(node="NOPE")]), NODES)
    assert exposures == ()
    assert len(rejections) == 1
    assert "NOPE" in str(list(rejections)[0])


def test_missing_anchor_is_rejected_not_defaulted():
    """There is no sensible default anchor, and guessing one invents a deviation."""
    exposures, rejections = build_node_exposures(_frame([_row(anchor=None)]), NODES)
    assert exposures == ()
    assert "anchor" in str(list(rejections)[0]).lower()


def test_missing_elasticity_is_rejected_not_defaulted():
    exposures, rejections = build_node_exposures(_frame([_row(elasticity=None)]), NODES)
    assert exposures == ()
    assert "elasticity" in str(list(rejections)[0]).lower()


def test_a_symbol_with_no_node_is_not_applied_to_everything():
    """F11's worst branch: a missing key became a mask over every node."""
    exposures, rejections = build_node_exposures(_frame([_row(node="")]), NODES)
    assert exposures == ()
    assert len(rejections) == 1


def test_partially_filled_row_is_reported():
    """An anchor with no market is a half-finished edit, not a blank row."""
    row = _row(symbol="", elasticity=None)
    exposures, rejections = build_node_exposures(_frame([row]), NODES)
    assert exposures == ()
    assert "symbol" in str(list(rejections)[0]).lower()


@pytest.mark.parametrize("bad", [0.0, -10.0])
def test_non_positive_anchor_is_rejected(bad):
    _, rejections = build_node_exposures(_frame([_row(anchor=bad)]), NODES)
    assert len(rejections) == 1


def test_negative_elasticity_is_rejected_with_the_reason_named():
    """The acquired system produced this by accident via `- min(change, 0) * 0.5`."""
    _, rejections = build_node_exposures(_frame([_row(elasticity=-0.3)]), NODES)
    assert "hedge" in str(list(rejections)[0])


def test_duplicate_node_symbol_pair_is_rejected():
    _, rejections = build_node_exposures(_frame([_row(), _row()]), NODES)
    assert len(rejections) == 1


def test_two_markets_on_one_node_are_both_kept():
    exposures, rejections = build_node_exposures(
        _frame([_row(symbol="OIL"), _row(symbol="COPPER")]), NODES
    )
    assert not rejections
    assert {e.symbol for e in exposures} == {"OIL", "COPPER"}


def test_missing_node_id_column_is_reported_not_guessed():
    frame = pd.DataFrame({COL_EXP_SYMBOL: ["OIL"]})
    exposures, rejections = build_node_exposures(frame, NODES)
    assert exposures == ()
    assert len(rejections) == 1


def test_one_bad_row_does_not_discard_the_good_ones():
    exposures, rejections = build_node_exposures(
        _frame([_row(node="N1"), _row(node="GHOST"), _row(node="N2")]), NODES
    )
    assert {e.node_id for e in exposures} == {"N1", "N2"}
    assert len(rejections) == 1


# --------------------------------------------------------------------------- #
# Keeping the sheet aligned to the node set
# --------------------------------------------------------------------------- #


def test_align_preserves_entered_rows_when_a_node_is_added():
    aligned = align_exposure_frame(_frame([_row()]), (*NODES, "N4"))
    exposures, rejections = build_node_exposures(aligned, (*NODES, "N4"))
    assert not rejections
    assert exposures[0].symbol == "BRENT_CRUDE_OIL"
    assert "N4" in set(aligned[COL_EXP_NODE_ID])


def test_align_drops_rows_for_removed_nodes():
    """A stale row would make an unrelated edit fail in the engine's validation."""
    aligned = align_exposure_frame(_frame([_row(node="N3")]), ("N1", "N2"))
    assert "N3" not in set(aligned[COL_EXP_NODE_ID])
    _, rejections = build_node_exposures(aligned, ("N1", "N2"))
    assert not rejections


def test_align_handles_a_frame_it_has_never_seen():
    aligned = align_exposure_frame(pd.DataFrame({"nonsense": [1]}), NODES)
    assert list(aligned.columns) == list(adapters.EXPOSURE_COLUMNS)


# --------------------------------------------------------------------------- #
# Quotes
# --------------------------------------------------------------------------- #


def test_an_unavailable_quote_has_no_price_at_all():
    """The acquired table initialised every row to a plausible 100.00."""
    quotes = quotes_from_prices({"OIL": None})
    assert quotes.get("OIL").price is None
    assert not quotes.get("OIL").is_usable


@pytest.mark.parametrize("bad", [0.0, -5.0, math.nan, math.inf])
def test_an_unusable_price_is_treated_as_no_price(bad):
    quotes = quotes_from_prices({"OIL": bad})
    assert not quotes.get("OIL").is_usable


def test_quote_lookup_is_case_insensitive():
    assert quotes_from_prices({"oil": 90.0}).get("OIL").is_usable


def test_partial_availability_is_reported_per_symbol():
    """One global status would either hide the gap or discard the good readings."""
    quotes = quotes_from_prices({"OIL": 90.0, "COPPER": None})
    assert quotes.usable == ("OIL",)
    assert quotes.unavailable == ("COPPER",)
    assert quotes.any_live


def test_a_feed_missing_its_credential_says_so_per_symbol_and_invents_nothing(monkeypatch):
    """The keyed provider, driven through the market service.

    Formerly this exercised the only provider there was. The default feed now needs no
    credential, so the provider is named explicitly: an unconfigured *keyed* feed is
    still a state a real installation reaches, and it must still produce no prices.
    """
    monkeypatch.delenv("SCRCAE_MARKET_API_KEY", raising=False)

    quotes = fetch_quotes(["OIL", "COPPER"], provider=ApiNinjasProvider())

    assert quotes.usable == ()
    assert set(quotes.unavailable) == {"OIL", "COPPER"}
    for symbol in ("OIL", "COPPER"):
        assert quotes.get(symbol).price is None
        assert "SCRCAE_MARKET_API_KEY" in quotes.get(symbol).status


def test_the_provider_that_answered_is_named_in_every_status():
    """So a fallback to the default provider is legible on screen.

    Without this, an installation whose configured feed name was misspelled would show
    prices from a different source than its operator believed, and nothing would say so.
    """
    quotes = fetch_quotes(["BZ=F"], provider=StaticProvider(prices={"BZ=F": 90.0}))

    assert quotes.get("BZ=F").price == 90.0
    assert "[static]" in quotes.get("BZ=F").status


def test_a_symbol_the_provider_ignored_is_reported_rather_than_absent():
    """A missing key in the quote map reads downstream as "never asked for".

    A provider that silently drops a symbol would make a configured exposure look
    unconfigured, which is the one reading the market panel must never give.
    """

    class Forgetful:
        name = "forgetful"
        symbol_help = ""

        def latest(self, symbols, *, timeout=4.0):
            return ()

        def history(self, symbol, *, months=60, timeout=8.0):
            raise AssertionError("not used")

    quotes = fetch_quotes(["BZ=F"], provider=Forgetful())

    assert quotes.get("BZ=F").price is None
    assert "no reading" in quotes.get("BZ=F").status


def test_fetching_history_for_a_blank_symbol_makes_no_request():
    class Exploding:
        name = "exploding"
        symbol_help = ""

        def latest(self, symbols, *, timeout=4.0):
            raise AssertionError("the feed was contacted")

        def history(self, symbol, *, months=60, timeout=8.0):
            raise AssertionError("the feed was contacted")

    assert fetch_history("   ", provider=Exploding()).ok is False


def test_fetch_with_no_symbols_makes_no_claims():
    assert fetch_quotes([]).quotes == {}


# --------------------------------------------------------------------------- #
# The overlay
# --------------------------------------------------------------------------- #


def test_a_rise_above_anchor_raises_only_the_exposed_node():
    """The whole point of F11's replacement."""
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    overlay = build_overlay(exposures, quotes_from_prices({"OIL": 96.0}))
    # 96 vs 80 is +20%; elasticity 0.5 gives +0.10
    assert overlay.multipliers == {"N1": pytest.approx(1.10)}
    assert "N2" not in overlay.multipliers


def test_a_fall_below_anchor_does_nothing_when_one_sided():
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    overlay = build_overlay(exposures, quotes_from_prices({"OIL": 40.0}))
    assert overlay.multipliers == {}
    assert overlay.applied_effects[0].applied


def test_two_sided_mode_lets_a_fall_lower_the_multiplier():
    """The asymmetry is an assumption, so it has to be visibly switchable."""
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    overlay = build_overlay(
        exposures, quotes_from_prices({"OIL": 40.0}), one_sided=False
    )
    assert overlay.multipliers["N1"] == pytest.approx(0.75)


def test_several_exposures_on_one_node_are_summed():
    exposures = (
        NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),
        NodeExposure("N1", "COPPER", anchor=100.0, elasticity=0.2),
    )
    overlay = build_overlay(
        exposures, quotes_from_prices({"OIL": 96.0, "COPPER": 150.0})
    )
    # +0.10 from oil, +0.10 from copper
    assert overlay.multipliers["N1"] == pytest.approx(1.20)


def test_an_unavailable_quote_leaves_the_node_untouched_and_says_so():
    """The condition the acquired system shipped in permanently."""
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    overlay = build_overlay(exposures, quotes_from_prices({"OIL": None}))
    assert overlay.multipliers == {}
    assert len(overlay.skipped_effects) == 1
    assert not overlay
    assert "No market adjustment applied" in overlay.summary()


def test_an_exposure_with_no_quote_requested_is_reported_not_dropped():
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    overlay = build_overlay(exposures, QuoteSet())
    assert overlay.skipped_effects[0].node_id == "N1"


def test_a_working_and_a_broken_market_coexist():
    exposures = (
        NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),
        NodeExposure("N2", "COPPER", anchor=100.0, elasticity=0.5),
    )
    overlay = build_overlay(
        exposures, quotes_from_prices({"OIL": 96.0, "COPPER": None})
    )
    assert set(overlay.multipliers) == {"N1"}
    assert len(overlay.applied_effects) == 1
    assert len(overlay.skipped_effects) == 1
    assert "did nothing" in overlay.summary()


def test_multiplier_is_floored_and_the_floor_is_reported():
    """A zero multiplier would keep the cost and delete the entire benefit."""
    exposures = (NodeExposure("N1", "OIL", anchor=100.0, elasticity=5.0),)
    overlay = build_overlay(
        exposures, quotes_from_prices({"OIL": 1.0}), one_sided=False
    )
    assert overlay.multipliers["N1"] == MIN_MULTIPLIER
    assert overlay.clamped == ("N1",)
    assert "Floored" in overlay.summary()


def test_an_overlay_that_changes_nothing_is_empty_not_a_mapping_of_ones():
    """Empty is provably a no-op in the engine; a mapping of ones merely behaves so."""
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    overlay = build_overlay(exposures, quotes_from_prices({"OIL": 80.0}))
    assert overlay.multipliers == {}


def test_every_effect_carries_its_own_arithmetic():
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    effect = build_overlay(exposures, quotes_from_prices({"OIL": 96.0})).effects[0]
    assert effect.deviation == pytest.approx(0.20)
    assert effect.contribution == pytest.approx(0.10)
    assert "+20.0%" in effect.note


def test_no_exposure_configured_says_exactly_that():
    assert "No market exposure configured" in build_overlay((), QuoteSet()).summary()


# --------------------------------------------------------------------------- #
# The overlay is an overlay
# --------------------------------------------------------------------------- #


def test_building_an_overlay_does_not_mutate_the_exposures_or_quotes():
    """The acquired bridge wrote its output back over the user's input table, so
    every 30-minute sync compounded on the last and the entered figures were
    unrecoverable. Nothing here may write to its inputs."""
    exposures = (NodeExposure("N1", "OIL", anchor=80.0, elasticity=0.5),)
    quotes = quotes_from_prices({"OIL": 96.0})
    before_exposures = tuple(exposures)
    before_price = quotes.get("OIL").price

    build_overlay(exposures, quotes)
    build_overlay(exposures, quotes)
    third = build_overlay(exposures, quotes)

    assert tuple(exposures) == before_exposures
    assert quotes.get("OIL").price == before_price
    # Repeated application is idempotent: no compounding.
    assert third.multipliers["N1"] == pytest.approx(1.10)


def test_quote_and_exposure_objects_are_immutable():
    quote = MarketQuote(symbol="OIL", price=90.0, status="Live", is_live=True)
    with pytest.raises((AttributeError, TypeError)):
        quote.price = 100.0  # type: ignore[misc]
