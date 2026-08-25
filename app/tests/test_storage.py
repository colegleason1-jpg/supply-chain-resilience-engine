"""Tests for persistence.

The two properties worth the most here are isolation between owners and honesty about
staleness. Everything else is round-tripping, which matters mainly because the acquired
system's editors coerced blanks to zeros and a save/load cycle is a second chance to
make that mistake.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import adapters
from app.auth.models import ANONYMOUS, Principal
from app.services.portfolios import PortfolioService
from app.storage.models import RunProvenance, SavedPortfolio
from app.storage.serialization import (
    frame_to_records,
    records_to_frame,
    records_to_tables,
    tables_to_records,
)
from app.storage.sqlite_store import SqlitePortfolioStore


@pytest.fixture
def store(tmp_path):
    store = SqlitePortfolioStore(tmp_path / "test.db")
    yield store
    store.close()


@pytest.fixture
def alice(store):
    return PortfolioService(store, Principal(user_id="u-alice", email="alice@example.com"))


@pytest.fixture
def bob(store):
    return PortfolioService(store, Principal(user_id="u-bob", email="bob@example.com"))


def _tables():
    return {
        "nodes": adapters.default_nodes_frame(),
        "bundles": adapters.default_bundles_frame(),
        "dependencies": adapters.default_dependencies_frame(),
        "correlation": adapters.default_correlation_frame(["N1", "N2"]),
    }


# --------------------------------------------------------------------------- #
# Round-tripping
# --------------------------------------------------------------------------- #


def test_a_saved_portfolio_survives_a_new_store_object(tmp_path):
    """The actual requirement: closing the tab must not discard the portfolio."""
    path = tmp_path / "persist.db"
    first = SqlitePortfolioStore(path)
    service = PortfolioService(first, ANONYMOUS)
    saved = service.save("Q3 plan", tables=_tables(), settings={"budget": 750_000.0})
    first.close()

    second = SqlitePortfolioStore(path)
    reloaded = PortfolioService(second, ANONYMOUS).load(saved.portfolio_id)
    second.close()

    assert reloaded is not None
    assert reloaded.name == "Q3 plan"
    assert len(reloaded.tables["nodes"]) == 5
    assert reloaded.settings["budget"] == 750_000.0


def test_reloaded_tables_still_build_a_valid_network(alice):
    """A round trip must not turn usable input into unusable input, or vice versa."""
    saved = alice.save("plan", tables=_tables(), settings={})
    loaded = alice.load(saved.portfolio_id)

    network, rejections = adapters.build_network(loaded.tables["nodes"], 65.5)
    assert not rejections
    assert len(network.interventions) == 5


def test_a_blank_cell_reloads_as_blank_not_as_zero(alice):
    """The one that would be dangerous.

    `adapters.build_interventions` rejects a row with a missing cost, and that
    rejection is what stops a zero-cost intervention from being funded to the maximum
    at any budget. If a save/load cycle turned the blank into 0.0, the reload would
    launder an unusable row into an irresistible one.
    """
    frames = _tables()
    frames["nodes"].loc[0, adapters.COL_COST] = None

    saved = alice.save("with a gap", tables=frames, settings={})
    loaded = alice.load(saved.portfolio_id)

    _, rejections = adapters.build_interventions(loaded.tables["nodes"])
    assert len(rejections) == 1
    assert "Cost is missing" in rejections.items[0]


def test_nan_is_serialised_as_null(alice):
    frame = pd.DataFrame([{"a": float("nan"), "b": 1.0}])
    records = frame_to_records(frame)
    assert records[0]["a"] is None
    assert records[0]["b"] == 1.0


def test_numeric_columns_do_not_become_strings(alice):
    saved = alice.save("plan", tables=_tables(), settings={})
    loaded = alice.load(saved.portfolio_id)
    costs = loaded.tables["nodes"][adapters.COL_COST]
    assert all(isinstance(c, (int, float)) for c in costs)


def test_an_empty_table_reloads_with_its_column_headers(alice):
    """Otherwise the editor comes back with no columns and cannot be typed into."""
    frames = _tables()
    frames["bundles"] = pd.DataFrame(columns=frames["bundles"].columns)

    saved = alice.save("no bundles", tables=frames, settings={})
    loaded = alice.load(saved.portfolio_id)

    assert list(loaded.tables["bundles"].columns) == list(
        adapters.default_bundles_frame().columns
    )


def test_apostrophes_in_names_are_stored_verbatim(alice):
    """Parameterised queries make sanitising unnecessary, so nothing is stripped.

    The acquired system ran user strings through a `sanitize_input()` filter before
    interpolating them into SQL. Binding parameters means the driver never parses user
    data as SQL, so the name can simply be the name.
    """
    frames = _tables()
    frames["nodes"].loc[0, adapters.COL_NAME] = "O'Brien's Port; DROP TABLE portfolios;--"

    saved = alice.save("O'Brien's plan", tables=frames, settings={})
    loaded = alice.load(saved.portfolio_id)

    assert loaded.name == "O'Brien's plan"
    assert loaded.tables["nodes"].loc[0, adapters.COL_NAME].startswith("O'Brien's Port")
    assert alice.list()  # the table still exists


def test_unicode_and_emoji_survive(alice):
    frames = _tables()
    frames["nodes"].loc[0, adapters.COL_NAME] = "Kraków hub — 港"
    saved = alice.save("plan", tables=frames, settings={})
    loaded = alice.load(saved.portfolio_id)
    assert loaded.tables["nodes"].loc[0, adapters.COL_NAME] == "Kraków hub — 港"


def test_extra_user_added_columns_are_preserved(alice):
    frames = _tables()
    frames["nodes"]["Owner"] = "ops"
    saved = alice.save("plan", tables=frames, settings={})
    loaded = alice.load(saved.portfolio_id)
    assert "Owner" in loaded.tables["nodes"].columns


def test_records_to_frame_handles_none():
    assert records_to_frame(None, ["a", "b"]).empty
    assert list(records_to_frame(None, ["a", "b"]).columns) == ["a", "b"]


def test_tables_round_trip_through_records():
    original = _tables()
    restored = records_to_tables(tables_to_records(original))
    assert len(restored["nodes"]) == len(original["nodes"])


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #


def test_one_owner_cannot_read_anothers_portfolio(alice, bob):
    """Scoped in the WHERE clause, not checked in the view."""
    saved = alice.save("confidential", tables=_tables(), settings={})

    assert bob.load(saved.portfolio_id) is None
    assert bob.list() == []


def test_one_owner_cannot_delete_anothers_portfolio(alice, bob):
    saved = alice.save("confidential", tables=_tables(), settings={})

    assert bob.delete(saved.portfolio_id) is False
    assert alice.load(saved.portfolio_id) is not None


def test_a_wrong_owner_is_indistinguishable_from_a_missing_portfolio(alice, bob):
    """No oracle: probing ids must not reveal which ones exist."""
    real = alice.save("real", tables=_tables(), settings={})

    assert bob.load(real.portfolio_id) is None
    assert bob.load("00000000-0000-0000-0000-000000000000") is None


def test_two_owners_can_use_the_same_portfolio_name(alice, bob):
    alice.save("Q3 plan", tables=_tables(), settings={"budget": 1.0})
    bob.save("Q3 plan", tables=_tables(), settings={"budget": 2.0})

    assert len(alice.list()) == 1
    assert len(bob.list()) == 1
    assert alice.load(alice.list()[0].portfolio_id).settings["budget"] == 1.0


# --------------------------------------------------------------------------- #
# Naming and updates
# --------------------------------------------------------------------------- #


def test_saving_the_same_name_twice_updates_rather_than_duplicating(alice):
    """Two portfolios with one name are indistinguishable in a list."""
    first = alice.save("Q3 plan", tables=_tables(), settings={"budget": 100.0})
    second = alice.save("Q3 plan", tables=_tables(), settings={"budget": 200.0})

    assert first.portfolio_id == second.portfolio_id
    assert len(alice.list()) == 1
    assert alice.load(second.portfolio_id).settings["budget"] == 200.0


def test_an_update_preserves_the_original_creation_time(alice):
    first = alice.save("plan", tables=_tables(), settings={})
    second = alice.save("plan", tables=_tables(), settings={"budget": 1.0})
    assert second.created_at == first.created_at


def test_listing_is_most_recently_updated_first(alice):
    alice.save("older", tables=_tables(), settings={})
    alice.save("newer", tables=_tables(), settings={})
    alice.save("older", tables=_tables(), settings={"budget": 1.0})

    assert [p.name for p in alice.list()] == ["older", "newer"]


def test_names_are_stripped(alice):
    saved = alice.save("  padded  ", tables=_tables(), settings={})
    assert saved.name == "padded"


def test_a_nameless_portfolio_is_rejected():
    with pytest.raises(ValueError, match="needs a name"):
        SavedPortfolio(name="   ", owner_id="u-1")


def test_an_ownerless_portfolio_is_rejected():
    with pytest.raises(ValueError, match="needs an owner"):
        SavedPortfolio(name="plan", owner_id="")


def test_delete_removes_it(alice):
    saved = alice.save("temporary", tables=_tables(), settings={})
    assert alice.delete(saved.portfolio_id) is True
    assert alice.load(saved.portfolio_id) is None
    assert alice.delete(saved.portfolio_id) is False


# --------------------------------------------------------------------------- #
# Provenance: results are not stored, they are re-derived and checked
# --------------------------------------------------------------------------- #


def test_results_are_not_persisted(alice):
    """Only inputs and provenance. A stored risk figure ages into a false claim."""
    from app.tests.test_pipeline import _pipeline

    bundle, _ = _pipeline()
    saved = alice.save(
        "plan", tables=_tables(), settings={}, result=bundle.result
    )

    blob = str(saved.tables) + str(saved.settings)
    assert "allocations" not in blob
    assert str(round(bundle.result.net_capital)) not in blob
    # What is kept is enough to check reproducibility, and nothing more.
    assert saved.provenance.input_hash.startswith("sha256:")
    assert saved.provenance.output_hash.startswith("sha256:")


def test_provenance_recognises_a_reproduced_result(alice):
    from app.tests.test_pipeline import _pipeline

    bundle, _ = _pipeline()
    provenance = RunProvenance.from_result(bundle.result)

    again, _ = _pipeline()
    assert provenance.agrees_with(again.result)
    assert provenance.divergence(again.result) == ""


def test_provenance_detects_changed_inputs(alice):
    from app.tests.test_pipeline import _pipeline

    bundle, _ = _pipeline()
    provenance = RunProvenance.from_result(bundle.result)

    different, _ = _pipeline(budget=300_000.0)

    assert not provenance.agrees_with(different.result)
    assert "no longer hash to what was saved" in provenance.divergence(different.result)


def test_provenance_with_no_recorded_solve_says_so():
    from app.tests.test_pipeline import _pipeline

    bundle, _ = _pipeline()
    assert "nothing to compare against" in RunProvenance().divergence(bundle.result)


def test_an_engine_version_change_is_reported_as_such(alice):
    """Identical inputs, different answer: the engine moved under the saved portfolio."""
    from app.tests.test_pipeline import _pipeline

    bundle, _ = _pipeline()
    stale = RunProvenance(
        engine_version="0.0.1",
        input_hash=bundle.result.audit["input_hash"],
        output_hash="sha256:something-else-entirely",
    )

    assert not stale.agrees_with(bundle.result)
    message = stale.divergence(bundle.result)
    assert "0.0.1" in message
    assert "will not match anything" in message


def test_saving_without_a_result_is_allowed(alice):
    """An unsolved portfolio is still worth keeping."""
    saved = alice.save("draft", tables=_tables(), settings={})
    assert saved.provenance.output_hash == ""
    assert alice.load(saved.portfolio_id) is not None


# --------------------------------------------------------------------------- #
# The market exposure table is part of the portfolio
# --------------------------------------------------------------------------- #


def _exposure_frame():
    frame = adapters.align_exposure_frame(None, ["N1", "N2"])
    frame.loc[0, adapters.COL_EXP_SYMBOL] = "BRENT_CRUDE_OIL"
    frame.loc[0, adapters.COL_EXP_ANCHOR] = 80.0
    frame.loc[0, adapters.COL_EXP_ELASTICITY] = 0.2
    return frame


def test_the_exposure_table_round_trips_with_the_portfolio(alice):
    """An exposure mapping is an assertion the user made about their network. Losing
    it on reload would leave a saved portfolio priced differently from when it was
    saved, with nothing on screen indicating why."""
    tables = {**_tables(), "exposure": _exposure_frame()}
    saved = alice.save("with exposure", tables=tables, settings={})
    loaded = alice.load(saved.portfolio_id)

    restored = loaded.tables["exposure"]
    exposures, rejections = adapters.build_node_exposures(restored, ["N1", "N2"])
    assert not rejections
    assert len(exposures) == 1
    assert exposures[0].symbol == "BRENT_CRUDE_OIL"
    assert exposures[0].anchor == pytest.approx(80.0)
    assert exposures[0].elasticity == pytest.approx(0.2)


def test_a_portfolio_saved_without_exposure_loads_with_an_empty_table():
    """Older saved portfolios predate this table and must not fail to open."""
    tables = records_to_tables(tables_to_records(_tables()), {"exposure": list(adapters.EXPOSURE_COLUMNS)})
    exposures, rejections = adapters.build_node_exposures(
        tables.get("exposure"), ["N1", "N2"]
    )
    assert exposures == ()
    assert not rejections
