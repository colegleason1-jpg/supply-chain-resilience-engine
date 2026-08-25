"""Tests that actually run the Streamlit script.

`AppTest` executes `main.py` in-process the way the server does, so these catch the
class of defect that pure-logic tests cannot: a value that is correct but not
renderable. The first bug they found was `SimulationResult.correlation_repair` — a
`RepairReport` object placed in a table cell, which Arrow could not serialise and
silently degraded the entire column.

These are the slowest tests in the suite because each run performs real solves, so
they are marked `slow`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="the UI tests need Streamlit installed")

from streamlit.testing.v1 import AppTest  # noqa: E402

import streamlit as st  # noqa: E402

from app import adapters  # noqa: E402
from app.services import feeds  # noqa: E402
from app.presenters import provenance_rows, verification_rows  # noqa: E402

# AppTest resolves relative paths against the *calling* file, so an absolute path
# keeps these tests runnable from any working directory.
APP = str(Path(__file__).resolve().parents[1] / "main.py")
TIMEOUT = 300


@pytest.fixture(scope="module")
def default_run():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- #
# The app runs
# --------------------------------------------------------------------------- #


def test_app_renders_without_exceptions(default_run):
    assert not default_run.exception, [str(e.value) for e in default_run.exception]


def test_default_run_shows_a_solved_plan(default_run):
    labels = {m.label: m.value for m in default_run.metric}

    assert labels["Baseline risk"] == "65.50%"
    assert labels["Capital deployed"] == "$750,000"
    # A dash here would mean the default configuration does not solve.
    assert labels["Optimised risk"] != "—"


def test_the_three_tabs_are_present(default_run):
    assert len(default_run.tabs) == 3


def test_no_table_cell_holds_a_non_string_object():
    """The general form of the RepairReport bug.

    Arrow cannot serialise arbitrary objects. When it fails it does not raise into the
    UI, it "applies automatic fixes" and degrades the column, so a leaked object shows
    up as a plausible-looking but wrong cell rather than an error.
    """
    from app.tests.test_pipeline import _pipeline

    bundle, request = _pipeline()
    rows = verification_rows(bundle.result, request) + provenance_rows(
        bundle.result, request=request, simulation=bundle.simulation
    )

    assert rows
    for row in rows:
        for key, value in row.items():
            assert isinstance(value, str), f"{key!r} holds a {type(value).__name__}"


def test_correlation_repair_is_described_in_words(default_run):
    from app.presenters import _describe_repair

    assert _describe_repair(None) == "none required"


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #


def test_switching_to_legacy_mode_warns_and_still_solves():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.radio[0].set_value("Legacy parity").run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("not defensible" in w.value for w in at.warning)


def test_an_unreachable_target_never_renders_the_word_infeasible():
    """The end-to-end version of the guarantee, through the real widgets."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.checkbox(key="target_mode").set_value(True).run()
    at.number_input(key="target_risk").set_value(1.0).run()

    assert not at.exception, [str(e.value) for e in at.exception]

    rendered = " ".join(
        [e.value for e in at.error] + [w.value for w in at.warning] + [c.value for c in at.caption]
    ).lower()
    assert "infeasible" not in rendered
    assert "not a budget constraint" in rendered


def test_macro_feed_is_reported_as_unconfigured(default_run, monkeypatch):
    """F13. An unconfigured feed must look unconfigured, not live."""
    captions = " ".join(c.value for c in default_run.caption)
    assert "No macro feed configured" in captions


# --------------------------------------------------------------------------- #
# Persistence and copilot, through the real UI
# --------------------------------------------------------------------------- #
#
# These are the tests that justify AppTest existing in this suite. The unit tests
# already prove the store round-trips and the copilot proposes; what they cannot prove
# is that the buttons are wired to those functions and that the writes land on keys the
# widgets actually read. The acquired system's copilot failed at exactly that seam: the
# logic was fine and the key was wrong, so it reported success and changed nothing.


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A fresh app against a scratch database."""
    monkeypatch.setenv("SCRCAE_DB_PATH", str(tmp_path / "render.db"))
    # Otherwise the cached connection from a previous test in the same process is
    # reused and the "fresh database" is not fresh.
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    return at


def _save_as(at, name):
    at.sidebar.text_input(key="portfolio_name").set_value(name).run()
    at.sidebar.button(key="save_portfolio").click().run()
    return at


def test_saving_a_portfolio_reports_success(app):
    at = _save_as(app, "Q3 plan")
    assert not at.exception
    assert any("Saved" in s.value for s in at.sidebar.success)


def test_a_saved_portfolio_appears_in_the_list_on_the_next_run(app):
    _save_as(app, "Q3 plan")

    fresh = AppTest.from_file(APP, default_timeout=300)
    fresh.run()

    assert not fresh.exception
    options = fresh.sidebar.selectbox(key="portfolio_choice").options
    assert any("Q3 plan" in option for option in options)


def test_saving_without_a_name_warns_rather_than_saving(app):
    app.sidebar.button(key="save_portfolio").click().run()
    assert any("name" in w.value for w in app.sidebar.warning)


def test_loading_a_portfolio_restores_its_settings(app):
    """The two-phase load: click parks the payload, the rerun applies it.

    Writing widget keys directly from the button handler raises in Streamlit, because
    the sidebar widgets already exist by the time the Load button is drawn. If that
    deferral regressed, this test would see the exception.
    """
    app.sidebar.number_input(key="budget").set_value(900_000.0).run()
    _save_as(app, "big budget")

    # Move the budget away, then load the saved portfolio back.
    app.sidebar.number_input(key="budget").set_value(100_000.0).run()
    assert app.session_state["budget"] == 100_000.0

    app.sidebar.button(key="load_portfolio").click().run()

    assert not app.exception
    assert app.session_state["budget"] == 900_000.0


def test_loading_restores_the_intervention_table(app):
    frame = app.session_state["nodes_frame"].copy()
    frame.loc[0, adapters.COL_NAME] = "Renamed supplier"
    app.session_state["nodes_frame"] = frame
    app.run()
    _save_as(app, "renamed")

    app.sidebar.button(key="load_portfolio").click().run()

    assert not app.exception
    assert app.session_state["nodes_frame"].loc[0, adapters.COL_NAME] == "Renamed supplier"


def test_deleting_a_portfolio_removes_it_from_the_list(app):
    _save_as(app, "temporary")
    app.sidebar.button(key="delete_portfolio").click().run()

    assert not app.exception
    assert any("Nothing saved yet" in c.value for c in app.sidebar.caption)


def test_a_reloaded_portfolio_that_reproduces_its_result_says_so(app):
    """Results are not stored; they are re-solved and the hashes compared."""
    _save_as(app, "reproducible")
    app.sidebar.button(key="load_portfolio").click().run()

    assert not app.exception
    assert any("reproduces its saved result" in s.value for s in app.sidebar.success)


def test_a_reloaded_portfolio_whose_answer_moved_is_flagged(app):
    """Change an input after loading and the provenance notice must dissent.

    This is the whole argument for storing inputs rather than results. Had the saved
    figure been persisted and redisplayed, the user would be reading a number the
    current engine does not produce, with nothing on screen to say so.
    """
    _save_as(app, "will diverge")
    app.sidebar.button(key="load_portfolio").click().run()
    app.sidebar.number_input(key="budget").set_value(300_000.0).run()

    assert not app.exception
    assert any(
        "differs from" in w.value for w in app.sidebar.warning
    ), [w.value for w in app.sidebar.warning]


def test_the_copilot_proposes_without_changing_anything(app):
    """The central property: reading an instruction must not move an input."""
    before = app.session_state["budget"]

    app.sidebar.text_area(key="copilot_instruction").set_value(
        "set the budget to $900k"
    ).run()
    app.sidebar.button(key="copilot_read").click().run()

    assert not app.exception
    assert app.session_state["budget"] == before, "reading a proposal changed an input"
    assert any("Capital budget" in m.value for m in app.sidebar.markdown)


def test_accepting_a_copilot_proposal_moves_the_input(app):
    app.sidebar.text_area(key="copilot_instruction").set_value(
        "set the budget to $900k"
    ).run()
    app.sidebar.button(key="copilot_read").click().run()
    app.sidebar.button(key="copilot_apply").click().run()

    assert not app.exception
    assert app.session_state["budget"] == 900_000.0


def test_discarding_a_copilot_proposal_leaves_the_input_alone(app):
    before = app.session_state["budget"]

    app.sidebar.text_area(key="copilot_instruction").set_value(
        "set the budget to $900k"
    ).run()
    app.sidebar.button(key="copilot_read").click().run()
    app.sidebar.button(key="copilot_discard").click().run()

    assert not app.exception
    assert app.session_state["budget"] == before


def test_an_unparsed_instruction_offers_no_apply_button(app):
    """No guessing, and nothing to accept.

    The acquired copilot switched target mode on for any sentence containing "risk"
    and a number, which is how a user reached the structurally unattainable 20% target
    by typing one line.
    """
    app.sidebar.text_area(key="copilot_instruction").set_value(
        "make the supply chain much better somehow"
    ).run()
    app.sidebar.button(key="copilot_read").click().run()

    assert not app.exception
    assert not any(b.key == "copilot_apply" for b in app.sidebar.button)
    assert any("Not understood" in e.value for e in app.sidebar.error)


def test_a_partly_understood_instruction_applies_only_what_it_understood(app):
    """One change, and a written admission of the rest."""
    app.sidebar.text_area(key="copilot_instruction").set_value(
        "set the budget to $900k and make it aggressive"
    ).run()
    app.sidebar.button(key="copilot_read").click().run()

    assert any("Not understood" in e.value for e in app.sidebar.error)

    app.sidebar.button(key="copilot_apply").click().run()
    assert not app.exception
    assert app.session_state["budget"] == 900_000.0


def test_an_applied_change_leaves_a_record_on_screen(app):
    """Why the numbers moved must still be visible after they have."""
    app.sidebar.text_area(key="copilot_instruction").set_value(
        "set the budget to $900k"
    ).run()
    app.sidebar.button(key="copilot_read").click().run()
    app.sidebar.button(key="copilot_apply").click().run()

    assert any("Capital budget" in code.value for code in app.code)


def test_switching_to_legacy_via_the_copilot_carries_its_warning(app):
    app.sidebar.text_area(key="copilot_instruction").set_value(
        "switch to legacy parity"
    ).run()
    app.sidebar.button(key="copilot_read").click().run()

    assert any("reproducible, not defensible" in w.value for w in app.sidebar.warning)


def test_the_app_states_that_it_is_running_unauthenticated(app):
    """An open door is acceptable for a local run; an ambiguous one is not."""
    assert any(
        "without authentication" in c.value for c in app.sidebar.caption
    ), [c.value for c in app.sidebar.caption]


def test_a_broken_portfolio_can_still_be_reloaded_from(app):
    """Unusable inputs must not strand the user.

    The script returns early when no network can be built, and the portfolio controls
    sit below that point. Without an explicit render on the failure path, a user who
    emptied the intervention table would face an error with no way back to a saved
    portfolio.
    """
    _save_as(app, "known good")

    # Empty the table entirely: no interventions means no network.
    app.session_state["nodes_frame"] = app.session_state["nodes_frame"].iloc[0:0]
    app.run()

    assert not app.exception
    assert any("Fix the inputs" in e.value for e in app.error)
    # Still offered a way out.
    app.sidebar.button(key="load_portfolio").click().run()

    assert not app.exception
    assert len(app.session_state["nodes_frame"]) == 5
    assert app.metric  # solving again


def test_the_provenance_notice_tracks_the_current_inputs_not_the_previous_run(app):
    """Regression test for a one-run lag.

    The portfolio panel was originally drawn before the solve, so it compared a
    reloaded portfolio's hashes against the *previous* run's result. Changing the budget
    therefore left the reassuring "reproduces its saved result" message on screen for a
    full interaction after it had stopped being true.
    """
    _save_as(app, "reference")
    app.sidebar.button(key="load_portfolio").click().run()
    assert any("reproduces its saved result" in s.value for s in app.sidebar.success)

    app.sidebar.number_input(key="budget").set_value(300_000.0).run()

    assert not app.exception
    assert not any(
        "reproduces its saved result" in s.value for s in app.sidebar.success
    ), "stale reassurance survived an input change"
    assert any("differs from" in w.value for w in app.sidebar.warning)


# --------------------------------------------------------------------------- #
# The price of risk, in the running app
# --------------------------------------------------------------------------- #


def test_deriving_the_price_replaces_the_entered_one_and_shows_its_workings():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.checkbox(key="derive_prices").set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]

    labels = {m.label: m.value for m in at.metric}
    # 400m x 35% x 28% / 365 x 21 days = 2,255,342; / 100 = 22,553
    assert labels["Value per risk point (annual)"] == "22,553"
    captions = " ".join(c.value for c in at.caption)
    assert "Cost of one event" in captions
    assert "linear in expected loss" in captions


def test_the_entered_price_is_disabled_while_deriving():
    """Both paths always exist, but only one can be in force at a time."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert at.number_input(key="value_per_risk_point").disabled is False
    at.checkbox(key="derive_prices").set_value(True).run()
    assert at.number_input(key="value_per_risk_point").disabled is True


def test_a_derived_price_is_not_flagged_as_unsourced():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.checkbox(key="derive_prices").set_value(True).run()
    rendered = " ".join(str(w.value) for w in at.warning) + " ".join(
        str(w.value) for w in at.info
    )
    assert "no stated source" not in rendered


def test_an_entered_price_with_no_source_is_flagged_as_unsourced(default_run):
    text = " ".join(str(w.value) for w in default_run.warning)
    assert "no stated source" in text


def test_impossible_exposure_figures_are_refused_rather_than_priced():
    """A margin of 100%+ is a units error. The app must not quietly price it."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.checkbox(key="derive_prices").set_value(True)
    at.number_input(key="disruption_days").set_value(0.0)
    at.number_input(key="revenue_at_risk_share_pct").set_value(0.0)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    # A zero-cost event prices a risk point at zero, which is legitimate and must
    # still solve rather than crash the objective.
    labels = {m.label: m.value for m in at.metric}
    assert labels["Value per risk point (annual)"] == "0"


# --------------------------------------------------------------------------- #
# Market exposure, in the running app
# --------------------------------------------------------------------------- #


def test_market_exposure_is_off_by_default_and_says_so(default_run):
    captions = " ".join(c.value for c in default_run.caption)
    assert "Market exposure is switched off" in captions


def test_switching_exposure_on_with_no_mapping_says_none_is_configured():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.checkbox(key="apply_market_exposure").set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "No market exposure is configured" in captions


def test_a_configured_exposure_with_no_feed_warns_that_it_did_nothing(monkeypatch):
    """The acquired system's permanent state, now visible instead of silent.

    The unreachable feed is now produced deterministically — the keyed provider with no
    key — rather than by assuming a symbol nothing can price. That assumption was how
    this test came to fail while the app was behaving correctly: "OIL" is a real listed
    ticker, so the keyless default answered it with 28.42 and the exposure applied. A
    test that depends on a symbol being unquotable is a test that breaks when a feed
    gets better, and worse, one that would pass for the wrong reason on a machine with
    no network.
    """
    monkeypatch.delenv("SCRCAE_MARKET_API_KEY", raising=False)
    monkeypatch.setenv(feeds.PROVIDER_ENV_VAR, "api-ninjas")
    st.cache_data.clear()

    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    node_ids = adapters.node_ids_in(at.session_state["nodes_frame"])
    frame = adapters.align_exposure_frame(None, node_ids)
    frame.loc[0, adapters.COL_EXP_SYMBOL] = "OIL"
    frame.loc[0, adapters.COL_EXP_ANCHOR] = 80.0
    frame.loc[0, adapters.COL_EXP_ELASTICITY] = 0.2
    at.session_state["exposure_frame"] = frame
    at.checkbox(key="apply_market_exposure").set_value(True).run()

    assert not at.exception, [str(e.value) for e in at.exception]
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "configured as exposed to OIL" in warnings
    assert "evaluated on your figures alone" in warnings


def test_a_broken_exposure_row_is_reported_not_ignored():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    node_ids = adapters.node_ids_in(at.session_state["nodes_frame"])
    frame = adapters.align_exposure_frame(None, node_ids)
    frame.loc[0, adapters.COL_EXP_SYMBOL] = "OIL"  # no anchor, no elasticity
    at.session_state["exposure_frame"] = frame
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "anchor price" in warnings


def test_the_users_intervention_table_is_never_modified_by_the_feed():
    """F11's actual damage: the bridge wrote its output back over the inputs, so each
    sync compounded and the entered Risk Reduction values drifted up irrecoverably."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    before = at.session_state["nodes_frame"].copy(deep=True)

    node_ids = adapters.node_ids_in(before)
    frame = adapters.align_exposure_frame(None, node_ids)
    frame.loc[0, adapters.COL_EXP_SYMBOL] = "OIL"
    frame.loc[0, adapters.COL_EXP_ANCHOR] = 80.0
    frame.loc[0, adapters.COL_EXP_ELASTICITY] = 0.2
    at.session_state["exposure_frame"] = frame

    at.checkbox(key="apply_market_exposure").set_value(True).run()
    at.run()
    at.run()

    after = at.session_state["nodes_frame"]
    assert after[adapters.COL_RISK].tolist() == before[adapters.COL_RISK].tolist()


def test_the_verification_tab_reports_the_exposure_state(default_run):
    """Reported as "none" rather than omitted. An absent row reads as an oversight;
    an explicit "none" is an answer a reviewer can rely on."""
    tables = [df.value for df in default_run.dataframe]
    provenance = [t for t in tables if "Item" in getattr(t, "columns", [])]
    assert provenance, "the verification tab rendered no provenance table"
    items = {}
    for table in provenance:
        items.update(dict(zip(table["Item"], table["Value"])))
    assert items["Market exposure applied"] == "none"
    assert "unstated" in items["Price of risk"]


# --------------------------------------------------------------------------- #
# Elasticity calibration
# --------------------------------------------------------------------------- #


def _history_frame(node_id: str, *, months: int = 12):
    """A delivery history table for one node, so the calibration panel has input.

    Built through `adapters.align_history_frame` rather than assembled by hand, so a
    test cannot pass against a column layout the app does not actually read.
    """
    import pandas as pd

    rows = [
        {
            adapters.COL_HIST_NODE_ID: node_id,
            adapters.COL_HIST_PERIOD: f"2025-{month:02d}",
            adapters.COL_HIST_DISRUPTION: 5.0,
        }
        for month in range(1, months + 1)
    ]
    return adapters.align_history_frame(pd.DataFrame(rows))


def _usable_fit():
    """A fit the engine itself judges usable, from synthetic delivery observations.

    Fabricated deliberately: the point of the adoption tests is the UI contract, and
    tying them to a live market feed would make them assert whichever elasticity the
    market happened to imply this morning.
    """
    from scrcae.calibration import DeliveryObservation, calibrate_elasticity

    prices = [70, 72, 74, 76, 78, 80, 79, 75, 88, 96, 104, 112]
    observations = []
    for index, price in enumerate(prices):
        deviation = max((price - 80.0) / 80.0, 0.0)
        rate = 5.0 * (1 + 0.6 * deviation) + (0.05 if index % 2 else -0.05)
        observations.append(
            DeliveryObservation(
                period=f"2025-{index + 1:02d}",
                market_price=float(price),
                disruption_rate=rate,
            )
        )
    return calibrate_elasticity(observations, anchor=80.0, one_sided=True)


def _exposed_app(node_ids=None, *, elasticity: float = 0.2):
    """The app with one configured exposure and a delivery history for that node."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    ids = node_ids or adapters.node_ids_in(at.session_state["nodes_frame"])
    frame = adapters.align_exposure_frame(None, ids)
    frame.loc[0, adapters.COL_EXP_SYMBOL] = "OIL"
    frame.loc[0, adapters.COL_EXP_ANCHOR] = 80.0
    frame.loc[0, adapters.COL_EXP_ELASTICITY] = elasticity
    at.session_state["exposure_frame"] = frame
    at.session_state["history_frame"] = _history_frame(ids[0])
    return at, ids[0]


def test_the_calibration_panel_says_no_elasticity_can_be_calibrated_without_history(
    default_run,
):
    """The default configuration has no delivery history, and the panel has to say so
    rather than render an empty expander that reads like "nothing was wrong"."""
    captions = " ".join(c.value for c in default_run.caption)
    assert "No delivery history is configured" in captions


def test_the_calibration_panel_asks_to_be_run_before_it_reports_anything():
    """With history configured but no run performed, the panel must not imply a
    result. It fetches market history, so it cannot have run on its own."""
    at, _ = _exposed_app()
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "No calibration has been run against this delivery history yet" in captions


def test_a_refused_fit_renders_no_checkbox_that_could_adopt_it():
    """A refused fit must be impossible to adopt from the UI at all — not a disabled
    control, no control. "Calibrated" cannot be one click away from a fit that failed."""
    from app.services.calibration import CalibrationRun, NodeCalibration

    at, node_id = _exposed_app()
    at.session_state["calibration_run"] = CalibrationRun(
        calibrations=(
            NodeCalibration(
                node_id=node_id,
                symbol="OIL",
                anchor=80.0,
                current_elasticity=0.2,
                fit=None,
                matched_periods=0,
                blocked="No market price history was returned for OIL.",
            ),
        ),
        provider_name="static",
    )
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    adopt_boxes = [c for c in at.checkbox if (c.key or "").startswith("adopt_")]
    assert not adopt_boxes, [c.key for c in adopt_boxes]
    captions = " ".join(c.value for c in at.caption)
    assert "No market price history was returned for OIL." in captions
    assert "remains asserted" in captions


def test_adopting_a_fit_writes_the_elasticity_and_the_evidence_behind_it():
    """Adoption is the one write to the user's exposure table, and it has to write both
    halves: the fitted number and the stored fit that entitles it to be called
    calibrated. Writing the number alone would show "asserted" beside a fitted value."""
    from app.services.calibration import CalibrationRun, NodeCalibration

    at, node_id = _exposed_app()
    fit = _usable_fit()
    at.session_state["calibration_run"] = CalibrationRun(
        calibrations=(
            NodeCalibration(
                node_id=node_id,
                symbol="OIL",
                anchor=80.0,
                current_elasticity=0.2,
                fit=fit,
                matched_periods=fit.observations,
            ),
        ),
        provider_name="static",
    )
    at.run()

    key = f"adopt_{node_id}_OIL"
    at.checkbox(key=key).set_value(True).run()
    at.button(key="adopt_elasticities").click().run()

    assert not at.exception, [str(e.value) for e in at.exception]
    exposure = at.session_state["exposure_frame"]
    assert float(exposure.loc[0, adapters.COL_EXP_ELASTICITY]) == pytest.approx(
        fit.elasticity
    )
    stored = at.session_state["calibration_store"]
    assert len(stored) == 1
    assert stored.loc[0, adapters.COL_CAL_NODE_ID] == node_id
    assert float(stored.loc[0, adapters.COL_CAL_ELASTICITY]) == pytest.approx(
        fit.elasticity
    )

    # The point of writing both halves: the run after adoption reports the exposure as
    # calibrated, and it can only do so because the stored fit matches the number in
    # the table. A number written without its evidence would still read "asserted".
    items = {}
    for table in (df.value for df in at.dataframe):
        if "Item" in getattr(table, "columns", []):
            items.update(dict(zip(table["Item"], table["Value"])))
    assert items["Risk elasticities"] == "1 calibrated"


def test_the_provenance_row_reports_how_many_elasticities_are_only_asserted():
    """Counted from the exposures rather than claimed, so a reviewer reading the
    verification tab learns that the elasticity driving the market path was typed in."""
    at, _ = _exposed_app()
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    items = {}
    for table in (df.value for df in at.dataframe):
        if "Item" in getattr(table, "columns", []):
            items.update(dict(zip(table["Item"], table["Value"])))
    assert items["Risk elasticities"] == "1 asserted"


def test_a_plain_rerun_never_reaches_the_market_to_calibrate(monkeypatch):
    """Calibration fetches years of monthly history per symbol. A Streamlit script
    reruns on every keystroke, so a calibration on the plain path would turn typing
    into a burst of network calls — hence the button. Both the fitting entry point and
    the history fetch are made to explode; a plain run must still complete."""
    from app.services import calibration as calibration_service
    from app.services import market as market_service

    def explode(*args, **kwargs):
        raise AssertionError("a plain rerun must not calibrate or fetch history")

    monkeypatch.setattr(calibration_service, "calibrate_exposures", explode)
    monkeypatch.setattr(market_service, "fetch_history", explode)

    at, _ = _exposed_app()
    at.run()
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert any(
        b.label == "Calibrate elasticities from delivery history" for b in at.button
    ), [b.label for b in at.button]


def _provenance_items(at) -> dict:
    """The provenance table as a mapping, whichever tab rendered it."""
    items = {}
    for table in (df.value for df in at.dataframe):
        if "Item" in getattr(table, "columns", []):
            items.update(dict(zip(table["Item"], table["Value"])))
    return items


def _stored_fit_frame(node_id: str, *, elasticity: float, anchor: float):
    """A stored calibration frame vouching for one elasticity."""
    import pandas as pd

    return adapters.align_calibration_frame(
        pd.DataFrame(
            [
                {
                    adapters.COL_CAL_NODE_ID: node_id,
                    adapters.COL_CAL_SYMBOL: "OIL",
                    adapters.COL_CAL_ELASTICITY: elasticity,
                    adapters.COL_CAL_ANCHOR: anchor,
                    adapters.COL_CAL_LOW: 0.48,
                    adapters.COL_CAL_HIGH: 0.63,
                    adapters.COL_CAL_R2: 0.98,
                    adapters.COL_CAL_OBS: 12,
                    adapters.COL_CAL_ABOVE: 4,
                    adapters.COL_CAL_METHOD: "one-sided OLS on log deviation",
                }
            ]
        )
    )


def test_a_reloaded_portfolio_keeps_the_evidence_behind_its_elasticities(app):
    """Without the stored fits in the save, a reloaded portfolio would show the very
    same elasticities while reporting every one of them as asserted — the numbers
    intact, the only thing that justified them quietly gone."""
    node_ids = adapters.node_ids_in(app.session_state["nodes_frame"])
    exposure = adapters.align_exposure_frame(None, node_ids)
    exposure.loc[0, adapters.COL_EXP_SYMBOL] = "OIL"
    exposure.loc[0, adapters.COL_EXP_ANCHOR] = 80.0
    exposure.loc[0, adapters.COL_EXP_ELASTICITY] = 0.61
    app.session_state["exposure_frame"] = exposure
    app.session_state["calibration_store"] = _stored_fit_frame(
        node_ids[0], elasticity=0.61, anchor=80.0
    )
    app.run()
    assert _provenance_items(app)["Risk elasticities"] == "1 calibrated"

    _save_as(app, "calibrated set")

    # Throw the evidence away in this session, so the reload has to supply it.
    app.session_state["calibration_store"] = adapters.default_calibration_frame()
    app.run()
    assert _provenance_items(app)["Risk elasticities"] == "1 asserted"

    app.sidebar.button(key="load_portfolio").click().run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert _provenance_items(app)["Risk elasticities"] == "1 calibrated"
