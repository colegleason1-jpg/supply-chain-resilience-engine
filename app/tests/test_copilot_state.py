"""Tests for the copilot-field to widget-key translation.

This module is small and boring and is where the acquired system's copilot actually
failed. It wrote `val_total_budget` into session state; the widget read a different key;
the write landed nowhere and the copilot reported success. So the mapping gets tested
directly rather than only through the UI.
"""

from __future__ import annotations

import pytest

from app import adapters
from app.services.copilot_state import (
    MODE_LABELS,
    apply_to_state,
    snapshot,
    widget_updates,
)


# --------------------------------------------------------------------------- #
# Reading state
# --------------------------------------------------------------------------- #


def test_an_empty_state_yields_the_sidebar_defaults():
    """On the first run no widget exists yet, so every key is absent.

    Returning zeros here would have the copilot propose changes away from values the
    user never chose.
    """
    reading = snapshot({})
    assert reading["mode"] == adapters.MODE_MONETARY
    assert reading["budget"] == 750_000.0
    assert reading["target_risk_pts"] is None
    assert reading["baseline_risk_pts"] == 65.5
    assert reading["exponent"] == 0.85
    assert reading["iterations"] == 10_000
    assert reading["seed"] == 42


def test_the_mode_label_is_translated_to_a_mode_value():
    """The radio holds a sentence; the engine wants an identifier."""
    assert snapshot({"mode": "Legacy parity"})["mode"] == adapters.MODE_LEGACY
    assert (
        snapshot({"mode": "Monetary NPV (recommended)"})["mode"] == adapters.MODE_MONETARY
    )


def test_an_unrecognised_mode_label_falls_back_to_monetary():
    """Never to legacy. Legacy reproduces known defects and must be chosen explicitly."""
    assert snapshot({"mode": "something else"})["mode"] == adapters.MODE_MONETARY


def test_target_mode_off_reads_the_budget_widget():
    reading = snapshot({"target_mode": False, "budget": 900_000.0, "max_capital": 1.0})
    assert reading["budget"] == 900_000.0
    assert reading["target_risk_pts"] is None


def test_target_mode_on_reads_the_target_and_the_capped_maximum():
    """In target mode the budget is an optional ceiling under a different key."""
    reading = snapshot(
        {
            "target_mode": True,
            "target_risk": 40.0,
            "cap_spend": True,
            "max_capital": 400_000.0,
            "budget": 750_000.0,
        }
    )
    assert reading["target_risk_pts"] == 40.0
    assert reading["budget"] == 400_000.0


def test_target_mode_with_no_cap_reads_no_budget():
    """Uncapped target mode has no budget at all, which is not the same as zero."""
    reading = snapshot({"target_mode": True, "target_risk": 40.0, "cap_spend": False})
    assert reading["budget"] is None


# --------------------------------------------------------------------------- #
# Writing state
# --------------------------------------------------------------------------- #


def test_a_mode_change_writes_the_radio_label_not_the_value():
    """Writing "legacy" into the radio's key would be rejected as an invalid option."""
    writes = widget_updates({"mode": adapters.MODE_LEGACY})
    assert writes["mode"] == MODE_LABELS[adapters.MODE_LEGACY]
    assert writes["mode"] in ("Legacy parity",)


def test_setting_a_target_switches_target_mode_on():
    """One nullable number on the copilot's side, two widgets on the sidebar's."""
    writes = widget_updates({"target_risk_pts": 35.0})
    assert writes["target_mode"] is True
    assert writes["target_risk"] == 35.0


def test_clearing_a_target_switches_target_mode_off():
    writes = widget_updates({"target_risk_pts": None})
    assert writes["target_mode"] is False
    assert "target_risk" not in writes


def test_a_budget_in_spend_mode_goes_to_the_budget_widget():
    writes = widget_updates({"budget": 900_000.0, "target_risk_pts": None})
    assert writes["budget"] == 900_000.0
    assert "max_capital" not in writes


def test_a_budget_in_target_mode_goes_to_the_capped_maximum():
    """The destination depends on the mode *after* the proposal, not before it.

    A single instruction can set a target and a budget at once. Routing the budget by
    the pre-existing mode would put it in the widget that mode no longer reads.
    """
    writes = widget_updates({"budget": 400_000.0, "target_risk_pts": 30.0})
    assert writes["max_capital"] == 400_000.0
    assert writes["cap_spend"] is True
    assert "budget" not in writes


def test_clearing_the_budget_in_target_mode_unchecks_the_cap():
    writes = widget_updates({"budget": None, "target_risk_pts": 30.0})
    assert writes["cap_spend"] is False
    assert "max_capital" not in writes


def test_only_the_named_fields_are_written():
    """A proposal about the budget must not silently move anything else."""
    writes = widget_updates({"budget": 900_000.0})
    assert set(writes) == {"budget"}


def test_every_simple_field_maps_to_a_key():
    writes = widget_updates(
        {
            "exponent": 0.5,
            "value_per_risk_point": 60_000.0,
            "value_per_lead_time_day": 100.0,
            "iterations": 50_000,
            "seed": 7,
            "run_simulation": False,
            "show_sweep": False,
            "baseline_risk_pts": 70.0,
        }
    )
    assert writes == {
        "exponent": 0.5,
        "value_per_risk_point": 60_000.0,
        "value_per_day": 100.0,
        "iterations": 50_000,
        "seed": 7,
        "run_simulation": False,
        "show_sweep": False,
        "baseline_risk": 70.0,
    }


def test_apply_to_state_writes_and_reports():
    state: dict = {}
    written = apply_to_state(state, {"budget": 900_000.0})
    assert state["budget"] == 900_000.0
    assert written == {"budget": 900_000.0}


# --------------------------------------------------------------------------- #
# Round trip: the property that catches a drifted key
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction, state",
    [
        ("set the budget to $900k", {}),
        ("switch to legacy parity", {}),
        ("target 40% risk", {}),
        ("turn off the Monte Carlo", {}),
        ("use 50,000 iterations and seed 7", {}),
        ("set the exponent to 0.5", {}),
        # Clearing needs something to clear: against the default state the copilot
        # correctly proposes nothing, because no target is set.
        ("clear the target", {"target_mode": True, "target_risk": 40.0}),
        (
            "cap the spend at 400k",
            {"target_mode": True, "target_risk": 40.0, "cap_spend": False},
        ),
    ],
)
def test_an_applied_proposal_is_visible_in_the_next_snapshot(instruction, state):
    """The end-to-end property, and the one the acquired system violated.

    Interpret against a snapshot, apply, write to state, read back: the new snapshot
    must show the change. If any widget key were misspelled the write would land on a
    key nothing reads, and this assertion is what notices.
    """
    from app import copilot

    state = dict(state)
    before = snapshot(state)
    proposal = copilot.interpret(instruction, before)
    assert proposal.changes, f"nothing proposed for {instruction!r}"

    updated = copilot.apply_proposal(before, proposal)
    apply_to_state(state, updated)
    after = snapshot(state)

    for change in proposal.changes:
        assert after[change.field] == change.proposed, (
            f"{change.field} did not survive the round trip through session state"
        )


# --------------------------------------------------------------------------- #
# Pricing and exposure inputs survive a save and reload
# --------------------------------------------------------------------------- #


def test_the_pricing_inputs_round_trip_through_a_snapshot():
    """A reloaded portfolio must reopen at the price it was saved at.

    Without this, reopening a saved portfolio silently reverted the price of risk to
    the sidebar default and every currency figure changed, while the plan, the budget
    and the audit's reproducibility claim all looked untouched.
    """
    state = {
        "derive_prices": True,
        "annual_revenue": 250_000_000.0,
        "revenue_at_risk_share_pct": 42.0,
        "gross_margin_pct": 31.0,
        "disruption_days": 14.0,
        "fixed_cost_per_event": 750_000.0,
        "event_probability_pct": 55.0,
        "exposure_basis": "FY26 board pack",
        "price_source": "group risk model",
        "apply_macro": True,
        "apply_market_exposure": True,
    }
    taken = snapshot(state)
    writes = widget_updates(taken)
    for key, value in state.items():
        assert taken[key] == value
        assert writes[key] == value


def test_the_pricing_defaults_match_the_sidebar_on_a_first_run():
    taken = snapshot({})
    assert taken["derive_prices"] is False
    assert taken["apply_market_exposure"] is False
    assert taken["price_source"] == ""
    assert taken["annual_revenue"] == 400_000_000.0


def test_no_copilot_rule_can_propose_a_pricing_input():
    """They are persisted, not proposable. A copilot that could rewrite the revenue
    figure could move every currency number in the app from one sentence."""
    from app import copilot
    from app.services.copilot_state import PRICING_KEYS

    proposable = set()
    for field in PRICING_KEYS:
        draft = copilot.interpret(f"set {field} to 5", snapshot({}))
        if draft.changes:
            proposable.update(c.field for c in draft.changes)
    assert not proposable & set(PRICING_KEYS)
