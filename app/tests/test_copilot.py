"""Tests for the copilot, with emphasis on what it must refuse to do.

The acquired system's copilot is the reason this module exists, and three of its
habits are pinned here as prohibitions:

* It wrote straight into `st.session_state` and reran the script, so a sentence moved
  every number on screen with no record of what had been asked for. Nothing here
  mutates anything: `interpret` returns a proposal and `apply_proposal` returns a new
  dict.
* It guessed. Any sentence containing "risk" and a digit switched target mode on,
  which is how users reached the structurally unattainable 20% target of F14 in one
  line. An instruction this module does not recognise must produce zero changes.
* It scaled bare numbers by a million below a threshold of 1000, so the meaning of a
  sentence changed discontinuously at a boundary no user could see.

The tests assert on behaviour a reviewer would notice — which field moved, what the
user is warned about, what was admitted as not understood — rather than on wording,
except where the wording is the point.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect

import pytest

from app import copilot
from app.adapters import MODE_LEGACY, MODE_MONETARY
from app.copilot import (
    AGGRESSIVE_TARGET_PTS,
    Change,
    Proposal,
    StaleProposal,
    apply_proposal,
    interpret,
    parse_magnitude,
    parse_percent,
)

#: The sidebar's shipped defaults, so a proposal in these tests is a proposal against
#: the state a user actually starts from.
DEFAULTS: dict[str, object] = {
    "mode": MODE_MONETARY,
    "baseline_risk_pts": 65.5,
    "budget": 750_000.0,
    "target_risk_pts": None,
    "exponent": 0.85,
    "value_per_risk_point": 50_000.0,
    "iterations": 10_000,
    "seed": 42,
    "run_simulation": True,
    "show_sweep": True,
}


def snapshot(**overrides: object) -> dict[str, object]:
    return {**DEFAULTS, **overrides}


def only_change(proposal: Proposal) -> Change:
    """Assert exactly one change and return it.

    Used everywhere a single-subject instruction is tested, because "did not invent a
    second change" is half of what is being checked.
    """
    assert len(proposal.changes) == 1, [c.describe() for c in proposal.changes]
    return proposal.changes[0]


def fields(proposal: Proposal) -> set[str]:
    return {change.field for change in proposal.changes}


def warned_about(proposal: Proposal, fragment: str) -> bool:
    return any(fragment.lower() in warning.lower() for warning in proposal.warnings)


def unresolved_about(proposal: Proposal, fragment: str) -> bool:
    return any(fragment.lower() in entry.lower() for entry in proposal.unresolved)


# --------------------------------------------------------------------------- #
# The module's own boundaries
# --------------------------------------------------------------------------- #


def test_the_module_cannot_touch_the_interface_or_the_solver():
    """The defect was a parser with write access to session state and a rerun.

    Read off the parsed syntax tree rather than `sys.modules`, so the assertion holds
    regardless of what the rest of the suite has already imported, and off the tree
    rather than the source text because the prose in this module names
    `st.session_state` repeatedly. What matters is that the code cannot reach it.
    """
    tree = ast.parse(inspect.getsource(copilot))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert not {name.split(".")[0] for name in imported} & {"streamlit", "scrcae"}
    assert "app.engine_client" not in imported
    assert "session_state" not in attributes
    assert "rerun" not in attributes


def test_interpret_is_deterministic():
    """A proposal a user reviews has to be the proposal that gets applied."""
    first = interpret("increase the budget by 20% and hit 40% risk", snapshot())
    second = interpret("increase the budget by 20% and hit 40% risk", snapshot())

    assert first == second


def test_proposals_and_changes_are_immutable():
    """A reviewed proposal that can be edited afterwards is not a record of anything."""
    proposal = interpret("set the budget to 500k", snapshot())

    with pytest.raises(dataclasses.FrozenInstanceError):
        proposal.changes[0].proposed = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$500k", 500_000.0),
        ("500,000", 500_000.0),
        ("500000", 500_000.0),
        ("0.5m", 500_000.0),
        ("750K", 750_000.0),
        ("1.2m", 1_200_000.0),
        ("2 million", 2_000_000.0),
        ("1.5bn", 1_500_000_000.0),
        ("300 thousand", 300_000.0),
    ],
)
def test_money_is_parsed_in_every_form_a_person_writes_it(text, expected):
    assert parse_magnitude(text) == expected


def test_a_bare_number_is_taken_at_face_value():
    """The acquired copilot multiplied bare numbers under 1000 by a million.

    `target_val if target_val > 1000 else target_val * 1_000_000` means "budget 900"
    and "budget 1100" differ by six orders of magnitude, and nothing on screen said
    which reading had been used.
    """
    assert parse_magnitude("set the budget to 900") == 900.0
    assert parse_magnitude("set the budget to 1100") == 1100.0


@pytest.mark.parametrize("text", ["40%", "40 percent", "40 pct", "risk under 40%"])
def test_percentages_parse_from_symbol_and_word(text):
    assert parse_percent(text) == 40.0


def test_a_trailing_percent_sign_is_not_missed():
    """`\\b` after `%` demands a word character, so "by 20%" would silently fail.

    It failed as a *number*, not as a parse: the 20 was then read as twenty dollars
    and a 20% increase became a twenty-dollar one.
    """
    assert parse_percent("increase the budget by 20%") == 20.0


def test_a_sentence_with_no_number_parses_to_nothing():
    assert parse_magnitude("make it faster") is None
    assert parse_percent("reduce the budget") is None


# --------------------------------------------------------------------------- #
# The budget
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("set the budget to $500k", 500_000.0),
        ("budget 1.2m", 1_200_000.0),
        ("set the capital budget to 500,000", 500_000.0),
        ("cap the spend at 250k", 250_000.0),
        ("make the budget 2 million", 2_000_000.0),
    ],
)
def test_absolute_budget_instructions(instruction, expected):
    change = only_change(interpret(instruction, snapshot()))

    assert change.field == "budget"
    assert change.label == "Capital budget"
    assert change.current == 750_000.0
    assert change.proposed == expected


def test_a_budget_instruction_proposes_exactly_one_change():
    """Nothing is changed that was not asked for.

    The acquired copilot switched target mode off as a side effect of any budget
    instruction, so a sentence about money silently changed the objective.
    """
    proposal = interpret("set the budget to $500k", snapshot(target_risk_pts=40.0))

    assert fields(proposal) == {"budget"}
    assert warned_about(proposal, "does not switch target mode off")


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("increase the budget by 20%", 900_000.0),
        ("increase the budget by 20 percent", 900_000.0),
        ("cut the budget by 10%", 675_000.0),
        ("reduce the budget by 100k", 650_000.0),
        ("add 250,000 to the budget", 1_000_000.0),
        ("double the budget", 1_500_000.0),
        ("halve the budget", 375_000.0),
    ],
)
def test_relative_budget_instructions_resolve_against_the_snapshot(instruction, expected):
    change = only_change(interpret(instruction, snapshot()))

    assert change.field == "budget"
    assert change.proposed == pytest.approx(expected)
    assert "750,000" in change.rationale


def test_a_relative_change_with_no_budget_to_change_is_refused():
    """In target mode without a spend cap, `budget` is None and "by 20%" means nothing.

    The acquired copilot would have read the bare 20 through its own scaling rule and
    authorised twenty million dollars.
    """
    current = snapshot(budget=None, target_risk_pts=45.0)
    proposal = interpret("increase the budget by 20%", current)

    assert proposal.is_empty
    assert unresolved_about(proposal, "needs a budget to be relative to")


def test_a_relative_change_with_no_direction_is_refused():
    """"Change the budget by 20%" has two readings and neither is assumed."""
    proposal = interpret("change the budget by 20%", snapshot())

    assert proposal.is_empty
    assert unresolved_about(proposal, "without saying up or down")


def test_a_budget_word_with_no_figure_is_refused():
    proposal = interpret("set the budget", snapshot())

    assert proposal.is_empty
    assert unresolved_about(proposal, "no figure was given")


def test_a_negative_budget_is_clamped_and_the_clamp_is_stated():
    change = only_change(interpret("set the budget to -100000", snapshot()))

    assert change.proposed == 0.0


def test_a_zero_budget_says_what_a_zero_budget_produces():
    """An empty portfolio is not a recommendation to do nothing (README, F-series)."""
    proposal = interpret("set the budget to 0", snapshot())

    assert only_change(proposal).proposed == 0.0
    assert warned_about(proposal, "funds nothing")


def test_a_budget_already_at_that_figure_is_reported_as_a_no_op():
    """No Change may exist that changes nothing.

    A review list containing a rewrite of 750,000 to 750,000 teaches the user to
    approve without reading, which is how the consequential ones get through.
    """
    proposal = interpret("set the budget to 750,000", snapshot())

    assert proposal.changes == ()
    assert proposal.is_empty
    assert warned_about(proposal, "already $750,000")


# --------------------------------------------------------------------------- #
# Target mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("get risk under 40%", 40.0),
        ("hit 35% risk", 35.0),
        ("target 30% risk", 30.0),
        ("get the risk below 40 percent", 40.0),
        ("aim for risk at most 45%", 45.0),
    ],
)
def test_target_mode_is_enabled_with_the_target_that_was_stated(instruction, expected):
    change = only_change(interpret(instruction, snapshot()))

    assert change.field == "target_risk_pts"
    assert change.current is None
    assert change.proposed == expected


def test_target_mode_is_never_enabled_without_a_number():
    """This is F14's delivery mechanism.

    The acquired copilot set `val_target_mode = True` for any sentence containing
    "risk" and a digit, leaving `val_target_risk_goal` at whatever it held — its
    shipped 20.0, which no portfolio in the default network reaches at any budget.
    """
    proposal = interpret("get the risk down as low as you can", snapshot())

    assert proposal.changes == ()
    assert unresolved_about(proposal, "never enabled without a stated target")


def test_an_aggressive_target_warns_and_defers_to_the_frontier():
    """Warn about the F14 shape without pretending to have computed feasibility."""
    proposal = interpret("get risk under 20%", snapshot())

    assert only_change(proposal).proposed == 20.0
    assert warned_about(proposal, "attainable frontier")
    assert warned_about(proposal, "portfolio, rather than the budget")


def test_a_moderate_target_is_not_editorialised():
    proposal = interpret("get risk under 45%", snapshot())

    assert 45.0 > AGGRESSIVE_TARGET_PTS
    assert not warned_about(proposal, "attainable frontier")


def test_a_target_above_the_baseline_says_it_is_met_by_doing_nothing():
    """A target of 70% against a 65.5% baseline is not a target."""
    proposal = interpret("get risk under 70%", snapshot())

    assert warned_about(proposal, "met by funding nothing")
    assert warned_about(proposal, "65.5")


def test_an_out_of_range_target_is_clamped_and_the_clamp_is_stated():
    proposal = interpret("hit 140% risk", snapshot())

    assert only_change(proposal).proposed == 100.0
    assert warned_about(proposal, "outside 0–100%")


def test_enabling_target_mode_says_what_happens_to_the_budget():
    """Target mode minimises capital, so the budget stops being a spend goal."""
    proposal = interpret("get risk under 40%", snapshot())

    assert warned_about(proposal, "ceiling on what may be spent")


@pytest.mark.parametrize(
    "instruction",
    ["turn off target mode", "disable the risk target", "clear the target"],
)
def test_target_mode_is_switched_off_by_proposing_no_target(instruction):
    """There is no separate boolean: `target_risk_pts is None` is target mode off."""
    current = snapshot(budget=500_000.0, target_risk_pts=40.0)
    change = only_change(interpret(instruction, current))

    assert change.field == "target_risk_pts"
    assert change.proposed is None


def test_switching_target_mode_off_with_no_budget_says_so():
    current = snapshot(budget=None, target_risk_pts=40.0)
    proposal = interpret("turn off target mode", current)

    assert warned_about(proposal, "no capital budget is set")


def test_switching_off_target_mode_that_is_already_off_is_a_no_op():
    proposal = interpret("turn off target mode", snapshot())

    assert proposal.is_empty
    assert warned_about(proposal, "already not set")


# --------------------------------------------------------------------------- #
# Model mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("use legacy parity", MODE_LEGACY),
        ("switch to the legacy model", MODE_LEGACY),
        ("switch to monetary", MODE_MONETARY),
        ("use the NPV model", MODE_MONETARY),
    ],
)
def test_mode_switches_are_understood_in_both_directions(instruction, expected):
    start = MODE_LEGACY if expected == MODE_MONETARY else MODE_MONETARY
    change = only_change(interpret(instruction, snapshot(mode=start)))

    assert change.field == "mode"
    assert change.proposed == expected


def test_switching_to_legacy_repeats_what_legacy_costs():
    """Legacy parity is for reconciliation, not for decisions (README)."""
    proposal = interpret("use legacy parity", snapshot())

    assert warned_about(proposal, "reproducible, not defensible")


def test_legacy_with_a_target_warns_that_the_risk_cap_is_off():
    proposal = interpret("use legacy parity", snapshot(budget=None, target_risk_pts=40.0))

    assert warned_about(proposal, "without the risk cap")


def test_switching_to_the_mode_already_in_use_is_a_no_op():
    proposal = interpret("switch to monetary", snapshot())

    assert proposal.is_empty
    assert warned_about(proposal, "already monetary")


# --------------------------------------------------------------------------- #
# Toggles, iterations, exponent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction,field,expected",
    [
        ("turn off the monte carlo", "run_simulation", False),
        ("skip the simulation", "run_simulation", False),
        ("run the monte carlo", "run_simulation", True),
        ("turn off the budget sweep", "show_sweep", False),
        ("stop computing budget sensitivity", "show_sweep", False),
        ("show the budget sweep", "show_sweep", True),
    ],
)
def test_toggles_are_understood_in_both_directions(instruction, field, expected):
    start = {field: not expected}
    change = only_change(interpret(instruction, snapshot(**start)))

    assert change.field == field
    assert change.proposed is expected


def test_switching_off_the_simulation_says_what_is_lost():
    proposal = interpret("turn off the monte carlo", snapshot())

    assert warned_about(proposal, "no P50 or P90")


def test_switching_off_the_sweep_says_what_is_lost():
    """A saturated budget and a binding one are indistinguishable without the sweep."""
    proposal = interpret("turn off the budget sweep", snapshot())

    assert warned_about(proposal, "saturated")


def test_an_ambiguous_toggle_is_refused_rather_than_flipped():
    """"Monte carlo" is not an instruction, and a coin flip is not an interpretation."""
    proposal = interpret("monte carlo", snapshot())

    assert proposal.is_empty
    assert unresolved_about(proposal, "on or off")


@pytest.mark.parametrize(
    "instruction,expected",
    [("run 50k iterations", 50_000), ("set iterations to 25,000", 25_000)],
)
def test_iteration_counts_are_understood(instruction, expected):
    change = only_change(interpret(instruction, snapshot()))

    assert change.field == "iterations"
    assert change.proposed == expected
    assert isinstance(change.proposed, int)


@pytest.mark.parametrize(
    "instruction,expected",
    [("set iterations to 5", 1_000), ("run 5m iterations", 200_000)],
)
def test_iteration_counts_are_clamped_to_what_the_widget_accepts(instruction, expected):
    """`st.number_input` coerces out-of-range state without saying anything."""
    proposal = interpret(instruction, snapshot())

    assert only_change(proposal).proposed == expected
    assert warned_about(proposal, "outside the supported")


def test_iterations_with_the_simulation_off_says_it_will_do_nothing():
    proposal = interpret("run 50k iterations", snapshot(run_simulation=False))

    assert warned_about(proposal, "currently switched off")


def test_the_exponent_is_understood_and_flagged_as_uncalibrated():
    """F2's neighbour: the exponent shapes the response and is not measured."""
    proposal = interpret("set the exponent to 0.5", snapshot())

    assert only_change(proposal).proposed == 0.5
    assert warned_about(proposal, "no stated empirical basis")


@pytest.mark.parametrize(
    "instruction,expected",
    [("set the exponent to 2", 1.0), ("set the exponent to 0.01", 0.1)],
)
def test_the_exponent_is_clamped_to_its_range(instruction, expected):
    proposal = interpret(instruction, snapshot())

    assert only_change(proposal).proposed == expected
    assert warned_about(proposal, "outside the supported")


def test_the_value_per_risk_point_carries_its_provenance_warning():
    """Every currency figure in the app is only as defensible as this price."""
    proposal = interpret("set the value per risk point to 80k", snapshot())

    assert only_change(proposal).proposed == 80_000.0
    assert warned_about(proposal, "unsourced")


def test_the_seed_is_understood():
    change = only_change(interpret("set the seed to 7", snapshot()))

    assert change.field == "seed"
    assert change.proposed == 7


# --------------------------------------------------------------------------- #
# Refusals — the F4 discipline applied to intent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction",
    [
        "make it faster",
        "optimise everything",
        "do the needful",
        "make the portfolio better please",
    ],
)
def test_an_unrecognised_instruction_produces_no_changes(instruction):
    """Never a guessed change. A plausible invention is the worst possible output."""
    proposal = interpret(instruction, snapshot())

    assert proposal.changes == ()
    assert proposal.is_empty
    assert proposal.unresolved
    assert instruction in proposal.unresolved[0]


def test_partial_understanding_is_explicit():
    """One change plus a written admission about the half that was not understood."""
    proposal = interpret("set budget to 500k and make it faster", snapshot())

    assert only_change(proposal).field == "budget"
    assert unresolved_about(proposal, "make it faster")


def test_a_question_is_not_treated_as_an_instruction():
    """The old copilot answered questions from whatever local was in scope."""
    proposal = interpret("what is my current budget?", snapshot())

    assert proposal.changes == ()
    assert unresolved_about(proposal, "does not answer questions")


def test_the_old_optimisation_weight_is_refused_by_name():
    """`val_opt_weight` has no equivalent here, so no equivalent is invented."""
    proposal = interpret("set the optimisation weight to 0.8", snapshot())

    assert proposal.changes == ()
    assert unresolved_about(proposal, "no optimisation weight")


def test_the_baseline_risk_is_refused_rather_than_moved_from_a_sentence():
    """Every risk figure and every target is quoted against it (F18)."""
    proposal = interpret("set the baseline risk to 70%", snapshot())

    assert proposal.changes == ()
    assert unresolved_about(proposal, "baseline risk is not proposed from here")


@pytest.mark.parametrize("instruction", ["", "   ", "\n\t "])
def test_an_empty_instruction_is_an_empty_proposal(instruction):
    proposal = interpret(instruction, snapshot())

    assert proposal.is_empty
    assert proposal.changes == ()
    assert proposal.warnings == ()
    assert proposal.unresolved == ()


# --------------------------------------------------------------------------- #
# Several clauses at once
# --------------------------------------------------------------------------- #


def test_multiple_clauses_each_produce_their_own_change():
    proposal = interpret(
        "set the budget to 400k, get risk under 40% and turn off the sweep", snapshot()
    )

    assert fields(proposal) == {"budget", "target_risk_pts", "show_sweep"}


def test_a_thousands_separator_is_not_read_as_a_clause_break():
    """Splitting on every comma turns "500,000" into two unparseable halves."""
    proposal = interpret("set the budget to 1,250,000", snapshot())

    assert only_change(proposal).proposed == 1_250_000.0


def test_one_field_named_twice_yields_one_change_and_says_which_won():
    """Two rows for one field in a review list hide which line actually decided it."""
    proposal = interpret(
        "set the budget to 300k and then set the budget to 400k", snapshot()
    )

    change = only_change(proposal)
    assert change.proposed == 400_000.0
    assert warned_about(proposal, "named twice")


# --------------------------------------------------------------------------- #
# Applying a proposal
# --------------------------------------------------------------------------- #


def test_apply_proposal_returns_a_new_dict_and_leaves_the_original_alone():
    """The old snapshot is what makes an undo and an audit trail possible."""
    current = snapshot()
    before = dict(current)
    proposal = interpret("set the budget to 500k", current)

    updated = apply_proposal(current, proposal)

    assert current == before
    assert updated is not current
    assert updated["budget"] == 500_000.0


def test_apply_proposal_changes_only_the_proposed_fields():
    current = snapshot()
    updated = apply_proposal(current, interpret("hit 40% risk", current))

    moved = {key for key in current if current[key] != updated[key]}
    assert moved == {"target_risk_pts"}


def test_applying_an_empty_proposal_changes_nothing():
    current = snapshot()
    updated = apply_proposal(current, interpret("make it faster", current))

    assert updated == current


def test_applying_then_reinterpreting_proposes_nothing_further():
    """Idempotence. A proposal that keeps proposing itself is a loop, not a decision."""
    current = snapshot()
    instruction = "set the budget to 500k and get risk under 40%"

    updated = apply_proposal(current, interpret(instruction, current))
    again = interpret(instruction, updated)

    assert again.is_empty
    assert apply_proposal(updated, again) == updated


def test_a_proposal_applied_to_moved_inputs_is_refused():
    """A proposal is a decision about specific values, not a blind instruction.

    Applying it after the sidebar has moved would write a figure nobody reviewed —
    exactly the invisible rewrite this module replaced.
    """
    proposal = interpret("increase the budget by 20%", snapshot())

    with pytest.raises(StaleProposal):
        apply_proposal(snapshot(budget=100_000.0), proposal)


# --------------------------------------------------------------------------- #
# The summary a reviewer reads
# --------------------------------------------------------------------------- #


def test_the_summary_shows_every_change_with_both_values():
    proposal = interpret("set the budget to 500k and hit 40% risk", snapshot())
    text = proposal.summary()

    assert "$750,000 → $500,000" in text
    assert "not set → 40.00%" in text
    assert "Capital budget" in text
    assert "Target risk (%)" in text


def test_the_summary_repeats_what_was_not_understood():
    text = interpret("set budget to 500k and make it faster", snapshot()).summary()

    assert "Not understood" in text
    assert "make it faster" in text


def test_the_summary_of_an_unrecognised_instruction_says_nothing_will_change():
    text = interpret("do the needful", snapshot()).summary()

    assert "No changes proposed" in text


def test_every_change_explains_itself():
    """A rationale is the only thing tying a number on screen to a sentence."""
    proposal = interpret(
        "double the budget, use legacy parity and turn off the monte carlo", snapshot()
    )

    assert len(proposal.changes) == 3
    for change in proposal.changes:
        assert change.rationale
        assert change.label
