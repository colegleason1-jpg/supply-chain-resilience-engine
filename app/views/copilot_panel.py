"""The copilot, as a review step rather than an actor.

The acquired system's copilot read a sentence and wrote straight into session state,
then reran. This one has two buttons: one that produces a proposal, and one that
applies a proposal the user has read. Nothing changes between them.

The gap matters most for the cases the copilot gets partly right. "Set the budget to
900k and make it aggressive" yields one change and one unresolved clause; the acquired
system yielded one change and silence. Silence is what let a user reach the
structurally unattainable 20% target (F14) by typing one line and then spend an
afternoon wondering why the solver said infeasible next to a billion-dollar budget.
"""

from __future__ import annotations

import streamlit as st

from app import copilot
from app.services.copilot_state import apply_to_state, snapshot


def render() -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Copilot")
    st.sidebar.caption(
        "Describe a change in a sentence. Nothing moves until you accept the proposal."
    )

    instruction = st.sidebar.text_area(
        "Instruction",
        key="copilot_instruction",
        placeholder="e.g. set the budget to $900k and turn off the Monte Carlo",
        label_visibility="collapsed",
    )

    if st.sidebar.button("Read this", key="copilot_read", width="stretch"):
        if not instruction.strip():
            st.session_state.pop("copilot_proposal", None)
        else:
            # The snapshot is stored with the proposal. `apply_proposal` refuses to
            # write against inputs that have moved since, so a proposal reviewed
            # against one set of numbers can never be applied to another.
            st.session_state["copilot_proposal"] = copilot.interpret(
                instruction, snapshot(st.session_state)
            )

    proposal = st.session_state.get("copilot_proposal")
    if proposal is None:
        return

    _render_proposal(proposal)


def _render_proposal(proposal: copilot.Proposal) -> None:
    if proposal.changes:
        st.sidebar.markdown("**Proposed**")
        for change in proposal.changes:
            st.sidebar.markdown(f"- {change.describe()}")
    else:
        st.sidebar.info("Nothing in that instruction would change any input.")

    for text in proposal.warnings:
        st.sidebar.warning(text)

    if proposal.unresolved:
        # Shown as prominently as the changes. A copilot that silently drops the half
        # of a sentence it could not parse is the acquired behaviour with nicer type.
        st.sidebar.error(
            "Not understood, and therefore not acted on:\n"
            + "\n".join(f"- {text}" for text in proposal.unresolved)
        )

    if proposal.is_empty:
        return

    accept, discard = st.sidebar.columns(2)
    if accept.button("Apply", key="copilot_apply", width="stretch", type="primary"):
        _apply(proposal)
    if discard.button("Discard", key="copilot_discard", width="stretch"):
        st.session_state.pop("copilot_proposal", None)
        st.rerun()


def _apply(proposal: copilot.Proposal) -> None:
    try:
        updated = copilot.apply_proposal(snapshot(st.session_state), proposal)
    except copilot.StaleProposal as exc:
        # Surfaced, not swallowed. The inputs moved between reading and accepting, so
        # applying would write a value nobody reviewed.
        st.sidebar.error(str(exc))
        st.session_state.pop("copilot_proposal", None)
        return

    apply_to_state(st.session_state, updated)
    st.session_state.pop("copilot_proposal", None)
    st.session_state["copilot_applied"] = proposal.summary()
    st.rerun()


def render_applied_note() -> None:
    """Show what the last accepted proposal did, in the main body.

    Kept after the rerun so the reason the numbers on screen moved is still visible
    once they have. This is the audit trail the acquired copilot had no equivalent of.
    """
    summary = st.session_state.get("copilot_applied")
    if not summary:
        return
    with st.expander("Last copilot change", expanded=False):
        st.code(summary, language=None)
