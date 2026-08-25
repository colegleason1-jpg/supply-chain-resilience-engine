"""What the delivery records say each elasticity should be, and what may be adopted.

The elasticity is the last freely typed number left in the market path. F11's
elasticity was a single hardcoded 0.15; the rebuild made it a per-row assumption a
person owns, and this panel is where that assumption can be replaced by a measurement
from the company's own delivery history — or, far more often, where the records are
shown to be unable to support one.

Two properties of this view are load-bearing rather than cosmetic:

**A refused fit has no control that adopts it.** Not a disabled control, not a
control that warns on click: no checkbox is drawn at all. A refused fit is not
evidence of anything, and the whole reason ``elasticity_source`` is derived from a
stored fit rather than typed is that "calibrated" must be impossible to claim without
the arithmetic behind it. Rendering a tick box next to "the interval includes zero"
would reintroduce exactly that claim, one click away.

**Nothing here writes anything.** This function returns the list of fits the user
ticked and pressed the button for; the caller performs the write. F11's actual damage
was ``st.session_state.nodes_df = bridge.apply_market_feedback(...)`` running as a
side effect of every rerun, compounding the user's own figures upward with no record
of it having happened. A view that returns a selection cannot do that.
"""

from __future__ import annotations

import streamlit as st

from app.services.calibration import CalibrationRun, NodeCalibration


def _adopt_key(calibration: NodeCalibration) -> str:
    """The checkbox key for one exposure.

    Keyed on node *and* symbol because a node may be exposed to several markets, and
    a key collision here would silently adopt one market's fit under another
    market's name — a mislabelled provenance being worse than no provenance.
    """
    return f"adopt_{calibration.node_id}_{calibration.symbol}"


def render(run: CalibrationRun | None, *, enabled: bool) -> list:
    """Show every calibration attempt; return the fits the user chose to adopt.

    ``enabled`` says whether there is any delivery history to calibrate from. An
    empty list is returned unless the user both ticked something and pressed the
    button in this run, so a plain rerun adopts nothing.
    """
    with st.expander("Elasticity calibration", expanded=False):
        if not enabled:
            st.caption(
                "No delivery history is configured, so no elasticity can be "
                "calibrated. Fill in the Delivery history table — one row per node "
                "per month — and the elasticities in the Market exposure table stay "
                "asserted until it is."
            )
            return []

        if run is None:
            st.caption(
                "No calibration has been run against this delivery history yet. "
                "Use the “Calibrate elasticities from delivery history” button; it "
                "fetches market history, so it runs only when asked."
            )
            return []

        st.caption(run.summary())
        if run.provider_name:
            st.caption(f"Market history read from the {run.provider_name} feed.")

        ticked: list[NodeCalibration] = []
        for calibration in run.calibrations:
            st.markdown(f"**{calibration.node_id} — {calibration.symbol}**")

            # The entered-versus-fitted line first, because it is the finding: the
            # interesting result of a calibration is usually that the number someone
            # typed a quarter ago is a multiple of what the records support.
            delta = calibration.delta_note()
            if delta:
                st.caption(delta)
            st.caption(calibration.headline())

            fit = calibration.fit
            if fit is not None and fit.notes:
                # Collapsed, but present. These notes carry the caveats that make the
                # estimate an association rather than a causal effect, and they
                # belong next to the number rather than in a docstring.
                with st.expander("What this fit does and does not establish"):
                    for note in fit.notes:
                        st.caption(note)

            if calibration.can_adopt:
                if st.checkbox(
                    f"Adopt {calibration.fitted_elasticity:.2f} for "
                    f"{calibration.node_id} / {calibration.symbol}",
                    key=_adopt_key(calibration),
                ):
                    ticked.append(calibration)
            else:
                # No checkbox on this branch, deliberately. See the module docstring.
                st.caption(
                    "Not adoptable: this fit cannot support calling the elasticity "
                    "calibrated, so the entered value stands and remains asserted."
                )

        if not run.adoptable:
            return []

        if st.button("Adopt selected elasticities", key="adopt_elasticities"):
            return ticked
        return []
