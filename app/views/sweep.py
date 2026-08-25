"""Budget sensitivity, computed by the engine.

The acquired system rebuilt an entire second LP here (F6), and the copy had drifted:
the headline clamped risk reduction at the baseline and this curve did not, so the
chart and the number above it were produced by different models. This module calls
`scrcae.optimization.sweep_budget`, which calls the same `solve()` the headline uses.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.presenters import sweep_note, sweep_rows


def render(sweep) -> None:
    st.subheader("Budget sensitivity")
    if sweep is None:
        st.info("Enable budget sensitivity in the sidebar to compute this.")
        return

    solved = sweep.solved_points
    if solved:
        chart_data = pd.DataFrame(
            {
                "Budget": [p.budget for p in solved],
                "Risk reduction (pts)": [p.risk_reduction_pts for p in solved],
            }
        ).set_index("Budget")
        st.line_chart(chart_data)

    note = sweep_note(sweep)
    (st.success if sweep.solved_points and not sweep.is_saturated else st.warning)(note)

    st.dataframe(pd.DataFrame(sweep_rows(sweep)), width="stretch", hide_index=True)
    st.caption(
        "Each row is a full solve by the same optimizer that produced the headline "
        "figures, so this curve and the plan above cannot disagree."
    )
