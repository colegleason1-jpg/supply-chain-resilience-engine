"""The intervention editor and the resulting plan.

Rejected rows are shown, always. The acquired system's editor coerced unparseable
values to zero, and a zero-cost intervention is irresistible to an optimizer: it gets
funded to the maximum at any budget. A plan built from nineteen good rows and one
silently zeroed row looked exactly like a plan built from twenty good rows.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import adapters
from app.presenters import Metric, portfolio_rows


#: Where the stored calibration table lives between runs. Not a widget key: it is
#: never edited by hand, only written when a fit is adopted.
CALIBRATION_STATE_KEY = "calibration_store"


def _stored_calibrations() -> pd.DataFrame | None:
    """The stored fits, taking a freshly reloaded portfolio's table over the session's.

    A portfolio load parks each saved table under ``"<key>_frame"``, so the reloaded
    calibrations arrive as ``calibration_frame`` while the live copy lives under
    ``calibration_store``. The reloaded table wins and is then consumed, because
    leaving it in place would let a portfolio loaded ten interactions ago keep
    overwriting fits adopted since — and a stale evidence table is worse than none:
    every elasticity it fails to vouch for silently drops back to asserted while the
    numbers on screen do not move.
    """
    if "calibration_frame" in st.session_state:
        st.session_state[CALIBRATION_STATE_KEY] = st.session_state.pop(
            "calibration_frame"
        )
    return st.session_state.get(CALIBRATION_STATE_KEY)


def _metric_row(metrics: list[Metric]) -> None:
    for column, metric in zip(st.columns(len(metrics)), metrics):
        column.metric(metric.label, metric.value, metric.delta, help=metric.help)


def render_editors() -> dict[str, pd.DataFrame]:
    """Interventions, bundles, dependencies, correlations and calibration evidence."""
    st.subheader("Interventions")
    nodes = st.data_editor(
        st.session_state.setdefault("nodes_frame", adapters.default_nodes_frame()),
        num_rows="dynamic",
        width="stretch",
        key="nodes_editor",
    )
    node_ids = adapters.node_ids_in(nodes)

    with st.expander("Bundles and dependencies"):
        st.caption(
            "Bundle membership is read from the Required Nodes column. Name the nodes "
            "explicitly — a discount is contingent on exactly the nodes listed."
        )
        bundles = st.data_editor(
            st.session_state.setdefault("bundles_frame", adapters.default_bundles_frame()),
            num_rows="dynamic",
            width="stretch",
            key="bundles_editor",
        )
        dependencies = st.data_editor(
            st.session_state.setdefault(
                "dependencies_frame", adapters.default_dependencies_frame()
            ),
            num_rows="dynamic",
            width="stretch",
            key="dependencies_editor",
        )

    with st.expander("Shock correlations"):
        st.caption(
            "How node outcomes move together under shock. Read by Node ID, so "
            "reordering the interventions above cannot transpose this matrix."
        )
        correlation = st.data_editor(
            adapters.align_correlation_frame(
                st.session_state.get("correlation_frame"), node_ids
            ),
            width="stretch",
            key="correlation_editor",
        )

    with st.expander("Market exposure"):
        st.caption(
            "Which commodity or index each intervention is sensitive to, and how "
            "strongly. Stated here by Node ID rather than inferred: the previous "
            "system guessed this relationship by comparing a market's sector name to "
            "an intervention's name, which matched nothing in the normal case and "
            "matched everything when the column was missing."
        )
        st.caption(
            "Leave a row blank if an intervention has no market exposure — that is "
            "the right answer for most of them. Elasticity is the proportional change "
            "in the intervention's effectiveness per unit proportional move of the "
            "price above its anchor, and is an assumption, not a measurement."
        )
        exposure = st.data_editor(
            adapters.align_exposure_frame(
                st.session_state.get("exposure_frame"), node_ids
            ),
            width="stretch",
            key="exposure_editor",
        )

    with st.expander("Delivery history"):
        st.caption(
            "What actually happened, month by month, per intervention. Period is a "
            "month written YYYY-MM. Disruption Rate % is the percentage of shipments "
            "that went wrong on whatever definition the business already uses — late, "
            "short, failed, diverted — applied consistently across every period. The "
            "estimator is scale-free in it, so the definition need not match anyone "
            "else's; it must only stay the same from month to month."
        )
        st.caption(
            "This table is evidence, not an input to the solve. It is the only thing "
            "that can move an elasticity from asserted to calibrated, so a row "
            "naming a node that no longer exists is reported rather than deleted."
        )
        history = st.data_editor(
            adapters.align_history_frame(
                st.session_state.setdefault(
                    "history_frame", adapters.default_history_frame()
                )
            ),
            num_rows="dynamic",
            width="stretch",
            key="history_editor",
        )

    with st.expander("Stored calibrations"):
        st.caption(
            "The fits that justify every elasticity currently called calibrated. "
            "Read-only on purpose: an editable evidence table is a table in which a "
            "user can type the word 'calibrated' next to a number nobody measured, "
            "which is the precise laundering this rebuild exists to stop. Rows are "
            "written only by adopting a fit in the Elasticity calibration panel."
        )
        calibration = adapters.align_calibration_frame(_stored_calibrations())
        st.dataframe(calibration, width="stretch", hide_index=True)

    return {
        "nodes": nodes,
        "bundles": bundles,
        "dependencies": dependencies,
        "correlation": correlation,
        "exposure": exposure,
        "history": history,
        "calibration": calibration,
    }


def render_rejections(rejections, *, label: str) -> None:
    """Never silent, never fatal."""
    if not rejections:
        return
    with st.expander(f"{len(rejections)} {label} could not be used", expanded=True):
        for reason in rejections:
            st.warning(reason, icon=":material/warning:")


def render_plan(result, metrics: list[Metric]) -> None:
    _metric_row(metrics)
    rows = portfolio_rows(result)
    if not rows:
        st.info("No funded plan to show.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if result.active_bundles:
        st.caption(
            "Bundle discounts applied: "
            + ", ".join(str(name) for name in result.active_bundles)
        )
