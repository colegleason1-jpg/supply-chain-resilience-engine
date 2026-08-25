"""Verification, provenance and disclosed assumptions.

Everything in this tab is read from the engine's own records. The acquired system's
equivalent printed "Zero fractional violations detected" and a truncated
"sha256:8f4c99a..." as string literals, with no verification pass and no hash behind
either of them (F4). If the engine has not checked something, this tab cannot claim it
has.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.presenters import (
    provenance_rows,
    simulation_metrics,
    uncalibrated_warnings,
    verification_rows,
)


def render(result, request, simulation=None, *, exposures=None) -> None:
    """``exposures`` are threaded in so the elasticity provenance is counted, not
    guessed: the tab reports how many of them were fitted and warns about only the
    ones that were not."""
    st.subheader("Constraint verification")
    report = result.constraint_report
    if report.is_feasible:
        st.success(report.summary())
    else:
        st.error(report.summary())
    st.dataframe(
        pd.DataFrame(verification_rows(result, request)), width="stretch", hide_index=True
    )

    st.subheader("Assumptions in force")
    warnings = uncalibrated_warnings(request, simulation, exposures=exposures)
    if not warnings:
        st.success("Every parameter in this run has a stated source.")
    for warning in warnings:
        st.warning(warning, icon=":material/science:")

    if simulation is not None:
        st.subheader("Stochastic outcomes")
        metrics = simulation_metrics(simulation)
        for column, metric in zip(st.columns(len(metrics)), metrics):
            column.metric(metric.label, metric.value, help=metric.help)

    st.subheader("Provenance")
    st.dataframe(
        pd.DataFrame(
            provenance_rows(
                result,
                request=request,
                simulation=simulation,
                exposures=exposures,
            )
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Hashes are computed from the actual inputs and outputs of this run. Two runs "
        "with the same input hash and the same engine version will agree."
    )
