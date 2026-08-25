"""Entry point. Layout and sequencing only.

Run with:

    streamlit run app/main.py

There is deliberately no arithmetic in this file, and no `pulp` or `numpy` import
anywhere in `app/` outside the adapter that builds a correlation matrix. The rule that
replaced the 1,146-line monolith is simple: the UI collects inputs, hands them to
`scrcae`, and renders what comes back. When a number is wrong there is now exactly one
place it can be wrong.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import adapters
from app.engine_client import analyse
from app.presenters import frontier_rows, headline_metrics, status_banner
from app.services import calibration as calibration_service
from app.services import market as market_service
from app.services.copilot_state import snapshot
from app.views import account as account_view
from app.views import audit as audit_view
from app.views import calibration_panel
from app.views import copilot_panel
from app.views import market_panel
from app.views import portfolio as portfolio_view
from app.views import sidebar as sidebar_view
from app.views import sweep as sweep_view

st.set_page_config(
    page_title="Supply Chain Resilience & Capital Allocation",
    layout="wide",
)


@st.cache_data(ttl=market_service.CACHE_TTL_SECONDS, show_spinner=False)
def _cached_quotes(symbols: tuple[str, ...]) -> market_service.QuoteSet:
    """Quotes fetched server-side and cached, so a rerun is not a new HTTP request.

    Keyed on the symbol tuple, so adding an exposure fetches the new market without
    discarding the readings already held for the others.
    """
    return market_service.fetch_quotes(symbols)


@st.cache_data(show_spinner=False)
def _cached_calibration(signature: tuple, *, _exposures, _records):
    """One calibration run per distinct set of exposures and delivery records.

    Cached on ``signature`` — the exposures and records reduced to primitives — while
    the objects themselves are passed underscore-prefixed so Streamlit does not try
    to hash them. Pressing the button twice without changing anything must not fetch
    five years of monthly history again.

    Caching is the second line of defence, not the first. The only call site is a
    button, because a calibration makes one network request per distinct symbol and a
    Streamlit script reruns on every keystroke.
    """
    return calibration_service.calibrate_exposures(_exposures, _records)


def _calibration_signature(exposures, records) -> tuple:
    """The inputs a calibration depends on, as something hashable.

    The anchor is part of the key as well as the elasticity: an elasticity is defined
    relative to its anchor, so moving the anchor invalidates the fit even though the
    delivery records have not changed.
    """
    return (
        tuple(
            (e.node_id, e.symbol, e.anchor, e.elasticity)
            for e in exposures
        ),
        tuple((r.node_id, r.period, r.disruption_rate_pct) for r in records),
    )


def _calibration_row(calibration) -> dict:
    """One adopted fit as a row of the stored calibration table.

    Written out field by field rather than summarised, because this row is the entire
    justification for calling the elasticity calibrated. `adapters.build_calibration_records`
    discards any row that cannot name its method, its fitted value and its anchor, so
    a partially written row degrades the elasticity to asserted rather than passing
    for evidence.
    """
    fit = calibration.fit
    return {
        adapters.COL_CAL_NODE_ID: calibration.node_id,
        adapters.COL_CAL_SYMBOL: calibration.symbol,
        adapters.COL_CAL_ELASTICITY: float(fit.elasticity),
        adapters.COL_CAL_ANCHOR: float(calibration.anchor),
        adapters.COL_CAL_LOW: fit.interval_low,
        adapters.COL_CAL_HIGH: fit.interval_high,
        adapters.COL_CAL_R2: fit.r_squared,
        adapters.COL_CAL_OBS: fit.observations,
        adapters.COL_CAL_ABOVE: fit.periods_above_anchor,
        adapters.COL_CAL_METHOD: fit.method,
    }


def _labels(frame, column: str, *, upper: bool = False):
    """One column as trimmed text, so a match is on the name and not on whitespace.

    Blank cells become empty strings rather than NaN: comparing a missing node id
    against a real one must be a plain "no", not a comparison whose result depends on
    the column's dtype.
    """
    text = frame[column].astype("string").fillna("").str.strip()
    return text.str.upper() if upper else text


def _adopt(adopted, node_ids) -> None:
    """Write adopted fits into the exposure table and record the evidence for them.

    This is the only code in the app that writes to the user's exposure table, and it
    runs only from the button in `calibration_panel`. F11's defect was
    ``st.session_state.nodes_df = bridge.apply_market_feedback(...)`` executing on
    every rerun: each refresh fed the market adjustment back over the user's own
    figures, so the entered risk reductions climbed with no record of it and no way
    back. The market overlay therefore still never touches an input, and the one
    write that does exist is a user's explicit decision, applied once.

    Both halves happen together or the label is a lie. The elasticity is called
    calibrated only while a stored fit records the same number and the same anchor,
    so writing the number without its evidence would show "asserted" beside a fitted
    value, and writing the evidence without the number would vouch for a figure that
    is not there.
    """
    exposure = adapters.align_exposure_frame(
        st.session_state.get("exposure_frame"), node_ids
    )
    stored = adapters.align_calibration_frame(
        st.session_state.get(portfolio_view.CALIBRATION_STATE_KEY)
    )

    for calibration in adopted:
        if calibration.fit is None or calibration.fit.elasticity is None:
            # Unreachable from the panel, which offers no control for an unusable
            # fit. Checked anyway rather than trusted, because the cost of being
            # wrong is a number in the exposure table with nothing behind it.
            continue
        rows = (_labels(exposure, adapters.COL_EXP_NODE_ID) == calibration.node_id) & (
            _labels(exposure, adapters.COL_EXP_SYMBOL, upper=True) == calibration.symbol
        )
        exposure.loc[rows.to_numpy(), adapters.COL_EXP_ELASTICITY] = float(
            calibration.fit.elasticity
        )

        row = _calibration_row(calibration)
        existing = (
            _labels(stored, adapters.COL_CAL_NODE_ID) == calibration.node_id
        ) & (_labels(stored, adapters.COL_CAL_SYMBOL, upper=True) == calibration.symbol)
        if bool(existing.any()):
            # Replaced, not appended. Two stored fits for one node and market would
            # leave which one vouches for the elasticity down to row order.
            stored = stored[~existing.to_numpy()]
        stored = pd.concat(
            [stored, pd.DataFrame([row])], ignore_index=True
        )

    st.session_state["exposure_frame"] = exposure
    st.session_state[portfolio_view.CALIBRATION_STATE_KEY] = (
        adapters.align_calibration_frame(stored)
    )
    # The editor's own pending edit diff is keyed by row and column and would be
    # replayed over the frame just written, silently restoring the value the user
    # asked to replace. Dropping the widget's state makes the new frame the record.
    if "exposure_editor" in st.session_state:
        del st.session_state["exposure_editor"]


def _render_banner(banner) -> None:
    renderer = {
        "success": st.success,
        "warning": st.warning,
        "error": st.error,
        "info": st.info,
    }[banner.level]
    renderer(f"**{banner.headline}**\n\n{banner.detail}")
    if banner.remedy:
        st.caption(banner.remedy)


def main() -> None:
    st.title("Supply Chain Resilience & Capital Allocation")

    # Order matters, and for one reason: Streamlit refuses to write a widget's key once
    # that widget has been instantiated in the current run. Everything that changes
    # inputs therefore happens before `sidebar_view.render()` creates them.
    loaded_name = account_view.apply_pending_load()
    if loaded_name:
        st.toast(f"Loaded “{loaded_name}”.")

    principal = account_view.render_identity()
    copilot_panel.render()

    inputs = sidebar_view.render()
    copilot_panel.render_applied_note()
    frames = portfolio_view.render_editors()

    network, rejections = adapters.build_network(
        frames["nodes"],
        inputs.baseline_risk_pts,
        bundles_frame=frames["bundles"],
        dependencies_frame=frames["dependencies"],
    )
    portfolio_view.render_rejections(rejections, label="inputs")

    if network is None:
        # No network means no solve. The alternative — solving an empty network — is
        # feasible, cheap and reads as a recommendation to do nothing.
        st.error("Fix the inputs above before an allocation can be computed.")
        # The portfolio panel is still drawn, because reloading a known-good portfolio
        # is the most likely way out of unusable inputs. Returning before it would
        # strand the user with a broken table and no route back.
        if principal is not None:
            account_view.render_portfolios(
                principal, frames=frames, settings=snapshot(st.session_state)
            )
        return

    node_ids = [i.node_id for i in network.interventions]
    correlation, correlation_rejections = adapters.build_correlation_matrix(
        frames["correlation"], node_ids
    )
    portfolio_view.render_rejections(correlation_rejections, label="correlation entries")

    exposures, exposure_rejections = adapters.build_node_exposures(
        frames["exposure"], node_ids
    )
    portfolio_view.render_rejections(exposure_rejections, label="market exposures")

    delivery_records, history_rejections = adapters.build_delivery_records(
        frames["history"], node_ids
    )
    portfolio_view.render_rejections(history_rejections, label="delivery history rows")

    calibration_records, calibration_rejections = adapters.build_calibration_records(
        frames["calibration"]
    )
    portfolio_view.render_rejections(
        calibration_rejections, label="stored calibrations"
    )

    # Resolved before the overlay is built, and before anything is displayed, because
    # every downstream statement about an elasticity is a statement about its
    # provenance. Deriving the label later would leave the market panel, the audit
    # tab and the saved portfolio describing the same number three different ways.
    exposures = adapters.resolve_elasticity_sources(exposures, calibration_records)

    quotes = market_service.QuoteSet()
    overlay = market_service.build_overlay((), quotes)
    if inputs.apply_market_exposure and exposures:
        quotes = _cached_quotes(tuple(sorted({e.symbol for e in exposures})))
        overlay = market_service.build_overlay(exposures, quotes)
    market_panel.render(overlay, quotes, enabled=inputs.apply_market_exposure)

    # Behind a button, not behind a condition. Calibration fetches monthly history per
    # symbol, and a Streamlit script reruns on every widget interaction, so anything
    # that reaches the network on the plain path would fire on each keystroke.
    if st.button(
        "Calibrate elasticities from delivery history", key="calibrate_elasticities"
    ):
        with st.spinner("Fitting elasticities from delivery history..."):
            st.session_state["calibration_run"] = _cached_calibration(
                _calibration_signature(exposures, delivery_records),
                _exposures=exposures,
                _records=delivery_records,
            )

    adopted = calibration_panel.render(
        st.session_state.get("calibration_run"), enabled=bool(delivery_records)
    )
    if adopted:
        # The rerun is what makes the adoption legible: the exposure editor, the
        # stored-calibration table, the provenance row and the solve all read the
        # written frames from the top rather than half of this run using the old ones.
        _adopt(adopted, node_ids)
        st.rerun()

    request = adapters.build_optimization_request(
        network,
        mode=inputs.mode,
        budget=inputs.budget,
        target_risk_pts=inputs.target_risk_pts,
        macro_multiplier=inputs.macro_multiplier,
        exponent=inputs.exponent,
        prices=inputs.prices,
        node_macro_multipliers=overlay.multipliers,
    )

    simulation_factory = None
    if inputs.run_simulation:
        # Deferred, because the simulation describes whichever portfolio the optimizer
        # ends up choosing.
        def simulation_factory(result):  # noqa: F811 - deliberate conditional binding
            return adapters.build_simulation_request(
                result,
                network,
                correlation,
                mode=inputs.mode,
                iterations=inputs.iterations,
                seed=inputs.seed,
            )

    with st.spinner("Solving..."):
        bundle = analyse(
            request,
            simulation_factory=simulation_factory,
            include_sweep=inputs.show_sweep,
        )

    # Drawn after the solve, though it appears in the sidebar. Both the provenance
    # comparison and the provenance recorded by a save refer to the result of *this*
    # run; reading them before the solve would attach the previous run's hashes to the
    # inputs currently on screen.
    if principal is not None:
        account_view.render_portfolios(
            principal,
            frames=frames,
            settings=snapshot(st.session_state),
            result=bundle.result,
        )
        account_view.render_provenance_notice(bundle.result)

    _render_banner(status_banner(bundle.result))

    if bundle.frontier is not None and not bundle.result.is_solved:
        # What the portfolio can actually reach, so the conversation moves on from
        # "why did this fail" to "what would we need".
        st.subheader("What this portfolio can reach")
        st.table(frontier_rows(bundle.frontier))

    plan_tab, sweep_tab, audit_tab = st.tabs(["Plan", "Sensitivity", "Verification"])

    with plan_tab:
        portfolio_view.render_plan(
            bundle.result, headline_metrics(bundle.result, request.macro_multiplier)
        )

    with sweep_tab:
        sweep_view.render(bundle.sweep)

    with audit_tab:
        audit_view.render(
            bundle.result, request, bundle.simulation, exposures=exposures
        )


if __name__ == "__main__":
    main()
