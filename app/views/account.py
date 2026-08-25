"""Sign-in and saved portfolios, in the sidebar.

Two properties this view must not violate:

**It never filters by owner itself.** `PortfolioService` holds the principal and every
query is scoped in SQL. A view that received an owner id could pass the wrong one; this
one cannot express the mistake.

**It states which provider is active.** With no Supabase and no local accounts the app
runs anonymously, which is right for a single-user local run and wrong to leave
ambiguous. "Not signed in" appears on screen rather than an empty space where a name
would go.
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from app.auth.models import Principal
from app.auth.providers import build_provider
from app.services.portfolios import PortfolioService
from app.storage.sqlite_store import SqlitePortfolioStore

DEFAULT_DB = "portfolios.db"


def database_path() -> str:
    """Where portfolios live. Overridable so tests get a scratch file.

    Read at call time rather than captured at import, because `AppTest` sets the
    environment after this module is already imported.
    """
    return os.environ.get("SCRCAE_DB_PATH", DEFAULT_DB)


@st.cache_resource(show_spinner=False)
def _store_for(path: str) -> SqlitePortfolioStore:
    """One connection per path per server, not one per rerun.

    `cache_resource` rather than `cache_data`: a database handle is not a value to be
    copied, and reconnecting on every keystroke would leak file handles. Keyed on the
    path so a test pointing elsewhere gets its own connection instead of the cached
    production one.
    """
    return SqlitePortfolioStore(path)


def _store() -> SqlitePortfolioStore:
    return _store_for(database_path())


def _provider():
    return build_provider(_store())


def current_principal() -> Principal | None:
    return st.session_state.get("principal")


#: Where a pending load parks its settings between runs. Not a widget key.
PENDING_LOAD = "pending_portfolio_load"


def apply_pending_load() -> str:
    """Apply a queued portfolio load. Must run before any widget is created.

    Streamlit refuses to write a widget's key once that widget has been instantiated in
    the current run, and the "Load" button necessarily sits below the sidebar widgets it
    needs to change. So loading is two-phase: the click parks the payload under a
    non-widget key and reruns, and this — called first thing in `main` — performs the
    writes while the keys are still free.

    The alternative, moving the load button above every input, would put a
    rarely-used control ahead of the ones users touch constantly.
    """
    pending = st.session_state.pop(PENDING_LOAD, None)
    if pending is None:
        return ""

    from app.services.copilot_state import apply_to_state

    for table, frame in pending["tables"].items():
        st.session_state[f"{table}_frame"] = frame
    apply_to_state(st.session_state, pending["settings"])
    st.session_state["loaded_provenance"] = pending["provenance"]
    return pending["name"]


def _sign_in_form(provider) -> None:
    st.sidebar.caption(f"Sign in ({provider.name})")
    with st.sidebar.form("sign_in", clear_on_submit=False):
        email = st.text_input("Email", key="sign_in_email")
        password = st.text_input("Password", type="password", key="sign_in_password")
        submitted = st.form_submit_button("Sign in")

    if not submitted:
        return

    outcome = provider.sign_in(email, password)
    if outcome.succeeded:
        st.session_state["principal"] = outcome.principal
        # Clear the password from session state immediately. Streamlit keeps widget
        # values there for the life of the session otherwise.
        st.session_state["sign_in_password"] = ""
        st.rerun()
    elif outcome.locked_out:
        st.sidebar.error(outcome.error)
    else:
        st.sidebar.error(outcome.error)


def render_identity() -> Principal | None:
    """Show who is signed in, or a sign-in form. Returns the principal, if any."""
    st.sidebar.divider()
    provider = _provider()

    principal = current_principal()
    if principal is None and provider.name == "anonymous":
        # No auth configured: adopt the local single-user principal so persistence
        # works without credentials, and say so.
        outcome = provider.sign_in()
        principal = outcome.principal
        st.session_state["principal"] = principal

    if principal is None:
        _sign_in_form(provider)
        st.sidebar.info("Sign in to save and reload portfolios.")
        return None

    if principal.is_anonymous:
        st.sidebar.caption(
            "Running without authentication — portfolios are saved locally to this "
            "machine and are not protected by a password."
        )
    else:
        st.sidebar.caption(f"Signed in as {principal.label}")
        if st.sidebar.button("Sign out", key="sign_out"):
            del st.session_state["principal"]
            st.rerun()

    return principal


def render_portfolios(
    principal: Principal,
    *,
    frames: dict[str, Any],
    settings: dict[str, Any],
    result=None,
) -> None:
    """Save the current inputs, or reload a saved set."""
    service = PortfolioService(_store(), principal)

    st.sidebar.divider()
    st.sidebar.subheader("Portfolios")

    # Shown after the rerun that a save triggers, because the success message would
    # otherwise be discarded by that rerun.
    just_saved = st.session_state.pop("portfolio_saved_note", "")
    if just_saved:
        st.sidebar.success(f"Saved “{just_saved}”.")

    saved = service.list()
    if saved:
        labels = {
            f"{p.name} — saved {p.updated_at:%d %b %H:%M}": p.portfolio_id for p in saved
        }
        chosen = st.sidebar.selectbox(
            "Saved portfolios", list(labels), key="portfolio_choice"
        )
        load, delete = st.sidebar.columns(2)
        if load.button("Load", key="load_portfolio", width="stretch"):
            _load(service, labels[chosen])
        if delete.button("Delete", key="delete_portfolio", width="stretch"):
            service.delete(labels[chosen])
            st.rerun()
    else:
        st.sidebar.caption("Nothing saved yet.")

    name = st.sidebar.text_input(
        "Name", key="portfolio_name", placeholder="e.g. Q3 resilience plan"
    )
    if st.sidebar.button("Save current inputs", key="save_portfolio", width="stretch"):
        if not name.strip():
            st.sidebar.warning("Give the portfolio a name before saving.")
        else:
            service.save(name, tables=frames, settings=settings, result=result)
            # Rerun so the new portfolio appears in the list above. The list was read
            # at the top of this function, before the save, so without this the user
            # saves something and does not see it until they touch another control —
            # which reads as the save having failed.
            st.session_state["portfolio_saved_note"] = name.strip()
            st.rerun()


def _load(service: PortfolioService, portfolio_id: str) -> None:
    """Queue a load for the next run. See `apply_pending_load`."""
    loaded = service.load(portfolio_id)
    if loaded is None:
        # Only reachable if it was deleted between listing and clicking.
        st.sidebar.error("That portfolio is no longer available.")
        return

    st.session_state[PENDING_LOAD] = {
        "name": loaded.name,
        "tables": dict(loaded.tables),
        "settings": loaded.settings,
        "provenance": loaded.portfolio.provenance,
    }
    st.rerun()


def render_provenance_notice(result) -> None:
    """Say so when a reloaded portfolio no longer produces the answer it was saved with.

    This is the reason results are not persisted. A saved portfolio stores its inputs
    and the hashes of the run it was saved from; on reload the engine solves again and
    this compares. Storing the numbers instead would have shown the user a figure from
    an older engine with nothing to indicate it was no longer being produced.

    Called from `main` *after* the solve, not from `render_portfolios`. The portfolio
    controls are drawn before the engine runs, so comparing there would compare against
    the previous run's result and the notice would lag a full interaction behind the
    inputs — telling the user a portfolio still reproduced its saved answer immediately
    after they had changed the budget. Writing to `st.sidebar` from here still places it
    in the sidebar regardless of where in the script it is called.
    """
    provenance = st.session_state.get("loaded_provenance")
    if provenance is None or result is None:
        return

    if provenance.agrees_with(result):
        st.sidebar.success("Reloaded portfolio reproduces its saved result exactly.")
        return

    divergence = provenance.divergence(result)
    st.sidebar.warning(
        "This portfolio's inputs were reloaded, but the current result differs from "
        f"the one saved with it. {divergence}"
    )
