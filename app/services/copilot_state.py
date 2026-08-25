"""Translating between copilot field names and Streamlit widget keys.

`copilot.py` speaks in `Inputs` field names because that is the vocabulary of the
decision. The sidebar speaks in widget keys because that is the vocabulary of the
framework. The two do not line up, and the mismatch is not cosmetic:

* ``mode`` is a value (``"monetary"``) in one and a radio *label* (``"Monetary NPV
  (recommended)"``) in the other.
* ``budget`` lives under ``"budget"`` in spend mode but under ``"max_capital"`` in
  target mode, gated behind the ``"cap_spend"`` checkbox — because in target mode a
  budget is an optional ceiling rather than the thing being spent.
* ``target_risk_pts`` is one nullable number in the copilot's world and two widgets in
  the sidebar's: a ``"target_mode"`` checkbox plus a ``"target_risk"`` number.

This module is pure and imports no streamlit, so the mapping is tested directly. The
acquired system had this translation nowhere — its copilot wrote `val_total_budget` and
`val_target_mode` straight into session state and hoped the widget names still matched.
When they drifted, the write landed on a key no widget read, and the copilot reported
success while changing nothing.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from app import adapters

MODE_LABELS: dict[str, str] = {
    adapters.MODE_MONETARY: "Monetary NPV (recommended)",
    adapters.MODE_LEGACY: "Legacy parity",
}

#: Fields whose widget key never moves.
SIMPLE_KEYS: dict[str, str] = {
    "exponent": "exponent",
    "value_per_risk_point": "value_per_risk_point",
    "value_per_lead_time_day": "value_per_day",
    "iterations": "iterations",
    "seed": "seed",
    "run_simulation": "run_simulation",
    "show_sweep": "show_sweep",
    "baseline_risk_pts": "baseline_risk",
}

#: Inputs that determine the price of risk and the market overlay.
#:
#: These are round-tripped so a saved portfolio reloads at the price it was saved at.
#: Without them, reopening a portfolio would silently fall back to the sidebar's
#: default price of risk, and every currency figure would change while the plan, the
#: budget and the audit's own claim of reproducibility all looked unaffected. No
#: copilot rule targets any of them, so they are persisted but not proposable.
PRICING_KEYS: dict[str, str] = {
    "derive_prices": "derive_prices",
    "annual_revenue": "annual_revenue",
    "revenue_at_risk_share_pct": "revenue_at_risk_share_pct",
    "gross_margin_pct": "gross_margin_pct",
    "disruption_days": "disruption_days",
    "fixed_cost_per_event": "fixed_cost_per_event",
    "event_probability_pct": "event_probability_pct",
    "exposure_basis": "exposure_basis",
    "price_source": "price_source",
    "apply_macro": "apply_macro",
    "apply_market_exposure": "apply_market_exposure",
}

#: Defaults matching the sidebar's, for the first run when no widget exists yet.
PRICING_DEFAULTS: dict[str, object] = {
    "derive_prices": False,
    "annual_revenue": 400_000_000.0,
    "revenue_at_risk_share_pct": 35.0,
    "gross_margin_pct": 28.0,
    "disruption_days": 21.0,
    "fixed_cost_per_event": 0.0,
    "event_probability_pct": 65.5,
    "exposure_basis": "",
    "price_source": "",
    "apply_macro": False,
    "apply_market_exposure": False,
}


def snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """Read the current inputs out of session state, in copilot vocabulary.

    Defaults mirror the sidebar's widget defaults, because on the very first run the
    script has not created the widgets yet and their keys are absent. Returning ``0``
    for an absent budget would have the copilot propose a change away from a value the
    user never chose.
    """
    target_on = bool(state.get("target_mode", False))

    if target_on:
        capped = bool(state.get("cap_spend", False))
        budget = float(state.get("max_capital", 750_000.0)) if capped else None
    else:
        budget = float(state.get("budget", 750_000.0))

    mode_label = state.get("mode", MODE_LABELS[adapters.MODE_MONETARY])
    mode = (
        adapters.MODE_LEGACY
        if mode_label == MODE_LABELS[adapters.MODE_LEGACY]
        else adapters.MODE_MONETARY
    )

    return {
        "mode": mode,
        "baseline_risk_pts": float(state.get("baseline_risk", 65.5)),
        "budget": budget,
        "target_risk_pts": float(state.get("target_risk", 45.0)) if target_on else None,
        "exponent": float(state.get("exponent", 0.85)),
        "value_per_risk_point": float(state.get("value_per_risk_point", 50_000.0)),
        "value_per_lead_time_day": float(state.get("value_per_day", 0.0)),
        "iterations": int(state.get("iterations", 10_000)),
        "seed": int(state.get("seed", 42)),
        "run_simulation": bool(state.get("run_simulation", True)),
        "show_sweep": bool(state.get("show_sweep", True)),
        **{
            field: state.get(key, PRICING_DEFAULTS[field])
            for field, key in PRICING_KEYS.items()
        },
    }


def widget_updates(updated: Mapping[str, Any]) -> dict[str, Any]:
    """Turn an applied snapshot into the session-state writes that realise it.

    Only the keys that need to change are returned, so a caller can see exactly what
    is about to be written — and a test can assert that a proposal about the budget
    does not quietly also move the target.
    """
    writes: dict[str, Any] = {}

    if "mode" in updated:
        writes["mode"] = MODE_LABELS.get(updated["mode"], MODE_LABELS[adapters.MODE_MONETARY])

    for field, key in {**SIMPLE_KEYS, **PRICING_KEYS}.items():
        if field in updated:
            writes[key] = updated[field]

    target = updated.get("target_risk_pts", ...)
    if target is not ...:
        writes["target_mode"] = target is not None
        if target is not None:
            writes["target_risk"] = float(target)

    budget = updated.get("budget", ...)
    if budget is not ...:
        # Which key a budget belongs in depends on the mode it will be read in, which
        # is the mode *after* this proposal is applied, not before.
        target_on = writes.get("target_mode", updated.get("target_risk_pts") is not None)
        if target_on:
            writes["cap_spend"] = budget is not None
            if budget is not None:
                writes["max_capital"] = float(budget)
        elif budget is not None:
            writes["budget"] = float(budget)

    return writes


def apply_to_state(state: MutableMapping[str, Any], updated: Mapping[str, Any]) -> dict[str, Any]:
    """Write the updates into session state and return what was written."""
    writes = widget_updates(updated)
    for key, value in writes.items():
        state[key] = value
    return writes
