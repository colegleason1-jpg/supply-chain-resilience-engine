"""What the market feed did, including when it did nothing.

The acquired system's market panel showed eleven rows of prices and a status of
either "Live" or "Offline (Holding Last Known)". Both were reachable without a
single successful fetch, because the table was initialised to 100.00 and the
failure path claimed to be holding a previous reading it had never taken.

This panel reports the feed and the exposure separately, and gives an inert
exposure the same prominence as an applied one. A mapping the user configured and
believes is live, silently doing nothing because a symbol is unavailable, is the
condition that made F11 invisible for as long as it existed.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.market import ExposureOverlay, QuoteSet


def render(overlay: ExposureOverlay, quotes: QuoteSet, *, enabled: bool) -> None:
    with st.expander("Market exposure", expanded=bool(overlay.skipped_effects)):
        if not enabled:
            st.caption(
                "Market exposure is switched off. Intervention effectiveness is "
                "taken from your figures alone."
            )
            return

        if not overlay.effects:
            st.caption(
                "No market exposure is configured. Add rows to the Market exposure "
                "table to link an intervention to a commodity or index."
            )
            return

        st.caption(overlay.summary())
        if not overlay.one_sided:
            st.caption(
                "Two-sided mode: prices below anchor reduce effectiveness as well."
            )

        if quotes.quotes:
            st.caption("**Feed**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Market": quote.symbol,
                            # No number at all when unavailable. A placeholder price
                            # is what made the acquired feed's failure invisible.
                            "Price": (
                                f"{quote.price:,.2f}"
                                if quote.price is not None
                                else "\u2014"
                            ),
                            "Status": quote.status,
                        }
                        for quote in sorted(
                            quotes.quotes.values(), key=lambda q: q.symbol
                        )
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        st.caption("**Adjustments**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Intervention": effect.node_id,
                        "Market": effect.symbol,
                        "Anchor": f"{effect.anchor:,.2f}",
                        "Deviation": (
                            f"{effect.deviation:+.1%}"
                            if effect.deviation is not None
                            else "\u2014"
                        ),
                        "Elasticity": f"{effect.elasticity:g}",
                        "Effect on effectiveness": (
                            f"{effect.contribution:+.1%}"
                            if effect.applied
                            else "none"
                        ),
                        "Applied": "Yes" if effect.applied else "No",
                        "Why": effect.note,
                    }
                    for effect in overlay.effects
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        for effect in overlay.skipped_effects:
            st.warning(
                f"{effect.node_id} is configured as exposed to {effect.symbol}, but "
                f"no adjustment was applied: {effect.note}. This intervention was "
                f"evaluated on your figures alone.",
                icon=":material/warning:",
            )

        if overlay.multipliers:
            st.caption(
                "These adjustments are applied when the allocation is solved. Your "
                "intervention table is not modified, so nothing accumulates between "
                "runs and the figures you entered remain the record."
            )
