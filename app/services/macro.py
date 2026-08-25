"""The macro multiplier, and honesty about where it came from.

The acquired system shipped this with the literal API key `'YOUR_FREE_API_KEY_HERE'`
in client-side code (F13). Every request failed, the bare `except` swallowed the
failure, and the UI displayed a multiplier of 1.0 next to a green "feed status" label.
The number was a default presented as telemetry.

Two changes. The credential is read from the environment, so it is deployment
configuration rather than source code. And the failure path is explicit: when the feed
is unavailable the returned reading says so, and the caller is expected to show that.
A stale or absent feed is useful information; a stale feed labelled "live" is not.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

#: Brent price the multiplier is anchored to. An assumption, recorded as one.
ANCHOR_BRENT_USD = 80.0
#: Sensitivity of the multiplier to a proportional move in Brent above the anchor.
MACRO_SENSITIVITY = 0.15
#: How long a reading stays usable before we try again.
CACHE_TTL_SECONDS = 1800.0


@dataclass(frozen=True, slots=True)
class MacroReading:
    """A macro multiplier and its provenance.

    ``is_live`` is the field that matters. The acquired system had no equivalent, so a
    hardcoded fallback of 1.0 was indistinguishable on screen from a real reading of
    1.0.
    """

    multiplier: float
    status: str
    is_live: bool
    brent_usd: float | None = None
    fetched_at: float | None = None

    @property
    def provenance(self) -> str:
        if self.is_live and self.brent_usd is not None:
            return (
                f"Brent ${self.brent_usd:,.2f} vs ${ANCHOR_BRENT_USD:,.2f} anchor, "
                f"sensitivity {MACRO_SENSITIVITY:g}"
            )
        return "No live reading. Multiplier held at 1.0 (no macro adjustment applied)."


def multiplier_from_brent(brent_usd: float) -> float:
    """Convert a Brent price into a risk multiplier.

    Deliberately one-sided: prices above the anchor raise risk, prices below it do not
    reduce it. That asymmetry is the acquired system's assumption, preserved because
    changing it would change results, but it is an assumption and not a finding about
    the world. Cheap oil making supply chains *safer* is not something anyone
    established.
    """
    if brent_usd <= ANCHOR_BRENT_USD:
        return 1.0
    excess = (brent_usd - ANCHOR_BRENT_USD) / ANCHOR_BRENT_USD
    return round(1.0 + excess * MACRO_SENSITIVITY, 4)


NO_FEED = MacroReading(
    multiplier=1.0,
    status="No macro feed configured",
    is_live=False,
)


def fetch_macro_reading(timeout: float = 4.0) -> MacroReading:
    """Read Brent from the configured feed, or say plainly that we could not.

    Never raises and never fabricates. The unconfigured case is distinguished from the
    failed case, because "nobody set this up" and "the provider is down" call for
    different actions.
    """
    api_key = os.environ.get("SCRCAE_MACRO_API_KEY", "").strip()
    if not api_key:
        return NO_FEED

    try:
        import requests

        response = requests.get(
            "https://api.api-ninjas.com/v1/commodityprice",
            params={"name": "brent_crude_oil"},
            headers={"X-Api-Key": api_key},
            timeout=timeout,
        )
        if response.status_code != 200:
            return MacroReading(
                multiplier=1.0,
                status=f"Feed returned HTTP {response.status_code}; no adjustment applied",
                is_live=False,
            )
        price = float(response.json()["price"])
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return MacroReading(
            multiplier=1.0,
            status=f"Feed unavailable ({type(exc).__name__}); no adjustment applied",
            is_live=False,
        )

    return MacroReading(
        multiplier=multiplier_from_brent(price),
        status="Live",
        is_live=True,
        brent_usd=price,
        fetched_at=time.time(),
    )
