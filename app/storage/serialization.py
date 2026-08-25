"""DataFrames to JSON-safe records and back.

Kept separate from the stores because both the SQLite and Supabase backends need
exactly the same treatment, and because this is the layer that decides what a blank
cell means — which is the question the acquired system got wrong.

The rule: a blank cell round-trips as ``None``, never as ``0``. `adapters.py` rejects a
row whose cost is missing, and that rejection is the safety property that stops a
zero-cost intervention from being funded to the maximum at any budget. If saving and
reloading turned blanks into zeros, a reload would launder unusable rows into
attractive ones.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd

#: ``history`` and ``calibration`` are saved alongside the inputs because they are
#: what makes a calibrated elasticity calibrated. An exposure counts as calibrated
#: only while a stored fit vouches for the exact number in the table, so a portfolio
#: that reloaded without its evidence would silently downgrade every fitted
#: elasticity to asserted while displaying the identical figures — the same numbers,
#: quietly stripped of the only thing that justified them.
TABLE_KEYS = (
    "nodes",
    "bundles",
    "dependencies",
    "correlation",
    "exposure",
    "history",
    "calibration",
)


def _clean(value: Any) -> Any:
    """Make one cell JSON-safe without changing what it means."""
    if value is None:
        return None
    # pandas uses several sentinels for "missing" depending on column dtype, and NaN
    # is not equal to itself, so identity checks are not enough.
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    # numpy scalars and anything else exotic: go through Python's own conversion
    # rather than str(), which would turn 180000.0 into "180000.0" and then into a
    # string column on reload.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _clean(item())
        except (ValueError, TypeError):
            pass
    return str(value)


def frame_to_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [
        {str(column): _clean(row[column]) for column in frame.columns}
        for _, row in frame.iterrows()
    ]


def records_to_frame(
    records: list[dict[str, Any]] | None, columns: list[str] | None = None
) -> pd.DataFrame:
    """Rebuild a frame, preserving column order where we know it.

    An empty saved table still needs its columns, or the editor comes back with no
    headers and the user cannot type anything into it.
    """
    if not records:
        return pd.DataFrame(columns=columns or [])
    frame = pd.DataFrame(records)
    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None
        # Keep any extra columns the user added rather than dropping their data.
        extras = [c for c in frame.columns if c not in columns]
        frame = frame[list(columns) + extras]
    return frame


def tables_to_records(frames: dict[str, pd.DataFrame | None]) -> dict[str, list[dict[str, Any]]]:
    return {key: frame_to_records(frames.get(key)) for key in TABLE_KEYS}


def records_to_tables(
    tables: dict[str, list[dict[str, Any]]] | None,
    columns: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    tables = tables or {}
    columns = columns or {}
    return {
        key: records_to_frame(tables.get(key), columns.get(key)) for key in TABLE_KEYS
    }


def isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def parse_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    if isinstance(raw, str) and raw:
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return datetime.now(UTC)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)
