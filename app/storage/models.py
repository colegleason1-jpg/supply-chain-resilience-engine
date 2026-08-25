"""What a saved portfolio is.

The central decision in this module is what *not* to store.

A saved portfolio holds **inputs only** — the intervention table, the bundles, the
dependencies, the correlations, and the settings that were in force. It does not hold
allocations, risk figures, capital totals or simulation outcomes, even though those are
the things the user was looking at when they hit save.

Storing results would be storing a claim that decays. The engine is versioned and
under active development: `AllocationConcaveResponse` could be recalibrated tomorrow,
and a stored "optimised risk: 25.25%" would then be a number attributed to the current
engine that the current engine does not produce. The acquired system's whole class of
defect was numbers presented as more established than they were, so a persistence
layer that silently ages results into fiction is the same mistake with a database
behind it.

What we do store is `provenance`: the engine version and the input hash of the run the
user was looking at when they saved. That is enough to answer the only question that
matters on reload — "does this still produce what it produced then?" — by re-solving
and comparing. Cheap to check, impossible to fake.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any


class StorageError(RuntimeError):
    """Raised when a store cannot satisfy a request.

    Distinct from "not found", which is represented by returning ``None``: a missing
    portfolio is an ordinary outcome, a broken store is not.
    """


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Which engine produced the figures the user was looking at when they saved."""

    engine_version: str = ""
    input_hash: str = ""
    output_hash: str = ""
    solver_status: str = ""
    saved_at: datetime = field(default_factory=_now)

    @classmethod
    def from_result(cls, result) -> RunProvenance:
        audit = result.audit or {}
        return cls(
            engine_version=str(audit.get("engine_version", "")),
            input_hash=str(audit.get("input_hash", "")),
            output_hash=str(audit.get("output_hash", "")),
            solver_status=str(result.status),
        )

    def agrees_with(self, result) -> bool:
        """Would re-solving today reproduce what was on screen at save time?

        Both hashes must match. A matching input hash with a differing output hash is
        the interesting case: identical inputs, different answer, which means the
        engine changed underneath the saved portfolio.
        """
        audit = result.audit or {}
        return (
            bool(self.output_hash)
            and self.input_hash == str(audit.get("input_hash", ""))
            and self.output_hash == str(audit.get("output_hash", ""))
        )

    def divergence(self, result) -> str:
        """Explain a mismatch in the terms that determine what the user should do."""
        audit = result.audit or {}
        if not self.output_hash:
            return "This portfolio was saved without a recorded solve, so there is nothing to compare against."
        if self.agrees_with(result):
            return ""
        if self.input_hash != str(audit.get("input_hash", "")):
            return (
                "The inputs no longer hash to what was saved, so something in the "
                "portfolio or its settings has been edited since. The figures below "
                "are from a fresh solve."
            )
        engine_now = str(audit.get("engine_version", ""))
        if self.engine_version and self.engine_version != engine_now:
            return (
                f"Identical inputs, different result: this was saved under engine "
                f"{self.engine_version} and re-solved under {engine_now}. The figures "
                "below are correct for the current engine and will not match anything "
                "recorded elsewhere from the earlier one."
            )
        return (
            "Identical inputs produced a different result under the same engine "
            "version. That should not happen and is worth reporting rather than "
            "working around."
        )


@dataclass(frozen=True, slots=True)
class SavedPortfolio:
    """A named set of inputs, owned by exactly one principal.

    ``owner_id`` is not decoration. Every repository method takes an owner and filters
    on it, so a caller cannot read another user's portfolio by guessing an id — the
    check lives below the UI rather than in it.
    """

    name: str
    owner_id: str
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    provenance: RunProvenance = field(default_factory=RunProvenance)
    portfolio_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a saved portfolio needs a name")
        if not self.owner_id.strip():
            raise ValueError("a saved portfolio needs an owner")

    def touched(self, **changes: Any) -> SavedPortfolio:
        """Return a copy with ``updated_at`` advanced."""
        return replace(self, updated_at=_now(), **changes)
