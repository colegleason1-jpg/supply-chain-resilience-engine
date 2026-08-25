"""Model audit records.

The acquired system printed a fixed string, ``sha256:8f4c99a...``, and called it
a verification hash. It was a literal in a formatted display string and hashed
nothing.

This module computes real content hashes over the actual inputs, so a result can
be tied to the exact data, parameters, model versions and seed that produced it.
That is the difference between an audit ledger and a badge.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

__all__ = ["ENGINE_VERSION", "canonical_json", "content_hash", "build_audit_record"]

ENGINE_VERSION = "0.1.0"


def _plain(value: Any) -> Any:
    """Convert arbitrary inputs into JSON-serialisable primitives, stably."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        # Round to 12 significant decimals so platform float formatting does not
        # change a hash for numerically identical inputs.
        return float(f"{value:.12g}")
    return repr(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding, stable across runs and platforms."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    """SHA-256 over the canonical encoding of ``value``."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_audit_record(
    *,
    network: Any,
    objective: Any,
    risk_response: Any,
    parameters: Mapping[str, Any],
    solver: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the provenance record attached to every result."""
    record: dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "solver": solver,
        "objective_version": getattr(objective, "version", repr(objective)),
        "objective_unit": getattr(objective, "unit", "unknown"),
        "objective_parameters_hash": content_hash(objective),
        # Objectives may publish readable provenance for the prices they embed.
        # Recorded outside the hash inputs on purpose: it is derived from fields
        # already hashed, so including it would change nothing but the appearance
        # of rigour.
        "objective_provenance": _plain(getattr(objective, "provenance", None)),
        "risk_response_version": getattr(risk_response, "version", repr(risk_response)),
        "risk_response_parameters_hash": content_hash(risk_response),
        "network_hash": content_hash(network),
        "parameters": _plain(parameters),
        "parameters_hash": content_hash(parameters),
    }
    if extra:
        record.update(_plain(extra))
    record["input_hash"] = content_hash(
        {
            "network": record["network_hash"],
            "objective": record["objective_parameters_hash"],
            "risk_response": record["risk_response_parameters_hash"],
            "parameters": record["parameters_hash"],
        }
    )
    return record
