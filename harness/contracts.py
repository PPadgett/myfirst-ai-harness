"""Shared contract and envelope models for harness outputs.

These models are intentionally small and additive so that existing JSON response
shape stays stable for callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEnvelope:
    """Standardized per-phase metadata."""

    phase: str
    next_action: str
    status: str = "ok"
    evidence_refs: list[str] = field(default_factory=list)
    error_code: str | None = None
    route_metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceEnvelope:
    """Evidence rows referenced by validators and final response contracts."""

    claim_id: str
    route_id: str
    evidence_type: str
    payload: dict[str, Any]
    source: str = "runtime"
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

