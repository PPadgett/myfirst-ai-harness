"""Evidence and claim primitives for verifier-aware output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvidenceRecord:
    evidence_id: str
    route_id: str
    source: str
    record: dict[str, Any]
    tool_name: str | None = None
    redacted_fields: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    route_id: str
    statement: str
    evidence_ids: list[str]
    status: str = "pending"
    created_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

