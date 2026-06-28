"""Checkpoint persistence primitives shared across harness execution paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PhaseState:
    phase: str
    run_id: str
    attempt: int
    route_id: str
    status: str = "ok"
    next_action: str | None = None
    error_code: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    route_metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Checkpoint(PhaseState):
    payload: dict[str, Any] = field(default_factory=dict)
    state_diffs: dict[str, Any] = field(default_factory=dict)

    def write(self, out_dir: Path, attempt: int) -> str:
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.run_id}-a{attempt}-{self.phase}.json"
        target = out_dir / filename
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(target)


def write_checkpoint(out_dir: Path, run_id: str, attempt: int, phase: str, route_id: str, **kwargs: Any) -> str:
    """Write a checkpoint and return the path."""
    checkpoint = Checkpoint(phase=phase, run_id=run_id, attempt=attempt, route_id=route_id, **kwargs)
    return checkpoint.write(out_dir, attempt)
