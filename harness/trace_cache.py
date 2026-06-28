from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
from typing import Any


@dataclass
class TraceEvent:
    request_id: str
    route: str
    model: str
    policy_version: str
    prompt_version: str
    status: str
    latency_ms: int
    stages: list[dict[str, Any]]
    request: dict[str, Any]
    result_summary: dict[str, Any]
    created_at: str = datetime.now(timezone.utc).isoformat()


class TraceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def write(self, event: TraceEvent) -> None:
        trace_file = self.path / f"{event.request_id}.json"
        with trace_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(event), f, indent=2)


class ResponseCache:
    def __init__(self, path: Path, max_entries: int = 1000) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(10, max_entries)

    def _key(self, data: dict[str, Any]) -> str:
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()

    def get(self, cache_key_payload: dict[str, Any]) -> dict[str, Any] | None:
        key = self._key(cache_key_payload)
        file = self.path / f"{key}.json"
        if not file.exists():
            return None
        with file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def put(self, cache_key_payload: dict[str, Any], value: dict[str, Any]) -> str:
        key = self._key(cache_key_payload)
        file = self.path / f"{key}.json"
        with file.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)

        # opportunistic cleanup: keep newest files only.
        entries = sorted(self.path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in entries[self.max_entries :]:
            old.unlink(missing_ok=True)
        return key

