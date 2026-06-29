from __future__ import annotations

from pathlib import Path

from harness.trace_cache import ResponseCache


def test_response_cache_cleanup_ignores_entries_removed_between_glob_and_stat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cache = ResponseCache(tmp_path)
    existing = tmp_path / "existing.json"
    missing = tmp_path / "missing.json"
    existing.write_text("{}", encoding="utf-8")

    original_glob = Path.glob

    def _fake_glob(self: Path, pattern: str):
        if self == tmp_path and pattern == "*.json":
            yield existing
            yield missing
            return
        yield from original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", _fake_glob)

    key = cache.put({"question": "hello"}, {"answer": "world"})

    assert key
