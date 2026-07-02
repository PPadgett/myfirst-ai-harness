"""Unit tests for the gauntlet quality/determinism reporter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from gauntlet_report import build_report, extract_answer, run_checks  # noqa: E402


def test_extract_answer_strips_think_block() -> None:
    answer, thinking_only = extract_answer("<think>reasoning here</think>\n\nThe answer is 43.")
    assert answer == "The answer is 43."
    assert thinking_only is False


def test_extract_answer_flags_unterminated_think() -> None:
    answer, thinking_only = extract_answer("<think>endless reasoning that never closes")
    assert answer == ""
    assert thinking_only is True


def test_extract_answer_passthrough_without_think() -> None:
    answer, thinking_only = extract_answer("Plain answer.")
    assert answer == "Plain answer."
    assert thinking_only is False


def test_run_checks_contains_and_regex() -> None:
    checks = {"c": [{"type": "nonempty_answer"}, {"type": "contains", "value": "43"}, {"type": "regex", "value": "answer"}]}
    ok, failed = run_checks("c", "The answer is 43, computed precisely.", False, checks)
    assert ok and not failed
    ok, failed = run_checks("c", "The answer is 42, computed precisely.", False, checks)
    assert not ok and failed == ["contains:43"]


def _row(case_id: str, backend: str, iteration: int, text: str, status: str = "ok") -> dict:
    return {
        "case_id": case_id,
        "selected_backend": backend,
        "iteration": iteration,
        "response_text": text,
        "status": status,
        "fallback_attempted": False,
        "provider_warning": "",
        "latency_ms": 100,
        "tokens_per_second": 10.0,
    }


def test_build_report_scoreboard_and_determinism(tmp_path: Path) -> None:
    checks = {"case_a": [{"type": "nonempty_answer"}, {"type": "contains", "value": "43"}]}
    good = "<think>x</think>\n\nThe answer is 43 and that is definitely final."
    for run_name, texts in (
        ("run1", [good, good]),
        ("run2", [good, good.replace("43", "44")]),  # drifting iteration 2
    ):
        d = tmp_path / run_name
        d.mkdir()
        payload = {"cases": [_row("case_a", "gpu", i + 1, t) for i, t in enumerate(texts)]}
        (d / "multidevice-benchmark-20260702T000000.json").write_text(json.dumps(payload), encoding="utf-8")

    report = build_report([tmp_path / "run1", tmp_path / "run2"], checks)
    gpu = report["backends"]["gpu"]
    assert gpu["rows"] == 4
    assert gpu["success_pct"] == 100.0
    assert gpu["quality_pass_pct"] == 75.0  # the "44" row fails contains:43
    case = report["cases"]["case_a"]
    assert case["distinct_outputs_by_run"] == {"run1": 1, "run2": 2}
    assert case["cross_run_identical"] is True  # iteration-1 texts match
