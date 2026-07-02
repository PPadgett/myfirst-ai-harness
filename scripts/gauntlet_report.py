#!/usr/bin/env python3
"""Combine seeded benchmark runs into a per-device scoreboard.

Adds what the raw benchmark JSON lacks: answer quality (post-<think> checks
from benchmark-quality-checks.json), thinking-truncation accounting, and
cross-run determinism. Usage:

    python scripts/gauntlet_report.py RUN_DIR [RUN_DIR ...] \
        [--checks scripts/benchmark-quality-checks.json] \
        [--output-json report.json] [--output-md report.md]

Each RUN_DIR is an -OutputDir from Benchmark-MultiDeviceHarness.ps1; the
newest multidevice-benchmark-*.json in each is used.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
NONEMPTY_MIN_CHARS = 20


def extract_answer(text: str) -> tuple[str, bool]:
    """Return (answer_text, thinking_only).

    Strips <think> blocks; an opened-but-unterminated block means the whole
    completion was reasoning (the truncation failure mode) => thinking_only.
    """
    raw = text or ""
    if THINK_OPEN not in raw:
        return raw.strip(), False
    if THINK_CLOSE not in raw:
        return "", True
    answer = raw.rsplit(THINK_CLOSE, 1)[1].strip()
    return answer, not answer


def run_checks(case_id: str, answer: str, thinking_only: bool, checks: dict[str, Any]) -> tuple[bool, list[str]]:
    case_checks = checks.get(case_id, [{"type": "nonempty_answer"}])
    failed: list[str] = []
    lowered = answer.lower()
    for check in case_checks:
        kind = str(check.get("type", ""))
        value = str(check.get("value", ""))
        if kind == "nonempty_answer":
            ok = not thinking_only and len(answer) >= NONEMPTY_MIN_CHARS
        elif kind == "contains":
            ok = value.lower() in lowered
        elif kind == "regex":
            ok = re.search(value, lowered, re.IGNORECASE) is not None
        else:
            ok = False
            kind = f"unknown:{kind}"
        if not ok:
            failed.append(f"{kind}:{value}" if value else kind)
    return (not failed), failed


def load_run(run_dir: Path) -> dict[str, Any]:
    files = sorted(run_dir.glob("multidevice-benchmark-*.json"))
    if not files:
        raise SystemExit(f"no multidevice-benchmark-*.json in {run_dir}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def p50(values: list[float]) -> float:
    return round(statistics.median(values), 1) if values else 0.0


def build_report(run_dirs: list[Path], checks: dict[str, Any]) -> dict[str, Any]:
    runs = {d.name: load_run(d) for d in run_dirs}

    per_backend: dict[str, dict[str, Any]] = {}
    per_case: dict[str, dict[str, Any]] = {}
    first_iteration_texts: dict[str, dict[str, str]] = {}

    for run_name, data in runs.items():
        for row in data.get("cases", []):
            case_id = str(row.get("case_id"))
            backend = str(row.get("selected_backend") or "unknown")
            answer, thinking_only = extract_answer(str(row.get("response_text") or ""))
            status_ok_early = row.get("status") == "ok"
            if status_ok_early:
                quality_pass, failed = run_checks(case_id, answer, thinking_only, checks)
            else:
                # Error rows carry harness error text, not model output; checking
                # it would produce noise. They already count as quality failures.
                quality_pass, failed = False, []

            b = per_backend.setdefault(backend, {
                "rows": 0, "ok": 0, "fallback": 0, "truncated": 0,
                "thinking_only": 0, "quality_pass": 0, "latencies": [], "tps": [],
            })
            b["rows"] += 1
            status_ok = row.get("status") == "ok"
            b["ok"] += int(status_ok)
            b["fallback"] += int(bool(row.get("fallback_attempted")))
            b["truncated"] += int("truncated" in str(row.get("provider_warning") or ""))
            b["thinking_only"] += int(thinking_only)
            b["quality_pass"] += int(status_ok and quality_pass)
            if status_ok:
                b["latencies"].append(float(row.get("latency_ms") or 0))
                tps = float(row.get("tokens_per_second") or 0)
                if tps:
                    b["tps"].append(tps)

            c = per_case.setdefault(case_id, {
                "rows": 0, "ok": 0, "quality_pass": 0, "failed_checks": {},
                "distinct_by_run": {}, "backend": backend,
            })
            c["rows"] += 1
            c["ok"] += int(status_ok)
            c["quality_pass"] += int(status_ok and quality_pass)
            for item in failed:
                c["failed_checks"][item] = c["failed_checks"].get(item, 0) + 1
            c["distinct_by_run"].setdefault(run_name, set()).add(str(row.get("response_text") or ""))
            if int(row.get("iteration") or 0) == 1:
                first_iteration_texts.setdefault(case_id, {})[run_name] = str(row.get("response_text") or "")

    backends_out = {}
    for name, b in sorted(per_backend.items()):
        rows = b["rows"] or 1
        backends_out[name] = {
            "rows": b["rows"],
            "success_pct": round(100.0 * b["ok"] / rows, 1),
            "fallback_pct": round(100.0 * b["fallback"] / rows, 1),
            "truncated_pct": round(100.0 * b["truncated"] / rows, 1),
            "thinking_only_pct": round(100.0 * b["thinking_only"] / rows, 1),
            "quality_pass_pct": round(100.0 * b["quality_pass"] / rows, 1),
            "latency_p50_ms": p50(b["latencies"]),
            "tokens_per_second_avg": round(statistics.fmean(b["tps"]), 2) if b["tps"] else 0.0,
        }

    cases_out = {}
    for name, c in sorted(per_case.items()):
        rows = c["rows"] or 1
        texts = first_iteration_texts.get(name, {})
        cases_out[name] = {
            "backend": c["backend"],
            "rows": c["rows"],
            "success_pct": round(100.0 * c["ok"] / rows, 1),
            "quality_pass_pct": round(100.0 * c["quality_pass"] / rows, 1),
            "top_failed_checks": sorted(c["failed_checks"], key=c["failed_checks"].get, reverse=True)[:3],
            "distinct_outputs_by_run": {run: len(s) for run, s in sorted(c["distinct_by_run"].items())},
            "cross_run_identical": len(set(texts.values())) <= 1 if len(texts) > 1 else None,
        }

    return {"runs": sorted(runs), "backends": backends_out, "cases": cases_out}


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Benchmark Gauntlet Report", "", f"Runs: {', '.join(report['runs'])}", ""]
    lines += ["## Device scoreboard", "",
              "| device | rows | success % | quality % | truncated % | thinking-only % | fallback % | p50 ms | tok/s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, b in report["backends"].items():
        lines.append(
            f"| {name} | {b['rows']} | {b['success_pct']} | {b['quality_pass_pct']} | "
            f"{b['truncated_pct']} | {b['thinking_only_pct']} | {b['fallback_pct']} | "
            f"{b['latency_p50_ms']} | {b['tokens_per_second_avg']} |"
        )
    lines += ["", "## Cases", "",
              "| case | device | success % | quality % | distinct/run | cross-run identical | top failed checks |",
              "|---|---|---|---|---|---|---|"]
    for name, c in report["cases"].items():
        distinct = ",".join(str(v) for v in c["distinct_outputs_by_run"].values())
        lines.append(
            f"| {name} | {c['backend']} | {c['success_pct']} | {c['quality_pass_pct']} | "
            f"{distinct} | {c['cross_run_identical']} | {', '.join(c['top_failed_checks']) or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--checks", type=Path, default=Path(__file__).parent / "benchmark-quality-checks.json")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    checks = json.loads(args.checks.read_text(encoding="utf-8")) if args.checks.exists() else {}
    report = build_report(args.run_dirs, checks)
    markdown = to_markdown(report)

    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
