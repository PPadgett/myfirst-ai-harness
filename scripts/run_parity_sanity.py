#!/usr/bin/env python3
"""
Parity script for the dual-track harness implementation.

Usage:
- real engine runs via RealHarnessEngine in real_ai_harness.py
- production engine runs via HarnessRuntime in harness/runtime.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from harness.adapters.base import BaseModelClient
from harness.config import load_runtime_config
from harness.runtime import HarnessRuntime
from harness.types import ModelGenerateRequest, ModelGenerateResult
from real_ai_harness import RealHarnessEngine


@dataclass
class Case:
    query: str
    expect: dict[str, Any]


class DeterministicMockModelClient(BaseModelClient):
    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        response_payload: str | dict[str, str]
        if req.response_schema and req.response_schema.get("required"):
            required = req.response_schema.get("required", [])
            response_payload = {name: f"{name}-ok" for name in required if isinstance(name, str)}
        elif req.tools:
            response_payload = {"tool_calls": [], "answer": "tool-plan-disabled"}
        else:
            response_payload = "ok"

        text = response_payload if isinstance(response_payload, str) else json.dumps(response_payload, ensure_ascii=False)
        return ModelGenerateResult(
            text=str(text),
            reasoning=None,
            raw={"text": str(text)},
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


def _load_cases(path: Path) -> list[Case]:
    if not path.exists():
        raise FileNotFoundError(f"Missing fixture: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Fixture must be a JSON list of {query, expect} objects.")
    cases: list[Case] = []
    for item in payload:
        if not isinstance(item, dict) or "query" not in item:
            raise ValueError("Fixture items must include a query and optional expect object.")
        query = str(item.get("query", "")).strip()
        expect = item.get("expect", {})
        if not isinstance(expect, dict):
            expect = {}
        if not query:
            raise ValueError("Fixture query must be a non-empty string.")
        cases.append(Case(query=query, expect=expect))
    return cases


def _route_category_real(route_id: str | None) -> str:
    route = str(route_id or "").strip().lower()
    if route in {"ask_clarification", "meta_route", "fallback_unknown"}:
        return "clarification"
    if route in {"run_tests", "schedule_email"}:
        return "tool_action"
    return "informational"


def _route_category_runtime(route: str | None, *, next_action: str | None = None) -> str:
    action = str(next_action or "").strip().lower()
    if action == "ask_clarification":
        return "clarification"

    value = str(route or "").strip().lower()
    if value in {"tool_required", "side_effecting_action", "high_risk"}:
        return "tool_action"
    if value in {"structured_extraction", "direct", "grounded_qa", "code_or_data"}:
        return "informational"
    return "other"


def _permission_blocked_real(result: dict[str, Any]) -> bool:
    status = str(result.get("status", "")).lower()
    final_response = result.get("final_response")
    guard = result.get("guard", {})
    missing_permissions = []
    if isinstance(final_response, dict):
        missing_permissions = final_response.get("missing_permissions", [])
    return status in {"blocked", "validation_failed"} and bool(
        (isinstance(missing_permissions, list) and missing_permissions)
        or (isinstance(guard, dict) and not guard.get("input", {}).get("allow", True))
    )


def _permission_blocked_runtime(result: dict[str, Any]) -> bool:
    status = str(result.get("status", "")).lower()
    validation = result.get("validation") or {}
    next_action = str(result.get("next_action", "")).lower()
    if status == "validation_block" and next_action == "ask_clarification":
        return False
    if isinstance(validation, dict) and validation.get("failed_route_ids"):
        return status in {"blocked", "validation_block"}
    return status in {"blocked", "validation_block"}


def _extract_error_codes(obj: dict[str, Any], keys: tuple[str, ...] = ("tool_calls", "tool_plan")) -> list[str]:
    codes: list[str] = []
    tool_calls = obj.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            code = call.get("error_code")
            if isinstance(code, str) and code:
                codes.append(code)
    tool_plan = obj.get("tool_plan")
    if isinstance(tool_plan, list):
        for call in tool_plan:
            if not isinstance(call, dict):
                continue
            code = call.get("error_code")
            if isinstance(code, str) and code:
                codes.append(code)
    for key in keys:
        if key == "tool_plan":
            continue
        payload = obj.get(key)
        if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
            for call in payload["tool_calls"]:
                if isinstance(call, dict):
                    code = call.get("error_code")
                    if isinstance(code, str) and code:
                        codes.append(code)
        if isinstance(payload, list):
            for call in payload:
                if isinstance(call, dict):
                    code = call.get("error_code")
                    if isinstance(code, str) and code:
                        codes.append(code)
    # Deduplicate while preserving deterministic order.
    unique: list[str] = []
    for code in codes:
        if code not in unique:
            unique.append(code)
    return unique


def _normalize_real(result: dict[str, Any]) -> dict[str, Any]:
    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    status = str(result.get("status", "")).lower()
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    return {
        "route": result.get("route_id"),
        "route_category": _route_category_real(result.get("route_id")),
        "status": status,
        "next_action": result.get("next_action"),
        "evidence_present": bool(result.get("evidence")),
        "evidence_count": int(len(result.get("evidence", []) or [])),
        "route_confidence": meta.get("route_confidence"),
        "route_confidence_gap": meta.get("route_confidence_gap"),
        "validation": validation,
        "validation_ok": bool(validation.get("ok")),
        "failed_route_ids": list(validation.get("failed_route_ids", [])) if isinstance(validation.get("failed_route_ids"), list) else [],
        "trace_count": int(result.get("trace_phase_count", 0)),
        "permission_blocked": _permission_blocked_real(result),
        "error_codes": _extract_error_codes(result, keys=("tool_calls", "trace")),
    }


def _normalize_runtime(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status", "")).lower()
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    return {
        "route": result.get("route"),
        "route_category": _route_category_runtime(result.get("route"), next_action=result.get("next_action")),
        "status": status,
        "next_action": result.get("next_action"),
        "evidence_present": bool(result.get("evidence")),
        "evidence_count": int(len(result.get("evidence", []) or [])),
        "route_confidence": meta.get("route_confidence"),
        "route_confidence_gap": meta.get("route_confidence_gap"),
        "validation": validation,
        "validation_ok": bool(validation.get("ok")),
        "failed_route_ids": list(validation.get("failed_route_ids", [])) if isinstance(validation.get("failed_route_ids"), list) else [],
        "trace_count": int(result.get("trace_checkpoint_count", 0)),
        "permission_blocked": _permission_blocked_runtime(result),
        "error_codes": _extract_error_codes(result, keys=("tool_calls", "stages")),
        "route_confidence": result.get("meta", {}).get("route_confidence") if isinstance(result.get("meta"), dict) else None,
    }


async def _run_real(engine: RealHarnessEngine, query: str) -> dict[str, Any]:
    return engine.run(query)


async def _run_runtime(runtime: HarnessRuntime, query: str, model: str) -> dict[str, Any]:
    return await runtime.process(
        {
            "messages": [{"role": "user", "content": query}],
            "model": model,
            "response_schema": None,
            "request_id": f"parity-{abs(hash(query))}",
            "toolset": ["calculator", "time_now", "run_tests", "new_uuid"],
        }
    )


def _check_case(
    query: str,
    real_actual: dict[str, Any],
    runtime_actual: dict[str, Any],
    real_expect: dict[str, Any],
    runtime_expect: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    real_norm = _normalize_real(real_actual)
    runtime_norm = _normalize_runtime(runtime_actual)

    checks: list[str] = []
    ok = True

    def _expect(expect: dict[str, Any], actual: dict[str, Any], field: str, engine: str) -> None:
        nonlocal ok
        if field not in expect:
            return
        if str(actual.get(field)) != str(expect[field]):
            checks.append(
                f"{engine}: expected {field}={expect[field]!r} but got {actual.get(field)!r}"
            )
            ok = False

    _expect(real_expect, real_norm, "status", "real")
    _expect(real_expect, real_norm, "route_category", "real")
    _expect(real_expect, real_norm, "next_action", "real")
    _expect(runtime_expect, runtime_norm, "status", "runtime")
    _expect(runtime_expect, runtime_norm, "route_category", "runtime")
    _expect(runtime_expect, runtime_norm, "next_action", "runtime")
    _expect(
        real_expect,
        real_norm,
        "validation_ok",
        "real",
    )
    _expect(
        runtime_expect,
        runtime_norm,
        "validation_ok",
        "runtime",
    )

    real_min = int(real_expect.get("trace_phase_count_min", 0))
    runtime_min = int(runtime_expect.get("trace_checkpoint_count_min", 0))
    if real_min and real_norm["trace_count"] < real_min:
        checks.append(
            f"real: expected trace_phase_count >= {real_min} but got {real_norm['trace_count']}"
        )
        ok = False
    if runtime_min and runtime_norm["trace_count"] < runtime_min:
        checks.append(
            f"runtime: expected trace_checkpoint_count >= {runtime_min} but got {runtime_norm['trace_count']}"
        )
        ok = False

    rb = bool(real_expect.get("permission_blocked", False)) == bool(real_norm.get("permission_blocked", False))
    rt = bool(runtime_expect.get("permission_blocked", False)) == bool(runtime_norm.get("permission_blocked", False))
    if not rb:
        checks.append(
            f"real: expected permission_blocked={real_expect.get('permission_blocked')} got {real_norm.get('permission_blocked')}"
        )
        ok = False
    if not rt:
        checks.append(
            f"runtime: expected permission_blocked={runtime_expect.get('permission_blocked')} got {runtime_norm.get('permission_blocked')}"
        )
        ok = False

    if real_norm["evidence_present"] != runtime_norm["evidence_present"]:
        checks.append(
            f"evidence presence mismatch: real={real_norm['evidence_present']} runtime={runtime_norm['evidence_present']}"
        )
        ok = False

    if real_norm["route_category"] != runtime_norm["route_category"]:
        checks.append(
            f"route category mismatch: real={real_norm['route_category']} runtime={runtime_norm['route_category']}"
        )
        ok = False

    if str(real_norm["next_action"]) != str(runtime_norm["next_action"]):
        checks.append(
            f"next_action mismatch: real={real_norm['next_action']!r} runtime={runtime_norm['next_action']!r}"
        )
        ok = False

    if real_norm["permission_blocked"] != runtime_norm["permission_blocked"]:
        checks.append(
            f"permission_blocked mismatch: real={real_norm['permission_blocked']} runtime={runtime_norm['permission_blocked']}"
        )
        ok = False

    if bool(real_norm["validation_ok"]) != bool(runtime_norm["validation_ok"]):
        checks.append(
            f"validation_ok mismatch: real={real_norm['validation_ok']} runtime={runtime_norm['validation_ok']}"
        )
        ok = False

    if real_norm.get("error_codes") != runtime_norm.get("error_codes"):
        checks.append(
            f"tool error code mismatch: real={real_norm['error_codes']!r} runtime={runtime_norm['error_codes']!r}"
        )
        ok = False

    if real_expect.get("expect_failed_route_ids"):
        expected = real_expect.get("expect_failed_route_ids")
        if isinstance(expected, list) and list(real_norm.get("failed_route_ids", [])) != expected:
            checks.append(
                f"real: expected failed_route_ids={expected!r} got {real_norm.get('failed_route_ids')!r}"
            )
            ok = False
    if runtime_expect.get("expect_failed_route_ids"):
        expected = runtime_expect.get("expect_failed_route_ids")
        if isinstance(expected, list) and list(runtime_norm.get("failed_route_ids", [])) != expected:
            checks.append(
                f"runtime: expected failed_route_ids={expected!r} got {runtime_norm.get('failed_route_ids')!r}"
            )
            ok = False

    re = bool(real_expect.get("expect_evidence_present"))
    if "expect_evidence_present" in real_expect and real_norm["evidence_present"] != re:
        checks.append(
            f"real: expected evidence_present={re} got={real_norm['evidence_present']}"
        )
        ok = False
    rte = bool(runtime_expect.get("expect_evidence_present"))
    if "expect_evidence_present" in runtime_expect and runtime_norm["evidence_present"] != rte:
        checks.append(
            f"runtime: expected evidence_present={rte} got={runtime_norm['evidence_present']}"
        )
        ok = False

    return {"query": query, "ok": ok, "checks": checks, "real": real_norm, "runtime": runtime_norm}, ok


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run real vs production harness parity checks")
    parser.add_argument("--real-fixture", default="tests/fixtures/baseline/real_ai/queries.json")
    parser.add_argument("--runtime-fixture", default="tests/fixtures/baseline/runtime/queries.json")
    parser.add_argument("--manifest", default="real_harness_routes.yaml", help="Real + production manifest path")
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    real_cases = _load_cases(Path(args.real_fixture))
    runtime_cases = _load_cases(Path(args.runtime_fixture))
    if not real_cases or not runtime_cases:
        raise ValueError("Fixture files must contain at least one case.")

    runtime_map = {case.query: case.expect for case in runtime_cases}
    real_map = {case.query: case.expect for case in real_cases}
    queries = [query for query in real_map.keys() if query in runtime_map][: args.max_cases]
    if not queries:
        raise ValueError("No overlapping queries found between the real/runtime fixture sets.")

    runtime_cfg = load_runtime_config()
    runtime_cfg.route_manifest_path = args.manifest
    runtime_cfg.feature_level = "basic"
    runtime_cfg.require_evidence = False
    runtime_cfg.tool_allowlist = (*tuple(runtime_cfg.tool_allowlist), "run_tests")
    runtime = HarnessRuntime(runtime_cfg, DeterministicMockModelClient())

    real = RealHarnessEngine(
        manifest_path=args.manifest,
        force_disable_llm=True,
        no_network=True,
        max_tool_calls_override=None,
    )

    failures: list[dict[str, Any]] = []
    for query in queries:
        real_actual = await _run_real(real, query)
        runtime_actual = await _run_runtime(runtime, query, str(runtime_cfg.model))
        result, ok = _check_case(query, real_actual, runtime_actual, real_map[query], runtime_map[query])
        if not ok:
            failures.append(result)

    summary = {
        "total": len(queries),
        "failed": len(failures),
        "passed": len(queries) - len(failures),
        "passes": [q for q in queries if q not in {item["query"] for item in failures}],
        "failures": failures,
    }

    if args.print_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"PARITY: {summary['passed']}/{summary['total']} passed.")
        for failure in failures:
            print(f"- {failure['query']}")
            for check in failure["checks"]:
                print(f"  - {check}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
