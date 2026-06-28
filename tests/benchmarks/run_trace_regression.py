#!/usr/bin/env python3
"""Simple benchmark harness for trace and phase-count regression."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

for candidate in Path(__file__).resolve().parents:
    if (candidate / "pyproject.toml").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

from harness.config import load_runtime_config
from harness.runtime import HarnessRuntime
from harness.types import ModelGenerateRequest, ModelGenerateResult
from harness.adapters.base import BaseModelClient
from real_ai_harness import RealHarnessEngine
from scripts.run_parity_sanity import _load_cases, _run_real, _run_runtime, _normalize_real, _normalize_runtime


class DeterministicMockModelClient(BaseModelClient):
    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        if req.response_schema and req.response_schema.get("required"):
            required = req.response_schema.get("required", [])
            payload = {name: f"{name}-ok" for name in required if isinstance(name, str)}
        elif req.tools:
            payload = {"tool_calls": [], "answer": "tool-plan-disabled"}
        else:
            payload = "ok"
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return ModelGenerateResult(
            text=str(text),
            reasoning=None,
            raw={"text": str(text)},
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Trace regression reporter for dual tracks")
    parser.add_argument("--real-fixture", default="tests/fixtures/baseline/real_ai/queries.json")
    parser.add_argument("--runtime-fixture", default="tests/fixtures/baseline/runtime/queries.json")
    parser.add_argument("--manifest", default="real_harness_routes.yaml")
    parser.add_argument("--out", default="tests/benchmarks/trace_regression.json")
    parser.add_argument("--route-category-threshold", type=float, default=0.90)
    parser.add_argument("--status-threshold", type=float, default=0.90)
    parser.add_argument("--next-action-threshold", type=float, default=0.90)
    parser.add_argument("--permission-threshold", type=float, default=0.95)
    parser.add_argument("--evidence-presence-threshold", type=float, default=0.95)
    parser.add_argument("--validation-threshold", type=float, default=0.95)
    args = parser.parse_args()

    real_cases = _load_cases(Path(args.real_fixture))
    runtime_cases = _load_cases(Path(args.runtime_fixture))
    common = [case for case in real_cases if any(r.query == case.query for r in runtime_cases)]
    if not common:
        raise ValueError("No overlapping queries found.")

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
    )

    rows: list[dict[str, object]] = []
    for case in common:
        real_result = _normalize_real(await _run_real(real, case.query))
        runtime_result = _normalize_runtime(
            await _run_runtime(runtime, case.query, str(runtime_cfg.model))
        )
        rows.append(
            {
                "query": case.query,
                "real": real_result,
                "runtime": runtime_result,
                "route_category_match": real_result["route_category"] == runtime_result["route_category"],
                "next_action_match": real_result["next_action"] == runtime_result["next_action"],
                "status_match": real_result["status"] == runtime_result["status"],
                "evidence_presence_match": bool(real_result["evidence_present"]) == bool(runtime_result["evidence_present"]),
                "validation_ok_match": bool(real_result["validation_ok"]) == bool(runtime_result["validation_ok"]),
                "permission_blocked_match": bool(real_result["permission_blocked"])
                == bool(runtime_result["permission_blocked"]),
            }
        )

    payload = {
        "total_cases": len(rows),
        "route_category_match_rate": (
            sum(1 for row in rows if row["route_category_match"]) / len(rows) if rows else 0.0
        ),
        "status_match_rate": (
            sum(1 for row in rows if row["status_match"]) / len(rows) if rows else 0.0
        ),
        "next_action_match_rate": (
            sum(1 for row in rows if row["next_action_match"]) / len(rows) if rows else 0.0
        ),
        "evidence_presence_match_rate": (
            sum(1 for row in rows if row["evidence_presence_match"]) / len(rows) if rows else 0.0
        ),
        "validation_ok_match_rate": (
            sum(1 for row in rows if row["validation_ok_match"]) / len(rows) if rows else 0.0
        ),
        "permission_blocked_match_rate": (
            sum(1 for row in rows if row["permission_blocked_match"]) / len(rows) if rows else 0.0
        ),
        "samples": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote benchmark summary to {out}")
    print(f"Route category match rate: {payload['route_category_match_rate']:.2f}")
    print(f"Status match rate: {payload['status_match_rate']:.2f}")
    print(f"Next action match rate: {payload['next_action_match_rate']:.2f}")
    print(f"Evidence presence match rate: {payload['evidence_presence_match_rate']:.2f}")
    print(f"Validation ok match rate: {payload['validation_ok_match_rate']:.2f}")
    print(f"Permission blocked match rate: {payload['permission_blocked_match_rate']:.2f}")

    if payload["route_category_match_rate"] < args.route_category_threshold:
        raise SystemExit(
            f"route_category_match_rate {payload['route_category_match_rate']:.2f} below threshold {args.route_category_threshold:.2f}"
        )
    if payload["status_match_rate"] < args.status_threshold:
        raise SystemExit(
            f"status_match_rate {payload['status_match_rate']:.2f} below threshold {args.status_threshold:.2f}"
        )
    if payload["next_action_match_rate"] < args.next_action_threshold:
        raise SystemExit(
            f"next_action_match_rate {payload['next_action_match_rate']:.2f} below threshold {args.next_action_threshold:.2f}"
        )
    if payload["evidence_presence_match_rate"] < args.evidence_presence_threshold:
        raise SystemExit(
            f"evidence_presence_match_rate {payload['evidence_presence_match_rate']:.2f} below threshold {args.evidence_presence_threshold:.2f}"
        )
    if payload["validation_ok_match_rate"] < args.validation_threshold:
        raise SystemExit(
            f"validation_ok_match_rate {payload['validation_ok_match_rate']:.2f} below threshold {args.validation_threshold:.2f}"
        )
    if payload["permission_blocked_match_rate"] < args.permission_threshold:
        raise SystemExit(
            f"permission_blocked_match_rate {payload['permission_blocked_match_rate']:.2f} below threshold {args.permission_threshold:.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
