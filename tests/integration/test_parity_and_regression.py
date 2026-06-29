from __future__ import annotations

import pytest
import json
from pathlib import Path

from harness.config import load_runtime_config
from harness.runtime import HarnessRuntime
from harness.types import ModelGenerateRequest, ModelGenerateResult
from real_ai_harness import RealHarnessEngine
from scripts.run_parity_sanity import _normalize_real, _normalize_runtime
from scripts.run_parity_sanity import _load_cases


class _DeterministicMockModelClient:
    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        if req.response_schema and req.response_schema.get("required"):
            required = req.response_schema.get("required", [])
            payload = {name: f"{name}-ok" for name in required if isinstance(name, str)}
        elif req.tools:
            payload = {"tool_calls": [], "answer": "tool-plan-disabled"}
        else:
            payload = "ok"
        return ModelGenerateResult(
            text=str(payload if isinstance(payload, str) else json.dumps(payload)),
            reasoning=None,
            raw={"text": str(payload)},
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


def _build_runtime() -> tuple[HarnessRuntime, RealHarnessEngine]:
    runtime_cfg = load_runtime_config()
    runtime_cfg.route_manifest_path = "real_harness_routes.yaml"
    runtime_cfg.feature_level = "basic"
    runtime_cfg.require_evidence = False
    runtime_cfg.tool_allowlist = (*tuple(runtime_cfg.tool_allowlist), "run_tests")
    runtime_cfg.advanced_router_enabled = True
    runtime = HarnessRuntime(runtime_cfg, _DeterministicMockModelClient())
    real = RealHarnessEngine(
        manifest_path="real_harness_routes.yaml",
        force_disable_llm=True,
        no_network=True,
        max_tool_calls_override=None,
    )
    return runtime, real


@pytest.mark.asyncio
async def test_parity_baseline_matches_runtime_contract_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RealHarnessEngine,
        "_tool_run_tests",
        lambda self, args: {"tool": "run_tests", "success": True, "output": {"returncode": 0}},
    )
    real_cases = _load_cases(Path("tests/fixtures/baseline/real_ai/queries.json"))
    runtime_cases = _load_cases(Path("tests/fixtures/baseline/runtime/queries.json"))
    common = [case.query for case in real_cases if any(r.query == case.query for r in runtime_cases)]
    assert common

    runtime, real = _build_runtime()
    runtime_cfg = runtime.config

    for query in common[:3]:
        real_result = real.run(query)
        runtime_result = await runtime.process(
            {
                "messages": [{"role": "user", "content": query}],
                "model": str(runtime_cfg.model),
                "response_schema": None,
                "request_id": f"integration-{abs(hash(query))}",
                "toolset": ["calculator", "time_now", "run_tests", "new_uuid"],
            }
        )
        normalized_real = _normalize_real(real_result)
        normalized_runtime = _normalize_runtime(runtime_result)

        assert normalized_real["route_category"] == normalized_runtime["route_category"]
        assert normalized_real["next_action"] == normalized_runtime["next_action"]
        assert normalized_real["validation_ok"] == normalized_runtime["validation_ok"]
        assert normalized_real["permission_blocked"] == normalized_runtime["permission_blocked"]
        assert bool(normalized_real["evidence_present"]) == bool(normalized_runtime["evidence_present"])


@pytest.mark.asyncio
async def test_parity_low_confidence_dual_track_behaves_as_clarification() -> None:
    runtime, real = _build_runtime()
    runtime.config.feature_level = "hardening"

    query = "qwer qwerty"
    real_result = real.run(query)
    runtime_result = await runtime.process(
        {
            "messages": [{"role": "user", "content": query}],
            "model": str(runtime.config.model),
            "response_schema": None,
            "request_id": "integration-low-confidence",
            "toolset": ["calculator", "time_now", "run_tests", "new_uuid"],
        }
    )

    normalized_real = _normalize_real(real_result)
    normalized_runtime = _normalize_runtime(runtime_result)

    assert normalized_real["next_action"] == "ask_clarification"
    assert normalized_runtime["next_action"] == "ask_clarification"
    assert normalized_real["route_category"] == "clarification"
    assert normalized_runtime["route_category"] == "clarification"
    assert normalized_real["permission_blocked"] is False
    assert normalized_runtime["permission_blocked"] is False
