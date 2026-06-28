from __future__ import annotations

import json
import pytest
from harness.tools import ToolCallResult
from harness.config import load_runtime_config
from harness.runtime import HarnessRuntime
from harness.router import Route, RoutePolicy
from harness.types import ModelGenerateRequest, ModelGenerateResult
from harness import runtime as runtime_module


class _FakeModelClient:
    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:  # pragma: no cover - not executed in validation tests
        raise RuntimeError("mock client should not be used in validation tests")


def _make_runtime(require_evidence: bool = False) -> HarnessRuntime:
    config = load_runtime_config()
    config.feature_level = "basic"
    config.require_evidence = require_evidence
    return HarnessRuntime(config, _FakeModelClient())


def _policy() -> RoutePolicy:
    return RoutePolicy(
        route=Route.DIRECT,
        use_retrieval=False,
        use_tools=False,
        strict_schema=False,
        require_verification=False,
        allow_reasoning=False,
        thinking_budget="low",
        max_model_calls=1,
        temperature=0.2,
        max_new_tokens=256,
        required_evidence_fields=("answer", "summary"),
        hard_fail_errors=("tool_return_nonzero",),
    )


def test_validate_route_evidence_fail_closed_when_required_and_enabled() -> None:
    runtime = _make_runtime(require_evidence=True)
    policy = _policy()
    result = runtime._validate_route_evidence(policy, tool_results=[], evidence_rows=[], parsed=None)
    assert result["ok"] is False
    assert result["missing_fields"] == ["answer", "summary"]
    assert result["failed_route_ids"] == ["direct"]


def test_validate_route_evidence_allows_missing_when_require_evidence_disabled_and_no_hard_fail() -> None:
    runtime = _make_runtime(require_evidence=False)
    policy = _policy()
    result = runtime._validate_route_evidence(policy, tool_results=[], evidence_rows=[], parsed=None)
    assert result["ok"] is True
    assert result["missing_fields"] == []
    assert result["failed_route_ids"] == []


def test_validate_route_evidence_respects_hard_fail_tool_error() -> None:
    runtime = _make_runtime(require_evidence=False)
    policy = _policy()
    tool_results = [
        _FakeToolResult(
            name="run_tests",
            error_code="tool_return_nonzero",
            success=False,
            args={},
            output=None,
            error="tool_return_nonzero",
            sandbox="local",
        )
    ]
    result = runtime._validate_route_evidence(policy, tool_results=tool_results, evidence_rows=[], parsed=None)
    assert result["ok"] is False
    assert result["failed_route_ids"] == ["direct"]


def test_validate_route_evidence_treats_unknown_tool_as_hard_fail() -> None:
    runtime = _make_runtime(require_evidence=False)
    policy = _policy()
    tool_results = [
        _FakeToolResult(
            name="not_a_tool",
            error_code="unknown_tool",
            success=False,
            args={},
            output=None,
            error="unknown_tool",
            sandbox="local",
        )
    ]
    result = runtime._validate_route_evidence(policy, tool_results=tool_results, evidence_rows=[], parsed=None)
    assert result["ok"] is False
    assert result["failed_route_ids"] == ["direct"]


def test_run_tool_in_sandbox_reports_unavailable_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _make_runtime(require_evidence=False)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _name: None)
    result = runtime_module._run_tool_in_sandbox(
        name="run_tests",
        args={},
        image="python:3.12-slim",
        timeout_seconds=2,
    )
    assert result.success is False
    assert result.error_code == "tool_sandbox_unavailable"
    assert result.sandbox == "docker"


def test_run_tool_in_sandbox_reports_docker_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _make_runtime(require_evidence=False)
    monkeypatch.setattr(runtime_module.shutil, "which", lambda _name: "docker")

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(runtime_module.subprocess, "run", lambda *_args, **_kwargs: _Proc())
    result = runtime_module._run_tool_in_sandbox(
        name="run_tests",
        args={},
        image="python:3.12-slim",
        timeout_seconds=2,
    )
    assert result.sandbox == "docker"
    assert result.success is True
    assert result.error_code is None
    assert result.error is None


@pytest.mark.asyncio
async def test_budget_exhaustion_includes_budget_guard_record(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MockClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
            self.calls += 1
            if req.tools:
                payload = {
                    "tool_calls": [
                        {"name": "run_tests", "arguments": {}},
                        {"name": "run_tests", "arguments": {}},
                    ],
                    "answer": "toolplan",
                }
                text = json.dumps(payload, ensure_ascii=False)
            else:
                text = "ok"
            return ModelGenerateResult(
                text=text,
                reasoning=None,
                raw={"text": text},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

    config = load_runtime_config()
    config.feature_level = "basic"
    config.require_evidence = False
    config.tool_allowlist = ("run_tests",)

    runtime = HarnessRuntime(config, _MockClient())
    forced_route = RoutePolicy(
        route=Route.TOOL_REQUIRED,
        use_retrieval=False,
        use_tools=True,
        strict_schema=False,
        require_verification=False,
        allow_reasoning=False,
        thinking_budget="low",
        max_model_calls=1,
        temperature=0.2,
        max_new_tokens=64,
        allowed_tools=("run_tests",),
        max_tool_calls_per_turn=1,
        required_evidence_fields=(),
        hard_fail_errors=(),
        tool_sandbox_required=("run_tests",),
        route_metadata={},
    )

    monkeypatch.setattr(runtime_module, "classify_route", lambda *_args, **_kwargs: forced_route)
    monkeypatch.setattr(
        runtime_module,
        "execute_tool",
        lambda name, args: ToolCallResult(
            name=name,
            args={},
            output={"returncode": 0},
            success=True,
            error=None,
            error_code=None,
            sandbox="local",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:00+00:00",
        ),
    )

    response = await runtime.process(
        {
            "messages": [{"role": "user", "content": "run tests please"}],
            "model": str(runtime.config.model),
            "response_schema": None,
            "request_id": "runtime-budget",
            "toolset": ["run_tests"],
        }
    )

    assert response["status"] == "ok"
    tool_calls = response.get("tool_calls", [])
    assert isinstance(tool_calls, list)
    assert tool_calls
    assert any(
        isinstance(item, dict) and item.get("name") == "tool_budget_guard" and item.get("error_code") == "tool_calls_budget_exceeded"
        for item in tool_calls
    )


@pytest.mark.asyncio
async def test_unknown_tool_is_hard_fail_in_runtime_with_validation_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MockClient:
        async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
            if req.tools:
                payload = {
                    "tool_calls": [
                        {"name": "ghost_tool", "arguments": {}},
                    ],
                    "answer": "toolplan",
                }
                text = json.dumps(payload, ensure_ascii=False)
            else:
                text = "tooling attempted"
            return ModelGenerateResult(
                text=text,
                reasoning=None,
                raw={"text": text},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

    config = load_runtime_config()
    config.feature_level = "basic"
    config.require_evidence = False
    config.tool_allowlist = ("ghost_tool",)

    runtime = HarnessRuntime(config, _MockClient())
    forced_route = RoutePolicy(
        route=Route.TOOL_REQUIRED,
        use_retrieval=False,
        use_tools=True,
        strict_schema=False,
        require_verification=False,
        allow_reasoning=False,
        thinking_budget="low",
        max_model_calls=1,
        temperature=0.2,
        max_new_tokens=64,
        allowed_tools=("ghost_tool",),
        max_tool_calls_per_turn=2,
        required_evidence_fields=(),
        hard_fail_errors=(),
        tool_sandbox_required=(),
        route_metadata={},
    )

    monkeypatch.setattr(runtime_module, "classify_route", lambda *_args, **_kwargs: forced_route)

    response = await runtime.process(
        {
            "messages": [{"role": "user", "content": "run unknown tool"}],
            "model": str(runtime.config.model),
            "response_schema": None,
            "request_id": "runtime-unknown-tool",
            "toolset": ["ghost_tool"],
        }
    )

    assert response["status"] == "validation_block"
    assert response["next_action"] == "ask_clarification"
    assert response["validation"]["ok"] is False
    assert "unknown_tool" in response["validation"].get("error_codes", [])
    assert response["validation"].get("failed_route_ids") == ["tool_required"]
    assert any(
        isinstance(item, dict)
        and item.get("name") == "ghost_tool"
        and item.get("error_code") == "unknown_tool"
        for item in response.get("tool_calls", [])
    )


@pytest.mark.asyncio
async def test_require_evidence_mode_blocks_success_without_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    class _MockClient:
        async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
            if req.tools:
                payload = {"tool_calls": [], "answer": "toolplan"}
                text = json.dumps(payload, ensure_ascii=False)
            else:
                text = "final grounding response"
            return ModelGenerateResult(
                text=text,
                reasoning=None,
                raw={"text": text},
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

    config = load_runtime_config()
    config.feature_level = "basic"
    config.require_evidence = True
    config.tool_allowlist = ()

    runtime = HarnessRuntime(config, _MockClient())
    forced_route = RoutePolicy(
        route=Route.DIRECT,
        use_retrieval=False,
        use_tools=False,
        strict_schema=False,
        require_verification=False,
        allow_reasoning=False,
        thinking_budget="low",
        max_model_calls=1,
        temperature=0.2,
        max_new_tokens=64,
        allowed_tools=(),
        max_tool_calls_per_turn=0,
        required_evidence_fields=("answer", "summary"),
        hard_fail_errors=(),
        tool_sandbox_required=(),
        route_metadata={},
    )

    monkeypatch.setattr(runtime_module, "classify_route", lambda *_args, **_kwargs: forced_route)

    response = await runtime.process(
        {
            "messages": [{"role": "user", "content": "provide a grounded explanation"}],
            "model": str(runtime.config.model),
            "response_schema": None,
            "request_id": "runtime-evidence-required",
            "toolset": [],
        }
    )

    assert response["status"] == "validation_block"
    assert response["next_action"] == "ask_clarification"
    assert response["validation"]["ok"] is False
    assert response["validation"].get("missing_fields", []) == ["summary"]


class _FakeToolResult:
    def __init__(self, name: str, error_code: str | None, success: bool, args: dict, output: object, error: str, sandbox: str | None) -> None:
        self.name = name
        self.error_code = error_code
        self.success = success
        self.args = args
        self.output = output
        self.error = error
        self.sandbox = sandbox
        self.started_at = "2026-01-01T00:00:00+00:00"
        self.finished_at = "2026-01-01T00:00:00+00:00"
