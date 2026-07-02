"""Regression tests for the reproducibility / parameter-plumbing fixes.

Covers: seed+sampling propagation through every model call, the response
cache actually hitting, per-request state isolation under concurrency,
schema repair on non-tool routes, per-trace timestamps, high-risk gating by
feature level, policy precedence over thinking-model detection, and the
provider's device/template/stop-token helpers.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from harness.config import BackendConfig, load_runtime_config
from harness.local_model_provider import (
    _derive_stop_tokens,
    _render_chat_prompt,
    _resolve_torch_device,
)
from harness.router import Route, classify_route
from harness.runtime import HarnessRuntime
from harness.trace_cache import TraceEvent
from harness.types import ModelGenerateRequest, ModelGenerateResult


class _CaptureClient:
    def __init__(self, name: str = "cap", responses: list[str] | None = None, delay: float = 0.0) -> None:
        self.name = name
        self.responses = list(responses or [])
        self.delay = delay
        self.calls: list[ModelGenerateRequest] = []

    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        self.calls.append(req)
        if self.delay:
            await asyncio.sleep(self.delay)
        text = self.responses.pop(0) if self.responses else f"{self.name} response"
        return ModelGenerateResult(
            text=text,
            reasoning=None,
            raw={"provider": {"generation_backend": self.name}},
            usage={"input_tokens": 2, "output_tokens": 4},
        )


def _runtime(tmp_path, client: _CaptureClient, *, enable_cache: bool = False) -> HarnessRuntime:
    config = load_runtime_config()
    config.enable_cache = enable_cache
    config.max_cache_entries = 50
    config.tool_allowlist = ()
    config.trace_dir = tmp_path / "traces"
    config.cache_dir = tmp_path / "cache"
    config.state_dir = tmp_path / "state"
    config.backends = (
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:11435/v1",
            backend_id="cpu",
            model="qwen3:4b",
            device="cpu",
            capabilities=("cpu",),
        ),
    )
    return HarnessRuntime(config, {"cpu": client})


SAMPLING = {"seed": 1337, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "stop": ["<|im_end|>"]}


@pytest.mark.asyncio
async def test_sampling_params_reach_final_generation(tmp_path) -> None:
    client = _CaptureClient()
    runtime = _runtime(tmp_path, client)
    response = await runtime.process(
        {
            "route": "direct",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 128,
            "temperature": 0.2,
            **SAMPLING,
        }
    )
    assert response["status"] == "ok"
    final_req = client.calls[-1]
    assert final_req.seed == 1337
    assert final_req.top_p == 0.9
    assert final_req.top_k == 40
    assert final_req.repeat_penalty == 1.1
    assert final_req.stop == ["<|im_end|>"]
    assert final_req.temperature == 0.2


@pytest.mark.asyncio
async def test_schema_repair_runs_without_tools_and_propagates_seed(tmp_path) -> None:
    # First response is malformed for the schema; repair returns valid JSON.
    client = _CaptureClient(responses=["not json at all", json.dumps({"answer": "42"})])
    runtime = _runtime(tmp_path, client)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    response = await runtime.process(
        {
            "messages": [{"role": "user", "content": "Extract the answer."}],
            "response_schema": schema,
            "max_tokens": 128,
            **SAMPLING,
        }
    )
    # structured_extraction has use_tools=False; before the fix the repair
    # retry was gated on tool_plan_answer and never ran here.
    assert len(client.calls) == 2, "repair generation did not run"
    assert response["parse"]["schema_valid"] is True
    assert client.calls[1].seed == 1337


@pytest.mark.asyncio
async def test_response_cache_hits_on_identical_request(tmp_path) -> None:
    client = _CaptureClient()
    runtime = _runtime(tmp_path, client, enable_cache=True)
    request = {
        "route": "direct",
        "messages": [{"role": "user", "content": "Cache me."}],
        "max_tokens": 128,
        "temperature": 0.0,
        "seed": 7,
    }
    first = await runtime.process(dict(request))
    second = await runtime.process(dict(request))
    assert first.get("cached") is not True
    assert second.get("cached") is True, "cache key still contains per-request volatiles"
    # Different sampling must miss the cache.
    third = await runtime.process({**request, "seed": 8})
    assert third.get("cached") is not True


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_checkpoint_state(tmp_path) -> None:
    client = _CaptureClient(delay=0.05)
    runtime = _runtime(tmp_path, client)
    request = {
        "route": "direct",
        "messages": [{"role": "user", "content": "Concurrency check."}],
        "max_tokens": 128,
        "temperature": 0.1,
    }
    first, second = await asyncio.gather(
        runtime.process(dict(request)), runtime.process(dict(request))
    )
    assert first["run_id"] != second["run_id"]
    # Same-shaped requests must produce identical checkpoint counts; under the
    # shared-instance-state bug the two lists interleaved and one response
    # reported roughly double the checkpoints.
    assert first["trace_checkpoint_count"] == second["trace_checkpoint_count"]
    assert first["run_id"] in str(first["checkpoint_id"])
    assert second["run_id"] in str(second["checkpoint_id"])


def test_trace_events_have_distinct_timestamps() -> None:
    def _event() -> TraceEvent:
        return TraceEvent(
            request_id="r",
            route="direct",
            model="m",
            policy_version="p",
            prompt_version="p",
            status="ok",
            latency_ms=1,
            stages=[],
            request={},
            result_summary={},
        )

    first = _event()
    time.sleep(0.002)
    second = _event()
    assert first.created_at != second.created_at, "created_at is a class-level constant"


def test_high_risk_blocks_only_in_hardening_mode() -> None:
    messages = [{"role": "user", "content": "Summarize this financial report."}]
    basic = classify_route(messages, feature_level="basic")
    hardened = classify_route(messages, feature_level="hardening")
    assert basic.route == Route.HIGH_RISK
    assert basic.next_action == "proceed"
    assert basic.require_confirmation is True
    assert hardened.route == Route.HIGH_RISK
    assert hardened.next_action == "ask_clarification"


@pytest.mark.asyncio
async def test_thinking_model_does_not_override_strict_schema_policy(tmp_path) -> None:
    client = _CaptureClient(responses=[json.dumps({"answer": "ok"})])
    runtime = _runtime(tmp_path, client)
    runtime.config.thinking_model_prefixes = ("qwen3",)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    await runtime.process(
        {
            "model": "qwen3:4b",
            "messages": [{"role": "user", "content": "Extract."}],
            "response_schema": schema,
            "max_tokens": 128,
        }
    )
    final_req = client.calls[-1]
    assert final_req.allow_reasoning is False, "model-name detection overrode strict-schema policy"


@pytest.mark.asyncio
async def test_execution_telemetry_reports_generation_time_not_pipeline(tmp_path) -> None:
    client = _CaptureClient()
    runtime = _runtime(tmp_path, client)
    response = await runtime.process(
        {
            "route": "direct",
            "messages": [{"role": "user", "content": "Telemetry check."}],
            "max_tokens": 128,
            "temperature": 0.1,
        }
    )
    execution = response["execution"]
    assert execution["ttft_ms"] is None, "fake TTFT reintroduced"
    assert isinstance(execution["generation_ms"], int)
    assert execution["generation_ms"] <= execution["runtime_ms"]


def test_resolve_torch_device_is_honest() -> None:
    assert _resolve_torch_device("cpu") == (-1, None)
    device, warning = _resolve_torch_device("npu")
    assert device == -1
    assert "npu_unsupported" in warning
    device, warning = _resolve_torch_device("quantum")
    assert device == -1
    assert "unknown_device" in warning
    # gpu resolves to cuda only when torch+cuda exist; otherwise warns.
    device, warning = _resolve_torch_device("gpu")
    assert device in (0, -1)
    if device == -1:
        assert warning in ("cuda_unavailable_running_cpu", "torch_unavailable_running_cpu")


def test_derive_stop_tokens_precedence_and_chatml() -> None:
    assert _derive_stop_tokens("chatml", None, None) == ["<|im_end|>"]
    assert _derive_stop_tokens(None, None, None) == ["</s>"]
    assert _derive_stop_tokens("chatml", ["REQ"], ["PARAM"]) == ["REQ"]
    assert _derive_stop_tokens("chatml", None, ["PARAM"]) == ["PARAM"]
    assert "\n\n\n" not in _derive_stop_tokens(None, None, None)


def test_render_chat_prompt_uses_template_when_available() -> None:
    class _FakeTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"

    messages = [{"role": "user", "content": "hello"}]
    prompt, applied = _render_chat_prompt(_FakeTokenizer(), messages, "fallback")
    assert applied is True
    assert "<|im_start|>" in prompt

    prompt, applied = _render_chat_prompt(None, messages, "fallback")
    assert applied is False
    assert prompt == "fallback"
