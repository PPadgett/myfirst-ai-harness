"""Regression tests for Phase C fixes: fallback-chain failover with real
attempts telemetry, deterministic (timestamp-free) tool prompts, and the
slimmed model-visible system prompt."""
from __future__ import annotations

import json

import pytest

from harness.config import BackendConfig, load_runtime_config
from harness.runtime import HarnessRuntime, _tool_result_prompt_payload
from harness.tools import ToolCallResult
from harness.types import ModelGenerateRequest, ModelGenerateResult


class _ScriptedClient:
    """Client that fails a set number of times, then succeeds."""

    def __init__(self, name: str, fail_first: int = 0, text: str | None = None) -> None:
        self.name = name
        self.fail_first = fail_first
        self.text = text if text is not None else f"{name} response"
        self.calls: list[ModelGenerateRequest] = []

    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        self.calls.append(req)
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("model_backend_unavailable: scripted failure")
        return ModelGenerateResult(
            text=self.text,
            reasoning=None,
            raw={"provider": {"generation_backend": self.name}},
            usage={"input_tokens": 2, "output_tokens": 4},
        )


def _runtime(tmp_path, clients: dict[str, object]) -> HarnessRuntime:
    config = load_runtime_config()
    config.enable_cache = False
    config.tool_allowlist = ()
    config.trace_dir = tmp_path / "traces"
    config.cache_dir = tmp_path / "cache"
    config.state_dir = tmp_path / "state"
    config.backends = tuple(
        BackendConfig(
            name="openai",
            base_url=f"http://127.0.0.1:1000{i}/v1",
            backend_id=key,
            model="qwen3:4b",
            device=key,
            capabilities=(key,),
        )
        for i, key in enumerate(clients)
    )
    return HarnessRuntime(config, clients)


@pytest.mark.asyncio
async def test_failover_advances_to_next_backend_and_records_attempts(tmp_path) -> None:
    npu = _ScriptedClient("npu", fail_first=100)  # always fails
    gpu = _ScriptedClient("gpu")
    runtime = _runtime(tmp_path, {"npu": npu, "gpu": gpu})

    response = await runtime.process(
        {
            "route": "direct",
            "execution_profile": "npu_only",
            "messages": [{"role": "user", "content": "Failover check."}],
            "max_tokens": 128,
            "temperature": 0.1,
        }
    )
    assert response["status"] == "ok"
    execution = response["execution"]
    assert execution["backend_id"] == "gpu"
    assert execution["fallback_attempted"] is True
    assert execution["fallback_reason"] == "model_backend_unavailable"
    assert [a["status"] for a in execution["attempts"]] == ["failed", "ok"]
    assert execution["attempts"][0]["backend_id"] == "npu"
    assert len(npu.calls) == 1 and len(gpu.calls) == 1


@pytest.mark.asyncio
async def test_failover_exhaustion_returns_backend_failure(tmp_path) -> None:
    npu = _ScriptedClient("npu", fail_first=100)
    gpu = _ScriptedClient("gpu", fail_first=100)
    runtime = _runtime(tmp_path, {"npu": npu, "gpu": gpu})

    response = await runtime.process(
        {
            "route": "direct",
            "messages": [{"role": "user", "content": "All down."}],
            "max_tokens": 128,
        }
    )
    assert response["status"] == "validation_block"
    assert response["error_code"] == "model_backend_unavailable"
    assert len(npu.calls) == 1 and len(gpu.calls) == 1


@pytest.mark.asyncio
async def test_no_fallback_fields_on_clean_run(tmp_path) -> None:
    cpu = _ScriptedClient("cpu")
    runtime = _runtime(tmp_path, {"cpu": cpu})
    response = await runtime.process(
        {
            "route": "direct",
            "messages": [{"role": "user", "content": "Clean run."}],
            "max_tokens": 128,
        }
    )
    execution = response["execution"]
    assert execution["fallback_attempted"] is False
    assert execution["fallback_reason"] == ""
    assert [a["status"] for a in execution["attempts"]] == ["ok"]


@pytest.mark.asyncio
async def test_tool_prompt_excludes_volatile_fields(tmp_path) -> None:
    plan = json.dumps({"tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}], "answer": ""})
    client = _ScriptedClient("cpu")
    client_texts = [plan, "The answer is 4."]

    class _SeqClient(_ScriptedClient):
        async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
            self.calls.append(req)
            return ModelGenerateResult(
                text=client_texts[min(len(self.calls) - 1, 1)],
                reasoning=None,
                raw={},
                usage={"input_tokens": 2, "output_tokens": 4},
            )

    seq = _SeqClient("cpu")
    runtime = _runtime(tmp_path, {"cpu": seq})
    runtime.config.tool_allowlist = ("calculator",)

    response = await runtime.process(
        {
            "route": "tool_required",
            "messages": [{"role": "user", "content": "calculate 2+2"}],
            "max_tokens": 128,
        }
    )
    assert response["status"] == "ok"
    # The model-visible tool record must not contain volatile timestamps...
    final_messages = seq.calls[-1].messages
    tool_msg = next(m["content"] for m in final_messages if "Tool execution results" in str(m.get("content", "")))
    assert "started_at" not in tool_msg and "finished_at" not in tool_msg
    # ...while the response/trace-facing records keep them.
    assert response["tool_calls"][0]["started_at"]


def test_tool_result_prompt_payload_shape() -> None:
    result = ToolCallResult(
        name="calculator",
        args={"expression": "2+2"},
        output={"value": 4},
        success=True,
        error=None,
        error_code=None,
        started_at="2026-07-02T00:00:00+00:00",
        finished_at="2026-07-02T00:00:01+00:00",
    )
    payload = _tool_result_prompt_payload(result)
    assert set(payload) == {"name", "success", "arguments", "output", "error", "error_code"}


@pytest.mark.asyncio
async def test_system_prompt_contains_no_internals(tmp_path) -> None:
    cpu = _ScriptedClient("cpu")
    runtime = _runtime(tmp_path, {"cpu": cpu})
    await runtime.process(
        {
            "route": "direct",
            "messages": [{"role": "user", "content": "Hello."}],
            "max_tokens": 128,
        }
    )
    system_text = str(cpu.calls[-1].messages[0]["content"])
    for leak in ("Route confidence", "Route metadata", "policy-controlled", "Execution route"):
        assert leak not in system_text
