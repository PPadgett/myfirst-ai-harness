from __future__ import annotations

import pytest

from harness.config import BackendConfig, load_runtime_config
from harness.runtime import HarnessRuntime
from harness.router import Route, RoutePolicy
from harness.types import ModelGenerateRequest, ModelGenerateResult


class _FakeClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[ModelGenerateRequest] = []

    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        self.calls.append(req)
        return ModelGenerateResult(
            text=f"{self.name} worker response",
            reasoning=None,
            raw={"provider": {"generation_backend": self.name}},
            usage={"input_tokens": 2, "output_tokens": 4},
        )


def _policy(**kwargs: object) -> RoutePolicy:
    defaults = dict(
        route=Route.DIRECT,
        use_retrieval=False,
        use_tools=False,
        strict_schema=False,
        require_verification=False,
        allow_reasoning=False,
        thinking_budget="low",
        max_model_calls=1,
        temperature=0.2,
        max_new_tokens=512,
        max_tool_calls_per_turn=0,
        allowed_tools=(),
        route_metadata={},
    )
    defaults.update(kwargs)
    return RoutePolicy(**defaults)


def _runtime(tmp_path=None) -> HarnessRuntime:
    config = load_runtime_config()
    config.enable_cache = False
    config.tool_allowlist = ()
    if tmp_path is not None:
        config.trace_dir = tmp_path / "traces"
        config.cache_dir = tmp_path / "cache"
        config.state_dir = tmp_path / "state"
    config.backends = [
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:11433/v1",
            backend_id="npu",
            model="qwen3:4b",
            device="npu",
            capabilities=["npu", "hybrid"],
            max_concurrency=1,
        ),
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:11434/v1",
            backend_id="gpu",
            model="qwen3:4b",
            device="gpu",
            capabilities=["gpu", "hybrid"],
            max_concurrency=2,
        ),
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:11435/v1",
            backend_id="cpu",
            model="qwen3:4b",
            device="cpu",
            capabilities=["cpu"],
            max_concurrency=4,
        ),
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:13305/api/v1",
            backend_id="hybrid",
            model="qwen3:4b",
            device="hybrid",
            device_mode="hybrid_npu_igpu",
            capabilities=["hybrid", "npu", "igpu", "gpu", "cpu"],
            max_concurrency=1,
        ),
    ]
    clients = {key: _FakeClient(key) for key in ("npu", "gpu", "cpu", "hybrid")}
    return HarnessRuntime(config, clients)


def test_execution_profile_planner_maps_task_shapes() -> None:
    runtime = _runtime()

    short_profile, _, _ = runtime._resolve_execution_profile(
        None,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        route=_policy(),
        requested_max_tokens=128,
        requested_context_tokens=24,
    )
    assert short_profile == "npu_only"

    tool_profile, _, _ = runtime._resolve_execution_profile(
        None,
        messages=[{"role": "user", "content": "Calculate 2 + 2."}],
        route=_policy(route=Route.TOOL_REQUIRED, use_tools=True, max_tool_calls_per_turn=1),
        requested_max_tokens=128,
        requested_context_tokens=24,
    )
    assert tool_profile == "cpu"

    long_profile, _, _ = runtime._resolve_execution_profile(
        None,
        messages=[{"role": "user", "content": "long context"}],
        route=_policy(),
        requested_max_tokens=1600,
        requested_context_tokens=2800,
    )
    assert long_profile == "gpu"

    complex_profile, _, score = runtime._resolve_execution_profile(
        None,
        messages=[{"role": "user", "content": "Design a detailed story with worldbuilding, character arc, a twist, and a revision plan."}],
        route=_policy(),
        requested_max_tokens=700,
        requested_context_tokens=120,
    )
    assert score >= 4
    assert complex_profile == "agentic_parallel"


@pytest.mark.asyncio
async def test_agentic_parallel_uses_workers_and_cpu_controller(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    route = _policy()
    runtime._apply_execution_profile_to_route(route, "agentic_parallel", "request_override", 5)
    prompt_messages = [{"role": "system", "content": "test harness"}]

    result, provider, binding, _plan, stages, usage = await runtime._run_agentic_parallel(
        route=route,
        model="qwen3:4b",
        prompt_messages=prompt_messages,
        original_messages=[{"role": "user", "content": "Design a detailed story with worldbuilding and a twist."}],
        requested_temp=0.2,
        requested_max_tokens=256,
        requested_context_tokens=100,
        final_schema=None,
    )

    assert result.text == "cpu worker response"
    assert binding.backend_id == "cpu"
    assert provider["execution"]["profile"] == "agentic_parallel"
    assert provider["execution"]["agentic_parallel"]["enabled"] is True
    assert provider["execution"]["agentic_parallel"]["controller_backend"] == "cpu"
    assert provider["execution"]["agentic_parallel"]["accepted_results"] >= 2
    assert len(provider["execution"]["agentic_parallel"]["workers"]) == 2
    assert stages[0]["stage"] == "agentic_parallel"
    assert usage["output_tokens"] >= 8


@pytest.mark.asyncio
async def test_request_execution_profile_forces_target_backend(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.config.agentic_parallel_enabled = True
    request_template = {
        "route": "direct",
        "messages": [
            {
                "role": "user",
                "content": "Design a quick short prompt so tokenization stays low while still exercising routing.",
            }
        ],
        "max_tokens": 128,
        "temperature": 0.2,
    }

    test_matrix = [
        ("cpu", "cpu"),
        ("gpu", "gpu"),
        ("npu_only", "npu"),
        ("hybrid_npu_igpu", "hybrid"),
        ("agentic_parallel", "cpu"),
    ]
    for requested_profile, expected_backend in test_matrix:
        request_payload = dict(request_template)
        request_payload["execution_profile"] = requested_profile
        response = await runtime.process(request_payload)
        execution = response["execution"]

        assert execution["profile"] == requested_profile
        assert execution["backend_id"] == expected_backend
        assert execution["profile_source"] == "request_override"
        if requested_profile != "agentic_parallel":
            assert execution["backend_id"] == expected_backend
        else:
            assert execution.get("agentic_parallel", {}).get("enabled") is True


def test_legacy_route_metadata_execution_profile_is_honored() -> None:
    runtime = _runtime()
    legacy_route = _policy(
        route=Route.DIRECT,
        route_metadata={"execution_profile": "gpu", "prefer_device": "gpu"},
        temperature=0.2,
    )

    candidates, _, _, _ = runtime._build_backend_candidates(
        legacy_route,
        requested_max_tokens=128,
        requested_context_tokens=24,
    )
    assert len(candidates) > 0
    assert candidates[0].backend_id == "gpu"

    runtime._apply_execution_profile_to_route(
        legacy_route,
        profile="hybrid_npu_igpu",
        reason="route_hint",
        complexity_score=1,
    )
    candidates, _, _, _ = runtime._build_backend_candidates(
        legacy_route,
        requested_max_tokens=128,
        requested_context_tokens=24,
    )
    assert candidates[0].backend_id == "hybrid"
