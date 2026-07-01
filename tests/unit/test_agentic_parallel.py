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
    config.backends = (
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:11433/v1",
            backend_id="npu",
            model="qwen3:4b",
            device="npu",
            capabilities=("npu", "hybrid"),
            max_concurrency=1,
        ),
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:11434/v1",
            backend_id="gpu",
            model="qwen3:4b",
            device="gpu",
            capabilities=("gpu", "hybrid"),
            max_concurrency=2,
        ),
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:11435/v1",
            backend_id="cpu",
            model="qwen3:4b",
            device="cpu",
            capabilities=("cpu",),
            max_concurrency=4,
        ),
        BackendConfig(
            name="openai",
            base_url="http://127.0.0.1:13305/api/v1",
            backend_id="hybrid",
            model="qwen3:4b",
            device="hybrid",
            device_mode="hybrid_npu_igpu",
            capabilities=("hybrid", "npu", "igpu", "gpu", "cpu"),
            max_concurrency=1,
        ),
    )
    clients = {key: _FakeClient(key) for key in ("npu", "gpu", "cpu", "hybrid")}
    return HarnessRuntime(config, clients)


def test_execution_profile_planner_maps_task_shapes() -> None:
    runtime = _runtime()
    runtime.config.agentic_parallel_enabled = True

    short_profile, short_source = runtime._resolve_execution_profile(
        requested_profile=None,
        route=_policy(),
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        requested_max_tokens=128,
    )
    assert short_profile == "npu_only"
    assert short_source == "heuristic_default"

    tool_profile, _ = runtime._resolve_execution_profile(
        requested_profile=None,
        route=_policy(route=Route.TOOL_REQUIRED, use_tools=True, max_tool_calls_per_turn=1),
        messages=[{"role": "user", "content": "Calculate 2 + 2."}],
        requested_max_tokens=128,
    )
    assert tool_profile == "cpu"

    code_profile, _ = runtime._resolve_execution_profile(
        requested_profile=None,
        route=_policy(route=Route.CODE_OR_DATA),
        messages=[{"role": "user", "content": "Write a pandas snippet to aggregate columns."}],
        requested_max_tokens=512,
    )
    assert code_profile == "gpu"

    long_profile, _ = runtime._resolve_execution_profile(
        requested_profile=None,
        route=_policy(),
        messages=[{"role": "user", "content": "long context request"}],
        requested_max_tokens=1600,
    )
    assert long_profile == "hybrid_npu_igpu"

    complex_profile, complex_source = runtime._resolve_execution_profile(
        requested_profile=None,
        route=_policy(),
        messages=[
            {
                "role": "user",
                "content": "Design a detailed story with worldbuilding, character arc, a twist, and a revision plan.",
            }
        ],
        requested_max_tokens=700,
    )
    assert complex_profile == "agentic_parallel"
    assert complex_source == "heuristic_complex_prompt"


@pytest.mark.asyncio
async def test_agentic_parallel_uses_workers_and_cpu_controller(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    bindings = runtime._backend_bindings()
    fallback_controller = bindings[0]

    result, meta, controller = await runtime._run_agentic_parallel(
        prompt_messages=[
            {"role": "system", "content": "test harness"},
            {"role": "user", "content": "Design a detailed story with worldbuilding and a twist."},
        ],
        requested_temp=0.2,
        requested_max_tokens=256,
        final_schema=None,
        fallback_controller=fallback_controller,
    )

    assert result.text == "cpu worker response"
    assert controller.backend_id == "cpu"
    assert meta["enabled"] is True
    assert meta["controller_backend"] == "cpu"
    assert meta["accepted_results"] >= 2
    assert len(meta["workers"]) == 2
    assert meta["stages"][0]["stage"] == "agentic_parallel"
    assert result.usage["output_tokens"] > 0


@pytest.mark.asyncio
async def test_request_execution_profile_forces_target_backend(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    runtime.config.agentic_parallel_enabled = True
    request_template = {
        "route": "direct",
        "messages": [
            {
                "role": "user",
                "content": "Give a quick short answer so tokenization stays low while still exercising routing.",
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
        assert execution["profile_source"] == "request_override"
        assert execution["backend_id"] == expected_backend
        if requested_profile == "agentic_parallel":
            assert execution.get("agentic_parallel", {}).get("enabled") is True


def test_legacy_route_metadata_execution_profile_is_honored() -> None:
    runtime = _runtime()
    legacy_route = _policy(route_metadata={"execution_profile": "gpu", "prefer_device": "gpu"})

    profile, source = runtime._resolve_execution_profile(
        requested_profile=None,
        route=legacy_route,
        messages=[{"role": "user", "content": "hello"}],
        requested_max_tokens=128,
    )
    assert profile == "gpu"
    assert source == "route_metadata"

    plan = runtime._ordered_backend_plan(legacy_route, profile)
    assert len(plan) > 0
    assert plan[0].backend_id == "gpu"

    hybrid_plan = runtime._ordered_backend_plan(legacy_route, "hybrid_npu_igpu")
    assert hybrid_plan[0].backend_id == "hybrid"
