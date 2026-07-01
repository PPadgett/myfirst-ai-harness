"""Production runtime orchestration for the local harness."""

from __future__ import annotations

import json
import shutil
import asyncio
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.adapters.base import BaseModelClient
from harness.config import BackendConfig, RuntimeConfig
from harness.manifest_schema import load_route_manifest
from harness.evidence import Claim
from harness.execution_state import write_checkpoint
from harness.guards import (
    GuardDecision,
    check_input_text,
    check_output_text,
    check_tool_output,
    check_tool_request_with_tool,
    sanitize_text,
    split_trusted_untrusted,
    redact_sensitive_args,
)
from harness.router import RoutePolicy, Route, classify_route
from harness.tools import ToolCallResult, execute_tool, specs_as_openai_tools
from harness.trace_cache import ResponseCache, TraceEvent, TraceStore
from harness.types import ModelGenerateRequest, ModelGenerateResult
from harness.validation import extract_first_json, validate_schema
from harness.retrieval import DirectoryCorpusRetriever, RetrievedDoc, SimpleReranker, pack_context

MODEL_BACKEND_UNAVAILABLE_CODE = "model_backend_unavailable"
MODEL_REQUEST_FAILED_CODE = "model_request_failed"
PROVIDER_DIAGNOSTIC_KEYS = (
    "model",
    "configured_backend",
    "generation_backend",
    "model_source",
    "model_source_type",
    "model_artifact_format",
    "provider_store",
    "manifest_path",
    "runtime_dependency",
    "runtime_dependency_available",
    "local_model_loaded",
    "model_source_present",
    "model_load_attempted",
    "model_load_succeeded",
    "last_load_error",
    "last_generation_error",
    "template_applied",
    "fallback_active",
    "allow_fallback",
    "finish_reason",
    "truncated",
    "reasoning_extracted",
    "warnings",
    "provider_warning",
)

RESPONSE_NORMALIZATION_VERSION = "2026-06-29-qwen3-clean-answer-v1"


@dataclass
class ModelArtifact:
    trace_id: str
    request_id: str
    route: RoutePolicy
    retrieved_doc_ids: list[str]
    tool_calls: list[dict[str, Any]]
    prompt_payload: list[dict[str, str]]
    model_result: str
    model_reasoning: str | None
    usage: dict[str, int]
    latency_ms: int


@dataclass
class BackendBinding:
    backend_id: str
    device: str
    device_mode: str
    model: str
    client: BaseModelClient
    config: BackendConfig | None = None


class HarnessRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        model_client: BaseModelClient | dict[str, BaseModelClient],
    ) -> None:
        self.config = config
        if isinstance(model_client, dict):
            self.model_clients = dict(model_client)
            first_client = next(iter(self.model_clients.values()), None)
            if first_client is None:
                raise ValueError("At least one model client is required")
            self.model_client = first_client
        else:
            self.model_clients = {}
            self.model_client = model_client
        self.retriever = DirectoryCorpusRetriever(config.corpus_dir)
        self.reranker = SimpleReranker()
        self.trace_store = TraceStore(config.trace_dir)
        self.cache = ResponseCache(config.cache_dir, config.max_cache_entries) if config.enable_cache else None
        self.state_dir = Path(config.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.feature_level = (config.feature_level or "basic").lower()
        self.route_manifests = load_route_manifest(
            str(config.route_manifest_path),
            strict=self.feature_level == "hardening",
        )
        self.require_evidence = bool(config.require_evidence)
        self.route_overrides = config.route_overrides
        self.tool_sandbox_mode = os.getenv("HARNESS_TOOL_SANDBOX", "off").lower()
        self.sandbox_image = os.getenv("HARNESS_TOOL_SANDBOX_IMAGE", "python:3.12-slim")
        self.sandbox_timeout = int(os.getenv("HARNESS_TOOL_SANDBOX_TIMEOUT", "120"))
        self.checkpoint_refs: list[str] = []
        self.last_run_id: str | None = None
        self.trace_id: str | None = None
        self._last_checkpoint_state: dict[str, Any] = {}

    def _backend_bindings(self) -> list[BackendBinding]:
        if self.config.backends:
            bindings: list[BackendBinding] = []
            for backend in self.config.backends:
                backend_id = backend.backend_id or backend.device or backend.name
                client = (
                    self.model_clients.get(backend_id)
                    or self.model_clients.get(backend.device)
                    or self.model_clients.get(backend.name)
                )
                if client is None:
                    continue
                bindings.append(
                    BackendBinding(
                        backend_id=backend_id,
                        device=backend.device or backend_id,
                        device_mode=backend.device_mode or backend.device or backend_id,
                        model=backend.model or self.config.model,
                        client=client,
                        config=backend,
                    )
                )
            if bindings:
                return bindings
        backend = self.config.backend
        return [
            BackendBinding(
                backend_id=backend.backend_id or backend.device or backend.name,
                device=backend.device or backend.backend_id or backend.name,
                device_mode=backend.device_mode or backend.device or backend.backend_id or backend.name,
                model=backend.model or self.config.model,
                client=self.model_client,
                config=backend,
            )
        ]

    @staticmethod
    def _normalize_execution_profile(raw: Any) -> str:
        text = str(raw or "").strip().lower().replace("-", "_")
        aliases = {
            "": "",
            "npu": "npu_only",
            "npu_only": "npu_only",
            "npuonly": "npu_only",
            "gpu": "gpu",
            "gpu_only": "gpu",
            "cpu": "cpu",
            "cpu_only": "cpu",
            "hybrid": "hybrid_npu_igpu",
            "npu_igpu": "hybrid_npu_igpu",
            "hybrid_npu_igpu": "hybrid_npu_igpu",
            "agentic": "agentic_parallel",
            "agentic_parallel": "agentic_parallel",
        }
        return aliases.get(text, text)

    @staticmethod
    def _device_to_profile(device: str) -> str:
        normalized = str(device or "").strip().lower().replace("-", "_")
        if normalized == "npu":
            return "npu_only"
        if normalized == "hybrid":
            return "hybrid_npu_igpu"
        if normalized in {"cpu", "gpu"}:
            return normalized
        return normalized

    @staticmethod
    def _profile_to_device(profile: str) -> str:
        normalized = HarnessRuntime._normalize_execution_profile(profile)
        if normalized == "npu_only":
            return "npu"
        if normalized == "hybrid_npu_igpu":
            return "hybrid"
        if normalized == "agentic_parallel":
            return "cpu"
        if normalized in {"cpu", "gpu"}:
            return normalized
        return normalized

    @staticmethod
    def _estimate_message_tokens(messages: list[dict[str, str]]) -> int:
        chars = 0
        for message in messages:
            chars += len(str(message.get("content", "")))
        return max(1, int((chars + 3) // 4))

    def _route_backend_defaults(self, route: RoutePolicy) -> dict[str, Any]:
        defaults = self.config.route_backend_defaults
        if not isinstance(defaults, dict):
            return {}
        route_defaults = defaults.get(route.route.value, {})
        return route_defaults if isinstance(route_defaults, dict) else {}

    def _resolve_execution_profile(
        self,
        *,
        requested_profile: Any,
        route: RoutePolicy,
        messages: list[dict[str, str]],
        requested_max_tokens: int,
    ) -> tuple[str, str]:
        explicit = self._normalize_execution_profile(requested_profile)
        if explicit:
            return explicit, "request_override"

        metadata_profile = self._normalize_execution_profile(route.route_metadata.get("execution_profile"))
        if metadata_profile:
            return metadata_profile, "route_metadata"

        route_defaults = self._route_backend_defaults(route)
        preferred_device = str(route_defaults.get("prefer_device", "") or "").strip()
        if preferred_device:
            return self._device_to_profile(preferred_device), "route_backend_defaults"

        text = _last_user_text(messages).lower()
        complex_markers = (
            "story",
            "worldbuilding",
            "character arc",
            "twist",
            "revision",
            "design",
            "strategy",
            "tradeoff",
            "architecture",
        )
        if self.config.agentic_parallel_enabled and any(marker in text for marker in complex_markers):
            return "agentic_parallel", "heuristic_complex_prompt"
        if route.route == Route.TOOL_REQUIRED:
            return "cpu", "heuristic_tool_route"
        if route.route in {Route.STRUCTURED_EXTRACTION, Route.CODE_OR_DATA}:
            return "gpu", "heuristic_route"
        if route.route == Route.GROUNDED_QA or requested_max_tokens >= 1024:
            return "hybrid_npu_igpu", "heuristic_context"
        return "npu_only", "heuristic_default"

    @staticmethod
    def _binding_matches_device(binding: BackendBinding, device: str) -> bool:
        target = str(device or "").strip().lower()
        backend_id = binding.backend_id.lower()
        primary_device = binding.device.lower()
        device_mode = binding.device_mode.lower()
        if target == "hybrid":
            return "hybrid" in {backend_id, primary_device} or "hybrid" in device_mode or "npu_igpu" in device_mode
        return target in {backend_id, primary_device, device_mode}

    def _ordered_backend_plan(self, route: RoutePolicy, profile: str) -> list[BackendBinding]:
        bindings = self._backend_bindings()
        route_defaults = self._route_backend_defaults(route)
        target_device = self._profile_to_device(profile)
        fallback_chain = route_defaults.get("fallback_chain", [])
        ordered_devices: list[str] = []
        if profile == "agentic_parallel":
            ordered_devices.extend(["cpu", "npu", "gpu", "hybrid"])
        elif target_device:
            ordered_devices.append(target_device)
        if isinstance(fallback_chain, list):
            ordered_devices.extend(str(item).strip().lower() for item in fallback_chain if str(item).strip())
        ordered_devices.extend(binding.device.lower() for binding in bindings)

        plan: list[BackendBinding] = []
        seen: set[str] = set()
        for device in ordered_devices:
            for binding in bindings:
                if binding.backend_id in seen:
                    continue
                if self._binding_matches_device(binding, device):
                    plan.append(binding)
                    seen.add(binding.backend_id)
        for binding in bindings:
            if binding.backend_id not in seen:
                plan.append(binding)
        return plan or bindings

    @staticmethod
    def _backend_plan_payload(plan: list[BackendBinding], selected_backend_id: str) -> list[dict[str, Any]]:
        return [
            {
                "backend_id": binding.backend_id,
                "device": binding.device,
                "device_mode": binding.device_mode,
                "model": binding.model,
                "selected": binding.backend_id == selected_backend_id,
            }
            for binding in plan
        ]

    async def _run_agentic_parallel(
        self,
        *,
        prompt_messages: list[dict[str, str]],
        requested_temp: float,
        requested_max_tokens: int,
        final_schema: dict[str, Any] | None,
        fallback_controller: BackendBinding,
    ) -> tuple[ModelGenerateResult, dict[str, Any], BackendBinding]:
        bindings = self._backend_bindings()
        controller = next((binding for binding in bindings if self._binding_matches_device(binding, "cpu")), fallback_controller)
        worker_candidates = [
            binding
            for binding in bindings
            if binding.backend_id != controller.backend_id
            and (
                self._binding_matches_device(binding, "npu")
                or self._binding_matches_device(binding, "gpu")
                or self._binding_matches_device(binding, "hybrid")
            )
        ]
        max_workers = max(1, int(self.config.agentic_parallel_max_workers or 2))
        worker_candidates = worker_candidates[:max_workers]

        async def _worker(binding: BackendBinding) -> dict[str, Any]:
            worker_messages = list(prompt_messages)
            worker_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Parallel worker task: produce a concise, high-signal draft for the final controller. "
                        "Focus on useful substance; do not mention benchmark internals."
                    ),
                }
            )
            req = ModelGenerateRequest(
                model=binding.model,
                messages=worker_messages,
                temperature=requested_temp,
                max_new_tokens=max(128, min(512, requested_max_tokens // 2)),
                response_schema=None,
                allow_reasoning=False,
            )
            started = time.perf_counter()
            try:
                result = await binding.client.generate(req)
                return {
                    "backend_id": binding.backend_id,
                    "device": binding.device,
                    "status": "ok",
                    "runtime_ms": int((time.perf_counter() - started) * 1000),
                    "usage": result.usage,
                    "text": result.text,
                }
            except RuntimeError as exc:
                return {
                    "backend_id": binding.backend_id,
                    "device": binding.device,
                    "status": "failed",
                    "runtime_ms": int((time.perf_counter() - started) * 1000),
                    "error": str(exc),
                    "text": "",
                }

        worker_results = await asyncio.gather(*[_worker(binding) for binding in worker_candidates])
        accepted = [result for result in worker_results if result.get("status") == "ok" and str(result.get("text", "")).strip()]
        worker_summaries = "\n\n".join(
            f"[{result['backend_id']}]\n{str(result.get('text', ''))[:3000]}" for result in accepted
        )
        controller_messages = list(prompt_messages)
        controller_messages.append(
            {
                "role": "user",
                "content": (
                    "Controller task: synthesize the best final answer from these parallel worker drafts. "
                    "Return only the final answer.\n\n"
                    f"{worker_summaries or '[no successful worker drafts]'}"
                ),
            }
        )
        controller_req = ModelGenerateRequest(
            model=controller.model,
            messages=controller_messages,
            temperature=requested_temp,
            max_new_tokens=requested_max_tokens,
            response_schema=final_schema,
            allow_reasoning=False,
        )
        controller_started = time.perf_counter()
        controller_result = await controller.client.generate(controller_req)
        controller_runtime_ms = int((time.perf_counter() - controller_started) * 1000)
        agentic_metadata = {
            "enabled": True,
            "controller_backend": controller.backend_id,
            "controller_device": controller.device,
            "worker_count": len(worker_results),
            "accepted_results": len(accepted),
            "workers": [
                {
                    key: value
                    for key, value in result.items()
                    if key in {"backend_id", "device", "status", "runtime_ms", "usage", "error"}
                }
                for result in worker_results
            ],
            "stages": [
                {
                    "stage": "agentic_parallel",
                    "controller_backend": controller.backend_id,
                    "worker_count": len(accepted),
                    "controller_runtime_ms": controller_runtime_ms,
                }
            ],
        }
        return controller_result, agentic_metadata, controller

    def _normalize_model_error(self, exc: BaseException) -> str:
        text = str(exc).lower()
        if MODEL_BACKEND_UNAVAILABLE_CODE in text:
            return MODEL_BACKEND_UNAVAILABLE_CODE
        if "model_request_failed" in text:
            return MODEL_REQUEST_FAILED_CODE
        if text.startswith("model_backend_unavailable"):
            return MODEL_BACKEND_UNAVAILABLE_CODE
        return MODEL_REQUEST_FAILED_CODE

    def _backend_failure_response(
        self,
        request_id: str,
        route: RoutePolicy,
        model: str,
        run_id: str,
        validation_reason: str,
        error_code: str,
        stage: str,
        *,
        usage: dict[str, int] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        base = _build_openai_like_response(
            request_id=request_id,
            trace_id=self.trace_id,
            model=model,
            text=f"Backend request failed during {stage}.",
            usage=usage or {"input_tokens": 0, "output_tokens": 0},
            reasoning=None,
            route=route,
            policy_version=self.config.policy_version,
            prompt_version=self.config.prompt_version,
            evidence=[],
            tool_plan="",
        )
        validation = {
            "ok": False,
            "missing_fields": ["model_output"],
            "failed_route_ids": [route.route.value],
            "error_codes": [error_code],
        }
        checkpoint_id = self.checkpoint_refs[-1] if self.checkpoint_refs else None
        base.update(
            {
                "status": "validation_block",
                "run_id": run_id,
                "checkpoint_id": checkpoint_id,
                "trace_checkpoint_count": len(self.checkpoint_refs),
                "route": route.route.value,
                "policy": route.as_dict(),
                "evidence_ids": [],
                "evidence": [],
                "claims": [],
                "validation": validation,
                "tool_calls": tool_calls or [],
                "next_action": "ask_clarification",
                "guard": {
                    "input": {"allow": True},
                    "output": {"allow": False, "reason": validation_reason},
                },
                "parse": {"schema_valid": False, "errors": [validation_reason]},
                "meta": {**base.get("meta", {}), "error_code": error_code, "backend_stage": stage},
                "error_code": error_code,
                "error": validation_reason,
                "feature_level": self.feature_level,
                "tool_sandbox_mode": self.tool_sandbox_mode,
            }
        )
        latency_ms = 0
        if usage is not None and isinstance(usage.get("input_tokens"), int):
            # Keep shape compatibility with final response; exact latency is still computed by caller.
            latency_ms = int(usage.get("input_tokens", 0) * 0 + usage.get("output_tokens", 0) * 0)
        trace = TraceEvent(
            request_id=request_id,
            route=route.route.value,
            model=model,
            policy_version=self.config.policy_version,
            prompt_version=self.config.prompt_version,
            status="validation_block",
            latency_ms=latency_ms,
            stages=[{"stage": stage, "status": "failed", "error_code": error_code}],
            checkpoints=self._load_checkpoint_payloads(),
            state_diffs={
                "checkpoint_count": len(self.checkpoint_refs),
                "latest_phase": self.checkpoint_refs[-1] if self.checkpoint_refs else None,
                "latest_state_diffs": {},
            },
            request=redact_sensitive_args({"route": route.as_dict(), "error_code": error_code}),
            result_summary={
                "status": "validation_block",
                "usage": usage or {},
                "validation": validation,
                "tool_calls": len(tool_calls or []),
            },
        )
        self.trace_store.write(trace)
        return base


    async def process(self, request: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        feature_level = (self.config.feature_level or "basic").lower()
        if feature_level != self.feature_level:
            self.feature_level = feature_level
            self.route_manifests = load_route_manifest(
                str(self.config.route_manifest_path),
                strict=self.feature_level == "hardening",
            )
        self.require_evidence = bool(self.config.require_evidence)
        self.route_overrides = self.config.route_overrides
        messages = request.get("messages", [])
        request_id = request.get("request_id") or str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        self.last_run_id = run_id
        self.trace_id = str(uuid.uuid4())
        self._last_checkpoint_state = {}
        model = request.get("model", self.config.model)
        response_schema = request.get("response_schema")
        route_override = request.get("route")
        self.checkpoint_refs = []

        raw_user_text = _last_user_text(messages)
        trusted_input, untrusted_input = split_trusted_untrusted(raw_user_text)
        user_text = sanitize_text(untrusted_input or trusted_input)
        route_manifest_payload: dict[str, Any] = {
            rid: (route.model_dump() if hasattr(route, "model_dump") else route)
            for rid, route in self.route_manifests.items()
        }
        route = classify_route(
            messages,
            response_schema=response_schema,
            route_override=route_override,
            route_manifest=route_manifest_payload,
            route_overrides=self.route_overrides,
            feature_level=self.feature_level,
            advanced_router_enabled=self.config.advanced_router_enabled,
        )

        self._checkpoint(
            phase="routing",
            run_id=run_id,
            attempt=1,
            route_id=route.route.value,
            status="ok",
            payload={"route": route.route.value, "confidence": route.confidence},
        )

        if route.next_action == "ask_clarification":
            validation = {"ok": False, "missing_fields": ["route_confidence"], "failed_route_ids": [route.route.value]}
            response = _build_openai_like_response(
                request_id=request_id,
                trace_id=self.trace_id,
                model=model,
                text="I need additional clarification before proceeding.",
                usage={"input_tokens": 0, "output_tokens": 0},
                reasoning=None,
                route=route,
                policy_version=self.config.policy_version,
                prompt_version=self.config.prompt_version,
                evidence=[],
                tool_plan="",
            )
            self._checkpoint(
                phase="validation",
                run_id=run_id,
                attempt=1,
                route_id=route.route.value,
                status="blocked",
                error_code="low_confidence",
                payload={"validation": validation, "next_action": "ask_clarification"},
            )
            response.update(
                {
                    "status": "validation_block",
                    "route": route.route.value,
                    "next_action": "ask_clarification",
                    "validation": validation,
                    "trace_checkpoint_count": len(self.checkpoint_refs),
                    "checkpoint_id": self.checkpoint_refs[-1] if self.checkpoint_refs else None,
                    "evidence": [],
                    "claims": [],
                    "trace": [],
                }
            )
            self.trace_store.write(
                TraceEvent(
                    request_id=request_id,
                    route=route.route.value,
                    model=model,
                    policy_version=self.config.policy_version,
                    prompt_version=self.config.prompt_version,
                    status=response["status"],
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    stages=[],
                    checkpoints=self._load_checkpoint_payloads(),
                    request=redact_sensitive_args({"messages": [{"role": "user", "content": raw_user_text}], "model": model, "route": route.route.value}),
                    result_summary={"status": response["status"], "validation": validation},
                )
            )
            return response

        input_decision = check_input_text(user_text)
        if not input_decision.allow:
            return _error_payload(
                self.trace_id,
                request_id,
                model,
                route,
                input_decision,
                {"reason": input_decision.reason},
            )

        if self.require_evidence:
            self._checkpoint(
                phase="input_guard",
                run_id=run_id,
                attempt=1,
                route_id=route.route.value,
                status="ok",
                payload={"decision": "passed"},
            )

        # Policy hooks from route class.
        model_name = str(model).lower()
        if any(prefix in model_name for prefix in self.config.thinking_model_prefixes):
            route.allow_reasoning = True
            if route.thinking_budget == "low":
                route.thinking_budget = "medium"

        requested_toolset = request.get("toolset")
        if isinstance(requested_toolset, list):
            requested_tools = tuple(sorted({str(x) for x in requested_toolset if isinstance(x, str)}))
            if requested_tools:
                allowed_by_request = [tool for tool in route.allowed_tools if tool in requested_tools]
                route.allowed_tools = tuple(allowed_by_request)

        # Global tool allowlist.
        route.allowed_tools = tuple(
            tool for tool in route.allowed_tools if tool in self.config.tool_allowlist
        )
        if not route.allowed_tools:
            route.use_tools = False
        requested_execution_profile = request.get("execution_profile")

        # Cache key excludes volatile outputs.
        cache_key_payload = {
            "response_normalization_version": RESPONSE_NORMALIZATION_VERSION,
            "model": model,
            "messages": messages,
            "policy": route.as_dict(),
            "schema": response_schema,
            "safety_profile": request.get("safety_profile", "default"),
            "route_override": route_override,
            "run_id": run_id,
            "toolset": requested_toolset or [],
            "execution_profile": requested_execution_profile or "",
        }

        cache_hit = None
        if self.cache is not None:
            cache_hit = self.cache.get(cache_key_payload)
            if cache_hit is not None:
                cache_hit = dict(cache_hit)
                cache_hit["id"] = f"ch_{cache_hit.get('id', request_id)}"
                cache_hit["cached"] = True
                cache_hit["trace_id"] = self.trace_id
                cache_hit["run_id"] = run_id
                cache_hit["checkpoint_id"] = None
                cache_hit["validation"] = {"ok": True, "missing_fields": [], "failed_route_ids": []}
                cache_hit["status"] = str(cache_hit.get("status") or "ok")
                cache_hit["next_action"] = "report"
                cache_hit["route"] = route.route.value
                cache_hit["trace_checkpoint_count"] = 0
                return cache_hit

        requested_max_tokens = int(request.get("max_tokens", route.max_new_tokens))
        requested_temp = float(request.get("temperature", route.temperature))
        requested_temp = min(2.0, max(0.0, requested_temp))
        requested_max_tokens = max(64, requested_max_tokens)
        execution_profile, profile_source = self._resolve_execution_profile(
            requested_profile=requested_execution_profile,
            route=route,
            messages=messages,
            requested_max_tokens=requested_max_tokens,
        )
        route.route_metadata["execution_profile"] = execution_profile
        route.route_metadata["execution_profile_source"] = profile_source
        route.route_metadata["requested_context_tokens"] = self._estimate_message_tokens(messages)
        backend_plan = self._ordered_backend_plan(route, execution_profile)
        selected_binding = backend_plan[0]
        selected_client = selected_binding.client
        selected_model = selected_binding.model or model
        selected_via = profile_source or "route_selection"

        tool_attempts = 0
        stage_events: list[dict[str, Any]] = []
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        parsed_payload: dict[str, Any] | None = None
        agentic_parallel_metadata: dict[str, Any] | None = None

        docs: list[RetrievedDoc] = []
        evidence_ids: list[str] = []
        if route.use_retrieval:
            docs = self.retriever.search(user_text, k=24)
            docs = self.reranker.rank(query=user_text, docs=docs, k=6)
            docs = pack_context(docs, max_tokens=1500)
            evidence_ids = [doc.doc_id for doc in docs]

        self._checkpoint(
            phase="retrieval",
            run_id=run_id,
            attempt=1,
            route_id=route.route.value,
            status="ok",
            payload={"count": len(docs)},
        )

        prompt_messages = _build_system_prompt(route, evidence_ids, [doc.text[:900] for doc in docs])
        prompt_messages.extend(messages)

        tool_results: list[ToolCallResult] = []
        tool_plan_answer = ""
        if route.use_tools and route.allowed_tools and tool_attempts < route.max_tool_calls_per_turn:
            tool_prompt = prompt_messages.copy()
            tool_prompt.append(
                {"role": "assistant", "content": _tool_planner_instruction(route.allowed_tools)},
            )
            tool_plan_schema = _tool_schema(route.allowed_tools)
            tool_plan_req = ModelGenerateRequest(
                model=selected_model,
                messages=tool_prompt,
                temperature=min(0.2, requested_temp),
                max_new_tokens=min(512, max(128, requested_max_tokens // 4)),
                response_schema=tool_plan_schema,
                tools=specs_as_openai_tools(route.allowed_tools),
                allow_reasoning=False,
            )
            try:
                tool_plan_res = await selected_client.generate(tool_plan_req)
            except RuntimeError as exc:
                error_code = self._normalize_model_error(exc)
                self._checkpoint(
                    phase="tool_plan",
                    run_id=run_id,
                    attempt=1,
                    route_id=route.route.value,
                    status="failed",
                    error_code=error_code,
                    payload={"error": str(exc)},
                )
                return self._backend_failure_response(
                    request_id=request_id,
                    route=route,
                    model=selected_model,
                    run_id=run_id,
                    validation_reason=str(exc),
                    error_code=error_code,
                    stage="tool_plan",
                    tool_calls=[],
                )

            tool_plan_stage = {"stage": "tool_plan", "model_usage": tool_plan_res.usage}
            tool_plan_provider = _provider_diagnostics_from_raw(tool_plan_res.raw)
            if tool_plan_provider:
                tool_plan_stage["provider"] = tool_plan_provider
            stage_events.append(tool_plan_stage)
            usage = _accumulate_usage(usage, tool_plan_res.usage)
            parsed_plan_ok, parsed_plan, parse_reason = extract_first_json(tool_plan_res.text)
            tool_plan_answer = tool_plan_res.text or ""
            self._checkpoint(
                phase="tool_plan",
                run_id=run_id,
                attempt=1,
                route_id=route.route.value,
                status="ok" if parsed_plan_ok else "parse_fail",
                error_code=None if parsed_plan_ok else f"tool_plan_parse_failed:{parse_reason}",
                payload={"plan": parsed_plan if isinstance(parsed_plan, dict) else None},
            )

            planned_calls: list[dict[str, Any]] = []
            if parsed_plan_ok and isinstance(parsed_plan, dict):
                planned_calls = parsed_plan.get("tool_calls", [])

            if not isinstance(planned_calls, list):
                planned_calls = []

            allowed_tool_names = set(route.allowed_tools)
            sandbox_tools = set(getattr(route, "tool_sandbox_required", ()))
            planned_count = len(planned_calls)
            for call in planned_calls:
                if not isinstance(call, dict):
                    continue
                if route.max_tool_calls_per_turn > 0 and tool_attempts >= route.max_tool_calls_per_turn:
                    break

                name = str(call.get("name", "")).strip()
                args = call.get("arguments", {})
                if not isinstance(args, dict):
                    args = {}
                if name not in allowed_tool_names:
                    tool_result = ToolCallResult(
                        name=name,
                        args=args if isinstance(args, dict) else {},
                        output=None,
                        success=False,
                        error=f"Unknown tool: {name}",
                        error_code="unknown_tool",
                        started_at=datetime.now(timezone.utc).isoformat(),
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    )
                else:
                    gate = check_tool_request_with_tool(args, tool_name=name)
                    if not gate.allow:
                        tool_result = ToolCallResult(
                            name=name,
                            args=args if isinstance(args, dict) else {},
                            output=None,
                            success=False,
                            error=gate.reason or "tool_request_blocked",
                            error_code="tool_request_blocked",
                            started_at=datetime.now(timezone.utc).isoformat(),
                            finished_at=datetime.now(timezone.utc).isoformat(),
                        )
                    else:
                        if self.tool_sandbox_mode == "docker" and (name in sandbox_tools or name == "run_tests"):
                            tool_result = _run_tool_in_sandbox(
                                name,
                                args,
                                self.sandbox_image,
                                self.sandbox_timeout,
                            )
                        else:
                            tool_result = execute_tool(name, args)

                        output_payload = (
                            json.dumps(tool_result.output, ensure_ascii=False, default=str)
                            if isinstance(tool_result.output, (dict, list))
                            else str(tool_result.output)
                        )
                        if not check_tool_output(output_payload).allow:
                            tool_result = ToolCallResult(
                                name=tool_result.name,
                                args=tool_result.args,
                                output=tool_result.output,
                                success=False,
                                error="tool_output_blocked",
                                error_code="tool_output_blocked",
                                started_at=tool_result.started_at,
                                finished_at=tool_result.finished_at or datetime.now(timezone.utc).isoformat(),
                            )

                tool_attempts += 1
                tool_results.append(tool_result)
                if not tool_result.success and route.hard_fail_errors:
                    hard_fail = any(code in route.hard_fail_errors for code in [tool_result.error_code or ""])
                    if hard_fail:
                        break

            budget_exhausted = (
                route.max_tool_calls_per_turn > 0
                and tool_attempts >= route.max_tool_calls_per_turn
                and planned_count > tool_attempts
            )
            if budget_exhausted:
                self._checkpoint(
                    phase="tool_execution",
                    run_id=run_id,
                    attempt=1,
                    route_id=route.route.value,
                    status="tool_budget_exhausted",
                    error_code="tool_calls_budget_exceeded",
                    payload={
                        "tool_calls": tool_attempts,
                        "tool_calls_budget_exceeded": True,
                        "branch": "tool_budget_guard",
                        "max_tool_calls_per_turn": route.max_tool_calls_per_turn,
                        "tool_plan_count": planned_count,
                    },
                )
                tool_results.append(
                    ToolCallResult(
                        name="tool_budget_guard",
                        args={},
                        output={"branch": "tool_calls_budget_exceeded"},
                        success=False,
                        error="tool_calls_budget_exceeded",
                        error_code="tool_calls_budget_exceeded",
                        started_at=datetime.now(timezone.utc).isoformat(),
                        finished_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
            else:
                self._checkpoint(
                    phase="tool_execution",
                    run_id=run_id,
                    attempt=1,
                    route_id=route.route.value,
                    status="ok" if all(result.success for result in tool_results) else "partial_fail",
                    payload={
                        "tool_calls": [_tool_result_payload(result) for result in tool_results],
                        "tool_calls_budget_exceeded": False,
                    },
                )
            prompt_messages.append(
                {
                    "role": "assistant",
                    "content": "Tool execution results: "
                    + json.dumps([_tool_result_payload(r) for r in tool_results], ensure_ascii=False),
                },
            )
        else:
            self._checkpoint(
                phase="tool_plan",
                run_id=run_id,
                attempt=1,
                route_id=route.route.value,
                status="skipped",
                payload={"tool_calls": 0},
            )
            self._checkpoint(
                phase="tool_execution",
                run_id=run_id,
                attempt=1,
                route_id=route.route.value,
                status="skipped",
                payload={"tool_calls": 0},
            )

        final_schema = response_schema if route.output_schema_required or route.strict_schema else None
        final_msg_prompt = prompt_messages.copy()
        try:
            if execution_profile == "agentic_parallel" and self.config.agentic_parallel_enabled:
                final_res, agentic_parallel_metadata, selected_binding = await self._run_agentic_parallel(
                    prompt_messages=final_msg_prompt,
                    requested_temp=requested_temp if not route.strict_schema else min(0.2, requested_temp),
                    requested_max_tokens=requested_max_tokens,
                    final_schema=final_schema,
                    fallback_controller=selected_binding,
                )
                selected_client = selected_binding.client
                selected_model = selected_binding.model or model
            else:
                final_req = ModelGenerateRequest(
                    model=selected_model,
                    messages=final_msg_prompt,
                    temperature=requested_temp if not route.strict_schema else min(0.2, requested_temp),
                    max_new_tokens=requested_max_tokens,
                    response_schema=final_schema,
                    allow_reasoning=route.allow_reasoning,
                    reasoning_budget_tokens=_budget_tokens(route.thinking_budget),
                )
                final_res = await selected_client.generate(final_req)
        except RuntimeError as exc:
            error_code = self._normalize_model_error(exc)
            self._checkpoint(
                phase="final_generate",
                run_id=run_id,
                attempt=1,
                route_id=route.route.value,
                status="failed",
                error_code=error_code,
                payload={"error": str(exc), "tool_calls": [_tool_result_payload(r) for r in tool_results]},
            )
            return self._backend_failure_response(
                request_id=request_id,
                route=route,
                model=selected_model,
                run_id=run_id,
                validation_reason=str(exc),
                error_code=error_code,
                stage="final_generate",
                tool_calls=[_tool_result_payload(r) for r in tool_results],
                usage=usage,
            )

        final_response_text = final_res.text
        final_reasoning = final_res.reasoning
        final_provider = _provider_diagnostics_from_raw(final_res.raw)
        usage = _accumulate_usage(usage, final_res.usage)
        final_stage = {
            "stage": "final_generate",
            "model_usage": final_res.usage,
            "reasoning": bool(final_reasoning),
        }
        if final_provider:
            final_stage["provider"] = final_provider
            if "finish_reason" in final_provider:
                final_stage["finish_reason"] = final_provider["finish_reason"]
            if "truncated" in final_provider:
                final_stage["truncated"] = final_provider["truncated"]
        stage_events.append(final_stage)

        schema_valid = True
        parse_error = []
        if final_schema is not None:
            parse_ok, parsed_payload = _parse_and_validate(final_response_text, final_schema)
            if not parse_ok:
                schema_valid = False
                parse_error = parsed_payload.get("errors", [])
                if tool_plan_answer:
                    repair_request = final_msg_prompt + [
                        {
                            "role": "assistant",
                            "content": "Previous answer malformed. Return strict JSON for schema only.",
                        }
                    ]
                    try:
                        repair_res = await selected_client.generate(
                            ModelGenerateRequest(
                                model=selected_model,
                                messages=repair_request,
                                temperature=0.0,
                                max_new_tokens=256,
                                response_schema=final_schema,
                                allow_reasoning=False,
                            )
                        )
                    except RuntimeError as exc:
                        error_code = self._normalize_model_error(exc)
                        self._checkpoint(
                            phase="repair",
                            run_id=run_id,
                            attempt=1,
                            route_id=route.route.value,
                            status="failed",
                            error_code=error_code,
                            payload={"error": str(exc)},
                        )
                        return self._backend_failure_response(
                            request_id=request_id,
                            route=route,
                            model=selected_model,
                            run_id=run_id,
                            validation_reason=str(exc),
                            error_code=error_code,
                            stage="repair",
                            tool_calls=[_tool_result_payload(r) for r in tool_results],
                            usage=usage,
                        )
                    parse_ok, parsed_payload = _parse_and_validate(repair_res.text, final_schema)
                    final_response_text = repair_res.text if parse_ok else final_response_text
                    if parse_ok:
                        repair_provider = _provider_diagnostics_from_raw(repair_res.raw)
                        if repair_provider:
                            final_provider = repair_provider
                    schema_valid = parse_ok
                    usage = _accumulate_usage(usage, repair_res.usage)
                    parse_error = parsed_payload.get("errors", []) if not parse_ok else []
                    repair_stage = {"stage": "repair_retry", "model_usage": repair_res.usage}
                    repair_provider = _provider_diagnostics_from_raw(repair_res.raw)
                    if repair_provider:
                        repair_stage["provider"] = repair_provider
                    stage_events.append(repair_stage)
                    tool_plan_answer = tool_plan_answer or repair_res.text

        evidence_rows = _build_evidence_rows(
            run_id=run_id,
            route_id=route.route.value,
            docs=docs,
            tool_results=tool_results,
            response=final_response_text,
        )
        evidence_ids_for_response = [row["evidence_id"] for row in evidence_rows]
        tool_call_payloads = [_tool_result_payload(result) for result in tool_results]
        validation = self._validate_route_evidence(route, tool_results, evidence_rows, parsed_payload)

        evidence_required_failed = self.require_evidence and route.required_evidence_fields and not validation["ok"]
        output_decision = GuardDecision(True, None)
        if evidence_required_failed:
            output_decision = GuardDecision(False, reason="required_evidence_missing")
            final_response_text = "I cannot provide a validated response without required evidence."
        else:
            output_decision = check_output_text(final_response_text)

        status = (
            "ok"
            if output_decision.allow and schema_valid and validation["ok"] and not evidence_required_failed
            else "validation_block"
        )

        claims = _build_claims(
            run_id=run_id,
            route_id=route.route.value,
            response=final_response_text,
            evidence_ids=evidence_ids_for_response,
            valid=not validation["missing_fields"],
            required_evidence=bool(route.required_evidence_fields),
        )

        self._checkpoint(
            phase="validation",
            run_id=run_id,
            attempt=1,
            route_id=route.route.value,
            status="ok" if status == "ok" and schema_valid else "failed",
            payload={"validation": validation, "output_allow": output_decision.allow},
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        tokens_per_second = 0.0
        if latency_ms > 0 and output_tokens > 0:
            tokens_per_second = round(output_tokens / (latency_ms / 1000.0), 4)
        attempts = [
            {
                "backend_id": selected_binding.backend_id,
                "device": selected_binding.device,
                "model": selected_model,
                "status": "ok" if status == "ok" else status,
            }
        ]
        execution = {
            "backend_id": selected_binding.backend_id,
            "device": selected_binding.device,
            "device_mode": selected_binding.device_mode,
            "model": selected_model,
            "profile": execution_profile,
            "profile_source": profile_source,
            "fallback_attempted": False,
            "fallback_reason": "",
            "attempts": attempts,
            "selected_via": selected_via,
            "wait_ms": 0,
            "runtime_ms": latency_ms,
            "ttft_ms": latency_ms,
            "tokens_per_second": tokens_per_second,
            "backend_plan": self._backend_plan_payload(backend_plan, selected_binding.backend_id),
        }
        if agentic_parallel_metadata is not None:
            execution["agentic_parallel"] = agentic_parallel_metadata

        final_provider = dict(final_provider or {})
        final_provider["execution"] = execution

        response = _build_openai_like_response(
            request_id=request_id,
            trace_id=self.trace_id,
            model=selected_model,
            text=final_response_text,
            usage=usage,
            reasoning=final_reasoning if route.allow_reasoning else None,
            route=route,
            policy_version=self.config.policy_version,
            prompt_version=self.config.prompt_version,
            evidence=evidence_ids_for_response,
            tool_plan=tool_plan_answer,
            provider=final_provider,
        )

        response.update(
            {
                "status": status,
                "run_id": run_id,
                "checkpoint_id": self.checkpoint_refs[-1] if self.checkpoint_refs else None,
                "trace_checkpoint_count": len(self.checkpoint_refs),
                "route": route.route.value,
                "policy": route.as_dict(),
                "evidence_ids": evidence_ids_for_response,
                "evidence": evidence_rows,
                "claims": [claim.to_dict() for claim in claims],
                "validation": validation,
                "tool_calls": tool_call_payloads,
                "next_action": "report" if status == "ok" else "ask_clarification",
                "guard": {"input": {"allow": True}, "output": {"allow": output_decision.allow, "reason": output_decision.reason}},
                "parse": {"schema_valid": schema_valid, "errors": parse_error},
                "stages": stage_events,
                "feature_level": self.feature_level,
                "tool_sandbox_mode": self.tool_sandbox_mode,
                "execution": execution,
                "provider": final_provider,
            }
        )
        if final_provider:
            response.update({key: final_provider.get(key) for key in PROVIDER_DIAGNOSTIC_KEYS if key in final_provider})
            finish_reason = final_provider.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                response["choices"][0]["finish_reason"] = finish_reason
            if "truncated" in final_provider:
                response["truncated"] = bool(final_provider["truncated"])
            if "reasoning_extracted" in final_provider:
                response["reasoning_extracted"] = bool(final_provider["reasoning_extracted"])
        if parsed_payload is not None and final_schema is not None:
            response["parsed"] = parsed_payload

        if self.cache is not None and status == "ok":
            cache_value = {
                "id": request_id,
                "object": "chat.completion",
                "model": selected_model,
                "choices": response["choices"],
                "usage": response["usage"],
                "meta": response.get("meta", {}),
                "policy": route.as_dict(),
                "route": route.route.value,
                "status": status,
                "cached": False,
                "run_id": run_id,
                "checkpoint_id": self.checkpoint_refs[-1] if self.checkpoint_refs else None,
                "validation": validation,
                "trace_checkpoint_count": len(self.checkpoint_refs),
                "execution": execution,
                "provider": final_provider,
            }
            if final_provider:
                cache_value.update({key: final_provider.get(key) for key in PROVIDER_DIAGNOSTIC_KEYS if key in final_provider})
            self.cache.put(cache_key_payload, cache_value)

        checkpoints_payload = self._load_checkpoint_payloads()
        trace = TraceEvent(
            request_id=request_id,
            route=route.route.value,
            model=selected_model,
            policy_version=self.config.policy_version,
            prompt_version=self.config.prompt_version,
            status=status,
            latency_ms=latency_ms,
            stages=stage_events,
            checkpoints=checkpoints_payload,
            state_diffs={
                "checkpoint_count": len(checkpoints_payload),
                "latest_phase": checkpoints_payload[-1].get("phase") if checkpoints_payload else None,
                "latest_state_diffs": checkpoints_payload[-1].get("state_diffs") if checkpoints_payload else {},
            },
            request=redact_sensitive_args({"messages": messages, "model": selected_model, "policy": route.as_dict(), "response_schema": response_schema}),
            result_summary={
                "status": status,
                "usage": usage,
                "evidence_ids": evidence_ids_for_response,
                "tool_calls": len(tool_call_payloads),
                "provider": final_provider,
            },
        )
        self.trace_store.write(trace)
        return response

    def _checkpoint(
        self,
        phase: str,
        run_id: str,
        attempt: int,
        route_id: str,
        status: str = "ok",
        next_action: str | None = None,
        error_code: str | None = None,
        evidence_refs: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        safe_payload = redact_sensitive_args(payload or {})
        safe_evidence_refs = list(evidence_refs or [])
        current_state = self._snapshot_checkpoint_payload(
            phase=phase,
            route_id=route_id,
            status=status,
            next_action=next_action,
            error_code=error_code,
            evidence_refs=safe_evidence_refs,
            payload=safe_payload,
        )
        state_diffs = self._payload_diff(self._last_checkpoint_state, current_state)
        path = write_checkpoint(
            out_dir=self.state_dir,
            run_id=run_id,
            attempt=attempt,
            phase=phase,
            route_id=route_id,
            status=status,
            next_action=next_action,
            error_code=error_code,
            evidence_refs=safe_evidence_refs,
            route_metadata=redact_sensitive_args({"payload": safe_payload}),
            state_diffs=state_diffs,
            payload=safe_payload,
        )
        self._last_checkpoint_state = current_state
        self.checkpoint_refs.append(path)

    @staticmethod
    def _snapshot_checkpoint_payload(
        phase: str,
        route_id: str,
        status: str,
        next_action: str | None,
        error_code: str | None,
        evidence_refs: list[str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "route_id": route_id,
            "status": status,
            "next_action": next_action,
            "error_code": error_code,
            "evidence_refs": evidence_refs,
            "payload": payload,
        }

    @staticmethod
    def _payload_diff(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        added: dict[str, Any] = {}
        removed: list[str] = []
        changed: dict[str, tuple[Any, Any]] = {}
        for key, value in current.items():
            if key not in previous:
                added[key] = value
            elif previous.get(key) != value:
                changed[key] = (previous.get(key), value)
        for key in previous:
            if key not in current:
                removed.append(key)
        return {"added": added, "removed": removed, "changed": changed}

    def _load_checkpoint_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in self.checkpoint_refs:
            try:
                with Path(path).open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if isinstance(payload, dict):
                payloads.append(redact_sensitive_args(payload))
        return payloads

    def _validate_route_evidence(
        self,
        route: RoutePolicy,
        tool_results: list[ToolCallResult],
        evidence_rows: list[dict[str, Any]],
        parsed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        required = [str(item) for item in route.required_evidence_fields]
        hard_fail_codes = {str(code).strip() for code in route.hard_fail_errors if isinstance(code, str)}
        hard_fail_codes.add("unknown_tool")
        failed_route = route.route.value
        result: dict[str, Any] = {"ok": True, "missing_fields": [], "failed_route_ids": []}

        tool_error_codes = [str(result.error_code or "") for result in tool_results if result.error_code]
        error_codes = []
        for code in tool_error_codes:
            if code and code not in error_codes:
                error_codes.append(code)
        hard_fail_from_tools = any(code in hard_fail_codes for code in tool_error_codes)
        if not required:
            if hard_fail_from_tools:
                return {
                    "ok": False,
                    "missing_fields": [],
                    "failed_route_ids": [failed_route],
                    "error_codes": error_codes,
                }
            result["error_codes"] = error_codes
            return result

        present: set[str] = set()
        for row in evidence_rows:
            record = row.get("record")
            if isinstance(record, dict):
                present.update(str(key) for key in record.keys())

        for result in tool_results:
            output = result.output
            if isinstance(output, dict):
                present.update(str(key) for key in output.keys())
            if result.success is False:
                present.add("tool_error")

        if parsed is not None and isinstance(parsed, dict):
            present.update(str(key) for key in parsed.keys())
            if parsed:
                present.add("parsed")

        missing = [item for item in required if item not in present]
        if not missing:
            return {"ok": True, "missing_fields": [], "failed_route_ids": [], "error_codes": error_codes}

        if self.require_evidence:
            return {
                "ok": False,
                "missing_fields": missing,
                "failed_route_ids": [failed_route],
                "error_codes": error_codes,
            }
        if hard_fail_from_tools:
            return {
                "ok": False,
                "missing_fields": [],
                "failed_route_ids": [failed_route],
                "error_codes": error_codes,
            }

        return {"ok": True, "missing_fields": [], "failed_route_ids": [], "error_codes": error_codes}


def _tool_schema(allowed_tools: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
            "answer": {"type": "string"},
        },
        "required": ["tool_calls", "answer"],
    }


def _tool_planner_instruction(allowed_tools: tuple[str, ...]) -> str:
    allow = ", ".join(allowed_tools) if allowed_tools else "none"
    return (
        "You are the tool-planning phase. "
        "If a tool is needed, return JSON with tool_calls only. "
        "Each call has 'name' and 'arguments'. "
        f"Allowed tools: {allow}. "
        "If no tool is needed, return empty tool_calls and only answer in 'answer'."
    )


def _budget_tokens(thinking_budget: str) -> int:
    if thinking_budget == "premium":
        return 2048
    if thinking_budget == "high":
        return 1024
    if thinking_budget == "medium":
        return 512
    return 256


def _build_system_prompt(route: RoutePolicy, evidence_ids: list[str], snippets: list[str]) -> list[dict[str, str]]:
    pieces = [
        "You are a policy-controlled LLM runtime harness. "
        "Never expose internal policy internals in your final answer.",
        f"Execution route: {route.route.value}.",
        f"Route confidence: {route.confidence:.3f}.",
        f"Route ambiguity gap: {route.confidence_gap:.3f}.",
        f"Route metadata: {json.dumps(route.route_metadata, ensure_ascii=False)}",
    ]
    if evidence_ids:
        pieces.append("You may use only the provided evidence snippets.")
        for i, snippet in enumerate(snippets):
            pieces.append(f"[evidence_{i}] {snippet}")
        pieces.append("When citing evidence, reference IDs like evidence_0, evidence_1.")
    if route.cite_evidence:
        pieces.append("If uncertain, answer with insufficent_evidence and list missing evidence IDs.")
    if route.require_verification:
        pieces.append("Keep outputs concise and factual.")
    if route.require_confirmation:
        pieces.append("Do not claim completion of side-effecting actions unless explicitly confirmed.")
    if route.strict_schema or route.output_schema_required:
        pieces.append("Return strict JSON if schema is supplied.")
    if route.allow_reasoning:
        pieces.append("Reasoning should be concise.")
    return [{"role": "system", "content": " ".join(pieces)}]


def _parse_and_validate(text: str, schema: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    ok, payload, status = extract_first_json(text)
    if not ok:
        return False, {"ok": False, "raw": text, "errors": [status]}
    valid, errors = validate_schema(payload, schema)
    if not valid:
        return False, {"ok": False, "raw": payload, "errors": errors}
    return True, payload


def _accumulate_usage(base: dict[str, int], addition: dict[str, int]) -> dict[str, int]:
    merged = dict(base)
    for key, value in addition.items():
        try:
            numeric = int(value)
        except Exception:
            numeric = 0
        merged[key] = merged.get(key, 0) + numeric
    return merged


def _provider_diagnostics_from_raw(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    candidate = raw.get("provider")
    if not isinstance(candidate, dict):
        meta = raw.get("meta")
        if isinstance(meta, dict):
            candidate = meta.get("provider")
    if not isinstance(candidate, dict):
        candidate = {key: raw.get(key) for key in PROVIDER_DIAGNOSTIC_KEYS if key in raw}

    diagnostics = {
        key: candidate.get(key)
        for key in PROVIDER_DIAGNOSTIC_KEYS
        if key in candidate and candidate.get(key) is not None
    }
    if not diagnostics:
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in diagnostics.items():
        sanitized[key] = redact_sensitive_args(value) if isinstance(value, str) else value
    return sanitized


def _tool_result_payload(res: ToolCallResult) -> dict[str, Any]:
    safe_output = redact_sensitive_args(res.output)
    safe_args = redact_sensitive_args(res.args)
    return {
        "name": res.name,
        "tool": res.name,
        "success": res.success,
        "arguments": safe_args,
        "output": safe_output,
        "error": res.error,
        "error_code": res.error_code,
        "sandbox": res.sandbox,
        "started_at": res.started_at,
        "finished_at": res.finished_at,
    }


def _build_evidence_rows(
    run_id: str,
    route_id: str,
    docs: list[RetrievedDoc],
    tool_results: list[ToolCallResult],
    response: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        rows.append(
            {
                "evidence_id": f"{run_id}-retrieval-{index}",
                "route_id": route_id,
                "source": "retrieval",
                "record": redact_sensitive_args(
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text[:1000],
                        "score": doc.score,
                        "metadata": doc.metadata,
                    }
                ),
            }
        )
    for index, result in enumerate(tool_results):
        result_payload = _tool_result_payload(result)
        rows.append(
            {
                "evidence_id": f"{run_id}-tool-{index}",
                "route_id": route_id,
                "source": "tool",
                "record": redact_sensitive_args(result_payload),
                "tool": result.name,
            }
        )
    if response is not None:
        rows.append(
            {
                "evidence_id": f"{run_id}-response-final",
                "route_id": route_id,
                "source": "final_response",
                "record": {
                    "answer": redact_sensitive_args(
                        response[:220] if isinstance(response, str) else str(response),
                    ),
                },
            }
        )
    return rows


def _build_claims(
    run_id: str,
    route_id: str,
    response: str,
    evidence_ids: list[str],
    valid: bool,
    required_evidence: bool,
) -> list[Claim]:
    status = "verified" if valid else "unverified"
    claim_ids = [f"{run_id}-claim-0"]
    claim = Claim(
        claim_id=claim_ids[0],
        route_id=route_id,
        statement=redact_sensitive_args(response[:220]) if isinstance(response, str) else "",
        evidence_ids=evidence_ids if required_evidence else [],
        status=status,
    )
    return [claim]


def _build_openai_like_response(
    request_id: str,
    trace_id: str | None,
    model: str,
    text: str,
    usage: dict[str, int],
    reasoning: str | None,
    route: RoutePolicy,
    policy_version: str,
    prompt_version: str,
    evidence: list[str],
    tool_plan: str,
    provider: dict[str, Any] | None = None,
) -> dict[str, Any]:
    msg = {"role": "assistant", "content": text}
    if route.allow_reasoning and reasoning:
        msg["reasoning"] = reasoning
    if tool_plan and route.use_tools:
        msg["tool_plan"] = tool_plan
    meta = {
        "route": route.route.value,
        "trace_id": trace_id,
        "policy_version": policy_version,
        "prompt_version": prompt_version,
        "evidence_count": len(evidence),
        "route_confidence": route.confidence,
        "route_confidence_gap": route.confidence_gap,
        "route_metadata": route.route_metadata,
    }
    if provider:
        meta["provider"] = provider
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
        "meta": meta,
    }


def _error_payload(
    trace_id: str,
    request_id: str,
    model: str,
    route: RoutePolicy,
    decision: GuardDecision,
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Request blocked by input policy."}, "finish_reason": "policy_block"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "status": "blocked",
        "route": route.route.value,
        "trace_id": trace_id,
        "next_action": "ask_clarification",
        "guard": {"input": {"allow": decision.allow, "reason": decision.reason}, "output": {"allow": True}},
        "evidence": [],
        "claims": [],
        "validation": {"ok": False, "missing_fields": [], "failed_route_ids": [route.route.value]},
        **extra,
    }


def _truncate_text(value: str, max_len: int = 2000) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _run_tool_in_sandbox(
    name: str,
    args: dict[str, Any],
    image: str,
    timeout_seconds: int,
) -> ToolCallResult:
    # Docker-backed sandbox path for declared heavy tools.
    start = datetime.now(timezone.utc).isoformat()
    if name != "run_tests":
        return ToolCallResult(
            name=name,
            args=dict(args),
            output={"sandbox_mode": "docker", "status": "unsupported_tool"},
            success=False,
            error="tool_sandbox_exec_error",
            error_code="tool_sandbox_exec_error",
            sandbox="docker",
            started_at=start,
            finished_at=start,
        )

    if shutil.which("docker") is None:
        return ToolCallResult(
            name=name,
            args=dict(args),
            output={"sandbox_mode": "docker", "status": "unavailable"},
            success=False,
            error="docker executable unavailable",
            error_code="tool_sandbox_unavailable",
            sandbox="docker",
            started_at=start,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    scope = ""
    raw_scope = args.get("scope")
    if isinstance(raw_scope, str):
        scope = raw_scope.strip()
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{Path.cwd()}:/workspace",
        "-w",
        "/workspace",
        image,
        "pytest",
    ]
    if scope:
        cmd.append(scope)
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(5, timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        return ToolCallResult(
            name=name,
            args=dict(args),
            output={
                "sandbox_mode": "docker",
                "status": "timeout",
                "command": cmd,
                "stdout": _truncate_text(str(exc.stdout or "")),
                "stderr": _truncate_text(str(exc.stderr or "")),
            },
            success=False,
            error="tool_sandbox_timeout",
            error_code="tool_sandbox_timeout",
            sandbox="docker",
            started_at=start,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except OSError as exc:
        return ToolCallResult(
            name=name,
            args=dict(args),
            output={"sandbox_mode": "docker", "status": "exec_error", "error": str(exc)},
            success=False,
            error="tool_sandbox_exec_error",
            error_code="tool_sandbox_exec_error",
            sandbox="docker",
            started_at=start,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

    payload = {
        "sandbox_mode": "docker",
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": _truncate_text(proc.stdout or ""),
        "stderr": _truncate_text(proc.stderr or ""),
    }
    if proc.returncode == 0:
        return ToolCallResult(
            name=name,
            args=dict(args),
            output=payload,
            success=True,
            error=None,
            error_code=None,
            sandbox="docker",
            started_at=start,
            finished_at=datetime.now(timezone.utc).isoformat(),
            )
    return ToolCallResult(
        name=name,
        args=dict(args),
        output=payload,
        success=False,
        error="tool_return_nonzero",
        error_code="tool_return_nonzero",
        sandbox="docker",
        started_at=start,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""
