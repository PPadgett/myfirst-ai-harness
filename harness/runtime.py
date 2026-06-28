from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any
import hashlib
import json
import time
import uuid

from harness.adapters.base import BaseModelClient
from harness.config import RuntimeConfig
from harness.guards import GuardDecision, check_input_text, check_output_text
from harness.retrieval import DirectoryCorpusRetriever, RetrievedDoc, SimpleReranker, pack_context
from harness.router import RoutePolicy, Route, classify_route
from harness.tools import ToolCallResult, execute_tool, specs_as_openai_tools
from harness.trace_cache import ResponseCache, TraceEvent, TraceStore
from harness.types import ModelGenerateRequest, ModelGenerateResult
from harness.validation import extract_first_json, validate_schema


@dataclass
class ModelArtifact:
    trace_id: str
    request_id: str
    route: RoutePolicy
    policy: dict[str, Any]
    retrieved_doc_ids: list[str]
    tool_calls: list[dict[str, Any]]
    prompt_payload: list[dict[str, str]]
    model_result: str
    model_reasoning: str | None
    usage: dict[str, int]
    latency_ms: int


class HarnessRuntime:
    def __init__(
        self,
        config: RuntimeConfig,
        model_client: BaseModelClient,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.retriever = DirectoryCorpusRetriever(config.corpus_dir)
        self.reranker = SimpleReranker()
        self.trace_store = TraceStore(config.trace_dir)
        self.cache = ResponseCache(config.cache_dir, config.max_cache_entries) if config.enable_cache else None

    async def process(self, request: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        request_id = request.get("request_id") or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        messages = request.get("messages", [])
        model = request.get("model", self.config.model)
        response_schema = request.get("response_schema")
        route_override = request.get("route")
        user_safety_profile = request.get("safety_profile", "default")
        route = classify_route(messages, response_schema=response_schema, route_override=route_override)
        model_name = str(model).lower()
        if any(prefix in model_name for prefix in self.config.thinking_model_prefixes):
            route.allow_reasoning = True
            if route.thinking_budget == "low":
                route.thinking_budget = "medium"

        # input guard
        user_text = _last_user_text(messages)
        input_decision = check_input_text(user_text)
        if not input_decision.allow:
            return _error_payload(trace_id, request_id, model, route, input_decision, {})

        # cache key excludes tool outputs and final response
        cache_hit = None
        cache_key_payload = {
            "model": model,
            "messages": messages,
            "policy": route.as_dict(),
            "schema": response_schema,
            "safety_profile": user_safety_profile,
        }
        requested_toolset = request.get("toolset")
        if isinstance(requested_toolset, list):
            requested_tools = tuple(sorted({str(x) for x in requested_toolset if isinstance(x, str)}))
            if requested_tools:
                route.allowed_tools = tuple(t for t in route.allowed_tools if t in requested_tools)

        # merge route and user constraints for cache key
        cache_key_payload["policy"]["allowed_tools"] = list(route.allowed_tools)
        if self.cache is not None:
            cache_hit = self.cache.get(cache_key_payload)
        if cache_hit is not None:
            cache_hit["id"] = f"ch_{cache_hit.get('id', request_id)}"
            cache_hit["cached"] = True
            cache_hit["trace_id"] = trace_id
            return cache_hit

        requested_temp = float(request.get("temperature", route.temperature))
        requested_max_tokens = int(request.get("max_tokens", route.max_new_tokens))
        if requested_temp < 0:
            requested_temp = 0.0
        if requested_temp > 2:
            requested_temp = 2.0

        if requested_max_tokens < 64:
            requested_max_tokens = 64
        evidence_ids: list[str] = []
        docs: list[RetrievedDoc] = []
        if route.use_retrieval:
            docs = self.retriever.search(user_text, k=24)
            docs = self.reranker.rank(query=user_text, docs=docs, k=6)
            docs = pack_context(docs, max_tokens=1500)
            evidence_ids = [d.doc_id for d in docs]

        stage_events = []
        tool_results: list[ToolCallResult] = []
        prompt_messages = _build_system_prompt(route, evidence_ids, [d.text[:900] for d in docs])
        prompt_messages.extend(messages)

        tool_attempts = 0
        final_response_text = ""
        final_reasoning = None
        usage = {"input_tokens": 0, "output_tokens": 0}

        if route.use_tools and tool_attempts < route.max_model_calls:
            tool_prompt = prompt_messages.copy()
            tool_prompt.append(
                {
                    "role": "assistant",
                    "content": _tool_planner_instruction(route.allowed_tools),
                }
            )
            tool_request_schema = _tool_schema(route.allowed_tools)
            tool_plan_req = ModelGenerateRequest(
                model=model,
                messages=tool_prompt,
                temperature=min(0.2, requested_temp),
                max_new_tokens=min(512, max(128, requested_max_tokens // 4)),
                response_schema=tool_request_schema,
                tools=specs_as_openai_tools(route.allowed_tools),
            )
            tool_plan_res = await self.model_client.generate(tool_plan_req)
            stage_events.append({"stage": "tool_plan", "model_usage": tool_plan_res.usage})
            usage = _accumulate_usage(usage, tool_plan_res.usage)

            parsed_plan_ok, parsed_plan, _ = extract_first_json(tool_plan_res.text)
            tool_calls: list[dict[str, Any]] = []
            if parsed_plan_ok and isinstance(parsed_plan, dict):
                tool_calls = parsed_plan.get("tool_calls", [])
            if tool_calls:
                for item in tool_calls[: route.max_tool_calls_per_turn]:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", ""))
                    args = item.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}
                    if name not in route.allowed_tools:
                        continue
                    res = execute_tool(name, args)
                    tool_results.append(res)
                    tool_attempts += 1

            if tool_results:
                results_payload = [
                    {
                        "tool": t.name,
                        "success": t.success,
                        "arguments": t.args,
                        "output": t.output,
                        "error": t.error,
                    }
                    for t in tool_results
                ]
                prompt_messages.append(
                    {
                        "role": "assistant",
                        "content": "Tool execution results: " + json.dumps(results_payload, ensure_ascii=False),
                    }
                )

        final_msg_prompt = prompt_messages.copy()
        final_schema = response_schema if route.output_schema_required or route.strict_schema else None
        final_temp = requested_temp if not route.strict_schema else min(0.2, requested_temp)
        final_tokens = requested_max_tokens
        final_req = ModelGenerateRequest(
            model=model,
            messages=final_msg_prompt,
            temperature=final_temp,
            max_new_tokens=final_tokens,
            response_schema=final_schema,
            allow_reasoning=route.allow_reasoning,
            reasoning_budget_tokens=_budget_tokens(route.thinking_budget),
        )
        final_res = await self.model_client.generate(final_req)
        final_response_text = final_res.text
        final_reasoning = final_res.reasoning
        usage = _accumulate_usage(usage, final_res.usage)
        stage_events.append({"stage": "final_generate", "model_usage": final_res.usage, "reasoning": bool(final_reasoning)})

        parsed = None
        parse_errors: list[str] = []
        schema_valid = False
        parsed_ok = True
        if final_schema is not None:
            parsed_ok, parsed = _parse_and_validate(final_response_text, final_schema)
            parse_errors = parsed.get("errors", []) if not parsed_ok else []
            schema_valid = parsed_ok
            if not parsed_ok:
                repair_request_schema = {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                    },
                    "required": ["answer"],
                }
                repair_msg = final_msg_prompt + [
                    {"role": "assistant", "content": "Previous answer malformed. Return only strict JSON for schema and keep values simple."},
                ]
                repair_req = ModelGenerateRequest(
                    model=model,
                    messages=repair_msg,
                    temperature=0.0,
                    max_new_tokens=512,
                    response_schema=repair_request_schema,
                )
                repair_res = await self.model_client.generate(repair_req)
                parsed_ok, parsed = _parse_and_validate(repair_res.text, final_schema)
                final_response_text = repair_res.text if parsed_ok else final_response_text
                schema_valid = parsed_ok
                usage = _accumulate_usage(usage, repair_res.usage)
                stage_events.append({"stage": "repair_retry", "model_usage": repair_res.usage})

        output_decision = check_output_text(final_response_text)
        if not output_decision.allow:
            final_response_text = "I can’t provide that content under current output policy."

        status = "ok" if output_decision.allow and (not final_schema or schema_valid) else "validation_block"

        latency_ms = int((time.perf_counter() - start) * 1000)
        response = _build_openai_like_response(
            request_id=request_id,
            trace_id=trace_id,
            model=model,
            text=final_response_text,
            usage=usage,
            reasoning=final_reasoning if route.allow_reasoning else None,
            route=route,
            policy_version=self.config.policy_version,
            prompt_version=self.config.prompt_version,
            evidence=evidence_ids,
        )
        response.update(
            {
                "status": status,
                "route": route.route.value,
                "policy": route.as_dict(),
                "evidence_ids": evidence_ids,
                "tool_calls": [r.__dict__ for r in tool_results],
                "trace_id": trace_id,
                "latency_ms": latency_ms,
                "guard": {"input": input_decision.__dict__, "output": output_decision.__dict__},
                "parse": {"schema_valid": schema_valid, "errors": parse_errors},
                "stages": stage_events,
            }
        )
        if parsed_ok and final_schema is not None:
            response["parsed"] = parsed

        if self.cache is not None:
            self.cache.put(
                cache_key_payload,
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "model": model,
                    "choices": response["choices"],
                    "usage": response["usage"],
                    "policy": route.as_dict(),
                    "route": route.route.value,
                    "status": status,
                    "cached": False,
                },
            )

        trace = TraceEvent(
            request_id=request_id,
            route=route.route.value,
            model=model,
            policy_version=self.config.policy_version,
            prompt_version=self.config.prompt_version,
            status=status,
            latency_ms=latency_ms,
            stages=stage_events,
            request={"messages": messages, "model": model, "policy": route.as_dict(), "response_schema": response_schema},
            result_summary={"status": status, "usage": usage, "evidence_ids": evidence_ids, "tool_calls": len(tool_results)},
        )
        self.trace_store.write(trace)
        return response


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


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
        "Allowed tools: " + allow + ". "
        "If no tool is needed, return empty tool_calls and only answer in 'answer'."
    )


def _budget_tokens(thinking_budget: str) -> int:
    if thinking_budget == "high":
        return 1024
    if thinking_budget == "medium":
        return 512
    if thinking_budget == "premium":
        return 2048
    return 256


def _build_system_prompt(route: RoutePolicy, evidence_ids: list[str], snippets: list[str]) -> list[dict[str, str]]:
    pieces = [
        "You are a policy-controlled LLM runtime harness. "
        "Never expose internal trace or policy internals in your final answer.",
        f"Execution route: {route.route.value}.",
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
        pieces.append("Reasoning should be careful and short; final answer should remain concise.")

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
        if not isinstance(value, int):
            value = 0
        merged[key] = merged.get(key, 0) + value
    return merged


def _build_openai_like_response(
    request_id: str,
    trace_id: str,
    model: str,
    text: str,
    usage: dict[str, int],
    reasoning: str | None,
    route: RoutePolicy,
    policy_version: str,
    prompt_version: str,
    evidence: list[str],
) -> dict[str, Any]:
    msg = {"role": "assistant", "content": text}
    if route.allow_reasoning and reasoning:
        msg["reasoning"] = reasoning
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
        "meta": {
            "route": route.route.value,
            "trace_id": trace_id,
            "policy_version": policy_version,
            "prompt_version": prompt_version,
            "evidence_count": len(evidence),
        },
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
        "guard": {"input": decision.__dict__, "output": {"allow": True}},
        **extra,
    }
