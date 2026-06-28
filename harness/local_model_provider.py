"""Local OpenAI-compatible model backend for self-hosted models.

This module intentionally supports two modes:
- a deterministic fallback mode (works everywhere, no extra deps)
- optional local Transformers mode when dependencies and downloaded model files are available

It exposes the same minimal OpenAI-compatible contract used by the harness runtime:
- GET /health
- GET /v1/models
- POST /v1/chat/completions
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


def _token_count(value: str) -> int:
    if not value:
        return 0
    return max(1, len(value.split()))


def _extract_last_user_message(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).lower() != "user":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            return content.strip()
    return ""


def _to_str_list(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = item.get("content")
        if isinstance(content, str):
            if content.strip():
                parts.append(f"[{role}] {content.strip()}")
    return "\n".join(parts)


def _load_json_model(payload: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class ModelPlan:
    model_id: str
    backend: str
    source: str | None = None


class _ModelBackendRuntime:
    def __init__(
        self,
        model_id: str,
        backend: str,
        source: str | None,
        *,
        device: str = "cpu",
        max_new_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.backend = backend
        self.source = source
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        prompt = _to_str_list(messages).strip()
        if prompt:
            return prompt
        return ""

    def _fallback_answer(self, question: str) -> str:
        if not question:
            return "I can help with that request."
        lowered = question.strip().lower()
        if "capital of france" in lowered:
            return "The capital of France is Paris."
        if "most played music video" in lowered:
            return "This is often cited as 'Gangnam Style' by Psy."
        if "one plus" in lowered:
            return "Could you clarify which OnePlus model you want to discuss?"
        return f"I received your question: {question}"

    async def generate(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        _ = tools

        if self.backend == "transformers":
            return await self._generate_with_transformers(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

        # deterministic fallback path keeps the harness usable without local GPU/weights.
        question = _extract_last_user_message({"messages": messages})
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            return json.dumps(
                {
                    "answer": self._fallback_answer(question),
                    "model": self.model_id,
                },
                ensure_ascii=False,
            )
        return self._fallback_answer(question)

    async def _generate_with_transformers(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> str:
        if not self.source:
            raise RuntimeError("transformer backend requires model source")

        def _run() -> str:
            if self._pipeline is None:
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                except Exception as exc:  # pragma: no cover - dependency path
                    raise RuntimeError(f"transformers backend unavailable: {exc}") from exc

                tokenizer = AutoTokenizer.from_pretrained(self.source, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(self.source, local_files_only=True)
                self._pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device.lower() != "cpu" else -1,
                )

            prompt = self._build_prompt(messages)
            params: dict[str, Any] = {
                "max_new_tokens": max_tokens or self.max_new_tokens,
                "do_sample": False,
                "return_full_text": False,
            }
            if temperature is not None:
                params["do_sample"] = True
                params["temperature"] = max(0.0, float(temperature))

            outputs = self._pipeline(prompt, **params)
            if not outputs:
                return ""
            output = outputs[0].get("generated_text", "") if isinstance(outputs[0], dict) else ""
            generated = str(output)
            if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
                fallback = self._fallback_answer(prompt)
                obj = {"answer": fallback, "model": self.model_id}
                if generated.strip():
                    structured = _load_json_model(generated)
                    if structured is not None:
                        obj = structured
                return json.dumps(obj, ensure_ascii=False)
            return generated.strip() if generated else self._fallback_answer(prompt)

        return await asyncio.to_thread(_run)


def _normalize_model_plan(raw: str, default_backend: str, default_source: str | None) -> ModelPlan:
    trimmed = (raw or "").strip()
    if "::" in trimmed:
        model_id, source = [part.strip() for part in trimmed.split("::", 1)]
        if model_id and source:
            return ModelPlan(model_id=model_id, backend=default_backend, source=source)
    if "=" in trimmed:
        model_id, source = [part.strip() for part in trimmed.split("=", 1)]
        if model_id and source:
            return ModelPlan(model_id=model_id, backend=default_backend, source=source)

    # if the entire string is empty, we still keep a named placeholder model.
    model_id = trimmed or "local-foundation:v1"
    return ModelPlan(model_id=model_id, backend=default_backend, source=default_source)


def _discover_default_models(model_root: str | None) -> list[str]:
    if not model_root:
        return []
    root = Path(model_root)
    if not root.exists() or not root.is_dir():
        return []

    discovered: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "config.json").exists():
            discovered.append(entry.name)
    return discovered


def _build_runtime_from_args(args: argparse.Namespace) -> tuple[dict[str, ModelPlan], str]:
    primary = _normalize_model_plan(
        args.model,
        default_backend=args.backend,
        default_source=args.model_path or None,
    )
    if not primary.source and args.models_root:
        primary_source_candidates = _discover_default_models(args.models_root)
        if args.model not in primary_source_candidates and primary.model_id not in primary_source_candidates:
            # keep explicit model id while still allowing fallback.
            primary = ModelPlan(model_id=primary.model_id, backend=primary.backend, source=None)
        else:
            # when a matching directory is found, point model source to it.
            candidate = (args.models_root.rstrip("/\\") + f"/{primary.model_id}").replace("\\", "/")
            primary = ModelPlan(model_id=primary.model_id, backend=primary.backend, source=candidate)

    catalog: dict[str, ModelPlan] = {primary.model_id: primary}

    for extra in args.extra_model or []:
        parsed = _normalize_model_plan(extra, default_backend=args.backend, default_source=None)
        if parsed.model_id in catalog and catalog[parsed.model_id].source:
            continue
        catalog[parsed.model_id] = parsed

    fallback_key = primary.model_id
    if args.model and "::" not in args.model and "=" not in args.model:
        fallback_key = args.model

    return catalog, fallback_key


def build_app(
    *,
    model: str = "local-foundation:v1",
    backend: str = "auto",
    model_path: str | None = None,
    models_root: str | None = None,
    extra_models: list[str] | None = None,
    device: str = "cpu",
    max_tokens: int = 256,
    local_only: bool = False,
) -> FastAPI:
    app = FastAPI(title="Local Model Provider")

    args = SimpleNamespace(
        model=model,
        backend=backend,
        model_path=model_path,
        models_root=models_root,
        extra_model=extra_models or [],
        device=device,
        local_only=local_only,
    )
    plans, fallback_model = _build_runtime_from_args(args)

    def _resolve_runtime(requested_model: str) -> _ModelBackendRuntime:
        selected = requested_model.strip() if isinstance(requested_model, str) else ""
        plan = plans.get(selected)
        if plan is None:
            if selected and selected in plans:
                plan = plans[selected]
            else:
                # keep exact fallback behavior for older payloads
                plan = plans.get(fallback_model)

        if plan is None:
            raise HTTPException(status_code=404, detail=f"Model not available: {requested_model}")

        selected_backend = plan.backend
        selected_source = plan.source

        if selected_backend == "transformers":
            if not selected_source:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Model '{plan.model_id}' has no local path."
                        " Configure --model-path or set an entry in --extra-model as '<id>::<path>'."
                    ),
                )
            return _ModelBackendRuntime(
                model_id=plan.model_id,
                backend="transformers",
                source=selected_source,
                device=device,
                max_new_tokens=max_tokens,
            )

        if selected_backend == "auto" and (selected_source or local_only):
            if selected_source:
                try:
                    return _ModelBackendRuntime(
                        model_id=plan.model_id,
                        backend="transformers",
                        source=selected_source,
                        device=device,
                        max_new_tokens=max_tokens,
                    )
                except Exception:
                    if local_only:
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                f"Model '{plan.model_id}' is configured for auto backend and could not be loaded from"
                                f" '{selected_source}' in local-only mode."
                            ),
                        )
            return _ModelBackendRuntime(model_id=plan.model_id, backend="fallback", source=selected_source, device=device)

        if selected_backend not in {"auto", "fallback", "transformers"}:
            raise HTTPException(status_code=400, detail=f"Unsupported backend '{selected_backend}'")

        if selected_backend == "transformers":
            return _ModelBackendRuntime(model_id=plan.model_id, backend="transformers", source=selected_source, device=device)
        return _ModelBackendRuntime(model_id=plan.model_id, backend="fallback", source=selected_source, device=device)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "local_model_provider",
            "models": len(plans),
            "backend": backend,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        data = [
            {
                "id": item.model_id,
                "object": "model",
                "owned_by": "local-stack",
            }
            for item in plans.values()
        ]
        return {
            "object": "list",
            "data": data,
        }

    @app.post("/v1/chat/completions")
    async def completions(req: Request) -> JSONResponse:
        payload = await req.json()
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "payload must be JSON object", "code": "bad_payload"}})

        requested_model = str(payload.get("model", model)).strip()
        if requested_model and requested_model not in plans and requested_model != fallback_model:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Model not available: {requested_model}",
                        "code": "model_not_available",
                    }
                },
            )

        try:
            runtime = _resolve_runtime(requested_model or fallback_model)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"message": exc.detail, "code": "model_not_available"}},
            )

        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "messages must be a list", "code": "bad_payload"}},
            )

        response_format = payload.get("response_format")
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else None
        temperature = payload.get("temperature")
        if temperature is not None:
            try:
                temperature = float(temperature)
            except Exception:
                temperature = None
        max_tokens_param = payload.get("max_tokens")
        if not isinstance(max_tokens_param, int):
            max_tokens_param = None

        try:
            content = await runtime.generate(
                messages=messages,
                response_format=response_format if isinstance(response_format, dict) else None,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens_param,
            )
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": f"Model backend generation failed: {exc}",
                        "code": "model_request_failed",
                    }
                },
            )

        question = _extract_last_user_message({"messages": messages})
        if tools:
            choice_obj = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [],
                    "content": content,
                },
                "finish_reason": "tool_calls",
            }
        else:
            choice_obj = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }

        response: dict[str, Any] = {
            "id": f"chatcmpl-local-{hash((requested_model or model).encode('utf-8')) & 0xFFFF}",
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": runtime.model_id,
            "choices": [choice_obj],
            "usage": {
                "prompt_tokens": _token_count(_to_str_list(messages)),
                "completion_tokens": _token_count(content),
                "total_tokens": _token_count(_to_str_list(messages)) + _token_count(content),
            },
        }

        return JSONResponse(content=response)

    return app


def _parse_backend(name: str) -> str:
    normalized = str(name or "auto").strip().lower()
    if normalized in {"auto", "fallback", "transformers"}:
        return normalized
    return "auto"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible model provider")
    parser.add_argument("--host", default="127.0.0.1", help="Listen host")
    parser.add_argument("--port", type=int, default=11435, help="Listen port")
    parser.add_argument("--model", default="local-foundation:v1", help="Primary model id")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "fallback", "transformers"),
        help="Backend runtime for model generation.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Optional absolute/relative path used for the primary model source.",
    )
    parser.add_argument(
        "--models-root",
        default="",
        help="Optional root directory to discover local model directories.",
    )
    parser.add_argument(
        "--model-root",
        dest="models_root",
        default="",
        help="Alias for --models-root.",
    )
    parser.add_argument(
        "--extra-model",
        action="append",
        default=None,
        help=(
            "Extra model in form '<model-id>' or '<model-id>::<local path>' or '<model-id>=<local path>' "
            "(repeatable)."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Torch device for transformers backend.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens when using transformer backend.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Reject non-local or network-backed model resolution."
    )
    args = parser.parse_args()

    backend = _parse_backend(args.backend)
    app = build_app(
        model=args.model,
        backend=backend,
        model_path=(args.model_path or None),
        models_root=(args.models_root or None),
        extra_models=args.extra_model,
        device=args.device,
        max_tokens=args.max_new_tokens,
        local_only=args.local_only,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
