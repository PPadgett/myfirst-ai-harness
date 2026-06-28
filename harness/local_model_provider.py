"""Minimal OpenAI-compatible local model backend for development and tests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi import Request
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


def _build_default_answer(question: str, route_hint: str | None = None) -> str:
    normalized = question.strip().lower()
    if "capital of france" in normalized:
        return "The capital of France is Paris."
    if "most played music video" in normalized:
        return "This is often cited as 'Gangnam Style' by Psy."
    if "one plus" in normalized:
        return "Could you clarify which OnePlus model you want to discuss?"
    if route_hint:
        return f"Processed request for {route_hint}."
    if not normalized:
        return "I can help with that question."
    return f"I received your question: {question}"


def _generate_payload(question: str, model: str, payload: dict[str, Any]) -> str:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        return json.dumps(
            {
                "answer": _build_default_answer(question, "structured response"),
                "model": model,
            },
            ensure_ascii=False,
        )

    if isinstance(payload.get("tools"), list):
        return json.dumps(
            {
                "tool_calls": [],
                "answer": _build_default_answer(question, "tool plan"),
            },
            ensure_ascii=False,
        )

    return _build_default_answer(question, "chat response")


def build_app(*, model: str = "local-foundation:v1", extra_models: list[str] | None = None) -> FastAPI:
    app = FastAPI(title="Local Model Provider")
    available_models = [model]
    if extra_models:
        for item in extra_models:
            if item and isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    available_models.append(normalized)
    # keep deterministic ordering and quick lookup
    model_catalog = [m for index, m in enumerate(dict.fromkeys(available_models)) if m]

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "local_model_provider",
            "models": len(model_catalog),
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": item,
                    "object": "model",
                    "owned_by": "local",
                }
                for item in model_catalog
            ],
        }

    @app.post("/v1/chat/completions")
    async def completions(req: Request) -> JSONResponse:
        payload = await req.json()
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"error": "payload must be JSON object"})

        requested_model = str(payload.get("model", model))
        if requested_model not in model_catalog:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Model not available: {requested_model}",
                        "code": "model_not_available",
                    }
                },
            )

        question = _extract_last_user_message(payload)
        content = _generate_payload(question, requested_model, payload)
        response: dict[str, Any] = {
            "id": "chatcmpl-local-" + str(hash(requested_model) % 100000),
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": _token_count(question),
                "completion_tokens": _token_count(content),
                "total_tokens": _token_count(question) + _token_count(content),
            },
        }
        return JSONResponse(content=response)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local OpenAI-compatible model provider")
    parser.add_argument("--host", default="127.0.0.1", help="Listen host")
    parser.add_argument("--port", type=int, default=11435, help="Listen port")
    parser.add_argument("--model", default="local-foundation:v1", help="Primary model id")
    parser.add_argument(
        "--extra-model",
        action="append",
        default=None,
        help="Additional model id (repeatable)",
    )
    args = parser.parse_args()
    app = build_app(model=args.model, extra_models=args.extra_model)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
