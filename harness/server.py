from __future__ import annotations

import asyncio
import argparse
import logging

import httpx
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from harness.adapters import LlamaCppClient, NvidiaNimClient, OpenAICompatibleClient
from harness.adapters.base import BaseModelClient
from harness.config import RuntimeConfig, load_runtime_config
from harness.runtime import HarnessRuntime

LOGGER = logging.getLogger(__name__)


def _should_validate_catalog(backend_name: str, llama_cpp_model_path: str | None) -> bool:
    backend_name = backend_name.strip().lower()
    if backend_name in {"ollama", "nvidia_nim"}:
        return True
    if backend_name == "auto":
        return not bool(llama_cpp_model_path)
    return False


def build_app(config: RuntimeConfig | None = None) -> FastAPI:
    cfg = config or load_runtime_config()
    app = FastAPI(title="Local LLM Harness")
    client: BaseModelClient
    backend_name = cfg.backend.name.lower()
    extra_headers = cfg.backend.extra_headers
    extra_body = cfg.backend.extra_body

    def _openai_client() -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            base_url=cfg.backend.base_url,
            model=cfg.model,
            api_key=cfg.backend.api_key,
            timeout_seconds=cfg.backend.timeout_seconds,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )

    if backend_name == "llamacpp":
        if not cfg.backend.llama_cpp_model_path:
            raise RuntimeError("llamacpp backend selected but llama_cpp_model_path is missing")
        client = LlamaCppClient(
            model_path=cfg.backend.llama_cpp_model_path,
            n_ctx=cfg.backend.llama_cpp_n_ctx,
            n_gpu_layers=cfg.backend.llama_cpp_ngl,
        )
    elif backend_name == "nvidia_nim":
        client = NvidiaNimClient(
            base_url=cfg.backend.base_url,
            model=cfg.model,
            api_key=cfg.backend.api_key,
            timeout_seconds=cfg.backend.timeout_seconds,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
    elif backend_name == "ollama":
        client = _openai_client()
    elif backend_name == "auto":
        model = cfg.backend.llama_cpp_model_path or cfg.model
        if str(model).lower().endswith(".gguf"):
            client = LlamaCppClient(
                model_path=model,
                n_ctx=cfg.backend.llama_cpp_n_ctx,
                n_gpu_layers=cfg.backend.llama_cpp_ngl,
            )
        else:
            client = _openai_client()
    else:
        client = _openai_client()

    runtime = HarnessRuntime(cfg, client)

    @app.on_event("startup")
    async def validate_backend_connection() -> None:
        if _should_validate_catalog(backend_name=backend_name, llama_cpp_model_path=cfg.backend.llama_cpp_model_path):
            await _wait_for_model_catalog(
                base_url=cfg.backend.base_url,
                model=cfg.model,
                timeout_seconds=cfg.backend.timeout_seconds,
                backend_label=backend_name,
            )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"ok": "true", "backend": cfg.backend.name, "model": cfg.model}

    @app.post("/v1/chat/completions")
    async def chat(req: Request) -> JSONResponse:
        data = await req.json()
        response = await runtime.process(
            {
                "messages": data.get("messages", []),
                "model": data.get("model", cfg.model),
                "temperature": data.get("temperature", 0.3),
                "max_tokens": data.get("max_tokens", cfg.backend.max_tokens),
                "response_schema": data.get("response_schema")
                or _response_schema_from_format(data.get("response_format")),
                "route": data.get("route"),
                "safety_profile": data.get("safety_profile", "default"),
                "request_id": data.get("request_id"),
                "toolset": data.get("tools", []),
            }
        )
        return JSONResponse(content=response)

    @app.post("/v1/answer")
    async def answer(req: Request) -> JSONResponse:
        data = await req.json()
        response = await runtime.process(
            {
                "messages": [{"role": "user", "content": data.get("input", "")}],
                "model": data.get("model", cfg.model),
                "response_schema": data.get("response_schema")
                or _response_schema_from_format(data.get("response_format")),
                "route": data.get("route"),
                "safety_profile": data.get("safety_profile", "default"),
                "request_id": data.get("request_id"),
            }
        )
        return JSONResponse(content=response)

    return app


def _response_schema_from_format(response_format: dict[str, object] | None) -> dict | None:
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") == "json_object":
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }
    return None


def _extract_model_names(model_payload: object) -> list[str]:
    if isinstance(model_payload, dict):
        for key in ("data", "models"):
            nested = model_payload.get(key, [])
            if isinstance(nested, list):
                model_payload = nested
                break
    if not isinstance(model_payload, list):
        return []

    names: list[str] = []
    for item in model_payload:
        if isinstance(item, str):
            if item.strip():
                names.append(item.strip())
            continue
        if isinstance(item, dict):
            for key in ("id", "name", "model"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    names.append(value.strip())
                    break
    return names


def _normalize_model_name(value: str) -> str:
    return value.strip().lower().replace(" ", "")


def _is_model_available(requested: str, available: list[str]) -> bool:
    normalized_requested = _normalize_model_name(requested)
    return any(_normalize_model_name(candidate) == normalized_requested for candidate in available)


def _build_model_catalog_url(base_url: str) -> str:
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        return f"{normalized_base_url}/models"
    return f"{normalized_base_url}/v1/models"


async def _validate_openai_model_catalog(
    *,
    base_url: str,
    model: str,
    timeout_seconds: int,
    backend_label: str,
) -> None:
    endpoint = _build_model_catalog_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
    except httpx.RequestError as exc:
        raise RuntimeError(f"{backend_label} endpoint unavailable at {endpoint}: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{backend_label} models endpoint returned {exc.response.status_code} at {endpoint}") from exc

    try:
        model_names = _extract_model_names(response.json())
    except ValueError as exc:
        raise RuntimeError(f"{backend_label} models response was not valid JSON at {endpoint}") from exc

    if not model_names:
        raise RuntimeError(
            f"{backend_label} models endpoint responded but returned no models; expected populated catalog at {endpoint}"
        )

    if model and not _is_model_available(model, model_names):
        raise RuntimeError(
            f"{backend_label} model '{model}' not found in catalog at {endpoint}. "
            f"Available: {', '.join(model_names[:20])}"
        )


async def _wait_for_model_catalog(
    *,
    base_url: str,
    model: str,
    timeout_seconds: int,
    backend_label: str,
    max_attempts: int = 18,
    retry_delay_seconds: float = 2.5,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            await _validate_openai_model_catalog(
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                backend_label=backend_label,
            )
            return
        except RuntimeError as exc:
            message = str(exc).lower()
            retryable = (
                "endpoint unavailable" in message
                or "endpoint returned" in message
                or "returned no models" in message
            )
            if not retryable or attempt >= max_attempts:
                raise
            LOGGER.warning(
                "%s model catalog not ready (attempt %s/%s); retrying in %ss",
                backend_label,
                attempt,
                max_attempts,
                retry_delay_seconds,
            )
            await asyncio.sleep(retry_delay_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local LLM Harness server")
    parser.add_argument("--config", default="harness.yaml", help="Path to config YAML (optional)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    config = load_runtime_config(None if args.config == "harness.yaml" else args.config)
    app = build_app(config)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
