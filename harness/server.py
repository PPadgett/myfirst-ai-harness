from __future__ import annotations

import asyncio
import argparse
import logging
import os

import httpx
import yaml

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from harness.adapters import LlamaCppClient, NvidiaNimClient, OpenAICompatibleClient
from harness.adapters.base import BaseModelClient
from harness.config import BackendConfig, RuntimeConfig, load_runtime_config
from harness.runtime import HarnessRuntime

LOGGER = logging.getLogger(__name__)

_KNOWN_RUNTIME_LABELS = {"openai", "ollama", "nvidia_nim", "auto"}


def _normalize_base_url(raw_url: str) -> str:
    normalized = str(raw_url or "").strip().rstrip("/")
    if not normalized:
        return ""
    if "://" not in normalized:
        return normalized
    return normalized


def _load_configured_backends(config_path: str | None) -> list[dict[str, object]]:
    if not config_path or not str(config_path).strip():
        return []

    candidate_path = os.path.abspath(config_path)
    if not os.path.exists(candidate_path):
        return []
    try:
        with open(candidate_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
    except Exception:
        return []

    if not isinstance(loaded, dict):
        return []

    raw_backends = loaded.get("backends")
    if not isinstance(raw_backends, list) or not raw_backends:
        fallback = loaded.get("backend")
        raw_backends = [fallback] if isinstance(fallback, dict) else []

    normalized = []
    for backend in raw_backends:
        if not isinstance(backend, dict):
            continue
        base_url = _normalize_base_url(str(backend.get("base_url", "") or "").strip())
        runtime_name = str(backend.get("runtime", "") or "").strip().lower()
        runtime_name = runtime_name if runtime_name in _KNOWN_RUNTIME_LABELS else "openai"
        model_name = str(backend.get("model", "") or "").strip()
        backend_id = str(backend.get("id", "") or backend.get("device", "") or runtime_name or "").strip()
        normalized.append(
            {
                "id": backend_id,
                "base_url": base_url,
                "runtime": runtime_name,
                "model": model_name,
                "required": bool(backend.get("required", True)),
            }
        )
    return normalized


def _build_backend_selection_errors(
    *,
    config_path: str | None,
    runtime_backend: str,
    runtime_model: str,
    runtime_base_url: str,
) -> list[tuple[bool, str, str, str, str]]:
    checks = _load_configured_backends(config_path=config_path)
    if not checks:
        return [(True, runtime_backend, runtime_model, runtime_base_url, f"single({runtime_backend})")]

    backend_checks = []
    for entry in checks:
        backend_id = str(entry.get("id", ""))
        base_url = _normalize_base_url(str(entry.get("base_url", "")))
        model = str(entry.get("model", "")).strip()
        runtime = str(entry.get("runtime", "openai"))
        required = bool(entry.get("required", True))
        if not base_url:
            if required:
                backend_checks.append((True, runtime, model or runtime_model, base_url, backend_id or "missing_base_url"))
            else:
                LOGGER.warning(
                    "Skipping optional backend '%s' during startup because base_url is missing in config '%s'.",
                    backend_id,
                    config_path,
                )
            continue
        backend_checks.append((required, runtime, model, base_url, backend_id or f"{runtime}@{base_url}"))
    return backend_checks

def _should_validate_catalog(backend_name: str, llama_cpp_model_path: str | None) -> bool:
    backend_name = backend_name.strip().lower()
    if backend_name in {"ollama", "nvidia_nim"}:
        return True
    if backend_name == "auto":
        return not bool(llama_cpp_model_path)
    return False


def build_app(config: RuntimeConfig | None = None, *, config_path: str | None = None) -> FastAPI:
    cfg = config or load_runtime_config()
    app = FastAPI(title="Local LLM Harness")
    backend_name = cfg.backend.name.lower()

    def _openai_client(backend: BackendConfig) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(
            base_url=backend.base_url,
            model=backend.model or cfg.model,
            api_key=backend.api_key,
            timeout_seconds=backend.timeout_seconds,
            extra_headers=backend.extra_headers,
            extra_body=backend.extra_body,
        )

    def _client_for_backend(backend: BackendConfig) -> BaseModelClient:
        name = backend.name.lower()
        if name == "llamacpp":
            if not backend.llama_cpp_model_path:
                raise RuntimeError("llamacpp backend selected but llama_cpp_model_path is missing")
            client = LlamaCppClient(
                model_path=backend.llama_cpp_model_path,
                n_ctx=backend.llama_cpp_n_ctx,
                n_gpu_layers=backend.llama_cpp_ngl,
            )
            return client
        if name == "nvidia_nim":
            return NvidiaNimClient(
                base_url=backend.base_url,
                model=backend.model or cfg.model,
                api_key=backend.api_key,
                timeout_seconds=backend.timeout_seconds,
                extra_headers=backend.extra_headers,
                extra_body=backend.extra_body,
            )
        if name == "auto":
            model = backend.llama_cpp_model_path or backend.model or cfg.model
            if str(model).lower().endswith(".gguf"):
                return LlamaCppClient(
                    model_path=model,
                    n_ctx=backend.llama_cpp_n_ctx,
                    n_gpu_layers=backend.llama_cpp_ngl,
                )
        return _openai_client(backend)

    if cfg.backends:
        clients: dict[str, BaseModelClient] = {}
        for backend in cfg.backends:
            backend_id = backend.backend_id or backend.device or backend.name
            clients[backend_id] = _client_for_backend(backend)
        client_or_clients: BaseModelClient | dict[str, BaseModelClient] = clients
    else:
        client_or_clients = _client_for_backend(cfg.backend)

    runtime = HarnessRuntime(cfg, client_or_clients)

    @app.on_event("startup")
    async def validate_backend_connection() -> None:
        configured_backends_present = _load_configured_backends(config_path=None if config_path == "harness.yaml" else config_path)
        backend_checks = _build_backend_selection_errors(
            config_path=None if config_path == "harness.yaml" else config_path,
            runtime_backend=backend_name,
            runtime_model=cfg.model,
            runtime_base_url=cfg.backend.base_url,
        )

        if not backend_checks:
            return

        required_failures: list[str] = []
        warnings: list[str] = []
        for required, candidate_backend, candidate_model, candidate_base_url, candidate_label in backend_checks:
            runtime_for_check = candidate_backend or backend_name
            model_for_check = candidate_model or cfg.model
            should_validate = _should_validate_catalog(
                backend_name=runtime_for_check,
                llama_cpp_model_path=cfg.backend.llama_cpp_model_path if runtime_for_check == backend_name else None,
            )
            if runtime_for_check == "openai" and configured_backends_present:
                should_validate = True
            if not should_validate:
                continue
            if not candidate_base_url:
                if required:
                    required_failures.append(f"{candidate_label} (missing base_url)")
                continue
            try:
                await _wait_for_model_catalog(
                    base_url=candidate_base_url,
                    model=model_for_check,
                    timeout_seconds=cfg.backend.timeout_seconds,
                    backend_label=f"{candidate_label}",
                )
            except RuntimeError as exc:
                message = str(exc)
                if required:
                    required_failures.append(message)
                else:
                    warnings.append(message)

        if warnings:
            for warning in warnings:
                LOGGER.warning("Optional model backend validation warning: %s", warning)

        if required_failures:
            raise RuntimeError("Model backend validation failed for one or more required backends: " + "; ".join(required_failures))

    def _health_backend_row(backend: BackendConfig) -> dict[str, object]:
        backend_id = backend.backend_id or backend.device or backend.name
        return {
            "id": backend_id,
            "backend_id": backend_id,
            "name": backend.name,
            "runtime": backend.runtime or backend.name,
            "device": backend.device or backend_id,
            "device_mode": backend.device_mode or backend.device or backend_id,
            "capabilities": list(backend.capabilities),
            "base_url": backend.base_url,
            "model": backend.model or cfg.model,
            "required": backend.required,
            "health": "ready",
            "status": "ready",
            "health_reachable": True,
            "health_endpoint": backend.health_endpoint,
            "max_context": backend.max_context,
            "max_output_tokens": backend.max_output_tokens,
            "max_concurrency": backend.max_concurrency,
        }

    @app.get("/health")
    async def health() -> dict[str, object]:
        configured_backends = list(cfg.backends) if cfg.backends else [cfg.backend]
        return {
            "ok": True,
            "backend": cfg.backend.name,
            "model": cfg.model,
            "backends": [_health_backend_row(backend) for backend in configured_backends],
        }

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
                "execution_profile": data.get("execution_profile"),
                "safety_profile": data.get("safety_profile", "default"),
                "request_id": data.get("request_id"),
                "toolset": data.get("toolset") or data.get("tools", []),
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
                "execution_profile": data.get("execution_profile"),
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
    app = build_app(config, config_path=args.config)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
