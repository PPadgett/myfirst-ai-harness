"""Shared helpers for running a one-shot query against the harness runtime."""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx

from harness.config import load_runtime_config


DEFAULT_CONFIG_PATH = "harness.yaml"
_BAD_EXPLICIT_MODELS = {
    "hf://meta-llama/Llama-3.1-8B-Instruct",
}


def validate_model_override(model: str | None) -> str | None:
    if model is None:
        return None

    normalized = str(model).strip()
    if not normalized:
        return None

    if normalized.lower() in {m.lower() for m in _BAD_EXPLICIT_MODELS}:
        raise ValueError(
            "Refusing known hardcoded model override hf://meta-llama/Llama-3.1-8B-Instruct. "
            "Use the runtime default from harness.yaml or pass a backend-compatible model."
        )

    return normalized


def resolve_model(config_path: str | None, explicit_model: str | None) -> str:
    explicit = validate_model_override(explicit_model)
    if explicit is not None:
        return explicit

    config_path = config_path or DEFAULT_CONFIG_PATH
    runtime = load_runtime_config(None if config_path == DEFAULT_CONFIG_PATH else config_path)
    return str(runtime.model)


def build_chat_payload(question: str, model_override: str | None = None) -> dict[str, Any]:
    normalized_question = str(question).strip()
    if not normalized_question:
        raise ValueError("question must not be empty")

    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": normalized_question,
            }
        ]
    }
    if model_override:
        payload["model"] = model_override
    return payload


def build_health_url(host: str, port: int) -> str:
    sanitized_host = str(host).strip().rstrip("/")
    if not sanitized_host:
        raise ValueError("host must be set")
    if ":" in sanitized_host and not sanitized_host.startswith(("http://", "https://")) and not sanitized_host.startswith("["):
        raise ValueError("use host without port when port argument is separate")
    if sanitized_host.startswith(("http://", "https://")):
        return f"{sanitized_host}:{port}/health"
    return f"http://{sanitized_host}:{port}/health"


def validate_oneshot_args(
    *,
    mode: str,
    question: str,
    host: str,
    port: int,
    startup_timeout_seconds: int,
    request_timeout_seconds: int,
) -> None:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"runtime", "demo"}:
        raise ValueError(f"unsupported mode '{mode}', expected 'runtime' or 'demo'")
    if not str(question).strip():
        raise ValueError("question is required")
    if not str(host).strip():
        raise ValueError("host is required")
    if int(port) <= 0:
        raise ValueError("port must be a positive integer")
    if int(startup_timeout_seconds) < 0:
        raise ValueError("StartupTimeoutSeconds must be >= 0")
    if int(request_timeout_seconds) <= 0:
        raise ValueError("RequestTimeoutSeconds must be > 0")


def wait_for_http_ready(
    url: str,
    *,
    timeout_seconds: int = 30,
    interval_seconds: float = 0.75,
    request_timeout_seconds: float = 2.0,
    request_fn: Callable[[str, float], Any] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> bool:
    request = request_fn or (lambda target, req_timeout: httpx.get(target, timeout=req_timeout))
    sleep = sleep_fn or time.sleep
    now = now_fn or time.perf_counter
    end_at = now() + float(timeout_seconds)
    last_error: Exception | None = None

    while True:
        if now() > end_at:
            raise TimeoutError(
                f"Timed out waiting for {url} after {timeout_seconds:.1f}s"
                + (f": {last_error}" if last_error else "")
            )
        try:
            response = request(url, request_timeout_seconds)
            if getattr(response, "status_code", None) == 200:
                return True
        except Exception as exc:  # pragma: no cover - runtime path only
            last_error = exc
        sleep(interval_seconds)
