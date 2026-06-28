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
MODEL_BACKEND_UNAVAILABLE_CODE = "model_backend_unavailable"
MODEL_CATALOG_UNAVAILABLE_CODE = "model_catalog_unavailable"
MODEL_CATALOG_PARSE_CODE = "model_catalog_parse_failed"
MODEL_CATALOG_MISSING_MODEL_CODE = "model_not_available"


class RuntimeBackendError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        config_path: str,
        backend_url: str,
        expected_model: str | None,
    ) -> None:
        self.code = code
        self.config_path = config_path
        self.backend_url = backend_url
        self.expected_model = expected_model
        self.message = message
        super().__init__(
            f"{message} | error_code={code} | config_path={config_path} | backend_url={backend_url}"
            + (f" | expected_model={expected_model}" if expected_model else "")
        )


def runtime_backend_context(config_path: str | None = None) -> dict[str, str]:
    config_path = config_path or DEFAULT_CONFIG_PATH
    runtime = load_runtime_config(None if config_path == DEFAULT_CONFIG_PATH else config_path)
    return {
        "config_path": config_path,
        "backend_name": runtime.backend.name,
        "backend_url": runtime.backend.base_url.rstrip("/"),
        "resolved_model": str(runtime.model),
    }


def _normalize_model_name(value: str) -> str:
    return value.strip().lower().replace(" ", "")


def _extract_model_names(payload: Any) -> list[str]:
    candidates: Any = payload
    if isinstance(payload, dict):
        for key in ("data", "models"):
            nested = payload.get(key)
            if isinstance(nested, list):
                candidates = nested
                break

    if not isinstance(candidates, list):
        return []

    names: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            name = item.strip()
            if name:
                names.append(name)
            continue
        if isinstance(item, dict):
            for key in ("id", "name", "model"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    names.append(value.strip())
                    break
    return names


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


def validate_runtime_backend(
    config_path: str | None = None,
    *,
    timeout_seconds: float = 2.0,
    expected_model: str | None = None,
    skip_backend_check: bool = False,
) -> None:
    config_path = config_path or DEFAULT_CONFIG_PATH
    if skip_backend_check:
        return

    runtime = load_runtime_config(None if config_path == DEFAULT_CONFIG_PATH else config_path)
    backend_name = str(runtime.backend.name).strip().lower()
    allowed_backends = {"openai", "ollama", "nvidia_nim", "auto"}
    if backend_name not in allowed_backends:
        raise RuntimeError(
            f"Unsupported backend '{backend_name}'. Supported backends: {', '.join(sorted(allowed_backends))}."
        )

    normalized_base_url = runtime.backend.base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        models_url = f"{normalized_base_url}/models"
    else:
        models_url = f"{normalized_base_url}/v1/models"

    normalized_expected = None
    if expected_model is not None:
        normalized_expected = str(expected_model)

    try:
        response = httpx.get(models_url, timeout=timeout_seconds)
    except httpx.RequestError as exc:
        raise RuntimeError(
            str(
                RuntimeBackendError(
                    MODEL_BACKEND_UNAVAILABLE_CODE,
                    f"Model backend is unavailable at {models_url}",
                    config_path=config_path,
                    backend_url=models_url,
                    expected_model=normalized_expected,
                )
            )
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            str(
                RuntimeBackendError(
                    MODEL_CATALOG_UNAVAILABLE_CODE,
                    f"Model backend catalog check failed at {models_url} with status {response.status_code}",
                    config_path=config_path,
                    backend_url=models_url,
                    expected_model=normalized_expected,
                )
            )
        )

    if not expected_model:
        return

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            str(
                RuntimeBackendError(
                    MODEL_CATALOG_PARSE_CODE,
                    f"Model backend catalog response at {models_url} was not valid JSON",
                    config_path=config_path,
                    backend_url=models_url,
                    expected_model=normalized_expected,
                )
            )
        ) from exc

    model_names = _extract_model_names(payload)
    normalized_expected = _normalize_model_name(expected_model)

    normalized_catalog = {_normalize_model_name(name) for name in model_names}
    if normalized_expected not in normalized_catalog:
        raise RuntimeError(
            str(
                RuntimeBackendError(
                    MODEL_CATALOG_MISSING_MODEL_CODE,
                    f"Model '{expected_model}' is not available in catalog at {models_url}",
                    config_path=config_path,
                    backend_url=models_url,
                    expected_model=expected_model,
                )
            )
        )


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
