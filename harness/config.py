from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os
import yaml
import json


_BACKEND_ALIASES: dict[str, str] = {
    "openai": "openai",
    "local_openai": "openai",
    "local-openai": "openai",
    "localopenai": "openai",
    "auto": "auto",
    "llamacpp": "llamacpp",
    "llama_cpp": "llamacpp",
    "llama-cpp": "llamacpp",
    "nim": "nvidia_nim",
    "nvidia": "nvidia_nim",
    "nvidia_nim": "nvidia_nim",
    "nvidia-nim": "nvidia_nim",
    "ollama": "ollama",
    "ollama_api": "ollama",
}
_KNOWN_BACKENDS = frozenset({"openai", "auto", "llamacpp", "nvidia_nim", "ollama"})


@dataclass
class BackendConfig:
    name: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: int = 120
    llama_cpp_model_path: str | None = None
    llama_cpp_ngl: int = 0
    llama_cpp_n_ctx: int = 4096
    max_tokens: int = 768
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    model: str = ""
    backend_id: str = ""
    device: str = ""
    device_mode: str = ""
    runtime: str = ""
    capabilities: tuple[str, ...] = ()
    required: bool = True
    max_context: int = 4096
    max_output_tokens: int = 768
    max_concurrency: int = 1
    health_endpoint: str = ""


@dataclass
class RuntimeConfig:
    backend: BackendConfig
    corpus_dir: Path = Path("corpus")
    trace_dir: Path = Path("traces")
    cache_dir: Path = Path(".cache")
    state_dir: Path = Path("state")
    enable_cache: bool = True
    max_cache_entries: int = 2000
    tool_allowlist: tuple[str, ...] = ("calculator", "time_now")
    model: str = "gpt-oss:latest"
    prompt_version: str = "2026.06.27-core"
    policy_version: str = "2026.06.27-policy-v1"
    feature_level: str = "basic"
    advanced_router_enabled: bool = False
    route_manifest_path: str | None = None
    require_evidence: bool = False
    thinking_model_prefixes: tuple[str, ...] = (
        "deepseek",
        "qwen2.5",
        "gpt-4",
        "glm-4",
        "llama-3.1",
        "mistral-large",
        "granite3.2",
    )
    route_overrides: dict[str, Any] = field(default_factory=dict)
    route_backend_defaults: dict[str, Any] = field(default_factory=dict)
    agentic_parallel_enabled: bool = False
    agentic_parallel_max_workers: int = 2
    agentic_parallel_max_repair_loops: int = 1
    agentic_parallel_max_wall_clock_seconds: int = 120
    backends: tuple[BackendConfig, ...] = ()


DEFAULT_CONFIG_PATH = "harness.yaml"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _to_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if value == "":
        return default
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on", "y"}


def _to_str_dict(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if key is None or item is None:
            continue
        normalized[str(key)] = str(item)
    return normalized


def _to_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return (text,) if text else ()


def _load_route_overrides(data: dict[str, Any]) -> dict[str, Any]:
    route_overrides: dict[str, Any] = {}
    configured = data.get("route_overrides", {})
    if isinstance(configured, dict):
        route_overrides.update(configured)

    env_overrides = os.getenv("HARNESS_ROUTE_OVERRIDES")
    if not env_overrides:
        return route_overrides
    try:
        parsed = json.loads(env_overrides)
    except Exception:
        return route_overrides
    if isinstance(parsed, dict):
        merged = dict(route_overrides)
        merged.update(parsed)
        return merged
    return route_overrides


def _normalize_backend_name(raw_name: Any) -> str:
    raw = str(raw_name).strip().lower().replace(" ", "_")
    return _BACKEND_ALIASES.get(raw, raw)


def _resolve_backend_defaults(backend_name: str) -> str:
    if backend_name == "openai":
        return "http://127.0.0.1:11435/v1"
    if backend_name == "nvidia_nim":
        return "http://127.0.0.1:8000/v1"
    if backend_name == "ollama":
        return "http://127.0.0.1:11434/v1"
    return "http://127.0.0.1:11435/v1"


def _resolve_backend_api_key(data: dict[str, Any], backend_name: str) -> str | None:
    api_key = data.get("api_key")
    if api_key is not None:
        return api_key

    explicit = os.getenv("HARNESS_API_KEY")
    if explicit:
        return explicit

    if backend_name == "nvidia_nim":
        return os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
    if backend_name == "ollama":
        return os.getenv("OLLAMA_API_KEY")
    return None


def _backend_config_from_data(
    backend_data: dict[str, Any],
    *,
    default_backend_name: str = "openai",
    default_base_url: str | None = None,
    default_model: str = "",
) -> BackendConfig:
    backend_name = _normalize_backend_name(backend_data.get("name", backend_data.get("backend", default_backend_name)))
    if backend_name not in _KNOWN_BACKENDS:
        runtime_name = str(backend_data.get("runtime", "") or "").strip().lower()
        if runtime_name in {"local_provider", "lemonade_ryzenai"}:
            backend_name = "openai"
        else:
            raise ValueError(f"Unsupported backend '{backend_name}'. Supported: {sorted(_KNOWN_BACKENDS)}")

    base_url = str(
        backend_data.get(
            "base_url",
            default_base_url if default_base_url is not None else _env("HARNESS_BASE_URL", _resolve_backend_defaults(backend_name)),
        )
    ).strip()
    if not base_url:
        raise ValueError(f"backend '{backend_name}' requires a non-empty base_url")

    model = str(backend_data.get("model", default_model or "") or "").strip()
    runtime_name = str(backend_data.get("runtime", "") or "").strip()
    device = str(backend_data.get("device", "") or "").strip()
    backend_id = str(backend_data.get("id", "") or backend_data.get("backend_id", "") or device or runtime_name or backend_name).strip()
    device_mode = str(backend_data.get("device_mode", "") or device or backend_id).strip()

    extra_headers = _to_str_dict(backend_data.get("extra_headers"))
    extra_body = backend_data.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = None

    return BackendConfig(
        name=backend_name,
        base_url=base_url,
        api_key=_resolve_backend_api_key(backend_data, backend_name),
        timeout_seconds=int(backend_data.get("timeout_seconds", _env("HARNESS_TIMEOUT", "120"))),
        llama_cpp_model_path=backend_data.get("llama_cpp_model_path"),
        llama_cpp_ngl=int(backend_data.get("llama_cpp_ngl", "0")),
        llama_cpp_n_ctx=int(backend_data.get("llama_cpp_n_ctx", "4096")),
        max_tokens=int(backend_data.get("max_tokens", backend_data.get("max_output_tokens", "768"))),
        extra_headers=extra_headers,
        extra_body=extra_body,
        model=model,
        backend_id=backend_id,
        device=device or backend_id,
        device_mode=device_mode,
        runtime=runtime_name or backend_name,
        capabilities=_to_str_tuple(backend_data.get("capabilities")),
        required=_to_bool(backend_data.get("required", True), True),
        max_context=int(backend_data.get("max_context", backend_data.get("llama_cpp_n_ctx", "4096"))),
        max_output_tokens=int(backend_data.get("max_output_tokens", backend_data.get("max_tokens", "768"))),
        max_concurrency=int(backend_data.get("max_concurrency", "1")),
        health_endpoint=str(backend_data.get("health_endpoint", "") or "").strip(),
    )


def _load_backend_list(data: dict[str, Any]) -> tuple[BackendConfig, ...]:
    raw_backends = data.get("backends")
    if not isinstance(raw_backends, list):
        return ()

    backends: list[BackendConfig] = []
    for item in raw_backends:
        if not isinstance(item, dict):
            continue
        backends.append(_backend_config_from_data(item, default_backend_name="openai"))
    return tuple(backends)


def load_runtime_config(path: str | None = None) -> RuntimeConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        with Path(config_path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded

    backend_data = data.get("backend", {})
    if not isinstance(backend_data, dict):
        backend_data = {}

    backends = _load_backend_list(data)
    configured_model = str(data.get("model", "") or "").strip()
    env_model = str(_env("HARNESS_MODEL", "") or "").strip()
    first_backend_model = next((backend.model for backend in backends if backend.model), "")
    model = configured_model or env_model or first_backend_model or "gpt-oss:latest"

    if backends:
        backend = backends[0]
    else:
        backend = _backend_config_from_data(
            backend_data,
            default_backend_name=_env("HARNESS_BACKEND", "openai"),
            default_model=model,
        )
        if backend.name in {"ollama", "nvidia_nim"} and not str(model).strip():
            raise ValueError(f"backend '{backend.name}' requires a non-empty model name")

    runtime = RuntimeConfig(
        backend=backend,
        backends=backends,
        corpus_dir=Path(data.get("corpus_dir", _env("HARNESS_CORPUS", "corpus"))),
        trace_dir=Path(data.get("trace_dir", _env("HARNESS_TRACE_DIR", "traces"))),
        cache_dir=Path(data.get("cache_dir", _env("HARNESS_CACHE_DIR", ".cache"))),
        state_dir=Path(data.get("state_dir", _env("HARNESS_STATE_DIR", "state"))),
        enable_cache=_to_bool(data.get("enable_cache", _env("HARNESS_CACHE", "true")), True),
        max_cache_entries=int(data.get("max_cache_entries", "2000")),
        tool_allowlist=tuple(data.get("tool_allowlist", ("calculator", "time_now"))),
        model=model,
        prompt_version=data.get("prompt_version", _env("HARNESS_PROMPT_VERSION", "2026.06.27-core")),
        policy_version=data.get("policy_version", _env("HARNESS_POLICY_VERSION", "2026.06.27-policy-v1")),
        feature_level=data.get("feature_level", _env("HARNESS_FEATURE_LEVEL", "basic")),
        advanced_router_enabled=_to_bool(data.get("advanced_router", _env("HARNESS_ENABLE_ADVANCED_ROUTER", "0")), False),
        route_manifest_path=data.get(
            "route_manifest_path",
            _env("HARNESS_ROUTE_MANIFEST", "real_harness_routes.yaml"),
        )
        or "real_harness_routes.yaml",
        require_evidence=_to_bool(data.get("require_evidence", _env("HARNESS_REQUIRE_EVIDENCE", "false")), False),
        thinking_model_prefixes=tuple(data.get("thinking_model_prefixes", (
            "deepseek",
            "qwen2.5",
            "gpt-4",
            "glm-4",
            "llama-3.1",
            "mistral-large",
            "granite3.2",
        ))),
        route_overrides=_load_route_overrides(data),
        route_backend_defaults=data.get("route_backend_defaults", {}) if isinstance(data.get("route_backend_defaults"), dict) else {},
        agentic_parallel_enabled=_to_bool(data.get("agentic_parallel_enabled", _env("HARNESS_AGENTIC_PARALLEL", "false")), False),
        agentic_parallel_max_workers=int(data.get("agentic_parallel_max_workers", "2")),
        agentic_parallel_max_repair_loops=int(data.get("agentic_parallel_max_repair_loops", "1")),
        agentic_parallel_max_wall_clock_seconds=int(data.get("agentic_parallel_max_wall_clock_seconds", "120")),
    )
    return runtime
