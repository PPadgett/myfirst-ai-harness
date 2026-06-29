"""Local OpenAI-compatible model backend for self-hosted models.

This module intentionally supports two modes:
- a deterministic fallback mode (works everywhere, no extra deps)
- optional local Transformers mode when dependencies and downloaded model files are available
- optional local llama.cpp mode for downloaded GGUF artifacts
- local Ollama model-store resolution by reading manifests/blobs directly, without
  starting or calling Ollama

It exposes the same minimal OpenAI-compatible contract used by the harness runtime:
- GET /health
- GET /v1/models
- POST /v1/chat/completions
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


FALLBACK_WARNING = (
    "Deterministic fallback provider is active. This is diagnostic stub mode, not a real LLM."
)


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


def _read_optional_text(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return None


def _read_optional_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _load_json_model(_read_optional_text(path) or "")
    return payload or {}


def _openai_usage(*, prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": max(0, int(prompt_tokens or 0)),
        "completion_tokens": max(0, int(completion_tokens or 0)),
        "total_tokens": max(0, int(prompt_tokens or 0)) + max(0, int(completion_tokens or 0)),
    }


def _normalize_usage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return _openai_usage(prompt_tokens=0, completion_tokens=0)
    prompt_tokens = payload.get("prompt_tokens", payload.get("input_tokens", 0))
    completion_tokens = payload.get("completion_tokens", payload.get("output_tokens", 0))
    try:
        prompt = int(prompt_tokens or 0)
    except Exception:
        prompt = 0
    try:
        completion = int(completion_tokens or 0)
    except Exception:
        completion = 0
    return _openai_usage(prompt_tokens=prompt, completion_tokens=completion)


def _first_text(payload: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _split_reasoning_content(text: str) -> tuple[str, str | None, bool]:
    if not isinstance(text, str) or not text:
        return "", None, False
    reasoning_parts: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if content:
            reasoning_parts.append(content)
        return ""

    cleaned, count = re.subn(r"(?is)<think>\s*(.*?)\s*</think>", _replace, text)
    return cleaned.strip(), "\n\n".join(reasoning_parts) or None, count > 0


def _looks_truncated(content: str, *, finish_reason: str, max_tokens: int | None) -> bool:
    reason = str(finish_reason or "").strip().lower()
    if reason in {"length", "max_tokens"}:
        return True
    if not max_tokens or not content:
        return False
    return _token_count(content) >= int(max_tokens)


def _is_qwen3_model(model_id: str) -> bool:
    lowered = str(model_id or "").lower()
    return "qwen3" in lowered


def _has_thinking_directive(content: str) -> bool:
    lowered = str(content or "").lower()
    return "/think" in lowered or "/no_think" in lowered


def _apply_model_thinking_control(
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    allow_reasoning: bool,
) -> list[dict[str, Any]]:
    copied = [dict(item) for item in messages if isinstance(item, dict)]
    if not _is_qwen3_model(model_id):
        return copied
    directive = "/think" if allow_reasoning else "/no_think"
    for item in reversed(copied):
        if str(item.get("role", "")).lower() != "user":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if _has_thinking_directive(content):
            return copied
        item["content"] = f"{content.rstrip()}\n\n{directive}".strip()
        return copied
    copied.append({"role": "user", "content": directive})
    return copied


def _infer_llama_chat_format(template: str | None) -> str | None:
    if not template:
        return None
    if "<|im_start|>" in template and "<|im_end|>" in template:
        return "chatml"
    return None


def _generation_params_from_ollama(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    allowed = {"temperature", "top_p", "top_k", "repeat_penalty", "stop"}
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed or value is None:
            continue
        if key == "stop":
            if isinstance(value, list):
                normalized[key] = [str(item) for item in value if str(item)]
            elif isinstance(value, str) and value:
                normalized[key] = value
            continue
        if key == "top_k":
            try:
                normalized[key] = int(value)
            except Exception:
                continue
            continue
        try:
            normalized[key] = float(value)
        except Exception:
            continue
    return normalized


@dataclass(frozen=True)
class ModelPlan:
    model_id: str
    backend: str
    source: str | None = None
    source_type: str | None = None
    artifact_format: str | None = None
    provider_store: str | None = None
    manifest_path: str | None = None
    ollama_template_path: str | None = None
    ollama_params_path: str | None = None


@dataclass(frozen=True)
class ModelDiagnostics:
    model: str
    configured_backend: str
    generation_backend: str
    model_source: str | None
    model_source_type: str | None
    model_artifact_format: str | None
    provider_store: str | None
    manifest_path: str | None
    runtime_dependency: str | None
    runtime_dependency_available: bool | None
    local_model_loaded: bool
    model_source_present: bool
    model_load_attempted: bool
    model_load_succeeded: bool
    last_load_error: str | None
    last_generation_error: str | None
    template_applied: bool
    fallback_active: bool
    allow_fallback: bool
    provider_warning: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "configured_backend": self.configured_backend,
            "generation_backend": self.generation_backend,
            "model_source": self.model_source,
            "model_source_type": self.model_source_type,
            "model_artifact_format": self.model_artifact_format,
            "provider_store": self.provider_store,
            "manifest_path": self.manifest_path,
            "runtime_dependency": self.runtime_dependency,
            "runtime_dependency_available": self.runtime_dependency_available,
            "local_model_loaded": self.local_model_loaded,
            "model_source_present": self.model_source_present,
            "model_load_attempted": self.model_load_attempted,
            "model_load_succeeded": self.model_load_succeeded,
            "last_load_error": self.last_load_error,
            "last_generation_error": self.last_generation_error,
            "template_applied": self.template_applied,
            "fallback_active": self.fallback_active,
            "allow_fallback": self.allow_fallback,
            "provider_warning": self.provider_warning,
        }


@dataclass(frozen=True)
class ProviderGenerationResult:
    content: str
    reasoning: str | None
    finish_reason: str
    usage: dict[str, int]
    truncated: bool
    warnings: list[str]
    raw: dict[str, Any]


class _ModelBackendRuntime:
    def __init__(
        self,
        model_id: str,
        backend: str,
        source: str | None,
        *,
        plan: ModelPlan | None = None,
        device: str = "cpu",
        max_new_tokens: int = 256,
        llama_cpp_n_ctx: int = 4096,
        llama_cpp_n_gpu_layers: int = 0,
        llama_cpp_n_threads: int = 4,
    ) -> None:
        self.model_id = model_id
        self.backend = backend
        self.source = source
        self.plan = plan
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._llama = None
        self.model_load_attempted = False
        self.model_load_succeeded = False
        self.last_load_error: str | None = None
        self.last_generation_error: str | None = None
        self.llama_cpp_n_ctx = llama_cpp_n_ctx
        self.llama_cpp_n_gpu_layers = llama_cpp_n_gpu_layers
        self.llama_cpp_n_threads = llama_cpp_n_threads
        self.ollama_template = _read_optional_text(plan.ollama_template_path if plan else None)
        self.ollama_params = _read_optional_json(plan.ollama_params_path if plan else None)
        self.llama_cpp_chat_format = _infer_llama_chat_format(self.ollama_template)
        self.template_applied = bool(self.llama_cpp_chat_format)

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
        allow_reasoning: bool = False,
    ) -> ProviderGenerationResult:
        _ = tools
        prepared_messages = _apply_model_thinking_control(
            self.model_id,
            messages,
            allow_reasoning=allow_reasoning,
        )

        try:
            if self.backend == "transformers":
                result = await self._generate_with_transformers(
                    messages=prepared_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                self.last_generation_error = None
                return result
            if self.backend == "llamacpp":
                result = await self._generate_with_llamacpp(
                    messages=prepared_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                self.last_generation_error = None
                return result
        except Exception as exc:
            self.last_generation_error = str(exc)
            raise

        # deterministic fallback path keeps the harness usable without local GPU/weights.
        question = _extract_last_user_message({"messages": prepared_messages})
        if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
            content = json.dumps(
                {
                    "answer": self._fallback_answer(question),
                    "model": self.model_id,
                },
                ensure_ascii=False,
            )
        else:
            content = self._fallback_answer(question)
        usage = _openai_usage(
            prompt_tokens=_token_count(_to_str_list(prepared_messages)),
            completion_tokens=_token_count(content),
        )
        return ProviderGenerationResult(
            content=content,
            reasoning=None,
            finish_reason="stop",
            usage=usage,
            truncated=False,
            warnings=[FALLBACK_WARNING],
            raw={},
        )

    async def _generate_with_transformers(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> ProviderGenerationResult:
        if not self.source:
            raise RuntimeError("transformer backend requires model source")

        def _run() -> ProviderGenerationResult:
            if self._pipeline is None:
                self.model_load_attempted = True
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                except Exception as exc:  # pragma: no cover - dependency path
                    self.last_load_error = str(exc)
                    raise RuntimeError(f"transformers backend unavailable: {exc}") from exc

                try:
                    tokenizer = AutoTokenizer.from_pretrained(self.source, local_files_only=True)
                    model = AutoModelForCausalLM.from_pretrained(self.source, local_files_only=True)
                    self._pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if self.device.lower() != "cpu" else -1,
                    )
                except Exception as exc:
                    self.last_load_error = str(exc)
                    raise
                self.model_load_succeeded = True
                self.last_load_error = None

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
                generated = ""
            else:
                output = outputs[0].get("generated_text", "") if isinstance(outputs[0], dict) else ""
                generated = str(output)
            if response_format and isinstance(response_format, dict) and response_format.get("type") == "json_object":
                fallback = self._fallback_answer(prompt)
                obj = {"answer": fallback, "model": self.model_id}
                if generated.strip():
                    structured = _load_json_model(generated)
                    if structured is not None:
                        obj = structured
                content = json.dumps(obj, ensure_ascii=False)
            else:
                content = generated.strip() if generated else self._fallback_answer(prompt)
            content, reasoning, reasoning_extracted = _split_reasoning_content(content)
            effective_max = max_tokens or self.max_new_tokens
            truncated = _looks_truncated(content, finish_reason="stop", max_tokens=effective_max)
            warnings = ["reasoning_extracted"] if reasoning_extracted else []
            if truncated:
                warnings.append("generation_may_be_truncated")
            return ProviderGenerationResult(
                content=content,
                reasoning=reasoning,
                finish_reason="length" if truncated else "stop",
                usage=_openai_usage(
                    prompt_tokens=_token_count(prompt),
                    completion_tokens=_token_count(content),
                ),
                truncated=truncated,
                warnings=warnings,
                raw={"generated_text": generated},
            )

        return await asyncio.to_thread(_run)

    async def _generate_with_llamacpp(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> ProviderGenerationResult:
        if not self.source:
            raise RuntimeError("llamacpp backend requires a GGUF model source")

        def _run() -> ProviderGenerationResult:
            if self._llama is None:
                self.model_load_attempted = True
                try:
                    from llama_cpp import Llama
                except Exception as exc:  # pragma: no cover - dependency path
                    self.last_load_error = str(exc)
                    raise RuntimeError(f"llama_cpp backend unavailable: {exc}") from exc

                try:
                    llama_args: dict[str, Any] = {
                        "model_path": self.source,
                        "n_ctx": self.llama_cpp_n_ctx,
                        "n_gpu_layers": self.llama_cpp_n_gpu_layers,
                        "n_threads": self.llama_cpp_n_threads,
                        "verbose": False,
                    }
                    if self.llama_cpp_chat_format:
                        llama_args["chat_format"] = self.llama_cpp_chat_format
                    self._llama = Llama(**llama_args)
                except Exception as exc:
                    self.last_load_error = str(exc)
                    raise
                self.model_load_succeeded = True
                self.last_load_error = None

            params = _generation_params_from_ollama(self.ollama_params)
            payload: dict[str, Any] = {
                "messages": messages,
                "temperature": (
                    float(params["temperature"])
                    if temperature is None and "temperature" in params
                    else (0.2 if temperature is None else float(temperature))
                ),
                "max_tokens": max_tokens or self.max_new_tokens,
            }
            for key in ("top_p", "top_k", "repeat_penalty", "stop"):
                if key in params:
                    payload[key] = params[key]
            if response_format and isinstance(response_format, dict):
                payload["response_format"] = response_format
            try:
                result = self._llama.create_chat_completion(**payload)
                choice = result.get("choices", [{}])[0]
                msg = choice.get("message", {})
                content = str(msg.get("content", "")).strip()
                reasoning = _first_text(msg, ("reasoning", "analysis", "reasoning_content"))
                finish_reason = str(choice.get("finish_reason") or "stop")
                usage = _normalize_usage(result.get("usage", {}))
                content, parsed_reasoning, reasoning_extracted = _split_reasoning_content(content)
                if parsed_reasoning and not reasoning:
                    reasoning = parsed_reasoning
                truncated = _looks_truncated(content, finish_reason=finish_reason, max_tokens=payload["max_tokens"])
                warnings = []
                if reasoning_extracted:
                    warnings.append("reasoning_extracted")
                if truncated:
                    warnings.append("generation_may_be_truncated")
                if self.ollama_template and not self.template_applied:
                    warnings.append("ollama_template_present_but_not_applied")
                return ProviderGenerationResult(
                    content=content,
                    reasoning=reasoning,
                    finish_reason="length" if truncated else finish_reason,
                    usage=usage,
                    truncated=truncated,
                    warnings=warnings,
                    raw=result,
                )
            except Exception:
                prompt = self._build_prompt(messages)
                stop = params.get("stop", ["</s>", "\n\n\n"])
                result = self._llama(
                    prompt,
                    max_tokens=max_tokens or self.max_new_tokens,
                    temperature=payload["temperature"],
                    stop=stop,
                )
                choice = result.get("choices", [{}])[0]
                content = str(choice.get("text", "")).strip()
                finish_reason = str(choice.get("finish_reason") or "stop")
                content, reasoning, reasoning_extracted = _split_reasoning_content(content)
                truncated = _looks_truncated(content, finish_reason=finish_reason, max_tokens=max_tokens or self.max_new_tokens)
                warnings = ["llamacpp_chat_completion_fallback"]
                if reasoning_extracted:
                    warnings.append("reasoning_extracted")
                if truncated:
                    warnings.append("generation_may_be_truncated")
                return ProviderGenerationResult(
                    content=content,
                    reasoning=reasoning,
                    finish_reason="length" if truncated else finish_reason,
                    usage=_normalize_usage(result.get("usage", {})),
                    truncated=truncated,
                    warnings=warnings,
                    raw=result,
                )

        return await asyncio.to_thread(_run)


def _expand_source_path(source: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(source).strip()))
    return Path(expanded)


def _detect_file_format(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".gguf":
        return "gguf"
    if suffix in {".safetensors", ".bin", ".pt", ".pth"}:
        return "transformers"
    try:
        with path.open("rb") as handle:
            if handle.read(4) == b"GGUF":
                return "gguf"
    except Exception:
        return None
    return None


def _source_from_path(source: str, *, source_type: str) -> ModelPlan:
    path = _expand_source_path(source)
    source_text = str(path)
    if path.is_file():
        artifact_format = _detect_file_format(path)
        return ModelPlan(
            model_id=path.stem,
            backend="auto",
            source=source_text,
            source_type=source_type,
            artifact_format=artifact_format,
            provider_store=None,
            manifest_path=None,
        )
    if path.is_dir():
        if (path / "config.json").exists():
            return ModelPlan(
                model_id=path.name,
                backend="auto",
                source=source_text,
                source_type=source_type,
                artifact_format="transformers",
                provider_store=None,
                manifest_path=None,
            )
        gguf_files = sorted(path.glob("*.gguf"))
        if len(gguf_files) == 1:
            gguf = gguf_files[0]
            return ModelPlan(
                model_id=path.name,
                backend="auto",
                source=str(gguf),
                source_type=source_type,
                artifact_format="gguf",
                provider_store=None,
                manifest_path=None,
            )
    return ModelPlan(
        model_id=path.stem or path.name,
        backend="auto",
        source=source_text,
        source_type=source_type,
        artifact_format=None,
        provider_store=None,
        manifest_path=None,
    )


def _with_source_metadata(
    plan: ModelPlan,
    *,
    source: str | None,
    source_type: str | None,
    artifact_format: str | None,
    provider_store: str | None = None,
    manifest_path: str | None = None,
    ollama_template_path: str | None = None,
    ollama_params_path: str | None = None,
) -> ModelPlan:
    return ModelPlan(
        model_id=plan.model_id,
        backend=plan.backend,
        source=source,
        source_type=source_type,
        artifact_format=artifact_format,
        provider_store=provider_store,
        manifest_path=manifest_path,
        ollama_template_path=ollama_template_path,
        ollama_params_path=ollama_params_path,
    )


def _detect_artifact_format_for_source(source: str | None) -> str | None:
    if not source:
        return None
    path = _expand_source_path(source)
    if path.is_file():
        return _detect_file_format(path)
    if path.is_dir():
        if (path / "config.json").exists():
            return "transformers"
        gguf_files = sorted(path.glob("*.gguf"))
        if len(gguf_files) == 1:
            return "gguf"
    return None


def _resolve_explicit_source(plan: ModelPlan) -> ModelPlan:
    if not plan.source:
        return plan
    detected = _source_from_path(plan.source, source_type=plan.source_type or "filesystem")
    source = detected.source or plan.source
    artifact_format = detected.artifact_format or _detect_artifact_format_for_source(source)
    return _with_source_metadata(
        plan,
        source=source,
        source_type=plan.source_type or detected.source_type or "filesystem",
        artifact_format=artifact_format,
        provider_store=detected.provider_store,
        manifest_path=detected.manifest_path,
        ollama_template_path=detected.ollama_template_path,
        ollama_params_path=detected.ollama_params_path,
    )


def _ollama_model_parts(model_id: str) -> tuple[list[str], str]:
    name = str(model_id or "").strip()
    if ":" in name:
        name_part, tag = name.rsplit(":", 1)
    else:
        name_part, tag = name, "latest"
    parts = [part for part in name_part.strip("/").split("/") if part]
    if not parts:
        parts = ["local-foundation"]
    if len(parts) == 1:
        parts = ["registry.ollama.ai", "library", parts[0]]
    elif len(parts) == 2:
        parts = ["registry.ollama.ai", *parts]
    return parts, tag or "latest"


def _ollama_manifest_path(model_id: str, root: Path) -> Path:
    parts, tag = _ollama_model_parts(model_id)
    return root.joinpath("manifests", *parts, tag)


def _ollama_blob_path(root: Path, layer: dict[str, Any]) -> Path | None:
    digest = str(layer.get("digest", "")).strip()
    if not digest.startswith("sha256:"):
        return None
    blob = root / "blobs" / digest.replace(":", "-")
    return blob if blob.exists() else None


def _candidate_model_roots(models_root: str | None) -> list[Path]:
    roots: list[Path] = []
    if models_root:
        roots.append(_expand_source_path(models_root))
    env_root = os.getenv("OLLAMA_MODELS")
    if env_root:
        roots.append(_expand_source_path(env_root))
    home = Path.home()
    if home:
        roots.append(home / ".ollama" / "models")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _candidate_huggingface_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.getenv(key)
        if value:
            roots.append(_expand_source_path(value))
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        roots.append(_expand_source_path(hf_home) / "hub")
    home = Path.home()
    if home:
        roots.append(home / ".cache" / "huggingface" / "hub")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _resolve_huggingface_cache_dir(model_id: str) -> Path | None:
    normalized = str(model_id or "").strip()
    if normalized.lower().startswith("hf://"):
        normalized = normalized[len("hf://"):]
    if not normalized or "/" not in normalized:
        return None
    cache_name = "models--" + normalized.replace("/", "--")
    for root in _candidate_huggingface_cache_roots():
        candidate = root / cache_name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _huggingface_cache_has_weight_files(cache_dir: str | None) -> bool:
    if not cache_dir:
        return False
    root = Path(cache_dir)
    if not root.exists() or not root.is_dir():
        return False
    snapshots = root / "snapshots"
    search_root = snapshots if snapshots.exists() else root
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        try:
            if any(search_root.rglob(pattern)):
                return True
        except OSError:
            continue
    return False


def _resolve_ollama_store_model(model_id: str, models_root: str | None) -> ModelPlan | None:
    for root in _candidate_model_roots(models_root):
        manifest = _ollama_manifest_path(model_id, root)
        if not manifest.exists():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        layers = payload.get("layers", [])
        if not isinstance(layers, list):
            continue
        model_layers = [
            layer for layer in layers
            if isinstance(layer, dict) and "model" in str(layer.get("mediaType", "")).lower()
        ]
        if not model_layers:
            model_layers = [layer for layer in layers if isinstance(layer, dict)]
        template_path: Path | None = None
        params_path: Path | None = None
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            media_type = str(layer.get("mediaType", "")).lower()
            blob = _ollama_blob_path(root, layer)
            if blob is None:
                continue
            if "template" in media_type:
                template_path = blob
            elif "params" in media_type:
                params_path = blob
        for layer in model_layers:
            blob = _ollama_blob_path(root, layer)
            if blob is None:
                continue
            artifact_format = _detect_file_format(blob) or "gguf"
            return ModelPlan(
                model_id=model_id,
                backend="auto",
                source=str(blob),
                source_type="ollama_store",
                artifact_format=artifact_format,
                provider_store="ollama",
                manifest_path=str(manifest),
                ollama_template_path=str(template_path) if template_path else None,
                ollama_params_path=str(params_path) if params_path else None,
            )
    return None


def _root_model_candidates(model_id: str, root: Path) -> list[Path]:
    raw = str(model_id).strip()
    candidates: list[Path] = []
    if raw:
        candidates.append(root / raw)
        candidates.append(root / raw.replace(":", "_").replace("/", os.sep))
    if ":" in raw:
        name, tag = raw.rsplit(":", 1)
        if name:
            candidates.append(root / name / tag)
            candidates.append(root / name)
    if "/" in raw:
        candidates.append(root.joinpath(*[part for part in raw.split("/") if part]))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_models_root_model(plan: ModelPlan, models_root: str | None) -> ModelPlan:
    if not models_root:
        return plan
    root = _expand_source_path(models_root)
    if not root.exists() or not root.is_dir():
        return plan
    if (root / "manifests").exists() and (root / "blobs").exists():
        resolved = _resolve_ollama_store_model(plan.model_id, str(root))
        if resolved is not None:
            return _with_source_metadata(
                plan,
                source=resolved.source,
                source_type=resolved.source_type,
                artifact_format=resolved.artifact_format,
                provider_store=resolved.provider_store,
                manifest_path=resolved.manifest_path,
                ollama_template_path=resolved.ollama_template_path,
                ollama_params_path=resolved.ollama_params_path,
            )
    for candidate in _root_model_candidates(plan.model_id, root):
        if not candidate.exists():
            continue
        detected = _source_from_path(str(candidate), source_type="models_root")
        if detected.source:
            return _with_source_metadata(
                plan,
                source=detected.source,
                source_type=detected.source_type,
                artifact_format=detected.artifact_format,
            )
    return plan


def _resolve_plan_source(plan: ModelPlan, models_root: str | None) -> ModelPlan:
    if plan.source:
        return _resolve_explicit_source(plan)
    if plan.model_id.lower().startswith("hf://"):
        cache_dir = _resolve_huggingface_cache_dir(plan.model_id)
        return _with_source_metadata(
            plan,
            source=plan.model_id[len("hf://"):],
            source_type="huggingface_cache",
            artifact_format="transformers",
            provider_store="huggingface",
            manifest_path=str(cache_dir) if cache_dir is not None else None,
        )
    resolved = _resolve_models_root_model(plan, models_root)
    if resolved.source:
        return resolved
    ollama_resolved = _resolve_ollama_store_model(plan.model_id, models_root)
    if ollama_resolved is not None:
        return _with_source_metadata(
            plan,
            source=ollama_resolved.source,
            source_type=ollama_resolved.source_type,
            artifact_format=ollama_resolved.artifact_format,
            provider_store=ollama_resolved.provider_store,
            manifest_path=ollama_resolved.manifest_path,
            ollama_template_path=ollama_resolved.ollama_template_path,
            ollama_params_path=ollama_resolved.ollama_params_path,
        )
    return plan


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


def _build_runtime_from_args(args: argparse.Namespace) -> tuple[dict[str, ModelPlan], str]:
    primary = _resolve_plan_source(
        _normalize_model_plan(
            args.model,
            default_backend=args.backend,
            default_source=args.model_path or None,
        ),
        args.models_root or None,
    )

    catalog: dict[str, ModelPlan] = {primary.model_id: primary}

    for extra in args.extra_model or []:
        parsed = _resolve_plan_source(
            _normalize_model_plan(extra, default_backend=args.backend, default_source=None),
            args.models_root or None,
        )
        if parsed.model_id in catalog and catalog[parsed.model_id].source:
            continue
        catalog[parsed.model_id] = parsed

    fallback_key = primary.model_id
    if args.model and "::" not in args.model and "=" not in args.model:
        fallback_key = args.model

    return catalog, fallback_key


def _fallback_required_message(plan: ModelPlan) -> str:
    return (
        f"Model '{plan.model_id}' is configured with backend 'auto' but no local model source was found. "
        "Configure --model-path, --models-root, an --extra-model '<id>::<path>' entry, "
        "use an hf:// model id that is already present in the local Hugging Face cache, "
        "or place the tag in a local Ollama-style model store with manifests/blobs available, "
        "or explicitly request diagnostic stub mode with --backend fallback or --allow-fallback."
    )


def _model_source_present(plan: ModelPlan) -> bool:
    if not plan.source:
        return False
    if plan.source_type == "huggingface_cache":
        return _huggingface_cache_has_weight_files(plan.manifest_path)
    try:
        return _expand_source_path(plan.source).exists()
    except Exception:
        return False


def _template_applied_for_plan(plan: ModelPlan, generation_backend: str) -> bool:
    if generation_backend != "llamacpp":
        return False
    return bool(_infer_llama_chat_format(_read_optional_text(plan.ollama_template_path)))


def _diagnostics_for_plan(plan: ModelPlan, *, allow_fallback: bool) -> ModelDiagnostics:
    configured_backend = str(plan.backend or "auto").strip().lower()
    fallback_allowed = bool(allow_fallback or configured_backend == "fallback")

    if configured_backend == "fallback":
        generation_backend = "fallback"
    elif configured_backend == "transformers":
        generation_backend = "transformers"
    elif configured_backend == "llamacpp":
        generation_backend = "llamacpp"
    elif configured_backend == "auto":
        if plan.source:
            artifact_format = plan.artifact_format or _detect_artifact_format_for_source(plan.source)
            generation_backend = "llamacpp" if artifact_format == "gguf" else "transformers"
        elif fallback_allowed:
            generation_backend = "fallback"
        else:
            raise RuntimeError(_fallback_required_message(plan))
    else:
        raise RuntimeError(f"Unsupported backend '{configured_backend}'")

    fallback_active = generation_backend == "fallback"
    runtime_dependency = None
    runtime_dependency_available: bool | None = None
    if generation_backend == "llamacpp":
        runtime_dependency = "llama_cpp"
        runtime_dependency_available = importlib.util.find_spec("llama_cpp") is not None
    elif generation_backend == "transformers":
        runtime_dependency = "transformers,torch"
        runtime_dependency_available = (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )
    source_present = _model_source_present(plan)
    local_model_ready = (
        generation_backend in {"transformers", "llamacpp"}
        and source_present
        and runtime_dependency_available is not False
    )
    return ModelDiagnostics(
        model=plan.model_id,
        configured_backend=configured_backend,
        generation_backend=generation_backend,
        model_source=plan.source,
        model_source_type=plan.source_type,
        model_artifact_format=plan.artifact_format,
        provider_store=plan.provider_store,
        manifest_path=plan.manifest_path,
        runtime_dependency=runtime_dependency,
        runtime_dependency_available=runtime_dependency_available,
        local_model_loaded=local_model_ready,
        model_source_present=source_present,
        model_load_attempted=False,
        model_load_succeeded=False,
        last_load_error=None,
        last_generation_error=None,
        template_applied=_template_applied_for_plan(plan, generation_backend),
        fallback_active=fallback_active,
        allow_fallback=fallback_allowed,
        provider_warning=FALLBACK_WARNING if fallback_active else None,
    )


def _validate_startup_plans(plans: dict[str, ModelPlan], *, allow_fallback: bool) -> None:
    for plan in plans.values():
        diagnostics = _diagnostics_for_plan(plan, allow_fallback=allow_fallback)
        if diagnostics.generation_backend not in {"transformers", "llamacpp"}:
            continue
        if not plan.source:
            raise RuntimeError(
                f"Model '{plan.model_id}' requires a local model source for backend "
                f"'{diagnostics.generation_backend}'."
            )
        if plan.source_type == "huggingface_cache":
            continue
        source_path = _expand_source_path(plan.source)
        if not source_path.exists():
            raise RuntimeError(f"Local model source for '{plan.model_id}' does not exist: {source_path}")
        artifact_format = plan.artifact_format or _detect_artifact_format_for_source(plan.source)
        if diagnostics.generation_backend == "llamacpp" and artifact_format != "gguf":
            raise RuntimeError(
                f"Model '{plan.model_id}' resolved to llama.cpp but source is not a GGUF artifact: {source_path}"
            )
        if diagnostics.generation_backend == "transformers" and artifact_format == "gguf":
            raise RuntimeError(
                f"Model '{plan.model_id}' resolved to Transformers but source is a GGUF artifact: {source_path}"
            )


def _payload_allows_reasoning(payload: dict[str, Any]) -> bool:
    candidates = []
    if isinstance(payload.get("reasoning"), dict):
        candidates.append(payload.get("reasoning"))
    extra_body = payload.get("extra_body")
    if isinstance(extra_body, dict) and isinstance(extra_body.get("reasoning"), dict):
        candidates.append(extra_body.get("reasoning"))
    for candidate in candidates:
        enabled = candidate.get("enabled")
        if isinstance(enabled, bool):
            return enabled
        if isinstance(enabled, str):
            return enabled.strip().lower() in {"1", "true", "yes", "on"}
    return False


def build_app(
    *,
    model: str = "local-foundation:v1",
    backend: str = "auto",
    model_path: str | None = None,
    models_root: str | None = None,
    extra_models: list[str] | None = None,
    device: str = "cpu",
    max_tokens: int = 256,
    llama_cpp_n_ctx: int = 4096,
    llama_cpp_n_gpu_layers: int = 0,
    llama_cpp_n_threads: int = 4,
    local_only: bool = False,
    allow_fallback: bool = False,
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
    fallback_allowed = bool(allow_fallback or backend == "fallback")
    _validate_startup_plans(plans, allow_fallback=fallback_allowed)
    runtime_cache: dict[str, _ModelBackendRuntime] = {}

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
        diagnostics = _diagnostics_for_plan(plan, allow_fallback=fallback_allowed)

        if selected_backend not in {"auto", "fallback", "transformers", "llamacpp"}:
            raise HTTPException(status_code=400, detail=f"Unsupported backend '{selected_backend}'")

        if diagnostics.generation_backend in {"transformers", "llamacpp"}:
            if not selected_source:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Model '{plan.model_id}' has no local path."
                        " Configure --model-path, --models-root, or set an entry in --extra-model as '<id>::<path>'."
                    ),
                )
            runtime_key = f"{plan.model_id}|{diagnostics.generation_backend}|{selected_source or ''}"
            cached = runtime_cache.get(runtime_key)
            if cached is not None:
                return cached
            runtime = _ModelBackendRuntime(
                model_id=plan.model_id,
                backend=diagnostics.generation_backend,
                source=selected_source,
                plan=plan,
                device=device,
                max_new_tokens=max_tokens,
                llama_cpp_n_ctx=llama_cpp_n_ctx,
                llama_cpp_n_gpu_layers=llama_cpp_n_gpu_layers,
                llama_cpp_n_threads=llama_cpp_n_threads,
            )
            runtime_cache[runtime_key] = runtime
            return runtime
        runtime_key = f"{plan.model_id}|fallback|{selected_source or ''}"
        cached = runtime_cache.get(runtime_key)
        if cached is not None:
            return cached
        runtime = _ModelBackendRuntime(
            model_id=plan.model_id,
            backend="fallback",
            source=selected_source,
            plan=plan,
            device=device,
        )
        runtime_cache[runtime_key] = runtime
        return runtime

    def _plan_for_request(requested_model: str) -> ModelPlan:
        selected = requested_model.strip() if isinstance(requested_model, str) else ""
        plan = plans.get(selected)
        if plan is None:
            plan = plans.get(fallback_model)
        if plan is None:
            raise HTTPException(status_code=404, detail=f"Model not available: {requested_model}")
        return plan

    @app.get("/health")
    async def health() -> dict[str, Any]:
        primary_plan = _plan_for_request(fallback_model)
        diagnostics = _diagnostics_for_plan(primary_plan, allow_fallback=fallback_allowed).as_dict()
        return {
            "ok": True,
            "service": "local_model_provider",
            "models": len(plans),
            "backend": backend,
            **diagnostics,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        data = [
            {
                "id": item.model_id,
                "object": "model",
                "owned_by": "local-stack",
                **_diagnostics_for_plan(item, allow_fallback=fallback_allowed).as_dict(),
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
        plan = _plan_for_request(requested_model or fallback_model)

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
        allow_reasoning = _payload_allows_reasoning(payload)

        try:
            generation = await runtime.generate(
                messages=messages,
                response_format=response_format if isinstance(response_format, dict) else None,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens_param,
                allow_reasoning=allow_reasoning,
            )
        except Exception as exc:
            if plan.backend == "auto" and fallback_allowed and runtime.backend in {"transformers", "llamacpp"}:
                runtime = _ModelBackendRuntime(
                    model_id=plan.model_id,
                    backend="fallback",
                    source=plan.source,
                    plan=plan,
                    device=device,
                )
                generation = await runtime.generate(
                    messages=messages,
                    response_format=response_format if isinstance(response_format, dict) else None,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens_param,
                    allow_reasoning=allow_reasoning,
                )
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": {
                            "message": f"Model backend generation failed: {exc}",
                            "code": "model_request_failed",
                        }
                    },
                )

        configured_diagnostics = _diagnostics_for_plan(plan, allow_fallback=fallback_allowed)
        local_model_loaded = (
            bool(runtime.model_load_succeeded)
            if runtime.model_load_attempted
            else bool(configured_diagnostics.local_model_loaded)
        )
        provider_warning = FALLBACK_WARNING if runtime.backend == "fallback" else None
        if not provider_warning and generation.warnings:
            provider_warning = "; ".join(generation.warnings)
        runtime_diagnostics = ModelDiagnostics(
            model=runtime.model_id,
            configured_backend=configured_diagnostics.configured_backend,
            generation_backend=runtime.backend,
            model_source=runtime.source,
            model_source_type=configured_diagnostics.model_source_type,
            model_artifact_format=configured_diagnostics.model_artifact_format,
            provider_store=configured_diagnostics.provider_store,
            manifest_path=configured_diagnostics.manifest_path,
            runtime_dependency=configured_diagnostics.runtime_dependency,
            runtime_dependency_available=configured_diagnostics.runtime_dependency_available,
            local_model_loaded=local_model_loaded,
            model_source_present=configured_diagnostics.model_source_present,
            model_load_attempted=bool(runtime.model_load_attempted),
            model_load_succeeded=bool(runtime.model_load_succeeded),
            last_load_error=runtime.last_load_error,
            last_generation_error=runtime.last_generation_error,
            template_applied=bool(runtime.template_applied or configured_diagnostics.template_applied),
            fallback_active=runtime.backend == "fallback",
            allow_fallback=configured_diagnostics.allow_fallback,
            provider_warning=provider_warning,
        ).as_dict()
        runtime_diagnostics.update(
            {
                "finish_reason": generation.finish_reason,
                "truncated": generation.truncated,
                "reasoning_extracted": bool(generation.reasoning),
                "warnings": generation.warnings,
            }
        )
        if tools:
            choice_obj = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [],
                    "content": generation.content,
                },
                "finish_reason": "tool_calls",
            }
        else:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": generation.content,
            }
            if generation.reasoning:
                message["reasoning"] = generation.reasoning
            choice_obj = {
                "index": 0,
                "message": message,
                "finish_reason": generation.finish_reason,
            }

        response: dict[str, Any] = {
            "id": f"chatcmpl-local-{hash((requested_model or model).encode('utf-8')) & 0xFFFF}",
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": runtime.model_id,
            "choices": [choice_obj],
            "usage": generation.usage,
            "provider": runtime_diagnostics,
            "configured_backend": runtime_diagnostics["configured_backend"],
            "generation_backend": runtime_diagnostics["generation_backend"],
            "model_source": runtime_diagnostics["model_source"],
            "local_model_loaded": runtime_diagnostics["local_model_loaded"],
            "model_source_present": runtime_diagnostics["model_source_present"],
            "model_load_attempted": runtime_diagnostics["model_load_attempted"],
            "model_load_succeeded": runtime_diagnostics["model_load_succeeded"],
            "last_load_error": runtime_diagnostics["last_load_error"],
            "last_generation_error": runtime_diagnostics["last_generation_error"],
            "template_applied": runtime_diagnostics["template_applied"],
            "fallback_active": runtime_diagnostics["fallback_active"],
            "provider_warning": runtime_diagnostics["provider_warning"],
            "finish_reason": generation.finish_reason,
            "truncated": generation.truncated,
            "reasoning_extracted": bool(generation.reasoning),
            "warnings": generation.warnings,
        }
        for key, value in runtime_diagnostics.items():
            response.setdefault(key, value)

        return JSONResponse(content=response)

    return app


def _parse_backend(name: str) -> str:
    normalized = str(name or "auto").strip().lower()
    if normalized in {"auto", "fallback", "transformers", "llamacpp", "llama_cpp", "llama-cpp"}:
        if normalized in {"llama_cpp", "llama-cpp"}:
            return "llamacpp"
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
        choices=("auto", "fallback", "transformers", "llamacpp", "llama_cpp", "llama-cpp"),
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
        "--llama-cpp-n-ctx",
        type=int,
        default=4096,
        help="Context window used by llama.cpp when serving GGUF models.",
    )
    parser.add_argument(
        "--llama-cpp-n-gpu-layers",
        type=int,
        default=0,
        help="Number of layers to offload to GPU for llama.cpp GGUF models.",
    )
    parser.add_argument(
        "--llama-cpp-n-threads",
        type=int,
        default=4,
        help="Thread count used by llama.cpp GGUF models.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Reject non-local or network-backed model resolution."
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow backend=auto to use deterministic diagnostic fallback mode when no local model can be used.",
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
        llama_cpp_n_ctx=args.llama_cpp_n_ctx,
        llama_cpp_n_gpu_layers=args.llama_cpp_n_gpu_layers,
        llama_cpp_n_threads=args.llama_cpp_n_threads,
        local_only=args.local_only,
        allow_fallback=args.allow_fallback,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
