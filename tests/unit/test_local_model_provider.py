from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.local_model_provider import _ModelBackendRuntime, build_app


class _FakeLlama:
    instances: list["_FakeLlama"] = []
    calls: list[dict[str, object]] = []
    response_content = "Final answer"
    finish_reason = "stop"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def create_chat_completion(self, **payload: object) -> dict[str, object]:
        self.__class__.calls.append(payload)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": self.__class__.response_content},
                    "finish_reason": self.__class__.finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }


def _install_fake_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeLlama.instances = []
    _FakeLlama.calls = []
    _FakeLlama.response_content = "Final answer"
    _FakeLlama.finish_reason = "stop"
    module = types.ModuleType("llama_cpp")
    module.__spec__ = importlib.machinery.ModuleSpec("llama_cpp", loader=None)
    module.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", module)


def _write_ollama_store_with_template(root: Path) -> tuple[Path, Path, Path]:
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "4b"
    model_blob = root / "blobs" / "sha256-deadbeef"
    template_blob = root / "blobs" / "sha256-template"
    params_blob = root / "blobs" / "sha256-params"
    manifest.parent.mkdir(parents=True)
    model_blob.parent.mkdir(parents=True)
    model_blob.write_bytes(b"GGUFfake")
    template_blob.write_text("<|im_start|>user\n{{ .Content }}<|im_end|>\n<|im_start|>assistant", encoding="utf-8")
    params_blob.write_text(
        '{"repeat_penalty":1,"stop":["<|im_start|>","<|im_end|>"],"temperature":0.6,"top_k":20,"top_p":0.95}',
        encoding="utf-8",
    )
    manifest.write_text(
        (
            '{"layers":['
            '{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:deadbeef"},'
            '{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:template"},'
            '{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:params"}'
            ']}'
        ),
        encoding="utf-8",
    )
    return model_blob, template_blob, params_blob


def test_local_model_provider_builds_and_serves_health() -> None:
    app = build_app(model="local-foundation:v1", backend="fallback", max_tokens=32)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["service"] == "local_model_provider"
    assert payload["generation_backend"] == "fallback"
    assert payload["fallback_active"] is True
    assert payload["provider_warning"]


def test_local_model_provider_builds_models_payload() -> None:
    app = build_app(model="local-foundation:v1", backend="fallback")
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"] and isinstance(payload["data"], list)
    assert payload["data"][0]["id"] == "local-foundation:v1"
    assert payload["data"][0]["generation_backend"] == "fallback"
    assert payload["data"][0]["fallback_active"] is True


def test_local_model_provider_auto_without_model_source_requires_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "missing-ollama-store"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    with pytest.raises(RuntimeError, match="no local model source"):
        build_app(model="local-foundation:v1", backend="auto")


def test_local_model_provider_auto_resolves_gguf_model_path(tmp_path: Path) -> None:
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"GGUFfake")

    app = build_app(model="tiny:v1", backend="auto", model_path=str(gguf))
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["generation_backend"] == "llamacpp"
    assert payload["model_source"] == str(gguf)
    assert payload["model_source_type"] == "filesystem"
    assert payload["model_artifact_format"] == "gguf"
    assert payload["provider_store"] is None
    assert payload["runtime_dependency"] == "llama_cpp"
    assert isinstance(payload["runtime_dependency_available"], bool)
    assert payload["fallback_active"] is False


def test_local_model_provider_auto_resolves_downloaded_ollama_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "ollama-models"
    manifest = models_root / "manifests" / "registry.ollama.ai" / "library" / "qwen3" / "4b"
    blob = models_root / "blobs" / "sha256-deadbeef"
    manifest.parent.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    manifest.write_text(
        '{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:deadbeef"}]}',
        encoding="utf-8",
    )
    blob.write_bytes(b"GGUFfake")
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "unused"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    app = build_app(model="qwen3:4b", backend="auto", models_root=str(models_root))
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["generation_backend"] == "llamacpp"
    assert health["model_source"] == str(blob)
    assert health["model_source_type"] == "ollama_store"
    assert health["model_artifact_format"] == "gguf"
    assert health["provider_store"] == "ollama"
    assert health["manifest_path"] == str(manifest)
    assert health["runtime_dependency"] == "llama_cpp"

    catalog = client.get("/v1/models").json()
    entry = catalog["data"][0]
    assert entry["id"] == "qwen3:4b"
    assert entry["generation_backend"] == "llamacpp"
    assert entry["provider_store"] == "ollama"


def test_local_model_provider_uses_ollama_template_params_and_no_think(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_llama_cpp(monkeypatch)
    models_root = tmp_path / "ollama-models"
    model_blob, _template_blob, _params_blob = _write_ollama_store_with_template(models_root)
    _FakeLlama.response_content = "<think>hidden chain</think>\n\nClean final answer"

    app = build_app(model="qwen3:4b", backend="auto", models_root=str(models_root))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3:4b", "messages": [{"role": "user", "content": "Top 3 N64 games?"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Clean final answer"
    assert payload["choices"][0]["message"]["reasoning"] == "hidden chain"
    assert payload["reasoning_extracted"] is True
    assert payload["template_applied"] is True
    assert payload["model_load_attempted"] is True
    assert payload["model_load_succeeded"] is True
    assert payload["model_source"] == str(model_blob)
    assert _FakeLlama.instances[0].kwargs["chat_format"] == "chatml"
    request_payload = _FakeLlama.calls[0]
    assert request_payload["temperature"] == 0.6
    assert request_payload["top_p"] == 0.95
    assert request_payload["top_k"] == 20
    assert request_payload["repeat_penalty"] == 1.0
    assert request_payload["stop"] == ["<|im_start|>", "<|im_end|>"]
    assert request_payload["messages"][-1]["content"].endswith("/no_think")


def test_local_model_provider_reasoning_request_uses_qwen_think(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_llama_cpp(monkeypatch)
    models_root = tmp_path / "ollama-models"
    _write_ollama_store_with_template(models_root)

    app = build_app(model="qwen3:4b", backend="auto", models_root=str(models_root))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3:4b",
            "messages": [{"role": "user", "content": "Think this through"}],
            "extra_body": {"reasoning": {"enabled": True}},
        },
    )

    assert response.status_code == 200
    assert _FakeLlama.calls[0]["messages"][-1]["content"].endswith("/think")


def test_local_model_provider_reports_truncation_from_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_llama_cpp(monkeypatch)
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"GGUFfake")
    _FakeLlama.response_content = "Partial answer"
    _FakeLlama.finish_reason = "length"

    app = build_app(model="tiny:v1", backend="auto", model_path=str(gguf))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "tiny:v1", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["finish_reason"] == "length"
    assert payload["truncated"] is True
    assert "generation_may_be_truncated" in payload["warnings"]


def test_local_model_provider_auto_resolves_hf_cache_model_id() -> None:
    app = build_app(model="hf://meta-llama/Llama-3.1-8B-Instruct", backend="auto")
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["generation_backend"] == "transformers"
    assert payload["model_source"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert payload["model_source_type"] == "huggingface_cache"
    assert payload["model_artifact_format"] == "transformers"
    assert payload["provider_store"] == "huggingface"
    assert payload["runtime_dependency"] == "transformers,torch"
    assert isinstance(payload["runtime_dependency_available"], bool)
    assert payload["fallback_active"] is False


def test_local_model_provider_auto_llamacpp_failure_does_not_fallback_without_allow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"GGUFfake")

    async def _raise_load_failure(self: _ModelBackendRuntime, **_: object) -> str:
        raise RuntimeError("llama.cpp load failed")

    monkeypatch.setattr(_ModelBackendRuntime, "_generate_with_llamacpp", _raise_load_failure)
    app = build_app(model="tiny:v1", backend="auto", model_path=str(gguf))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "tiny:v1", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_request_failed"


def test_local_model_provider_auto_llamacpp_failure_can_explicitly_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gguf = tmp_path / "tiny.gguf"
    gguf.write_bytes(b"GGUFfake")

    async def _raise_load_failure(self: _ModelBackendRuntime, **_: object) -> str:
        raise RuntimeError("llama.cpp load failed")

    monkeypatch.setattr(_ModelBackendRuntime, "_generate_with_llamacpp", _raise_load_failure)
    app = build_app(model="tiny:v1", backend="auto", model_path=str(gguf), allow_fallback=True)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "tiny:v1", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "I received your question: hello"
    assert payload["provider"]["generation_backend"] == "fallback"
    assert payload["provider"]["fallback_active"] is True


def test_local_model_provider_auto_allow_fallback_serves_diagnostic_stub() -> None:
    app = build_app(model="local-foundation:v1", backend="auto", allow_fallback=True)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "local-foundation:v1", "messages": [{"role": "user", "content": "What are the top 3 N64 games of all time?"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "I received your question: What are the top 3 N64 games of all time?"
    assert payload["provider"]["configured_backend"] == "auto"
    assert payload["provider"]["generation_backend"] == "fallback"
    assert payload["provider"]["fallback_active"] is True
    assert payload["provider_warning"]


def test_local_model_provider_auto_transformer_failure_does_not_fallback_without_allow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    async def _raise_load_failure(self: _ModelBackendRuntime, **_: object) -> str:
        raise RuntimeError("transformers load failed")

    monkeypatch.setattr(_ModelBackendRuntime, "_generate_with_transformers", _raise_load_failure)
    app = build_app(model="local-foundation:v1", backend="auto", model_path=str(tmp_path))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "local-foundation:v1", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_request_failed"


def test_local_model_provider_auto_transformer_failure_can_explicitly_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    async def _raise_load_failure(self: _ModelBackendRuntime, **_: object) -> str:
        raise RuntimeError("transformers load failed")

    monkeypatch.setattr(_ModelBackendRuntime, "_generate_with_transformers", _raise_load_failure)
    app = build_app(model="local-foundation:v1", backend="auto", model_path=str(tmp_path), allow_fallback=True)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "local-foundation:v1", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "I received your question: hello"
    assert payload["provider"]["generation_backend"] == "fallback"
    assert payload["provider"]["fallback_active"] is True
