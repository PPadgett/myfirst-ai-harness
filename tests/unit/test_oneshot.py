from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from harness.oneshot import (
    build_chat_payload,
    validate_runtime_backend,
    resolve_model,
    runtime_backend_context,
    validate_model_override,
    validate_oneshot_args,
    wait_for_http_ready,
)


def _write_config(path: Path, model: str = "qwen2.5:7b", backend: str = "openai") -> str:
    content = (
        f"backend:\n"
        f"  name: {backend}\n"
        f"  base_url: \"http://127.0.0.1:11434/v1\"\n"
        "  api_key: null\n"
        "  timeout_seconds: 120\n"
        f"\nmodel: \"{model}\"\n"
    )
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return str(path)


def test_resolve_model_prefers_config_default(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml")
    assert resolve_model(config_path, None) == "qwen2.5:7b"


def test_resolve_model_prefers_explicit_override(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", model="qwen2.5:7b")
    assert resolve_model(config_path, "custom-model:v1") == "custom-model:v1"


def test_build_chat_payload_omits_model_by_default() -> None:
    payload = build_chat_payload("What is the weather today?")
    assert payload["messages"] == [{"role": "user", "content": "What is the weather today?"}]
    assert "model" not in payload


def test_build_chat_payload_includes_explicit_model_only() -> None:
    payload = build_chat_payload("What is the weather today?", model_override="hf-like-test")
    assert payload["messages"][0]["content"] == "What is the weather today?"
    assert payload["model"] == "hf-like-test"


def test_invalid_hardcoded_model_is_rejected_before_request_path() -> None:
    with pytest.raises(ValueError, match="hardcoded model override"):
        resolve_model(None, "hf://meta-llama/Llama-3.1-8B-Instruct")


def test_validate_model_override_empty_is_treated_as_none() -> None:
    assert validate_model_override("") is None
    assert validate_model_override("   ") is None


def test_validate_oneshot_args_enforces_timeouts_and_question() -> None:
    validate_oneshot_args(
        mode="runtime",
        question="hello",
        host="127.0.0.1",
        port=8080,
        startup_timeout_seconds=1,
        request_timeout_seconds=1,
    )
    with pytest.raises(ValueError, match="question is required"):
        validate_oneshot_args(
            mode="runtime",
            question="",
            host="127.0.0.1",
            port=8080,
            startup_timeout_seconds=1,
            request_timeout_seconds=1,
        )


def test_validate_runtime_backend_rejects_unreachable_catalog_endpoint(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", backend="openai")

    def _raise(_: str, timeout: float) -> object:
        request = httpx.Request("GET", "http://127.0.0.1:11434/v1/models")
        raise httpx.RequestError("connection failed", request=request)

    monkeypatch.setattr("harness.oneshot.httpx.get", _raise)

    with pytest.raises(RuntimeError, match="model_backend_unavailable") as exc_info:
        validate_runtime_backend(config_path, timeout_seconds=0.5, expected_model="qwen2.5:7b")

    assert "error_code=model_backend_unavailable" in str(exc_info.value)
    assert "backend_url=http://127.0.0.1:11434/v1/models" in str(exc_info.value)
    assert "expected_model=qwen2.5:7b" in str(exc_info.value)


def test_validate_runtime_backend_can_be_skipped(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", backend="openai")

    def _raise(_: str, timeout: float) -> object:
        request = httpx.Request("GET", "http://127.0.0.1:11434/v1/models")
        raise httpx.RequestError("connection failed", request=request)

    monkeypatch.setattr("harness.oneshot.httpx.get", _raise)
    validate_runtime_backend(
        config_path,
        timeout_seconds=0.5,
        skip_backend_check=True,
    )


def test_validate_runtime_backend_normalizes_v1_base_url(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", backend="openai")

    class _Response:
        status_code = 200

    def _assert_url(url: str, timeout: float) -> object:
        assert url == "http://127.0.0.1:11434/v1/models"
        return _Response()

    monkeypatch.setattr("harness.oneshot.httpx.get", _assert_url)
    validate_runtime_backend(config_path, timeout_seconds=0.5)


def test_validate_runtime_backend_accepts_config_model_when_present_in_catalog(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", model="qwen2.5:7b", backend="openai")

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "data": [
                    {"id": "gpt-oss:latest"},
                    {"id": "qwen2.5:7b"},
                ]
            }

    def _assert_url(url: str, timeout: float) -> object:
        assert url == "http://127.0.0.1:11434/v1/models"
        return _Response()

    monkeypatch.setattr("harness.oneshot.httpx.get", _assert_url)
    validate_runtime_backend(config_path, timeout_seconds=0.5, expected_model="qwen2.5:7b")


def test_runtime_backend_context_reports_config_backend_fields(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml")
    context = runtime_backend_context(str(config_path))
    assert context["backend_name"] == "openai"
    assert context["backend_url"] == "http://127.0.0.1:11434/v1"
    assert context["resolved_model"] == "qwen2.5:7b"


def test_validate_runtime_backend_fails_when_expected_model_missing_from_catalog(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", model="qwen2.5:7b", backend="openai")

    class _Response:
        status_code = 200

        def json(self) -> dict:
            return {
                "data": [
                    {"id": "llama3.1:8b"},
                    {"id": "gpt-oss:latest"},
                ]
            }

    def _assert_url(url: str, timeout: float) -> object:
        assert url == "http://127.0.0.1:11434/v1/models"
        return _Response()

    monkeypatch.setattr("harness.oneshot.httpx.get", _assert_url)
    with pytest.raises(RuntimeError, match="is not available in catalog"):
        validate_runtime_backend(config_path, timeout_seconds=0.5, expected_model="qwen2.5:7b")


def test_validate_runtime_backend_rejects_non_200_catalog(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", backend="openai")

    class _Response:
        status_code = 503

    def _return(_: str, timeout: float) -> object:
        return _Response()

    monkeypatch.setattr("harness.oneshot.httpx.get", _return)

    with pytest.raises(RuntimeError, match="Model backend catalog check failed"):
        validate_runtime_backend(config_path, timeout_seconds=0.5)


def test_validate_runtime_backend_is_noop_for_unknown_backend_name(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "harness.yaml", backend="llamacpp")

    with pytest.raises(RuntimeError, match="Unsupported backend 'llamacpp'"):
        validate_runtime_backend(config_path, timeout_seconds=0.5)


def test_wait_for_http_ready_succeeds_after_status_200() -> None:
    call_count = {"count": 0}
    state = {"now": 0.0}

    class _Response:
        status_code: int = 503

    def _request(_url: str, _timeout: float) -> object:
        call_count["count"] += 1
        if call_count["count"] < 3:
            return _Response()
        response = _Response()
        response.status_code = 200
        return response

    def _sleep(_seconds: float) -> None:
        state["now"] += 0.1

    def _now() -> float:
        return state["now"]

    assert (
        wait_for_http_ready(
            "http://127.0.0.1:8080/health",
            timeout_seconds=10,
            interval_seconds=0.0,
            request_fn=_request,
            sleep_fn=_sleep,
            now_fn=_now,
        )
        is True
    )


def test_wait_for_http_ready_times_out_and_raises() -> None:
    state = {"now": 0.0}

    def _request(_url: str, _timeout: float) -> object:
        class _Response:
            status_code = 500

        return _Response()

    def _sleep(seconds: float) -> None:
        state["now"] += seconds

    def _now() -> float:
        return state["now"]

    with pytest.raises(TimeoutError, match="Timed out waiting"):
        wait_for_http_ready(
            "http://127.0.0.1:8080/health",
            timeout_seconds=1,
            interval_seconds=0.5,
            request_fn=_request,
            sleep_fn=_sleep,
            now_fn=_now,
        )
