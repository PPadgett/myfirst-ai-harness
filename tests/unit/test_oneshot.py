from __future__ import annotations

from pathlib import Path

import pytest

from harness.oneshot import (
    build_chat_payload,
    resolve_model,
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
