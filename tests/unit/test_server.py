from __future__ import annotations

from harness.server import _build_model_catalog_url, _should_validate_catalog


def test_catalog_validation_applies_to_openai_backends() -> None:
    assert _should_validate_catalog("openai", None) is False
    assert _should_validate_catalog("openai", "model.gguf") is False


def test_catalog_validation_applies_to_ollama_and_nvidia_nim() -> None:
    assert _should_validate_catalog("ollama", None) is True
    assert _should_validate_catalog("nvidia_nim", None) is True


def test_catalog_validation_skips_auto_only_when_llamacpp_model_path_is_present() -> None:
    assert _should_validate_catalog("auto", None) is True
    assert _should_validate_catalog("auto", "models/llama.gguf") is False


def test_catalog_validation_does_not_apply_to_llamacpp() -> None:
    assert _should_validate_catalog("llamacpp", None) is False


def test_build_model_catalog_url_handles_plain_base_url() -> None:
    assert _build_model_catalog_url("http://127.0.0.1:11435") == "http://127.0.0.1:11435/v1/models"


def test_build_model_catalog_url_strips_double_v1() -> None:
    assert _build_model_catalog_url("http://127.0.0.1:11435/v1") == "http://127.0.0.1:11435/v1/models"
