from __future__ import annotations

from pathlib import Path

from harness.config import load_runtime_config


def test_openai_default_base_url_uses_local_provider_port(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        "\n".join(
            [
                "backend:",
                "  name: openai",
                "  api_key: null",
                "model: local-foundation:v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runtime = load_runtime_config(str(config_path))
    assert runtime.backend.base_url == "http://127.0.0.1:11435/v1"
    assert runtime.backend.name == "openai"


def test_local_openai_alias_default_base_url_uses_local_provider_port(tmp_path: Path) -> None:
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(
        "\n".join(
            [
                "backend:",
                "  name: local_openai",
                "  api_key: null",
                "model: local-foundation:v1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runtime = load_runtime_config(str(config_path))
    assert runtime.backend.base_url == "http://127.0.0.1:11435/v1"
    assert runtime.backend.name == "openai"
