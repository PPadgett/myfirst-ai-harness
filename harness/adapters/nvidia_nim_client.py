from __future__ import annotations

from typing import Any

from harness.adapters.openai_compatible_client import OpenAICompatibleClient


class NvidiaNimClient(OpenAICompatibleClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        # NVIDIA NIM exposes OpenAI-compatible REST. We keep a dedicated client
        # so users can pass NIM-specific headers/body without touching other backends.
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            extra_body=extra_body,
            reasoning_fields=("reasoning", "analysis", "reasoning_content"),
        )
