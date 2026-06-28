from __future__ import annotations

from typing import Any

import httpx

from harness.adapters.base import BaseModelClient
from harness.types import ModelGenerateRequest, ModelGenerateResult


class OpenAICompatibleClient(BaseModelClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        reasoning_fields: tuple[str, ...] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.extra_headers = {str(k): str(v) for k, v in (extra_headers or {}).items() if v is not None}
        self.extra_body = extra_body if isinstance(extra_body, dict) else {}
        self.reasoning_fields = reasoning_fields or ("reasoning", "analysis", "reasoning_content")

    async def generate(self, req: ModelGenerateRequest) -> ModelGenerateResult:
        headers: dict[str, str] = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": req.model or self.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_new_tokens,
        }
        if req.tools:
            payload["tools"] = req.tools
        if req.response_schema is not None:
            # Keep a broad compatibility path: most local providers honor JSON mode for schema-constrained flows.
            payload["response_format"] = {"type": "json_object"}
        if req.allow_reasoning:
            # Optional extra body for runtimes that expose reasoning budgets.
            payload["extra_body"] = {
                "reasoning": {"enabled": True},
            }
            if req.reasoning_budget_tokens is not None:
                payload["extra_body"]["reasoning"]["max_tokens"] = req.reasoning_budget_tokens
        if req.extra:
            payload.update(req.extra)
        if self.extra_body:
            payload.update(self.extra_body)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.RequestError as exc:
            raise RuntimeError(f"model_backend_unavailable: failed to reach model backend: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else "unknown"
            raise RuntimeError(f"model_request_failed: backend returned status {status_code}") from exc

        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content", "")
        reasoning = self._extract_reasoning(msg)
        usage = data.get("usage", {})
        return ModelGenerateResult(
            text=text,
            reasoning=reasoning,
            raw=data,
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            },
        )

    def _extract_reasoning(self, message: dict[str, Any]) -> str | None:
        for field in self.reasoning_fields:
            value = message.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
        return None
