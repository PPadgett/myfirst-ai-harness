"""The reasoning wire-gap fix: route thinking intent must survive the HTTP
boundary (adapter -> provider) instead of dying silently or being guessed
from payload shape."""
from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

from harness.adapters.openai_compatible_client import OpenAICompatibleClient
from harness.types import ModelGenerateRequest


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class _FakeAsyncClient:
    captured: list[dict] = []

    def __init__(self, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def post(self, url, json=None, headers=None):
        self.__class__.captured.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse()


@pytest.fixture()
def fake_httpx(monkeypatch: pytest.MonkeyPatch):
    import harness.adapters.openai_compatible_client as mod

    _FakeAsyncClient.captured = []
    monkeypatch.setattr(mod.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


@pytest.mark.asyncio
async def test_adapter_sends_reasoning_headers_enabled(fake_httpx) -> None:
    client = OpenAICompatibleClient(base_url="http://x/v1", model="m")
    await client.generate(
        ModelGenerateRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            allow_reasoning=True,
            reasoning_budget_tokens=512,
        )
    )
    headers = fake_httpx.captured[0]["headers"]
    assert headers["x-harness-reasoning"] == "1"
    assert headers["x-harness-reasoning-budget"] == "512"
    # reasoning must NOT leak into the body (Lemonade rejects unknown fields)
    assert "reasoning" not in fake_httpx.captured[0]["json"]


@pytest.mark.asyncio
async def test_adapter_sends_reasoning_disabled_header(fake_httpx) -> None:
    client = OpenAICompatibleClient(base_url="http://x/v1", model="m")
    await client.generate(
        ModelGenerateRequest(model="m", messages=[{"role": "user", "content": "hi"}], allow_reasoning=False)
    )
    headers = fake_httpx.captured[0]["headers"]
    assert headers["x-harness-reasoning"] == "0"
    assert "x-harness-reasoning-budget" not in headers
