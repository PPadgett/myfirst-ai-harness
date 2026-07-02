from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ModelGenerateRequest:
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.2
    max_new_tokens: int = 768
    response_schema: dict[str, Any] | None = None
    allow_reasoning: bool = False
    reasoning_budget_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None
    # Explicit sampling controls. When None, the backend's own defaults apply
    # (which differ per engine - see docs/model-parameters-audit.md). Pin these
    # for reproducible cross-backend comparisons. Note: a fixed seed is
    # necessary but not sufficient for bit-identical outputs; batch composition
    # also affects results (arXiv:2601.06118), so benchmark with concurrency 1.
    seed: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    stop: list[str] | None = None


@dataclass
class ModelGenerateResult:
    text: str
    reasoning: str | None
    raw: dict[str, Any]
    usage: dict[str, int]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None

